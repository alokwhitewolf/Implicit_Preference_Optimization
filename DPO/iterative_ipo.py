#!/usr/bin/env python3
"""
Iterative IPO: Aligned with paper methodology
Key updates:
1. Category-specific evaluation prompts
2. Qwen model support
3. RewardBench evaluation integration
4. Paper-aligned hyperparameters
"""

import os
import json
import torch
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login, HfFolder
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from config_updates import (
    CATEGORY_EVALUATION_PROMPTS, 
    PAPER_HYPERPARAMETERS,
    get_updated_training_config
)

@dataclass
class IterationMetrics:
    """Metrics for a single iteration of IPO"""
    iteration: int
    train_loss: float
    eval_loss: float
    self_eval_accuracy: float
    cross_dataset_scores: Dict[str, float]
    preference_agreement: float
    response_diversity: float
    category_scores: Dict[str, float]  # New: category-specific scores
    timestamp: str

@dataclass
class ExperimentConfig:
    """Configuration for iterative IPO experiments"""
    model_id: str
    base_dataset: str
    eval_datasets: List[str]
    max_iterations: int = 25  # Increased to test limits
    samples_per_iteration: int = 1000
    early_stopping_patience: int = 5  # More patient to observe degradation
    performance_threshold: float = 0.005  # More sensitive threshold
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    wandb_project: str = "iterative-ipo"
    use_4bit: bool = True  # Paper uses quantization
    use_rewardbench: bool = True  # Evaluate on RewardBench
    
    # New parameters for research objectives
    save_all_checkpoints: bool = False  # Only save best and current models
    cross_dataset_eval: bool = True  # Evaluate on all datasets every iteration
    track_degradation: bool = True  # Detailed degradation tracking
    plateau_detection_window: int = 3  # Window for plateau detection
    forced_iterations: int = None  # Force specific number of iterations (ignore early stopping)
    instruction_batch_size: int = 4  # Number of instructions to batch together for GPU efficiency
    eval_batch_size: int = 16  # Number of evaluations to batch together for cross-dataset evaluation
    
class IterativeIPO:
    """Main class for iterative self-improvement experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.iteration_metrics: List[IterationMetrics] = []
        self.best_performance = 0.0
        self.best_iteration = 0
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for checkpoints and results"""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        # Create subdirectories for essential checkpoints only
        Path(os.path.join(self.config.checkpoint_dir, "current")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.config.checkpoint_dir, "best")).mkdir(parents=True, exist_ok=True)
        
    def authenticate_huggingface(self):
        """Authenticate with Hugging Face using stored token"""
        try:
            # Use the stored token from CLI login without prompting
            token = HfFolder.get_token()
            if token:
                login(token=token)
                print("✓ Successfully authenticated with Hugging Face using stored token")
            else:
                print("Warning: No stored Hugging Face token found")
                print("Run 'huggingface-cli login' first")
        except Exception as e:
            print(f"Warning: Could not authenticate with Hugging Face: {e}")
            print("Make sure you've run 'huggingface-cli login' first")
        
    def load_model_and_tokenizer(self, checkpoint_path: Optional[str] = None):
        """Load model and tokenizer with paper-aligned configuration"""
        # Authenticate with Hugging Face first
        self.authenticate_huggingface()
        
        model_path = checkpoint_path or self.config.model_id
        
        # Quantization config from paper
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Check if flash-attention is available
        try:
            import flash_attn
            attn_implementation = "flash_attention_2" if torch.cuda.is_available() else None
            print("✓ Flash attention available")
        except ImportError:
            attn_implementation = None
            print("⚠️ Flash attention not available, using standard attention")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            max_memory={0: "35GB"},  # Force using more GPU memory
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Handle tokenizer configuration
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Important for batch generation
        
        # Prepare model for training if using quantization
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)
            
        return model, tokenizer
    
    def run_sft_training(self, model, tokenizer, iteration: int):
        """SFT training on Dolly-15k before DPO (for base models only) - Paper Section 4.3"""
        # Only run SFT for base models (not instruct models)
        if "Instruct" in self.config.model_id or "instruct" in self.config.model_id.lower():
            print("✓ Skipping SFT - using pre-trained instruct model")
            return model
        
        print(f"🔧 Running SFT training on Dolly-15k (Iteration {iteration})...")
        
        # Load Dolly-15k dataset for SFT (as specified in paper)
        dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        print(f"✓ Loaded Dolly-15k with {len(dolly_dataset)} examples")
        
        # Format dataset for SFT training
        def format_dolly_for_sft(example):
            if "mistral" in self.config.model_id.lower():
                formatted_text = f"[INST] {example['instruction']} [/INST] {example['response']}"
            else:
                # Use chat template for other models
                messages = [
                    {"role": "user", "content": example['instruction']},
                    {"role": "assistant", "content": example['response']}
                ]
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            return {"text": formatted_text}
        
        sft_dataset = dolly_dataset.map(format_dolly_for_sft)
        
        # Split into train/eval
        sft_split = sft_dataset.train_test_split(test_size=0.1, seed=42)
        
        # SFT configuration (from paper appendix)
        sft_checkpoint_path = os.path.join(self.config.checkpoint_dir, f"sft_iteration_{iteration}")
        
        # LoRA configuration for SFT - match ours.py configuration  
        sft_peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # SFT training arguments - match ours.py configuration
        from transformers import TrainingArguments
        sft_args = TrainingArguments(
            output_dir=sft_checkpoint_path,
            num_train_epochs=3,  # Keep original 3 epochs for SFT
            per_device_train_batch_size=6,  # Match ours.py batch size
            per_device_eval_batch_size=3,  # Match ours.py batch size
            learning_rate=5e-5,  # Match ours.py learning rate
            optim="adamw_torch_fused",  # Match ours.py optimizer
            max_grad_norm=0.3,  # Match ours.py
            lr_scheduler_type="cosine",  # Match ours.py
            logging_steps=25,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            gradient_checkpointing=True,
            warmup_ratio=0.1,
            bf16=True,
            remove_unused_columns=False,
            report_to="wandb" if wandb.run else "none",
            run_name=f"sft_iteration_{iteration}",
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, sft_peft_config)
        model.print_trainable_parameters()
        
        # Create SFT trainer
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=sft_split["train"],
            eval_dataset=sft_split["test"],
            tokenizer=tokenizer,
            max_seq_length=1024,
        )
        
        # Train
        print("🚀 Starting SFT training...")
        sft_trainer.train()
        
        # Save SFT model
        sft_trainer.save_model(sft_checkpoint_path)
        tokenizer.save_pretrained(sft_checkpoint_path)
        
        print(f"✓ SFT training completed. Model saved to {sft_checkpoint_path}")
        
        # Return the SFT-trained model (still has LoRA adapters)
        return model
    
    def detect_category(self, instruction: str) -> str:
        """Detect instruction category for appropriate evaluation prompt"""
        instruction_lower = instruction.lower()
        
        if any(keyword in instruction_lower for keyword in ['code', 'function', 'program', 'script', 'debug']):
            return 'code'
        elif any(keyword in instruction_lower for keyword in ['calculate', 'solve', 'equation', 'math', 'number']):
            return 'math'
        elif any(keyword in instruction_lower for keyword in ['safe', 'ethical', 'harmful', 'dangerous']):
            return 'safety'
        elif any(keyword in instruction_lower for keyword in ['reason', 'explain why', 'logic', 'deduce']):
            return 'reasoning'
        else:
            return 'chat'
    
    def generate_responses(self, model, tokenizer, instruction: str, num_responses: int = 4):
        """Generate multiple responses with memory-efficient batching"""
        # Format instruction with proper template
        if "mistral" in self.config.model_id.lower():
            prompt = f"[INST] {instruction} [/INST]"
        else:
            # Use chat template for other models
            messages = [{"role": "user", "content": instruction}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        responses = []
        
        # Try batch generation first, fallback to sequential if OOM
        try:
            # Batch generation - try all 4 responses at once
            batch_size = num_responses  # Try all 4 at once
            for i in range(0, num_responses, batch_size):
                current_batch_size = min(batch_size, num_responses - i)
                prompts = [prompt] * current_batch_size
                inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=PAPER_HYPERPARAMETERS["max_new_tokens"],
                        temperature=PAPER_HYPERPARAMETERS["temperature"],
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode responses for this batch
                for output in outputs:
                    response = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    responses.append(response.strip())
                
                # Clear memory between batches
                del inputs, outputs
                torch.cuda.empty_cache()
                
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Fallback to sequential generation
            print("Batch generation failed, falling back to sequential...")
            responses = []
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            for _ in range(num_responses):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=PAPER_HYPERPARAMETERS["max_new_tokens"],
                        temperature=PAPER_HYPERPARAMETERS["temperature"],
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                responses.append(response.strip())
        
        return responses
    
    def evaluate_response(self, model, tokenizer, instruction: str, response: str, category: Optional[str] = None) -> float:
        """Evaluate response using category-specific prompt"""
        if category is None:
            category = self.detect_category(instruction)
        
        eval_prompt = CATEGORY_EVALUATION_PROMPTS[category].format(
            instruction=instruction,
            response=response
        )
        
        inputs = tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            
            # Use ours.py approach - more robust to tokenizer differences
            yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
            no_tokens = tokenizer.encode(" No", add_special_tokens=False)
            
            # Get full probability distribution
            probs = torch.softmax(logits, dim=-1)
            
            # Sum probabilities across all tokens for "Yes" and "No"
            yes_prob = sum(probs[token_id].item() for token_id in yes_tokens if token_id < len(probs))
            no_prob = sum(probs[token_id].item() for token_id in no_tokens if token_id < len(probs))
            
            # Normalize only over Yes/No probabilities (ours.py approach)
            total_prob = yes_prob + no_prob
            if total_prob > 0:
                yes_prob = yes_prob / total_prob
            # If total_prob == 0, yes_prob remains 0 (original ours.py behavior)
            
        return yes_prob
    
    def evaluate_responses_batch(self, model, tokenizer, instruction: str, responses: List[str], category: Optional[str] = None) -> List[float]:
        """Batch evaluate multiple responses for better GPU utilization"""
        if category is None:
            category = self.detect_category(instruction)
        
        # Create batch of evaluation prompts
        eval_prompts = []
        for response in responses:
            eval_prompt = CATEGORY_EVALUATION_PROMPTS[category].format(
                instruction=instruction,
                response=response
            )
            eval_prompts.append(eval_prompt)
        
        # Batch tokenization
        inputs = tokenizer(eval_prompts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        scores = []
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1]  # Get last token logits for all samples
            
            # Use ours.py approach - more robust to tokenizer differences
            yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
            no_tokens = tokenizer.encode(" No", add_special_tokens=False)
            
            # Debug: Print token info only once per evaluation
            if not hasattr(self, '_debug_printed'):
                print(f"Debug: 'Yes' tokens: {yes_tokens}, 'No' tokens: {no_tokens}")
                print(f"Debug: Tokenizer vocab: {tokenizer.vocab_size}, Model vocab: {logits.shape[-1]}")
                self._debug_printed = True
            
            # Process each sample in the batch using ours.py approach
            for i in range(len(responses)):
                # Get full probability distribution for this sample
                probs = torch.softmax(logits[i], dim=-1)
                
                # Sum probabilities across all tokens for "Yes" and "No"
                yes_prob = sum(probs[token_id].item() for token_id in yes_tokens if token_id < len(probs))
                no_prob = sum(probs[token_id].item() for token_id in no_tokens if token_id < len(probs))
                
                # Normalize only over Yes/No probabilities (ours.py approach)
                total_prob = yes_prob + no_prob
                if total_prob > 0:
                    yes_prob = yes_prob / total_prob
                # If total_prob == 0, yes_prob remains 0 (original ours.py behavior)
                
                scores.append(yes_prob)
        
        return scores
    
    def generate_self_preferences(self, model, tokenizer, dataset, iteration: int):
        """Generate preference pairs with instruction-level batching for better GPU utilization"""
        preferences = []
        category_counts = {'code': 0, 'math': 0, 'chat': 0, 'safety': 0, 'reasoning': 0}
        
        # Get batch size from config (can be overridden by command line)
        instruction_batch_size = getattr(self.config, 'instruction_batch_size', PAPER_HYPERPARAMETERS.get("instruction_batch_size", 4))
        num_responses = PAPER_HYPERPARAMETERS.get("num_responses", 4)
        
        # Process instructions in batches
        dataset_list = list(dataset)
        total_batches = (len(dataset_list) + instruction_batch_size - 1) // instruction_batch_size
        
        for batch_start in tqdm(range(0, len(dataset_list), instruction_batch_size), 
                               desc=f"Generating preferences (Iteration {iteration})", 
                               total=total_batches):
            
            batch_end = min(batch_start + instruction_batch_size, len(dataset_list))
            instruction_batch = dataset_list[batch_start:batch_end]
            
            # Try batched approach first, fallback to sequential if OOM
            try:
                batch_preferences = self._process_instruction_batch(
                    model, tokenizer, instruction_batch, num_responses, category_counts
                )
                preferences.extend(batch_preferences)
                
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"Batch processing failed ({e}), falling back to sequential processing...")
                # Fallback to sequential processing for this batch
                for example in instruction_batch:
                    instruction = example.get('instruction') or example.get('prompt', '')
                    category = self.detect_category(instruction)
                    category_counts[category] += 1
                    
                    # Generate responses sequentially
                    responses = self.generate_responses(model, tokenizer, instruction, num_responses)
                    scores = self.evaluate_responses_batch(model, tokenizer, instruction, responses, category)
                    
                    # Create preference pair
                    best_idx = np.argmax(scores)
                    worst_idx = np.argmin(scores)
                    
                    if best_idx != worst_idx and scores[best_idx] - scores[worst_idx] > 0.1:
                        preferences.append({
                            'instruction': instruction,
                            'chosen': responses[best_idx],
                            'rejected': responses[worst_idx],
                            'score_diff': scores[best_idx] - scores[worst_idx],
                            'category': category,
                            'chosen_score': scores[best_idx],
                            'rejected_score': scores[worst_idx]
                        })
        
        print(f"Category distribution: {category_counts}")
        print(f"📊 Batched processing: {instruction_batch_size} instructions × {num_responses} responses = {instruction_batch_size * num_responses} responses per batch")
        return Dataset.from_list(preferences)
    
    def transform_entry(self, entry):
        """Transform preference pair to DPO format (from original ours.py)"""
        return {
            "prompt": f"[INST] {entry['instruction']} [/INST]",
            "chosen": f"{entry['chosen']}<|eot_id|>",
            "rejected": f"{entry['rejected']}<|eot_id|>"
        }
    
    def prepare_dpo_data(self, preferences: Dataset, iteration: int):
        """Prepare DPO data in memory (no disk saving) for training"""
        # Convert Dataset to list for processing
        preferences_list = preferences.to_list() if hasattr(preferences, 'to_list') else list(preferences)
        
        # Transform to DPO format (prompt, chosen, rejected)
        transformed_data = []
        for entry in preferences_list:
            transformed = self.transform_entry(entry)
            transformed_data.append(transformed)
        
        # Split into train/test (90/10)
        train_size = int(0.9 * len(transformed_data))
        train_data = transformed_data[:train_size]
        test_data = transformed_data[train_size:]
        
        print(f"📊 Prepared DPO data: {len(train_data)} train, {len(test_data)} test examples")
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(test_data)
        
        return train_dataset, eval_dataset
    
    def _process_instruction_batch(self, model, tokenizer, instruction_batch, num_responses, category_counts):
        """Process a batch of instructions simultaneously for maximum GPU efficiency"""
        # Collect all prompts for this batch (num_responses per instruction)
        all_prompts = []
        instruction_metadata = []  # Track which responses belong to which instruction
        
        for batch_idx, example in enumerate(instruction_batch):
            instruction = example.get('instruction') or example.get('prompt', '')
            category = self.detect_category(instruction)
            category_counts[category] += 1
            
            # Format prompt according to model type
            if "mistral" in self.config.model_id.lower():
                prompt = f"[INST] {instruction} [/INST]"
            else:
                messages = [{"role": "user", "content": instruction}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Add num_responses copies of this prompt to the batch
            for response_idx in range(num_responses):
                all_prompts.append(prompt)
                instruction_metadata.append({
                    'batch_idx': batch_idx,
                    'response_idx': response_idx,
                    'instruction': instruction,
                    'category': category
                })
        
        # Generate ALL responses in one massive batch (instruction_batch_size × num_responses)
        print(f"🚀 Generating {len(all_prompts)} responses in single batch...")
        
        inputs = tokenizer(all_prompts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                # Clear CUDA cache before generation
                torch.cuda.empty_cache()
                
                # Add validation and safer generation parameters
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=PAPER_HYPERPARAMETERS["max_new_tokens"],
                    temperature=max(0.1, PAPER_HYPERPARAMETERS["temperature"]),  # Ensure min temperature
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,  # Add top_k for stability
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetition loops
                )
        except Exception as e:
            print(f"Batch generation failed: {e}")
            torch.cuda.empty_cache()  # Clear memory on error
            raise
        
        # Decode all responses
        all_responses = []
        for output in outputs:
            response = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            all_responses.append(response.strip())
        
        # Clear generation memory immediately
        del inputs, outputs
        torch.cuda.empty_cache()
        
        # Group responses back by instruction and evaluate each instruction's responses
        batch_preferences = []
        
        for inst_idx in range(len(instruction_batch)):
            # Get the num_responses responses for this instruction
            start_idx = inst_idx * num_responses
            end_idx = start_idx + num_responses
            responses = all_responses[start_idx:end_idx]
            
            # Get metadata for this instruction
            instruction = instruction_metadata[start_idx]['instruction']
            category = instruction_metadata[start_idx]['category']
            
            # Batch evaluate these responses
            scores = self.evaluate_responses_batch(model, tokenizer, instruction, responses, category)
            
            # Create preference pair
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)
            
            if best_idx != worst_idx and scores[best_idx] - scores[worst_idx] > 0.1:
                batch_preferences.append({
                    'instruction': instruction,
                    'chosen': responses[best_idx],
                    'rejected': responses[worst_idx],
                    'score_diff': scores[best_idx] - scores[worst_idx],
                    'category': category,
                    'chosen_score': scores[best_idx],
                    'rejected_score': scores[worst_idx]
                })
        
        return batch_preferences
    
    def train_iteration(self, model, tokenizer, train_dataset, eval_dataset, iteration: int):
        """Train one iteration with paper-aligned configuration"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration}")
        
        # LoRA configuration - match ours.py configuration
        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # DPO Training configuration - match ours.py exactly
        training_args = DPOConfig(
            output_dir=checkpoint_path,
            num_train_epochs=1,  # Match ours.py default
            dataloader_num_workers=4,  # Match ours.py
            per_device_train_batch_size=6,  # Match ours.py
            per_device_eval_batch_size=3,  # Match ours.py
            gradient_accumulation_steps=1,  # Match ours.py
            gradient_checkpointing=True,  # Match ours.py
            optim="adamw_torch_fused",  # Match ours.py
            learning_rate=5e-5,  # Match ours.py
            max_grad_norm=0.3,  # Match ours.py
            warmup_ratio=0.1,  # Match ours.py
            lr_scheduler_type="cosine",  # Match ours.py
            logging_steps=25,  # Match ours.py
            save_steps=500,  # Match ours.py
            save_total_limit=2,  # Match ours.py
            evaluation_strategy="steps",  # Match ours.py
            eval_steps=300,  # Match ours.py
            bf16=True,  # Match ours.py
            push_to_hub=False,  # Match ours.py
            beta=0.1,  # Match ours.py DPO beta
            max_length=1024,  # Keep DPO-specific params
            max_completion_length=256,
            max_prompt_length=512,
            report_to="wandb" if wandb.run else "none",
            run_name=f"ipo_iteration_{iteration}",
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # The dataset is already in DPO format from save_preference_data()
        # No need to reformat - it already has prompt, chosen, rejected keys
        print(f"✓ DPO dataset format: {train_dataset.column_names}")
        print(f"✓ Training on {len(train_dataset)} preference pairs")
        
        # Create trainer
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        
        # Train
        train_result = trainer.train()
        
        # Clear cache before evaluation
        torch.cuda.empty_cache()
        
        # Get metrics with memory management
        try:
            # Try evaluation with memory safety
            if hasattr(self.config, 'skip_dpo_eval') and self.config.skip_dpo_eval:
                print("⚠️ Skipping DPO evaluation due to memory constraints")
                eval_results = {"eval_loss": 0.0}
            else:
                eval_results = trainer.evaluate()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ Evaluation OOM - skipping evaluation this iteration")
                torch.cuda.empty_cache()
                eval_results = {"eval_loss": 0.0}
            else:
                raise e
        
        # Save model and tokenizer only if selective saving is enabled
        if getattr(self.config, 'save_all_checkpoints', True):
            # Original behavior - save all checkpoints
            trainer.save_model(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
        else:
            # Selective saving - only save current model temporarily for evaluation
            # We'll determine if it's the best later and save accordingly
            temp_checkpoint_path = os.path.join(self.config.checkpoint_dir, "temp_current")
            trainer.save_model(temp_checkpoint_path)
            tokenizer.save_pretrained(temp_checkpoint_path)
        
        # Clear trainer memory immediately after saving
        del trainer
        torch.cuda.empty_cache()
        
        return train_result.training_loss, eval_results.get('eval_loss', 0)
    
    def evaluate_on_rewardbench(self, model, tokenizer) -> Dict[str, float]:
        """Evaluate on RewardBench subsets"""
        if not self.config.use_rewardbench:
            return {}
        
        try:
            print("Loading RewardBench dataset...")
            # Load RewardBench dataset
            rewardbench = load_dataset("allenai/reward-bench", split="filtered")
            print(f"✓ Loaded RewardBench with {len(rewardbench)} examples")
            
            subset_scores = {}
            for subset in ['alpacaeval-easy', 'chat_easy', 'math-prm']:  # Sample subsets
                subset_data = rewardbench.filter(lambda x: x['subset'] == subset)
                if len(subset_data) == 0:
                    continue
                    
                correct = 0
                total = min(len(subset_data), 100)  # Limit evaluation size
                
                for example in subset_data.select(range(total)):
                    # Handle different field names in RewardBench dataset
                    instruction = example.get('instruction') or example.get('prompt') or example.get('question', '')
                    chosen = example.get('chosen', '')
                    rejected = example.get('rejected', '')
                    
                    if not instruction or not chosen or not rejected:
                        continue
                        
                    chosen_score = self.evaluate_response(
                        model, tokenizer, 
                        instruction, 
                        chosen
                    )
                    rejected_score = self.evaluate_response(
                        model, tokenizer,
                        instruction,
                        rejected
                    )
                    
                    if chosen_score > rejected_score:
                        correct += 1
                
                subset_scores[subset] = correct / total if total > 0 else 0
                
            return subset_scores
            
        except Exception as e:
            print(f"RewardBench evaluation failed: {e}")
            print(f"Skipping RewardBench evaluation and continuing...")
            return {}
    
    def run_experiment(self):
        """Run the full iterative IPO experiment with paper alignment
        
        Paper Implementation Flow (Section 4.3):
        1. Base Model → SFT on Dolly-15k (for base models only)
        2. Generate 4 responses per instruction from UltraFeedback
        3. Self-evaluate responses using category-specific Yes/No prompts
        4. Create preference pairs (best vs worst based on Yes probability)
        5. Save preference data to disk in DPO format
        6. Load saved data for DPO training
        7. Train model using DPO on self-generated preferences
        8. Repeat for multiple iterations
        """
        # Initialize wandb with paper-specific tags
        wandb.init(
            project=self.config.wandb_project,
            config=asdict(self.config),
            tags=["iterative-ipo", "paper-aligned", self.config.model_id.split("/")[-1]]
        )
        
        # Load evaluation datasets
        eval_datasets = {}
        # Load evaluation datasets with correct repository names
        dataset_configs = {
            "truthful_qa": {"repo": "domenicrosati/TruthfulQA", "config": "default", "split": "train[:500]"},
            "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "test[:500]"},
            "hellaswag": {"repo": "Rowan/hellaswag", "config": None, "split": "validation[:500]"},
        }
        
        for dataset_name in self.config.eval_datasets:
            try:
                config_info = dataset_configs.get(dataset_name, {"repo": dataset_name, "config": None, "split": "test[:500]"})
                repo_name = config_info.get("repo", dataset_name)
                
                print(f"Loading {dataset_name} from {repo_name}...")
                
                # Load dataset with proper config
                if config_info["config"]:
                    eval_datasets[dataset_name] = load_dataset(
                        repo_name, 
                        config_info["config"], 
                        split=config_info["split"],
                        trust_remote_code=True
                    )
                else:
                    eval_datasets[dataset_name] = load_dataset(
                        repo_name, 
                        split=config_info["split"],
                        trust_remote_code=True
                    )
                    
                print(f"✓ Loaded {dataset_name} with {len(eval_datasets[dataset_name])} examples")
                
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")
                print(f"Skipping {dataset_name}...")
        
        # Load base training dataset
        # For UltraFeedback-1k-Each, use split_1; for other datasets, use train split
        if "UltraFeedback-1k-Each" in self.config.base_dataset:
            base_dataset = load_dataset(self.config.base_dataset, split="split_1")
            print(f"✓ Using balanced UltraFeedback dataset (split_1) with 1k examples per category")
        else:
            base_dataset = load_dataset(self.config.base_dataset, split="train")
            print(f"✓ Using dataset: {self.config.base_dataset}")
        
        previous_prefs = None
        checkpoint_path = None
        
        for iteration in range(self.config.max_iterations):
            print(f"\n{'='*50}")
            print(f"Starting Iteration {iteration + 1}/{self.config.max_iterations}")
            print(f"{'='*50}")
            
            # Clear CUDA cache before loading new model
            torch.cuda.empty_cache()
            
            # Step 1: Load model (base or from previous iteration)
            model, tokenizer = self.load_model_and_tokenizer(checkpoint_path)
            
            # Step 2: SFT training (only for base models on first iteration)
            if iteration == 0:
                model = self.run_sft_training(model, tokenizer, iteration + 1)
            
            # Step 3: Generate self-preferences from UltraFeedback data
            train_prefs = self.generate_self_preferences(
                model, tokenizer, 
                base_dataset.select(range(self.config.samples_per_iteration)),
                iteration + 1
            )
            
            # Step 4: Prepare DPO data (in memory, no permanent disk saving)
            train_dataset, eval_dataset = self.prepare_dpo_data(train_prefs, iteration + 1)
            
            # Category distribution analysis
            category_dist = {}
            for cat in ['code', 'math', 'chat', 'safety', 'reasoning']:
                cat_count = len(train_prefs.filter(lambda x: x['category'] == cat))
                category_dist[cat] = cat_count / len(train_prefs) if len(train_prefs) > 0 else 0
            
            # Step 6: DPO training on self-generated preferences
            train_loss, eval_loss = self.train_iteration(
                model, tokenizer, train_dataset, eval_dataset, iteration + 1
            )
            
            # Evaluate performance on external datasets
            cross_dataset_scores = self.evaluate_cross_dataset(model, tokenizer, eval_datasets)
            rewardbench_scores = self.evaluate_on_rewardbench(model, tokenizer)
            self_eval_accuracy = np.mean(list(cross_dataset_scores.values())) if cross_dataset_scores else 0.5
            
            # Category-specific evaluation (using training preference data)
            category_scores = {}
            for category in ['code', 'math', 'chat', 'safety', 'reasoning']:
                cat_data = train_prefs.filter(lambda x: x['category'] == category)
                if len(cat_data) > 0:
                    cat_accuracy = np.mean([
                        1 if ex['chosen_score'] > ex['rejected_score'] else 0 
                        for ex in cat_data
                    ])
                    category_scores[category] = cat_accuracy
            
            # Calculate additional metrics
            preference_agreement = self.calculate_preference_agreement(train_prefs, previous_prefs)
            response_diversity = self.calculate_response_diversity(train_prefs)
            
            # Record metrics
            metrics = IterationMetrics(
                iteration=iteration + 1,
                train_loss=train_loss,
                eval_loss=eval_loss,
                self_eval_accuracy=self_eval_accuracy,
                cross_dataset_scores=cross_dataset_scores,
                preference_agreement=preference_agreement,
                response_diversity=response_diversity,
                category_scores=category_scores,
                timestamp=datetime.now().isoformat()
            )
            self.iteration_metrics.append(metrics)
            
            # Log to wandb
            wandb.log({
                "iteration": iteration + 1,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "self_eval_accuracy": self_eval_accuracy,
                "preference_agreement": preference_agreement,
                "response_diversity": response_diversity,
                **{f"cross_dataset/{k}": v for k, v in cross_dataset_scores.items()},
                **{f"rewardbench/{k}": v for k, v in rewardbench_scores.items()},
                **{f"category/{k}": v for k, v in category_scores.items()},
                **{f"category_dist/{k}": v for k, v in category_dist.items()}
            })
            
            # Save results
            self.save_results()
            
            # Handle selective model saving after evaluation
            if not getattr(self.config, 'save_all_checkpoints', True):
                self._handle_selective_saving(self_eval_accuracy, iteration + 1)
            
            # Update for next iteration
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration + 1}")
            previous_prefs = train_prefs
            
            # Check early stopping
            if self.should_stop_early():
                print(f"Stopping early at iteration {iteration + 1}")
                break
            
            # Clean up memory
            del model
            torch.cuda.empty_cache()
        
        # Generate final visualizations
        self.generate_plots()
        wandb.finish()
    
    def evaluate_cross_dataset(self, model, tokenizer, eval_datasets: Dict[str, Dataset]) -> Dict[str, float]:
        """Evaluate model performance across multiple datasets with batching for efficiency"""
        scores = {}
        eval_batch_size = getattr(self.config, 'eval_batch_size', PAPER_HYPERPARAMETERS.get("eval_batch_size", 16))
        
        for dataset_name, dataset in eval_datasets.items():
            try:
                # Use batched evaluation for efficiency
                accuracy = self._evaluate_dataset_batched(
                    model, tokenizer, dataset_name, dataset[:100], eval_batch_size
                )
                scores[dataset_name] = accuracy
                
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"Batched evaluation failed for {dataset_name} ({e}), falling back to sequential...")
                # Fallback to sequential evaluation
                scores[dataset_name] = self._evaluate_dataset_sequential(
                    model, tokenizer, dataset_name, dataset[:100]
                )
        
        return scores
    
    def _evaluate_dataset_batched(self, model, tokenizer, dataset_name: str, dataset, batch_size: int) -> float:
        """Batch evaluate a dataset for maximum GPU efficiency"""
        # Collect all evaluation pairs
        eval_pairs = []
        
        for example in dataset:
            # Handle different dataset formats
            if isinstance(example, str):
                continue  # Skip string-only examples
            
            instruction = example.get('question') or example.get('instruction') or example.get('prompt', '')
            
            # Handle TruthfulQA format
            if 'Best Answer' in example and 'Incorrect Answers' in example and instruction:
                incorrect_answers = example['Incorrect Answers'].split(';') if example['Incorrect Answers'] else []
                if incorrect_answers:
                    eval_pairs.append({
                        'instruction': instruction,
                        'chosen': example['Best Answer'],
                        'rejected': incorrect_answers[0].strip()  # Use first incorrect answer
                    })
            # Handle standard preference format
            elif 'chosen' in example and 'rejected' in example and instruction:
                eval_pairs.append({
                    'instruction': instruction,
                    'chosen': example['chosen'],
                    'rejected': example['rejected']
                })
        
        if not eval_pairs:
            return 0.0
        
        print(f"🚀 Batch evaluating {len(eval_pairs)} preference pairs for {dataset_name}")
        
        # Process evaluation pairs in batches
        correct = 0
        total = len(eval_pairs)
        
        for batch_start in tqdm(range(0, len(eval_pairs), batch_size), 
                               desc=f"Evaluating {dataset_name}"):
            
            batch_end = min(batch_start + batch_size, len(eval_pairs))
            batch_pairs = eval_pairs[batch_start:batch_end]
            
            # Collect all evaluation prompts for this batch
            chosen_prompts = []
            rejected_prompts = []
            instructions = []
            categories = []
            
            for pair in batch_pairs:
                instruction = pair['instruction']
                category = self.detect_category(instruction)
                
                instructions.append(instruction)
                categories.append(category)
                
                # Create evaluation prompts for chosen and rejected responses
                chosen_eval_prompt = CATEGORY_EVALUATION_PROMPTS[category].format(
                    instruction=instruction,
                    response=pair['chosen']
                )
                rejected_eval_prompt = CATEGORY_EVALUATION_PROMPTS[category].format(
                    instruction=instruction,
                    response=pair['rejected']
                )
                
                chosen_prompts.append(chosen_eval_prompt)
                rejected_prompts.append(rejected_eval_prompt)
            
            # Batch evaluate all chosen responses
            chosen_scores = self._batch_evaluate_prompts(model, tokenizer, chosen_prompts)
            
            # Batch evaluate all rejected responses  
            rejected_scores = self._batch_evaluate_prompts(model, tokenizer, rejected_prompts)
            
            # Compare scores for this batch
            for chosen_score, rejected_score in zip(chosen_scores, rejected_scores):
                if chosen_score > rejected_score:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"📊 {dataset_name}: {correct}/{total} = {accuracy:.3f} accuracy")
        return accuracy
    
    def _batch_evaluate_prompts(self, model, tokenizer, eval_prompts: List[str]) -> List[float]:
        """Batch evaluate a list of evaluation prompts"""
        if not eval_prompts:
            return []
        
        # Batch tokenization
        inputs = tokenizer(eval_prompts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        scores = []
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1]  # Get last token logits for all samples
            
            # Use ours.py approach - more robust to tokenizer differences
            yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
            no_tokens = tokenizer.encode(" No", add_special_tokens=False)
            
            # Process each sample in the batch
            for i in range(len(eval_prompts)):
                # Get full probability distribution for this sample
                probs = torch.softmax(logits[i], dim=-1)
                
                # Sum probabilities across all tokens for "Yes" and "No"
                yes_prob = sum(probs[token_id].item() for token_id in yes_tokens if token_id < len(probs))
                no_prob = sum(probs[token_id].item() for token_id in no_tokens if token_id < len(probs))
                
                # Normalize only over Yes/No probabilities (ours.py approach)
                total_prob = yes_prob + no_prob
                if total_prob > 0:
                    yes_prob = yes_prob / total_prob
                # If total_prob == 0, yes_prob remains 0 (original ours.py behavior)
                
                scores.append(yes_prob)
        
        # Clear memory
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return scores
    
    def _evaluate_dataset_sequential(self, model, tokenizer, dataset_name: str, dataset) -> float:
        """Fallback sequential evaluation if batching fails"""
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc=f"Evaluating {dataset_name} (sequential)"):
            # Handle different dataset formats
            if isinstance(example, str):
                continue
                
            instruction = example.get('question') or example.get('instruction') or example.get('prompt', '')
            
            # Handle TruthfulQA format
            if 'Best Answer' in example and 'Incorrect Answers' in example and instruction:
                incorrect_answers = example['Incorrect Answers'].split(';') if example['Incorrect Answers'] else []
                if incorrect_answers:
                    chosen_score = self.evaluate_response(
                        model, tokenizer, instruction, example['Best Answer']
                    )
                    rejected_score = self.evaluate_response(
                        model, tokenizer, instruction, incorrect_answers[0].strip()
                    )
                    
                    if chosen_score > rejected_score:
                        correct += 1
                    total += 1
                    
            # Handle standard preference format
            elif 'chosen' in example and 'rejected' in example and instruction:
                chosen_score = self.evaluate_response(
                    model, tokenizer, instruction, example['chosen']
                )
                rejected_score = self.evaluate_response(
                    model, tokenizer, instruction, example['rejected']
                )
                
                if chosen_score > rejected_score:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def calculate_preference_agreement(self, current_prefs: Dataset, previous_prefs: Dataset) -> float:
        """Calculate agreement between current and previous preference rankings"""
        if previous_prefs is None or len(previous_prefs) == 0:
            return 1.0
        
        try:
            agreements = []
            # Convert datasets to lists for easier comparison
            current_list = current_prefs.to_list() if hasattr(current_prefs, 'to_list') else list(current_prefs)
            previous_list = previous_prefs.to_list() if hasattr(previous_prefs, 'to_list') else list(previous_prefs)
            
            # Limit to first 100 for performance
            current_sample = current_list[:100]
            
            for curr in current_sample:
                # Find matching instruction in previous preferences
                matching = [p for p in previous_list if p.get('instruction') == curr.get('instruction')]
                if matching:
                    prev = matching[0]
                    # Check if preference order is maintained (considering score differences)
                    curr_prefers_chosen = curr.get('chosen_score', 0) > curr.get('rejected_score', 0)
                    prev_prefers_chosen = prev.get('chosen_score', 0) > prev.get('rejected_score', 0)
                    agreements.append(float(curr_prefers_chosen == prev_prefers_chosen))
            
            return np.mean(agreements) if agreements else 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate preference agreement: {e}")
            return 0.0
    
    def calculate_response_diversity(self, dataset: Dataset) -> float:
        """Calculate diversity of generated responses"""
        from collections import Counter
        import string
        
        all_responses = []
        for example in dataset[:100]:
            # Handle both dict and string formats
            if isinstance(example, dict):
                all_responses.extend([example['chosen'], example['rejected']])
            # Skip if example is not a dict (shouldn't happen in our preference dataset)
        
        # Calculate unique n-grams
        all_ngrams = []
        for response in all_responses:
            # Clean and tokenize
            words = response.lower().translate(str.maketrans('', '', string.punctuation)).split()
            # Generate 3-grams
            ngrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            all_ngrams.extend(ngrams)
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        # Also calculate average response length variance
        lengths = [len(r.split()) for r in all_responses]
        length_variance = np.var(lengths) if lengths else 0
        
        diversity_score = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        
        # Combine diversity metrics
        return diversity_score * (1 + min(length_variance / 1000, 1))  # Weighted by length variance
    
    def should_stop_early(self) -> bool:
        """Enhanced early stopping with detailed degradation tracking for research"""
        # If forced iterations specified, ignore early stopping
        if self.config.forced_iterations and len(self.iteration_metrics) < self.config.forced_iterations:
            return False
            
        if len(self.iteration_metrics) < self.config.early_stopping_patience:
            return False
        
        # For research: Track but don't stop early unless explicitly degrading
        if not self.config.track_degradation:
            return False
        
        recent_metrics = self.iteration_metrics[-self.config.early_stopping_patience:]
        all_metrics = self.iteration_metrics
        
        # Plateau detection with configurable window
        window = self.config.plateau_detection_window
        if len(all_metrics) >= window:
            recent_scores = [m.self_eval_accuracy for m in all_metrics[-window:]]
            if np.std(recent_scores) < self.config.performance_threshold:
                print(f"Plateau detected: Performance variance {np.std(recent_scores):.4f} < threshold {self.config.performance_threshold}")
                # For research: continue to observe degradation
                if len(all_metrics) >= self.config.max_iterations * 0.8:  # Only stop if near max iterations
                    return True
        
        # Severe degradation detection (>20% drop from peak)
        if len(all_metrics) >= 3:
            peak_performance = max([m.self_eval_accuracy for m in all_metrics])
            current_performance = all_metrics[-1].self_eval_accuracy
            degradation_ratio = (peak_performance - current_performance) / peak_performance
            
            if degradation_ratio > 0.2:  # 20% degradation
                print(f"Severe degradation detected: {degradation_ratio:.1%} drop from peak {peak_performance:.3f}")
                return True
        
        # Catastrophic forgetting detection (enhanced)
        for category in ['code', 'math', 'chat']:
            if len(all_metrics) >= 3:
                category_scores = [m.category_scores.get(category, 0) for m in all_metrics]
                if any(s > 0 for s in category_scores):  # Only check if category has data
                    peak_score = max(category_scores)
                    current_score = category_scores[-1]
                    if peak_score > 0 and (peak_score - current_score) / peak_score > 0.5:  # 50% drop
                        print(f"Catastrophic forgetting in {category}: {current_score:.3f} vs peak {peak_score:.3f}")
                        return True
        
        # Cross-dataset performance degradation
        if len(all_metrics) >= 3:
            for dataset_name in all_metrics[0].cross_dataset_scores.keys():
                dataset_scores = [m.cross_dataset_scores.get(dataset_name, 0) for m in all_metrics]
                if len(dataset_scores) >= 3:
                    peak_dataset_score = max(dataset_scores)
                    current_dataset_score = dataset_scores[-1]
                    if peak_dataset_score > 0 and (peak_dataset_score - current_dataset_score) / peak_dataset_score > 0.3:
                        print(f"Severe degradation on {dataset_name}: {current_dataset_score:.3f} vs peak {peak_dataset_score:.3f}")
                        # Don't stop immediately, just warn
        
        return False
    
    def convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable types to JSON-compatible types"""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self):
        """Save experiment results with detailed degradation analysis for research"""
        if not self.iteration_metrics:
            return
            
        # Calculate detailed performance analysis
        self_eval_scores = [m.self_eval_accuracy for m in self.iteration_metrics]
        peak_performance = max(self_eval_scores)
        peak_iteration = self_eval_scores.index(peak_performance) + 1
        final_performance = self_eval_scores[-1]
        
        # Calculate degradation patterns
        degradation_analysis = {
            "peak_performance": peak_performance,
            "peak_iteration": peak_iteration,
            "final_performance": final_performance,
            "total_degradation": (peak_performance - final_performance) / peak_performance if peak_performance > 0 else 0,
            "iterations_after_peak": len(self.iteration_metrics) - peak_iteration,
            "plateau_detected": False,
            "catastrophic_forgetting": {}
        }
        
        # Detect plateau
        if len(self.iteration_metrics) >= 5:
            last_5_scores = self_eval_scores[-5:]
            degradation_analysis["plateau_detected"] = np.std(last_5_scores) < self.config.performance_threshold
        
        # Category-wise degradation analysis
        for category in ['code', 'math', 'chat', 'safety', 'reasoning']:
            category_scores = [m.category_scores.get(category, 0) for m in self.iteration_metrics]
            if any(s > 0 for s in category_scores):
                cat_peak = max(category_scores)
                cat_final = category_scores[-1]
                degradation_analysis["catastrophic_forgetting"][category] = {
                    "peak_score": cat_peak,
                    "final_score": cat_final,
                    "degradation_ratio": (cat_peak - cat_final) / cat_peak if cat_peak > 0 else 0,
                    "peak_iteration": category_scores.index(cat_peak) + 1
                }
        
        # Cross-dataset performance analysis
        cross_dataset_analysis = {}
        if self.iteration_metrics[0].cross_dataset_scores:
            for dataset_name in self.iteration_metrics[0].cross_dataset_scores.keys():
                dataset_scores = [m.cross_dataset_scores.get(dataset_name, 0) for m in self.iteration_metrics]
                ds_peak = max(dataset_scores) if dataset_scores else 0
                ds_final = dataset_scores[-1] if dataset_scores else 0
                cross_dataset_analysis[dataset_name] = {
                    "peak_score": ds_peak,
                    "final_score": ds_final,
                    "degradation_ratio": (ds_peak - ds_final) / ds_peak if ds_peak > 0 else 0,
                    "peak_iteration": dataset_scores.index(ds_peak) + 1 if ds_peak > 0 else 1
                }
        
        results = {
            "config": asdict(self.config),
            "hyperparameters": PAPER_HYPERPARAMETERS,
            "metrics": [asdict(m) for m in self.iteration_metrics],
            "model_family": self.config.model_id.split("/")[0],
            "degradation_analysis": degradation_analysis,
            "cross_dataset_analysis": cross_dataset_analysis,
            "final_performance": {
                "iterations_completed": len(self.iteration_metrics),
                "peak_performance": peak_performance,
                "final_performance": final_performance,
                "performance_trajectory": self_eval_scores
            }
        }
        
        # Convert to JSON-serializable format
        results = self.convert_to_json_serializable(results)
        
        results_file = os.path.join(self.config.results_dir, "iteration_metrics.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save detailed degradation summary
        degradation_file = os.path.join(self.config.results_dir, "degradation_analysis.json")
        degradation_data = {
            "degradation_analysis": degradation_analysis,
            "cross_dataset_analysis": cross_dataset_analysis
        }
        degradation_data = self.convert_to_json_serializable(degradation_data)
        with open(degradation_file, 'w') as f:
            json.dump(degradation_data, f, indent=2)
        
        # Export to CSV for easy analysis
        self.export_metrics_to_csv()
    
    def export_metrics_to_csv(self):
        """Export iteration metrics to CSV files for easy analysis and plotting"""
        if not self.iteration_metrics:
            return
        
        # Main iteration metrics CSV
        iteration_data = []
        for metric in self.iteration_metrics:
            row = {
                'iteration': metric.iteration,
                'eval_loss': metric.eval_loss,
                'self_eval_accuracy': metric.self_eval_accuracy,
                'preference_agreement': metric.preference_agreement,
                'response_diversity': metric.response_diversity,
                'model_id': self.config.model_id,
                'experiment_type': getattr(self.config, 'experiment_type', 'unknown'),
                'max_iterations': self.config.max_iterations,
                'timestamp': getattr(metric, 'timestamp', datetime.now().isoformat())
            }
            
            # Add category scores as separate columns
            for category, score in metric.category_scores.items():
                row[f'category_{category}'] = score
            
            # Add cross-dataset scores
            for dataset, score in metric.cross_dataset_scores.items():
                dataset_clean = dataset.replace('/', '_').replace('-', '_')
                row[f'dataset_{dataset_clean}'] = score
            
            # Add RewardBench scores if available
            if hasattr(metric, 'rewardbench_scores'):
                for subset, score in metric.rewardbench_scores.items():
                    subset_clean = subset.replace('-', '_').replace('/', '_')
                    row[f'rewardbench_{subset_clean}'] = score
            
            iteration_data.append(row)
        
        # Save main metrics CSV
        df_metrics = pd.DataFrame(iteration_data)
        metrics_csv = os.path.join(self.config.results_dir, "iteration_metrics.csv")
        df_metrics.to_csv(metrics_csv, index=False)
        
        # Category performance CSV (for detailed analysis)
        category_data = []
        for metric in self.iteration_metrics:
            for category, score in metric.category_scores.items():
                category_data.append({
                    'iteration': metric.iteration,
                    'category': category,
                    'score': score,
                    'model_id': self.config.model_id
                })
        
        if category_data:
            df_categories = pd.DataFrame(category_data)
            categories_csv = os.path.join(self.config.results_dir, "category_performance.csv")
            df_categories.to_csv(categories_csv, index=False)
        
        # Cross-dataset performance CSV
        dataset_data = []
        for metric in self.iteration_metrics:
            for dataset, score in metric.cross_dataset_scores.items():
                dataset_data.append({
                    'iteration': metric.iteration,
                    'dataset': dataset,
                    'score': score,
                    'model_id': self.config.model_id
                })
        
        if dataset_data:
            df_datasets = pd.DataFrame(dataset_data)
            datasets_csv = os.path.join(self.config.results_dir, "cross_dataset_performance.csv")
            df_datasets.to_csv(datasets_csv, index=False)
        
        # Summary statistics CSV
        if len(self.iteration_metrics) > 1:
            self_eval_scores = [m.self_eval_accuracy for m in self.iteration_metrics]
            peak_performance = max(self_eval_scores)
            peak_iteration = self_eval_scores.index(peak_performance) + 1
            final_performance = self_eval_scores[-1]
            
            summary_data = [{
                'model_id': self.config.model_id,
                'total_iterations': len(self.iteration_metrics),
                'peak_performance': peak_performance,
                'peak_iteration': peak_iteration,
                'final_performance': final_performance,
                'total_degradation': (peak_performance - final_performance) / peak_performance if peak_performance > 0 else 0,
                'iterations_after_peak': len(self.iteration_metrics) - peak_iteration,
                'experiment_start': self.iteration_metrics[0].timestamp if hasattr(self.iteration_metrics[0], 'timestamp') else 'unknown',
                'experiment_end': self.iteration_metrics[-1].timestamp if hasattr(self.iteration_metrics[-1], 'timestamp') else 'unknown'
            }]
            
            df_summary = pd.DataFrame(summary_data)
            summary_csv = os.path.join(self.config.results_dir, "experiment_summary.csv")
            df_summary.to_csv(summary_csv, index=False)
        
        print(f"📊 CSV exports saved:")
        print(f"   - Main metrics: {metrics_csv}")
        print(f"   - Category performance: {os.path.join(self.config.results_dir, 'category_performance.csv')}")
        print(f"   - Cross-dataset performance: {os.path.join(self.config.results_dir, 'cross_dataset_performance.csv')}")
        print(f"   - Experiment summary: {os.path.join(self.config.results_dir, 'experiment_summary.csv')}")
        
        # Also save to a global results directory for comparison across experiments
        self.save_to_global_csv(df_metrics, df_summary if len(self.iteration_metrics) > 1 else None)
    
    def save_to_global_csv(self, df_metrics, df_summary=None):
        """Save experiment data to global CSV files for cross-experiment comparison"""
        global_results_dir = "./results/all_experiments"
        Path(global_results_dir).mkdir(parents=True, exist_ok=True)
        
        # Global metrics file (append mode)
        global_metrics_file = os.path.join(global_results_dir, "all_experiments_metrics.csv")
        df_metrics['experiment_id'] = f"{self.config.model_id.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Add experiment metadata
        for col in ['experiment_id', 'model_id', 'experiment_type', 'max_iterations']:
            if col not in df_metrics.columns:
                if col == 'experiment_id':
                    df_metrics[col] = f"{self.config.model_id.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                elif col == 'experiment_type':
                    df_metrics[col] = 'limit_testing'
        
        # Save or append to global file
        if os.path.exists(global_metrics_file):
            df_metrics.to_csv(global_metrics_file, mode='a', header=False, index=False)
        else:
            df_metrics.to_csv(global_metrics_file, index=False)
        
        # Global summary file
        if df_summary is not None:
            global_summary_file = os.path.join(global_results_dir, "all_experiments_summary.csv")
            df_summary['experiment_id'] = df_metrics['experiment_id'].iloc[0]
            
            if os.path.exists(global_summary_file):
                df_summary.to_csv(global_summary_file, mode='a', header=False, index=False)
            else:
                df_summary.to_csv(global_summary_file, index=False)
        
        print(f"📈 Global comparison files updated: {global_results_dir}")
    
    def generate_plots(self):
        """Generate comprehensive visualizations including category-specific analysis"""
        iterations = [m.iteration for m in self.iteration_metrics]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall performance trajectory
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(iterations, [m.train_loss for m in self.iteration_metrics], 'b-', label='Train Loss')
        ax1.plot(iterations, [m.eval_loss for m in self.iteration_metrics], 'r-', label='Eval Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Self-evaluation accuracy with confidence
        ax2 = plt.subplot(3, 3, 2)
        accuracies = [m.self_eval_accuracy for m in self.iteration_metrics]
        ax2.plot(iterations, accuracies, 'g-', marker='o', linewidth=2)
        
        # Add confidence band based on cross-dataset variance
        if self.iteration_metrics[0].cross_dataset_scores:
            variances = []
            for m in self.iteration_metrics:
                scores = list(m.cross_dataset_scores.values())
                variances.append(np.std(scores) if scores else 0)
            
            ax2.fill_between(iterations,
                           [a - v for a, v in zip(accuracies, variances)],
                           [a + v for a, v in zip(accuracies, variances)],
                           alpha=0.3, color='green')
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Self-Evaluation Accuracy with Variance')
        ax2.grid(True)
        
        # 3. Category-specific performance
        ax3 = plt.subplot(3, 3, 3)
        categories = ['code', 'math', 'chat', 'safety', 'reasoning']
        for category in categories:
            scores = [m.category_scores.get(category, 0) for m in self.iteration_metrics]
            if any(s > 0 for s in scores):
                ax3.plot(iterations, scores, marker='o', label=category)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Category Accuracy')
        ax3.set_title('Performance by Task Category')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Cross-dataset performance
        ax4 = plt.subplot(3, 3, 4)
        if self.iteration_metrics[0].cross_dataset_scores:
            for dataset_name in self.iteration_metrics[0].cross_dataset_scores.keys():
                scores = [m.cross_dataset_scores.get(dataset_name, 0) for m in self.iteration_metrics]
                ax4.plot(iterations, scores, marker='o', label=dataset_name)
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Cross-Dataset Performance')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Preference stability vs diversity
        ax5 = plt.subplot(3, 3, 5)
        agreements = [m.preference_agreement for m in self.iteration_metrics]
        diversities = [m.response_diversity for m in self.iteration_metrics]
        
        ax5_twin = ax5.twinx()
        line1 = ax5.plot(iterations, agreements, 'b-', label='Preference Agreement')
        line2 = ax5_twin.plot(iterations, diversities, 'r-', label='Response Diversity')
        
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Preference Agreement', color='b')
        ax5_twin.set_ylabel('Response Diversity', color='r')
        ax5.set_title('Stability vs Diversity Trade-off')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels)
        ax5.grid(True)
        
        # 6. Category distribution over iterations
        ax6 = plt.subplot(3, 3, 6)
        category_dists = []
        for m in self.iteration_metrics:
            # Extract from logged data or calculate
            dist = {}
            for cat in categories:
                dist[cat] = m.category_scores.get(cat, 0)
            category_dists.append(dist)
        
        # Stack plot for category distribution
        if category_dists:
            cat_data = {cat: [d.get(cat, 0) for d in category_dists] for cat in categories}
            bottom = np.zeros(len(iterations))
            for cat, color in zip(categories, plt.cm.Set3.colors):
                values = cat_data[cat]
                ax6.bar(iterations, values, bottom=bottom, label=cat, color=color)
                bottom += np.array(values)
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Category Distribution')
        ax6.set_title('Task Category Distribution')
        ax6.legend()
        
        # 7. Performance degradation analysis
        ax7 = plt.subplot(3, 3, 7)
        if len(self.iteration_metrics) > 1:
            # Calculate iteration-over-iteration changes
            performance_changes = []
            for i in range(1, len(self.iteration_metrics)):
                change = self.iteration_metrics[i].self_eval_accuracy - self.iteration_metrics[i-1].self_eval_accuracy
                performance_changes.append(change)
            
            colors = ['green' if c >= 0 else 'red' for c in performance_changes]
            ax7.bar(iterations[1:], performance_changes, color=colors)
            ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax7.set_xlabel('Iteration')
            ax7.set_ylabel('Performance Change')
            ax7.set_title('Iteration-over-Iteration Performance Change')
            ax7.grid(True, axis='y')
        
        # 8. Score distribution analysis
        ax8 = plt.subplot(3, 3, 8)
        # Collect score differences from latest iteration
        if hasattr(self, '_latest_preferences') and self._latest_preferences:
            score_diffs = [p['score_diff'] for p in self._latest_preferences[:100]]
            ax8.hist(score_diffs, bins=20, alpha=0.7, color='purple')
            ax8.set_xlabel('Score Difference (Chosen - Rejected)')
            ax8.set_ylabel('Count')
            ax8.set_title('Preference Score Distribution')
            ax8.grid(True, axis='y')
        
        # 9. RewardBench performance (if available)
        ax9 = plt.subplot(3, 3, 9)
        if any(hasattr(m, 'rewardbench_scores') for m in self.iteration_metrics):
            # Create heatmap of RewardBench subset performance
            rb_data = []
            rb_subsets = []
            for m in self.iteration_metrics:
                if hasattr(m, 'rewardbench_scores'):
                    for subset, score in m.rewardbench_scores.items():
                        if subset not in rb_subsets:
                            rb_subsets.append(subset)
                    rb_data.append([m.rewardbench_scores.get(s, 0) for s in rb_subsets])
            
            if rb_data:
                im = ax9.imshow(np.array(rb_data).T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
                ax9.set_xticks(range(len(iterations)))
                ax9.set_xticklabels(iterations)
                ax9.set_yticks(range(len(rb_subsets)))
                ax9.set_yticklabels(rb_subsets)
                ax9.set_xlabel('Iteration')
                ax9.set_title('RewardBench Subset Performance')
                plt.colorbar(im, ax=ax9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, 'iteration_metrics_comprehensive.png'), dpi=300)
        plt.close()
        
        # Additional plots for paper-specific analysis
        self.generate_paper_specific_plots()
    
    def generate_paper_specific_plots(self):
        """Generate plots specifically aligned with paper's analysis"""
        # Model family comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance trajectory with model info
        iterations = [m.iteration for m in self.iteration_metrics]
        accuracies = [m.self_eval_accuracy for m in self.iteration_metrics]
        
        ax1.plot(iterations, accuracies, 'b-', marker='o', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Self-Evaluation Accuracy')
        ax1.set_title(f'IPO Performance: {self.config.model_id.split("/")[-1]}')
        ax1.grid(True)
        
        # Add annotations for key events
        peak_idx = np.argmax(accuracies)
        ax1.annotate(f'Peak: {accuracies[peak_idx]:.3f}',
                    xy=(iterations[peak_idx], accuracies[peak_idx]),
                    xytext=(iterations[peak_idx]+0.5, accuracies[peak_idx]+0.02),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        # Category-specific degradation
        categories = ['code', 'math', 'chat']
        degradation_data = {}
        
        for category in categories:
            scores = [m.category_scores.get(category, 0) for m in self.iteration_metrics]
            if any(s > 0 for s in scores):
                # Calculate degradation from peak
                peak_score = max(scores)
                peak_iter = scores.index(peak_score)
                if peak_iter < len(scores) - 1:
                    final_score = scores[-1]
                    degradation = (peak_score - final_score) / peak_score if peak_score > 0 else 0
                    degradation_data[category] = degradation
        
        if degradation_data:
            categories = list(degradation_data.keys())
            degradations = list(degradation_data.values())
            colors = ['red' if d > 0.1 else 'green' for d in degradations]
            
            ax2.bar(categories, degradations, color=colors)
            ax2.set_ylabel('Degradation from Peak (%)')
            ax2.set_title('Category-Specific Performance Degradation')
            ax2.set_ylim(-0.1, max(degradations) * 1.2 if degradations else 0.5)
            
            # Add percentage labels
            for i, (cat, deg) in enumerate(zip(categories, degradations)):
                ax2.text(i, deg + 0.01, f'{deg*100:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, 'paper_aligned_analysis.png'), dpi=300)
        plt.close()
    
    def _handle_selective_saving(self, current_performance: float, iteration: int):
        """Handle selective model saving - only keep current and best models"""
        import shutil
        
        temp_path = os.path.join(self.config.checkpoint_dir, "temp_current")
        current_path = os.path.join(self.config.checkpoint_dir, "current")
        best_path = os.path.join(self.config.checkpoint_dir, "best")
        
        # Always save as current model
        if os.path.exists(current_path):
            shutil.rmtree(current_path)
        shutil.move(temp_path, current_path)
        
        # Check if this is the best model so far
        if current_performance > self.best_performance:
            print(f"🌟 New best model! Performance: {current_performance:.4f} (prev: {self.best_performance:.4f})")
            self.best_performance = current_performance
            self.best_iteration = iteration
            
            # Save as best model
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(current_path, best_path)
        else:
            print(f"📊 Current: {current_performance:.4f}, Best: {self.best_performance:.4f} (iteration {self.best_iteration})")

def main():
    parser = argparse.ArgumentParser(description="Iterative IPO Experiments - Research on Self-Improvement Limits")
    
    # Basic model and dataset config
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--base_dataset", type=str, default="Ayush-Singh/UltraFeedback-1k-Each")
    parser.add_argument("--eval_datasets", nargs="+", default=["truthful_qa", "gsm8k", "hellaswag"])
    
    # Research-focused parameters
    parser.add_argument("--max_iterations", type=int, default=25, help="Maximum iterations to test limits")
    parser.add_argument("--forced_iterations", type=int, default=None, help="Force exact number of iterations (ignore early stopping)")
    parser.add_argument("--samples_per_iteration", type=int, default=1000)
    
    # Research control parameters
    parser.add_argument("--track_degradation", action="store_true", default=True, help="Enable detailed degradation tracking")
    parser.add_argument("--cross_dataset_eval", action="store_true", default=True, help="Evaluate on all datasets every iteration")
    parser.add_argument("--save_all_checkpoints", action="store_true", default=True, help="Save checkpoint every iteration")
    parser.add_argument("--plateau_window", type=int, default=3, help="Window size for plateau detection")
    
    # Output directories
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/self_improvement_limits")
    parser.add_argument("--results_dir", type=str, default="./results/self_improvement_limits")
    parser.add_argument("--wandb_project", type=str, default="ipo-self-improvement-limits")
    
    # Technical parameters
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--use_rewardbench", action="store_true", default=True)
    parser.add_argument("--use_qwen", action="store_true", help="Test with Qwen models (paper's top performer)")
    parser.add_argument("--instruction_batch_size", type=int, default=4, help="Number of instructions to batch together for GPU efficiency")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Number of evaluations to batch together for cross-dataset evaluation")
    
    # Research experiment types
    parser.add_argument("--experiment_type", type=str, default="limit_testing", 
                       choices=["limit_testing", "cross_dataset_transfer", "both"],
                       help="Type of research experiment to run")
    
    args = parser.parse_args()
    
    # Override model if testing Qwen
    if args.use_qwen:
        args.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print("Testing with Qwen model as per paper's findings...")
    
    # Create experiment name based on configuration
    model_name = args.model_id.split("/")[-1]
    exp_name = f"{model_name}_{args.experiment_type}_{args.max_iterations}iter"
    if args.forced_iterations:
        exp_name += f"_forced{args.forced_iterations}"
    
    config = ExperimentConfig(
        model_id=args.model_id,
        base_dataset=args.base_dataset,
        eval_datasets=args.eval_datasets,
        max_iterations=args.max_iterations,
        samples_per_iteration=args.samples_per_iteration,
        checkpoint_dir=f"{args.checkpoint_dir}/{exp_name}",
        results_dir=f"{args.results_dir}/{exp_name}",
        wandb_project=args.wandb_project,
        use_4bit=args.use_4bit,
        use_rewardbench=args.use_rewardbench,
        
        # Research-specific configs
        save_all_checkpoints=args.save_all_checkpoints,
        cross_dataset_eval=args.cross_dataset_eval,
        track_degradation=args.track_degradation,
        plateau_detection_window=args.plateau_window,
        forced_iterations=args.forced_iterations,
        instruction_batch_size=args.instruction_batch_size,
        eval_batch_size=args.eval_batch_size
    )
    
    print(f"🔬 Starting Research Experiment: {exp_name}")
    print(f"📊 Research Questions:")
    print(f"   1. How many iterations before performance plateaus/degrades?")
    print(f"   2. How does training on {args.base_dataset} affect {args.eval_datasets}?")
    print(f"📝 Experiment will run for up to {args.max_iterations} iterations")
    if args.forced_iterations:
        print(f"   ⚠️ FORCED MODE: Will run exactly {args.forced_iterations} iterations")
    print(f"📋 Paper-Aligned Methodology:")
    if "Instruct" not in args.model_id:
        print(f"   1. SFT training on Dolly-15k (base model)")
        print(f"   2. Generate preferences from {args.base_dataset}")
        print(f"   3. Save preference data to disk")
        print(f"   4. DPO training on self-generated preferences")
    else:
        print(f"   1. Generate preferences from {args.base_dataset} (skip SFT)")
        print(f"   2. Save preference data to disk")
        print(f"   3. DPO training on self-generated preferences")
    print(f"🚀 GPU Optimization:")
    print(f"   - Preference generation: {args.instruction_batch_size} instructions × 4 responses = {args.instruction_batch_size * 4} responses per batch")
    print(f"   - Cross-dataset evaluation: {args.eval_batch_size} evaluations per batch")
    print(f"💾 Results will be saved to: {config.results_dir}")
    print(f"💾 Preference data will be saved to: {config.results_dir}/preference_data/")
    
    experiment = IterativeIPO(config)
    experiment.run_experiment()

if __name__ == "__main__":
    main()