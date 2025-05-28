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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login, HfFolder
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
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
    save_all_checkpoints: bool = True  # Save every iteration for analysis
    cross_dataset_eval: bool = True  # Evaluate on all datasets every iteration
    track_degradation: bool = True  # Detailed degradation tracking
    plateau_detection_window: int = 3  # Window for plateau detection
    forced_iterations: int = None  # Force specific number of iterations (ignore early stopping)
    
class IterativeIPO:
    """Main class for iterative self-improvement experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.iteration_metrics: List[IterationMetrics] = []
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for checkpoints and results"""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
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
    
    def generate_responses(self, model, tokenizer, instruction: str, num_responses: int = 8):
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
            # Aggressive batch generation - we have 40GB GPU, use more!
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
            
            # Get token IDs for Yes/No
            yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = tokenizer.encode("No", add_special_tokens=False)
            
            # Handle different tokenizer behaviors
            yes_token_id = yes_tokens[0] if yes_tokens else tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = no_tokens[0] if no_tokens else tokenizer.encode("no", add_special_tokens=False)[0]
            
            yes_logit = logits[yes_token_id].item()
            no_logit = logits[no_token_id].item()
            
            # Softmax to get probability
            yes_prob = torch.nn.functional.softmax(
                torch.tensor([yes_logit, no_logit]), dim=0
            )[0].item()
            
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
            
            # Get token IDs for Yes/No
            yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = tokenizer.encode("No", add_special_tokens=False)
            
            # Handle different tokenizer behaviors
            yes_token_id = yes_tokens[0] if yes_tokens else tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = no_tokens[0] if no_tokens else tokenizer.encode("no", add_special_tokens=False)[0]
            
            # Process each sample in the batch
            for i in range(len(responses)):
                yes_logit = logits[i, yes_token_id].item()
                no_logit = logits[i, no_token_id].item()
                
                # Softmax to get probability
                yes_prob = torch.nn.functional.softmax(
                    torch.tensor([yes_logit, no_logit]), dim=0
                )[0].item()
                
                scores.append(yes_prob)
        
        return scores
    
    def generate_self_preferences(self, model, tokenizer, dataset, iteration: int):
        """Generate preference pairs with category awareness"""
        preferences = []
        category_counts = {'code': 0, 'math': 0, 'chat': 0, 'safety': 0, 'reasoning': 0}
        
        for example in tqdm(dataset, desc=f"Generating preferences (Iteration {iteration})"):
            instruction = example.get('instruction') or example.get('prompt', '')
            category = self.detect_category(instruction)
            category_counts[category] += 1
            
            # Generate multiple responses - increase to 8 for better GPU utilization
            responses = self.generate_responses(model, tokenizer, instruction, num_responses=8)
            
            # Self-evaluate responses with category-specific prompt (batch evaluation)
            scores = self.evaluate_responses_batch(model, tokenizer, instruction, responses, category)
            
            # Create preference pair
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)
            
            if best_idx != worst_idx and scores[best_idx] - scores[worst_idx] > 0.1:  # Min difference threshold
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
        return Dataset.from_list(preferences)
    
    def train_iteration(self, model, tokenizer, train_dataset, eval_dataset, iteration: int):
        """Train one iteration with paper-aligned configuration"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration}")
        
        # LoRA configuration from paper - use specific linear layers
        peft_config = LoraConfig(
            r=PAPER_HYPERPARAMETERS["lora_r"],
            lora_alpha=PAPER_HYPERPARAMETERS["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=PAPER_HYPERPARAMETERS["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Training configuration
        training_config = get_updated_training_config()
        training_args = DPOConfig(
            output_dir=checkpoint_path,
            beta=PAPER_HYPERPARAMETERS["dpo_beta"],
            max_length=1024,
            max_completion_length=256,
            max_prompt_length=512,
            **training_config,
            report_to="wandb" if wandb.run else "none",
            run_name=f"ipo_iteration_{iteration}",
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Format dataset for DPO
        def format_for_dpo(example):
            # Ensure proper format for DPO trainer
            return {
                "prompt": example["instruction"],
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }
        
        train_dataset = train_dataset.map(format_for_dpo)
        eval_dataset = eval_dataset.map(format_for_dpo)
        
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
        
        # Save model and tokenizer
        trainer.save_model(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Get metrics
        eval_results = trainer.evaluate()
        
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
        """Run the full iterative IPO experiment with paper alignment"""
        # Initialize wandb with paper-specific tags
        wandb.init(
            project=self.config.wandb_project,
            config=asdict(self.config),
            tags=["iterative-ipo", "paper-aligned", self.config.model_id.split("/")[-1]]
        )
        
        # Load evaluation datasets
        eval_datasets = {}
        # Define proper splits and configs for different datasets
        dataset_configs = {
            "truthful_qa": {"config": "generation", "split": "validation[:500]"},
            "gsm8k": {"config": "main", "split": "test[:500]"}, 
            "hellaswag": {"config": None, "split": "validation[:500]"},
            "tatsu-lab/alpaca": {"config": None, "split": "train[:500]"}  # Fix: use train split
        }
        
        for dataset_name in self.config.eval_datasets:
            try:
                config_info = dataset_configs.get(dataset_name, {"config": None, "split": "test[:500]"})
                
                # Load dataset with proper config and trust_remote_code
                if config_info["config"]:
                    eval_datasets[dataset_name] = load_dataset(
                        dataset_name, 
                        config_info["config"], 
                        split=config_info["split"],
                        trust_remote_code=True
                    )
                else:
                    eval_datasets[dataset_name] = load_dataset(
                        dataset_name, 
                        split=config_info["split"],
                        trust_remote_code=True
                    )
                    
                print(f"✓ Loaded {dataset_name} with {len(eval_datasets[dataset_name])} examples")
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                print(f"Skipping {dataset_name}...")
        
        # Load base training dataset
        base_dataset = load_dataset(self.config.base_dataset, split="train")
        
        previous_prefs = None
        checkpoint_path = None
        
        for iteration in range(self.config.max_iterations):
            print(f"\n{'='*50}")
            print(f"Starting Iteration {iteration + 1}/{self.config.max_iterations}")
            print(f"{'='*50}")
            
            # Load model
            model, tokenizer = self.load_model_and_tokenizer(checkpoint_path)
            
            # Generate self-preferences
            train_prefs = self.generate_self_preferences(
                model, tokenizer, 
                base_dataset.select(range(self.config.samples_per_iteration)),
                iteration + 1
            )
            
            # Category distribution analysis
            category_dist = {}
            for cat in ['code', 'math', 'chat', 'safety', 'reasoning']:
                cat_count = len(train_prefs.filter(lambda x: x['category'] == cat))
                category_dist[cat] = cat_count / len(train_prefs) if len(train_prefs) > 0 else 0
            
            # Split into train/eval
            train_size = int(0.9 * len(train_prefs))
            train_data = train_prefs.select(range(train_size))
            eval_data = train_prefs.select(range(train_size, len(train_prefs)))
            
            # Train iteration
            train_loss, eval_loss = self.train_iteration(
                model, tokenizer, train_data, eval_data, iteration + 1
            )
            
            # Evaluate performance
            cross_dataset_scores = self.evaluate_cross_dataset(model, tokenizer, eval_datasets)
            rewardbench_scores = self.evaluate_on_rewardbench(model, tokenizer)
            self_eval_accuracy = np.mean(list(cross_dataset_scores.values()))
            
            # Category-specific evaluation
            category_scores = {}
            for category in ['code', 'math', 'chat', 'safety', 'reasoning']:
                cat_data = eval_data.filter(lambda x: x['category'] == category)
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
        """Evaluate model performance across multiple datasets"""
        scores = {}
        
        for dataset_name, dataset in eval_datasets.items():
            correct = 0
            total = 0
            
            for example in tqdm(dataset[:100], desc=f"Evaluating on {dataset_name}"):
                # Handle different dataset formats
                if isinstance(example, str):
                    instruction = example
                else:
                    instruction = example.get('question') or example.get('instruction') or example.get('prompt', '')
                
                # Only evaluate if we have preference pairs
                if not isinstance(example, str) and 'chosen' in example and 'rejected' in example:
                    chosen_score = self.evaluate_response(
                        model, tokenizer, instruction, example['chosen']
                    )
                    rejected_score = self.evaluate_response(
                        model, tokenizer, instruction, example['rejected']
                    )
                    
                    if chosen_score > rejected_score:
                        correct += 1
                    total += 1
            
            scores[dataset_name] = correct / total if total > 0 else 0.0
        
        return scores
    
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
        
        results_file = os.path.join(self.config.results_dir, "iteration_metrics.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save detailed degradation summary
        degradation_file = os.path.join(self.config.results_dir, "degradation_analysis.json")
        with open(degradation_file, 'w') as f:
            json.dump({
                "degradation_analysis": degradation_analysis,
                "cross_dataset_analysis": cross_dataset_analysis
            }, f, indent=2)
    
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

def main():
    parser = argparse.ArgumentParser(description="Iterative IPO Experiments - Research on Self-Improvement Limits")
    
    # Basic model and dataset config
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--base_dataset", type=str, default="databricks/databricks-dolly-15k")
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
        forced_iterations=args.forced_iterations
    )
    
    print(f"🔬 Starting Research Experiment: {exp_name}")
    print(f"📊 Research Questions:")
    print(f"   1. How many iterations before performance plateaus/degrades?")
    print(f"   2. How does training on {args.base_dataset} affect {args.eval_datasets}?")
    print(f"📝 Experiment will run for up to {args.max_iterations} iterations")
    if args.forced_iterations:
        print(f"   ⚠️ FORCED MODE: Will run exactly {args.forced_iterations} iterations")
    print(f"💾 Results will be saved to: {config.results_dir}")
    
    experiment = IterativeIPO(config)
    experiment.run_experiment()

if __name__ == "__main__":
    main()