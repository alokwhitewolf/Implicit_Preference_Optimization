#!/usr/bin/env python3
"""
Preference generation for self-improvement
"""

import torch
import numpy as np
from datasets import Dataset
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from config_updates import CATEGORY_EVALUATION_PROMPTS, PAPER_HYPERPARAMETERS

class PreferenceGenerator:
    """Generates self-preferences from model responses"""
    
    def __init__(self, config):
        self.config = config
    
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
    
    def generate_responses(self, model, tokenizer, instruction: str, num_responses: int = 4) -> List[str]:
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
            batch_size = num_responses
            prompts = [prompt] * batch_size
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
    
    def generate_self_preferences(self, model, tokenizer, dataset, iteration: int) -> Dataset:
        """Generate preference pairs with instruction-level batching for better GPU utilization"""
        preferences = []
        category_counts = {'code': 0, 'math': 0, 'chat': 0, 'safety': 0, 'reasoning': 0}
        
        # Get batch size from config
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
        
        # Log preference generation statistics to wandb
        if len(preferences) > 0:
            import wandb
            
            # Calculate generation statistics
            score_diffs = [p['score_diff'] for p in preferences]
            chosen_scores = [p['chosen_score'] for p in preferences]
            rejected_scores = [p['rejected_score'] for p in preferences]
            
            generation_stats = {
                f"preference_gen/iteration_{iteration}/total_preferences": len(preferences),
                f"preference_gen/iteration_{iteration}/avg_score_diff": sum(score_diffs) / len(score_diffs),
                f"preference_gen/iteration_{iteration}/min_score_diff": min(score_diffs),
                f"preference_gen/iteration_{iteration}/max_score_diff": max(score_diffs),
                f"preference_gen/iteration_{iteration}/high_confidence_pairs": sum(1 for d in score_diffs if d > 0.3),
                f"preference_gen/iteration_{iteration}/avg_chosen_score": sum(chosen_scores) / len(chosen_scores),
                f"preference_gen/iteration_{iteration}/avg_rejected_score": sum(rejected_scores) / len(rejected_scores),
            }
            
            # Add category distribution
            for category, count in category_counts.items():
                generation_stats[f"preference_gen/iteration_{iteration}/category_{category}"] = count
                generation_stats[f"preference_gen/iteration_{iteration}/category_{category}_pct"] = (count / len(preferences)) * 100
            
            wandb.log(generation_stats)
        
        return Dataset.from_list(preferences)
    
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