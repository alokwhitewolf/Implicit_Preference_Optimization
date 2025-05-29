#!/usr/bin/env python3
"""
Optimized external dataset evaluation with batching and parallelization
"""

import torch
import numpy as np
import re
from datasets import load_dataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import concurrent.futures
from functools import partial

class FastExternalEvaluator:
    """Fast evaluation with batching and parallelization"""
    
    def __init__(self, config):
        self.config = config
        self.eval_datasets = getattr(config, 'external_eval_datasets', ['gsm8k', 'truthful_qa', 'hellaswag'])
        self.batch_size = getattr(config, 'eval_batch_size', 8)  # Batch size for evaluation
        
    def evaluate_all_datasets(self, model, tokenizer, iteration: int) -> Dict[str, float]:
        """Evaluate on all configured external datasets with parallelization"""
        results = {}
        
        print(f"🚀 Fast evaluation on external benchmarks (iteration {iteration})...")
        
        # Option 1: Parallel dataset evaluation (if enough GPU memory)
        if getattr(self.config, 'parallel_eval', False):
            results = self._evaluate_parallel(model, tokenizer)
        else:
            # Option 2: Sequential but batched evaluation
            results = self._evaluate_sequential_batched(model, tokenizer)
        
        return results
    
    def _evaluate_sequential_batched(self, model, tokenizer) -> Dict[str, float]:
        """Sequential evaluation with batching optimizations"""
        results = {}
        
        if 'gsm8k' in self.eval_datasets:
            results['gsm8k_accuracy'] = self._evaluate_gsm8k_batched(model, tokenizer)
        
        if 'truthful_qa' in self.eval_datasets:
            results['truthful_qa_score'] = self._evaluate_truthful_qa_batched(model, tokenizer)
        
        if 'hellaswag' in self.eval_datasets:
            results['hellaswag_accuracy'] = self._evaluate_hellaswag_batched(model, tokenizer)
        
        return results
    
    def _evaluate_gsm8k_batched(self, model, tokenizer, num_samples: int = None) -> float:
        """Batched GSM8K evaluation"""
        if num_samples is None:
            num_samples = getattr(self.config, 'external_eval_samples', 100)
            
        print(f"📊 Evaluating GSM8K (batched, {num_samples} samples)...")
        
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            # Prepare all prompts
            prompts = []
            correct_answers = []
            
            for example in dataset:
                question = example['question']
                correct_answer = self._extract_answer(example['answer'])
                prompt = f"Question: {question}\nAnswer: Let me think step by step."
                
                prompts.append(prompt)
                correct_answers.append(correct_answer)
            
            # Batch generation
            responses = self._generate_batch_responses(
                model, tokenizer, prompts, max_tokens=512
            )
            
            # Batch evaluation
            correct = 0
            for response, correct_answer in zip(responses, correct_answers):
                predicted_answer = self._extract_numeric_answer(response)
                if predicted_answer is not None and abs(predicted_answer - correct_answer) < 1e-3:
                    correct += 1
            
            accuracy = correct / len(correct_answers)
            print(f"✓ GSM8K (batched): {correct}/{len(correct_answers)} = {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ GSM8K evaluation failed: {e}")
            return 0.0
    
    def _evaluate_truthful_qa_batched(self, model, tokenizer, num_samples: int = None) -> float:
        """Batched TruthfulQA evaluation using logit-based scoring"""
        if num_samples is None:
            num_samples = getattr(self.config, 'external_eval_samples', 100)
            
        print(f"📊 Evaluating TruthfulQA (batched logits, {num_samples} samples)...")
        
        try:
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            # Process in batches for memory efficiency
            for i in tqdm(range(0, len(dataset), self.batch_size), desc="TruthfulQA"):
                batch = dataset[i:i+self.batch_size]
                batch_correct = self._evaluate_truthful_qa_batch_logits(
                    model, tokenizer, batch
                )
                correct += batch_correct
                total += len(batch) if isinstance(batch, list) else 1
            
            accuracy = correct / total if total > 0 else 0.0
            print(f"✓ TruthfulQA (batched): {correct}/{total} = {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ TruthfulQA evaluation failed: {e}")
            return 0.0
    
    def _evaluate_truthful_qa_batch_logits(self, model, tokenizer, batch) -> int:
        """Evaluate TruthfulQA batch using logit probabilities"""
        correct = 0
        
        # Handle both single examples and batches
        examples = batch if isinstance(batch, list) else [batch]
        
        for example in examples:
            question = example['question']
            choices = example['mc1_targets']['choices']
            correct_idx = example['mc1_targets']['labels'].index(1)
            
            # Create context for all choices
            context = f"Question: {question}\nAnswer:"
            
            # Get logit probabilities for each choice
            choice_logits = []
            
            for choice in choices:
                full_prompt = f"{context} {choice}"
                
                # Tokenize and get logits
                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # Get probability of the choice tokens
                choice_tokens = tokenizer.encode(f" {choice}", add_special_tokens=False)
                context_len = len(tokenizer.encode(context, add_special_tokens=False))
                
                choice_logit = outputs.logits[0, context_len-1:context_len+len(choice_tokens)-1].mean().item()
                choice_logits.append(choice_logit)
            
            # Predict choice with highest logit
            predicted_idx = np.argmax(choice_logits)
            
            if predicted_idx == correct_idx:
                correct += 1
        
        return correct
    
    def _evaluate_hellaswag_batched(self, model, tokenizer, num_samples: int = None) -> float:
        """Batched HellaSwag evaluation using perplexity scoring"""
        if num_samples is None:
            num_samples = getattr(self.config, 'external_eval_samples', 100)
            
        print(f"📊 Evaluating HellaSwag (batched perplexity, {num_samples} samples)...")
        
        try:
            dataset = load_dataset("hellaswag", split="validation")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            
            for example in tqdm(dataset, desc="HellaSwag"):
                context = example['ctx']
                choices = example['endings']
                correct_idx = int(example['label'])
                
                # Calculate perplexity for each choice
                perplexities = []
                
                for choice in choices:
                    full_text = context + choice
                    perplexity = self._calculate_perplexity(model, tokenizer, full_text)
                    perplexities.append(perplexity)
                
                # Lower perplexity = more likely = better choice
                predicted_idx = np.argmin(perplexities)
                
                if predicted_idx == correct_idx:
                    correct += 1
            
            accuracy = correct / len(dataset)
            print(f"✓ HellaSwag (batched): {correct}/{len(dataset)} = {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ HellaSwag evaluation failed: {e}")
            return 0.0
    
    def _generate_batch_responses(self, model, tokenizer, prompts: List[str], max_tokens: int = 256) -> List[str]:
        """Generate responses in batches for efficiency"""
        responses = []
        
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+self.batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy for consistency
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode batch responses
            for j, output in enumerate(outputs):
                # Skip input tokens
                input_len = inputs['input_ids'][j].shape[0]
                response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                responses.append(response.strip())
        
        return responses
    
    def _calculate_perplexity(self, model, tokenizer, text: str) -> float:
        """Calculate perplexity for a text"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    # Keep the helper methods from the original
    def _extract_answer(self, answer_text: str) -> float:
        """Extract numeric answer from GSM8K answer text"""
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_text)
        if match:
            return float(match.group(1))
        return 0.0
    
    def _extract_numeric_answer(self, response: str) -> float:
        """Extract numeric answer from model response"""
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None