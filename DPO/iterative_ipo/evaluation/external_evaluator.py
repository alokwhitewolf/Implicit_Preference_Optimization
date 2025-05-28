#!/usr/bin/env python3
"""
External dataset evaluation for measuring real performance improvements
"""

import torch
import numpy as np
import re
from datasets import load_dataset
from typing import Dict, List, Tuple
from tqdm import tqdm

class ExternalEvaluator:
    """Evaluates model performance on external benchmarks"""
    
    def __init__(self, config):
        self.config = config
        self.eval_datasets = ['gsm8k', 'truthful_qa', 'hellaswag']
        
    def evaluate_all_datasets(self, model, tokenizer, iteration: int) -> Dict[str, float]:
        """Evaluate on all configured external datasets"""
        results = {}
        
        print(f"🔍 Evaluating on external benchmarks (iteration {iteration})...")
        
        # GSM8K - Math reasoning
        if 'gsm8k' in self.eval_datasets:
            results['gsm8k_accuracy'] = self._evaluate_gsm8k(model, tokenizer)
        
        # TruthfulQA - Truthfulness
        if 'truthful_qa' in self.eval_datasets:
            results['truthful_qa_score'] = self._evaluate_truthful_qa(model, tokenizer)
        
        # HellaSwag - Commonsense reasoning
        if 'hellaswag' in self.eval_datasets:
            results['hellaswag_accuracy'] = self._evaluate_hellaswag(model, tokenizer)
        
        return results
    
    def _evaluate_gsm8k(self, model, tokenizer, num_samples: int = 100) -> float:
        """Evaluate mathematical reasoning on GSM8K"""
        print("📊 Evaluating GSM8K (math reasoning)...")
        
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for example in tqdm(dataset, desc="GSM8K"):
                question = example['question']
                correct_answer = self._extract_answer(example['answer'])
                
                # Generate response
                prompt = f"Question: {question}\nAnswer: Let me think step by step."
                response = self._generate_response(model, tokenizer, prompt, max_tokens=512)
                
                # Extract predicted answer
                predicted_answer = self._extract_numeric_answer(response)
                
                if predicted_answer is not None and abs(predicted_answer - correct_answer) < 1e-3:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            print(f"✓ GSM8K: {correct}/{total} = {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ GSM8K evaluation failed: {e}")
            return 0.0
    
    def _evaluate_truthful_qa(self, model, tokenizer, num_samples: int = 100) -> float:
        """Evaluate truthfulness on TruthfulQA"""
        print("📊 Evaluating TruthfulQA (truthfulness)...")
        
        try:
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for example in tqdm(dataset, desc="TruthfulQA"):
                question = example['question']
                choices = example['mc1_targets']['choices']
                correct_idx = example['mc1_targets']['labels'].index(1)  # Find the correct answer
                
                # Create multiple choice prompt
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Answer:"
                
                # Generate response
                response = self._generate_response(model, tokenizer, prompt, max_tokens=10)
                
                # Extract predicted choice
                predicted_idx = self._extract_choice(response, len(choices))
                
                if predicted_idx == correct_idx:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            print(f"✓ TruthfulQA: {correct}/{total} = {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ TruthfulQA evaluation failed: {e}")
            return 0.0
    
    def _evaluate_hellaswag(self, model, tokenizer, num_samples: int = 100) -> float:
        """Evaluate commonsense reasoning on HellaSwag"""
        print("📊 Evaluating HellaSwag (commonsense)...")
        
        try:
            dataset = load_dataset("hellaswag", split="validation")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for example in tqdm(dataset, desc="HellaSwag"):
                context = example['ctx']
                choices = example['endings']
                correct_idx = int(example['label'])
                
                # Create multiple choice prompt
                prompt = f"Context: {context}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Which continuation makes the most sense? Answer:"
                
                # Generate response
                response = self._generate_response(model, tokenizer, prompt, max_tokens=10)
                
                # Extract predicted choice
                predicted_idx = self._extract_choice(response, len(choices))
                
                if predicted_idx == correct_idx:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            print(f"✓ HellaSwag: {correct}/{total} = {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ HellaSwag evaluation failed: {e}")
            return 0.0
    
    def _generate_response(self, model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
        """Generate response from model"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Use greedy decoding for consistency
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _extract_answer(self, answer_text: str) -> float:
        """Extract numeric answer from GSM8K answer text"""
        # GSM8K answers end with "#### X" where X is the answer
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_text)
        if match:
            return float(match.group(1))
        return 0.0
    
    def _extract_numeric_answer(self, response: str) -> float:
        """Extract numeric answer from model response"""
        # Look for numbers in the response, prefer the last one
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None
    
    def _extract_choice(self, response: str, num_choices: int) -> int:
        """Extract choice (A, B, C, D) from response"""
        response = response.upper().strip()
        
        # Look for letter choices
        for i in range(num_choices):
            letter = chr(65 + i)  # A, B, C, D...
            if letter in response[:5]:  # Check first few characters
                return i
        
        # Fallback: look for numbers
        for i in range(num_choices):
            if str(i) in response[:5] or str(i+1) in response[:5]:
                return i
        
        # Random guess if no clear answer
        return 0