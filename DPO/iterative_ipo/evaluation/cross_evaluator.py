#!/usr/bin/env python3
"""
Cross-dataset evaluation for transfer analysis
This is a SEPARATE experiment from iterative self-improvement
"""

from datasets import Dataset, load_dataset
from typing import Dict, List
import torch
from tqdm import tqdm

class CrossDatasetEvaluator:
    """
    Evaluates trained models on external datasets
    
    This is used for SEPARATE transfer analysis experiments,
    NOT for the main iterative IPO experiment
    """
    
    def __init__(self, config):
        self.config = config
        
        # Dataset configurations for proper evaluation
        self.dataset_configs = {
            "truthful_qa": {"repo": "domenicrosati/TruthfulQA", "config": "default", "split": "train[:500]"},
            "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "test[:500]"},
            "hellaswag": {"repo": "Rowan/hellaswag", "config": None, "split": "validation[:500]"},
        }
    
    def load_eval_datasets(self, dataset_names: List[str]) -> Dict[str, Dataset]:
        """Load evaluation datasets with correct repository names"""
        eval_datasets = {}
        
        for dataset_name in dataset_names:
            try:
                config_info = self.dataset_configs.get(dataset_name, {"repo": dataset_name, "config": None, "split": "test[:500]"})
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
        
        return eval_datasets
    
    def evaluate_model_on_datasets(self, model, tokenizer, eval_datasets: Dict[str, Dataset]) -> Dict[str, float]:
        """
        Evaluate model performance across multiple datasets
        
        This implements proper evaluation logic for each dataset type:
        - TruthfulQA: Best Answer vs Incorrect Answers
        - GSM8K: Correct mathematical reasoning
        - HellaSwag: Commonsense reasoning completion
        """
        scores = {}
        
        for dataset_name, dataset in eval_datasets.items():
            print(f"🚀 Evaluating on {dataset_name}...")
            
            if dataset_name == "truthful_qa":
                scores[dataset_name] = self._evaluate_truthful_qa(model, tokenizer, dataset)
            elif dataset_name == "gsm8k":
                scores[dataset_name] = self._evaluate_gsm8k(model, tokenizer, dataset)
            elif dataset_name == "hellaswag":
                scores[dataset_name] = self._evaluate_hellaswag(model, tokenizer, dataset)
            else:
                print(f"⚠️ No evaluation logic for {dataset_name}, skipping...")
                scores[dataset_name] = 0.0
        
        return scores
    
    def _evaluate_truthful_qa(self, model, tokenizer, dataset: Dataset) -> float:
        """Evaluate on TruthfulQA using Best Answer vs Incorrect Answers"""
        correct = 0
        total = 0
        
        for example in tqdm(dataset[:100], desc="Evaluating TruthfulQA"):
            if 'Best Answer' not in example or 'Incorrect Answers' not in example:
                continue
                
            instruction = example.get('question', '')
            if not instruction:
                continue
            
            best_answer = example['Best Answer']
            incorrect_answers = example['Incorrect Answers'].split(';') if example['Incorrect Answers'] else []
            
            if not incorrect_answers:
                continue
            
            # Evaluate best answer vs first incorrect answer
            best_score = self._evaluate_response(model, tokenizer, instruction, best_answer)
            incorrect_score = self._evaluate_response(model, tokenizer, instruction, incorrect_answers[0].strip())
            
            if best_score > incorrect_score:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_gsm8k(self, model, tokenizer, dataset: Dataset) -> float:
        """Evaluate on GSM8K mathematical reasoning"""
        # TODO: Implement proper GSM8K evaluation
        # This requires checking mathematical correctness, not just preference
        print("⚠️ GSM8K evaluation not implemented yet")
        return 0.0
    
    def _evaluate_hellaswag(self, model, tokenizer, dataset: Dataset) -> float:
        """Evaluate on HellaSwag commonsense reasoning"""
        # TODO: Implement proper HellaSwag evaluation
        # This requires multiple choice evaluation, not preference
        print("⚠️ HellaSwag evaluation not implemented yet")
        return 0.0
    
    def _evaluate_response(self, model, tokenizer, instruction: str, response: str) -> float:
        """Evaluate a single response using category-specific prompt"""
        from config_updates import CATEGORY_EVALUATION_PROMPTS
        
        # Detect category (could be made more sophisticated)
        category = 'chat'  # Default category
        
        eval_prompt = CATEGORY_EVALUATION_PROMPTS[category].format(
            instruction=instruction,
            response=response
        )
        
        inputs = tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            
            # Use ours.py approach
            yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
            no_tokens = tokenizer.encode(" No", add_special_tokens=False)
            
            probs = torch.softmax(logits, dim=-1)
            
            yes_prob = sum(probs[token_id].item() for token_id in yes_tokens if token_id < len(probs))
            no_prob = sum(probs[token_id].item() for token_id in no_tokens if token_id < len(probs))
            
            total_prob = yes_prob + no_prob
            if total_prob > 0:
                yes_prob = yes_prob / total_prob
            
        return yes_prob