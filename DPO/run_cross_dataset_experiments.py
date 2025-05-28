#!/usr/bin/env python3
"""
Run comprehensive cross-dataset experiments for iterative IPO
"""

import os
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations
import numpy as np
from datasets import load_dataset
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dataset configurations
DATASET_CONFIGS = {
    "dolly": {
        "name": "databricks/databricks-dolly-15k",
        "split": "train",
        "instruction_field": "instruction",
        "response_field": "response",
        "category": "instruction_following"
    },
    "alpaca": {
        "name": "tatsu-lab/alpaca",
        "split": "train",
        "instruction_field": "instruction",
        "response_field": "output",
        "category": "instruction_following"
    },
    "code_alpaca": {
        "name": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "instruction_field": "instruction",
        "response_field": "output",
        "category": "code"
    },
    "gsm8k": {
        "name": "gsm8k",
        "split": "train",
        "instruction_field": "question",
        "response_field": "answer",
        "category": "math"
    },
    "truthful_qa": {
        "name": "truthful_qa",
        "split": "validation",
        "instruction_field": "question",
        "response_field": "best_answer",
        "category": "truthfulness"
    },
    "hellaswag": {
        "name": "hellaswag",
        "split": "validation",
        "instruction_field": "ctx",
        "response_field": "endings",
        "category": "reasoning"
    }
}

class CrossDatasetExperiment:
    """Run cross-dataset transfer experiments"""
    
    def __init__(self, model_id: str, experiment_name: str):
        self.model_id = model_id
        self.experiment_name = experiment_name
        self.results_dir = Path(f"./results/cross_dataset/{experiment_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def load_dataset_sample(self, dataset_key: str, sample_size: int = 1000) -> List[Dict]:
        """Load a sample from a dataset"""
        config = DATASET_CONFIGS[dataset_key]
        dataset = load_dataset(config['name'], split=config['split'])
        
        # Sample if dataset is larger than sample_size
        if len(dataset) > sample_size:
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            dataset = dataset.select(indices)
        
        samples = []
        for item in dataset:
            instruction = item.get(config['instruction_field'], "")
            response = item.get(config['response_field'], "")
            
            # Handle special cases
            if dataset_key == "hellaswag":
                # For hellaswag, use the correct ending
                response = response[int(item.get('label', 0))]
            
            if instruction and response:
                samples.append({
                    'instruction': str(instruction),
                    'response': str(response),
                    'category': config['category']
                })
        
        return samples
    
    def evaluate_on_dataset(self, model, tokenizer, dataset_key: str, 
                          trained_on: str, iteration: int) -> Dict:
        """Evaluate model performance on a specific dataset"""
        print(f"Evaluating on {dataset_key} (model trained on {trained_on}, iteration {iteration})")
        
        samples = self.load_dataset_sample(dataset_key, sample_size=500)
        
        correct = 0
        total = 0
        category_scores = {}
        
        for sample in samples:
            # Generate response
            prompt = f"User: {sample['instruction']}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Evaluate response quality
            eval_prompt = f"""Evaluate if this response is appropriate and helpful.
User: {sample['instruction']}
Expected type of response: {sample['response'][:100]}...
Generated response: {generated}

Is the generated response appropriate? Answer Yes or No:"""
            
            eval_inputs = tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                eval_outputs = model(**eval_inputs)
                logits = eval_outputs.logits[0, -1]
                
                yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_token = tokenizer.encode("No", add_special_tokens=False)[0]
                
                yes_prob = torch.nn.functional.softmax(
                    torch.tensor([logits[yes_token], logits[no_token]]), dim=0
                )[0].item()
            
            if yes_prob > 0.5:
                correct += 1
            total += 1
            
            # Track category-specific scores
            category = sample['category']
            if category not in category_scores:
                category_scores[category] = {'correct': 0, 'total': 0}
            category_scores[category]['total'] += 1
            if yes_prob > 0.5:
                category_scores[category]['correct'] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate category accuracies
        category_accuracies = {}
        for cat, scores in category_scores.items():
            if scores['total'] > 0:
                category_accuracies[cat] = scores['correct'] / scores['total']
        
        return {
            'accuracy': accuracy,
            'total_samples': total,
            'category_accuracies': category_accuracies,
            'trained_on': trained_on,
            'evaluated_on': dataset_key,
            'iteration': iteration
        }
    
    def run_transfer_experiment(self, train_dataset: str, eval_datasets: List[str], 
                              max_iterations: int = 5) -> Dict:
        """Run a single transfer learning experiment"""
        print(f"\nStarting experiment: Training on {train_dataset}")
        
        # Import the iterative IPO class
        from iterative_ipo import IterativeIPO, ExperimentConfig
        
        config = ExperimentConfig(
            model_id=self.model_id,
            base_dataset=DATASET_CONFIGS[train_dataset]['name'],
            eval_datasets=[DATASET_CONFIGS[d]['name'] for d in eval_datasets],
            max_iterations=max_iterations,
            samples_per_iteration=500,
            checkpoint_dir=f"./checkpoints/cross_dataset/{self.experiment_name}/{train_dataset}",
            results_dir=f"./results/cross_dataset/{self.experiment_name}/{train_dataset}",
            wandb_project=f"cross-dataset-{self.experiment_name}"
        )
        
        # Run iterative IPO
        ipo = IterativeIPO(config)
        ipo.run_experiment()
        
        # Evaluate on all datasets after each iteration
        transfer_results = {
            'train_dataset': train_dataset,
            'iterations': []
        }
        
        for iteration in range(1, max_iterations + 1):
            checkpoint_path = f"./checkpoints/cross_dataset/{self.experiment_name}/{train_dataset}/iteration_{iteration}"
            
            if not os.path.exists(checkpoint_path):
                break
                
            model, tokenizer = self.load_model_from_checkpoint(checkpoint_path)
            
            iteration_results = {
                'iteration': iteration,
                'evaluations': {}
            }
            
            # Evaluate on all datasets
            for eval_dataset in DATASET_CONFIGS.keys():
                eval_result = self.evaluate_on_dataset(
                    model, tokenizer, eval_dataset, train_dataset, iteration
                )
                iteration_results['evaluations'][eval_dataset] = eval_result
            
            transfer_results['iterations'].append(iteration_results)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        return transfer_results
    
    def load_model_from_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def run_all_experiments(self, datasets_to_test: List[str], max_iterations: int = 5):
        """Run experiments training on each dataset and evaluating on all others"""
        
        all_results = {}
        
        for train_dataset in datasets_to_test:
            eval_datasets = [d for d in datasets_to_test if d != train_dataset]
            
            results = self.run_transfer_experiment(
                train_dataset, eval_datasets, max_iterations
            )
            
            all_results[train_dataset] = results
            
            # Save intermediate results
            self.save_results(all_results)
        
        # Generate comprehensive analysis
        self.analyze_transfer_patterns(all_results)
        
        return all_results
    
    def analyze_transfer_patterns(self, results: Dict):
        """Analyze transfer learning patterns across datasets"""
        
        # Create transfer matrix for final iteration
        datasets = list(results.keys())
        n_datasets = len(datasets)
        
        # Get maximum iteration across all experiments
        max_iter = max(len(results[d]['iterations']) for d in datasets)
        
        # Create transfer matrices for each iteration
        for iteration in range(1, max_iter + 1):
            transfer_matrix = np.zeros((n_datasets, n_datasets))
            
            for i, train_dataset in enumerate(datasets):
                if iteration <= len(results[train_dataset]['iterations']):
                    iter_data = results[train_dataset]['iterations'][iteration - 1]
                    
                    for j, eval_dataset in enumerate(datasets):
                        if eval_dataset in iter_data['evaluations']:
                            accuracy = iter_data['evaluations'][eval_dataset]['accuracy']
                            transfer_matrix[i, j] = accuracy
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                transfer_matrix,
                xticklabels=datasets,
                yticklabels=[f"Trained on {d}" for d in datasets],
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Accuracy'}
            )
            plt.title(f'Cross-Dataset Transfer Matrix (Iteration {iteration})')
            plt.xlabel('Evaluated On')
            plt.ylabel('Trained On')
            plt.tight_layout()
            plt.savefig(self.results_dir / f'transfer_matrix_iter_{iteration}.png', dpi=300)
            plt.close()
        
        # Analyze transfer degradation
        self.analyze_transfer_degradation(results)
        
        # Analyze category-specific transfer
        self.analyze_category_transfer(results)
    
    def analyze_transfer_degradation(self, results: Dict):
        """Analyze how transfer performance degrades over iterations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (train_dataset, data) in enumerate(results.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            # Plot performance on each evaluation dataset over iterations
            eval_datasets = set()
            for iter_data in data['iterations']:
                eval_datasets.update(iter_data['evaluations'].keys())
            
            for eval_dataset in eval_datasets:
                iterations = []
                accuracies = []
                
                for iter_data in data['iterations']:
                    if eval_dataset in iter_data['evaluations']:
                        iterations.append(iter_data['iteration'])
                        accuracies.append(iter_data['evaluations'][eval_dataset]['accuracy'])
                
                style = '-' if eval_dataset == train_dataset else '--'
                ax.plot(iterations, accuracies, style, marker='o', label=eval_dataset)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Transfer from {train_dataset}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'transfer_degradation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_category_transfer(self, results: Dict):
        """Analyze transfer patterns by task category"""
        
        category_transfer = {}
        
        for train_dataset, data in results.items():
            train_category = DATASET_CONFIGS[train_dataset]['category']
            
            if train_category not in category_transfer:
                category_transfer[train_category] = {}
            
            for iter_data in data['iterations']:
                for eval_dataset, eval_results in iter_data['evaluations'].items():
                    eval_category = DATASET_CONFIGS[eval_dataset]['category']
                    
                    if eval_category not in category_transfer[train_category]:
                        category_transfer[train_category][eval_category] = []
                    
                    category_transfer[train_category][eval_category].append(
                        eval_results['accuracy']
                    )
        
        # Calculate average transfer scores
        avg_transfer = {}
        for train_cat in category_transfer:
            avg_transfer[train_cat] = {}
            for eval_cat in category_transfer[train_cat]:
                scores = category_transfer[train_cat][eval_cat]
                avg_transfer[train_cat][eval_cat] = np.mean(scores)
        
        # Create category transfer heatmap
        categories = list(avg_transfer.keys())
        transfer_matrix = np.zeros((len(categories), len(categories)))
        
        for i, train_cat in enumerate(categories):
            for j, eval_cat in enumerate(categories):
                if eval_cat in avg_transfer[train_cat]:
                    transfer_matrix[i, j] = avg_transfer[train_cat][eval_cat]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            transfer_matrix,
            xticklabels=categories,
            yticklabels=[f"{cat} →" for cat in categories],
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Average Transfer Score'}
        )
        plt.title('Task Category Transfer Patterns')
        plt.xlabel('Evaluated On Category')
        plt.ylabel('Trained On Category')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'category_transfer.png', dpi=300)
        plt.close()
    
    def save_results(self, results: Dict):
        """Save experiment results"""
        results_file = self.results_dir / 'cross_dataset_results.json'
        
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(results_file, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"Results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Run cross-dataset IPO experiments")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name for this experiment run")
    parser.add_argument("--datasets", nargs="+", default=["dolly", "alpaca", "gsm8k", "truthful_qa"],
                       choices=list(DATASET_CONFIGS.keys()), help="Datasets to include in experiments")
    parser.add_argument("--max_iterations", type=int, default=5, help="Maximum iterations per experiment")
    
    args = parser.parse_args()
    
    experiment = CrossDatasetExperiment(args.model_id, args.experiment_name)
    results = experiment.run_all_experiments(args.datasets, args.max_iterations)
    
    print("\nExperiment completed! Check results in:", experiment.results_dir)

if __name__ == "__main__":
    main()