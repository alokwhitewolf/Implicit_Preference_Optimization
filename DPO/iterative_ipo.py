#!/usr/bin/env python3
"""
Iterative IPO: Testing the limits of self-improvement through multiple rounds
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType, get_peft_model
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    timestamp: str

@dataclass
class ExperimentConfig:
    """Configuration for iterative IPO experiments"""
    model_id: str
    base_dataset: str
    eval_datasets: List[str]
    max_iterations: int = 10
    samples_per_iteration: int = 1000
    early_stopping_patience: int = 3
    performance_threshold: float = 0.01
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    wandb_project: str = "iterative-ipo"
    
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
        
    def load_model_and_tokenizer(self, checkpoint_path: Optional[str] = None):
        """Load model and tokenizer from checkpoint or base model"""
        if checkpoint_path:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def generate_self_preferences(self, model, tokenizer, dataset, iteration: int):
        """Generate preference pairs using the model as its own judge"""
        preferences = []
        
        for example in tqdm(dataset, desc=f"Generating preferences (Iteration {iteration})"):
            instruction = example['instruction']
            
            # Generate multiple responses
            responses = self.generate_responses(model, tokenizer, instruction, num_responses=4)
            
            # Self-evaluate responses
            scores = []
            for response in responses:
                score = self.evaluate_response(model, tokenizer, instruction, response)
                scores.append(score)
            
            # Create preference pair
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)
            
            if best_idx != worst_idx:  # Ensure different responses
                preferences.append({
                    'instruction': instruction,
                    'chosen': responses[best_idx],
                    'rejected': responses[worst_idx],
                    'score_diff': scores[best_idx] - scores[worst_idx]
                })
        
        return Dataset.from_list(preferences)
    
    def generate_responses(self, model, tokenizer, instruction: str, num_responses: int = 4):
        """Generate multiple responses for a given instruction"""
        responses = []
        
        prompt = f"User: {instruction}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        for _ in range(num_responses):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def evaluate_response(self, model, tokenizer, instruction: str, response: str) -> float:
        """Evaluate a response using the model as judge"""
        eval_prompt = f"""Below is a user question and a response. 
Your task is to evaluate if the response is helpful, accurate, and appropriate.
Answer with just Yes or No.

User: {instruction}
Response: {response}

Is this response good? Answer:"""
        
        inputs = tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            
            # Get probabilities for Yes/No tokens
            yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
            
            yes_logit = logits[yes_token_id].item()
            no_logit = logits[no_token_id].item()
            
            # Convert to probability
            yes_prob = torch.nn.functional.softmax(
                torch.tensor([yes_logit, no_logit]), dim=0
            )[0].item()
            
        return yes_prob
    
    def train_iteration(self, model, tokenizer, train_dataset, eval_dataset, iteration: int):
        """Train one iteration of IPO"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration}")
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=256,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Configure DPO training
        training_args = DPOConfig(
            output_dir=checkpoint_path,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            learning_rate=5e-5,
            logging_steps=10,
            eval_steps=100,
            save_steps=100,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=f"ipo_iteration_{iteration}",
            bf16=True,
            remove_unused_columns=False,
        )
        
        # Create trainer
        model = get_peft_model(model, peft_config)
        
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            max_length=1024,
            max_target_length=256,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save model
        trainer.save_model(checkpoint_path)
        
        return train_result.training_loss, trainer.evaluate()['eval_loss']
    
    def evaluate_cross_dataset(self, model, tokenizer, eval_datasets: Dict[str, Dataset]) -> Dict[str, float]:
        """Evaluate model performance across multiple datasets"""
        scores = {}
        
        for dataset_name, dataset in eval_datasets.items():
            correct = 0
            total = 0
            
            for example in tqdm(dataset[:100], desc=f"Evaluating on {dataset_name}"):
                if 'chosen' in example and 'rejected' in example:
                    chosen_score = self.evaluate_response(
                        model, tokenizer, example['instruction'], example['chosen']
                    )
                    rejected_score = self.evaluate_response(
                        model, tokenizer, example['instruction'], example['rejected']
                    )
                    
                    if chosen_score > rejected_score:
                        correct += 1
                    total += 1
            
            scores[dataset_name] = correct / total if total > 0 else 0.0
        
        return scores
    
    def calculate_preference_agreement(self, current_prefs: Dataset, previous_prefs: Dataset) -> float:
        """Calculate agreement between current and previous preference rankings"""
        if not previous_prefs:
            return 1.0
        
        agreements = []
        for curr, prev in zip(current_prefs[:100], previous_prefs[:100]):
            if curr['instruction'] == prev['instruction']:
                # Check if preference order is maintained
                curr_agrees = curr['chosen'] == prev['chosen']
                agreements.append(float(curr_agrees))
        
        return np.mean(agreements) if agreements else 0.0
    
    def calculate_response_diversity(self, dataset: Dataset) -> float:
        """Calculate diversity of generated responses"""
        from collections import Counter
        
        all_responses = []
        for example in dataset[:100]:
            all_responses.extend([example['chosen'], example['rejected']])
        
        # Use unique n-grams as diversity measure
        all_ngrams = []
        for response in all_responses:
            words = response.split()
            ngrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            all_ngrams.extend(ngrams)
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early due to performance plateau or degradation"""
        if len(self.iteration_metrics) < self.config.early_stopping_patience:
            return False
        
        recent_metrics = self.iteration_metrics[-self.config.early_stopping_patience:]
        
        # Check for performance plateau
        eval_losses = [m.eval_loss for m in recent_metrics]
        if np.std(eval_losses) < self.config.performance_threshold:
            print("Early stopping: Performance plateau detected")
            return True
        
        # Check for consistent degradation
        self_eval_scores = [m.self_eval_accuracy for m in recent_metrics]
        if all(self_eval_scores[i] <= self_eval_scores[i-1] for i in range(1, len(self_eval_scores))):
            print("Early stopping: Consistent performance degradation detected")
            return True
        
        return False
    
    def run_experiment(self):
        """Run the full iterative IPO experiment"""
        # Initialize wandb
        wandb.init(project=self.config.wandb_project, config=asdict(self.config))
        
        # Load evaluation datasets
        eval_datasets = {}
        for dataset_name in self.config.eval_datasets:
            eval_datasets[dataset_name] = load_dataset(dataset_name, split="test[:500]")
        
        # Load base training dataset
        base_dataset = load_dataset(self.config.base_dataset, split="train")
        
        previous_prefs = None
        checkpoint_path = None
        
        for iteration in range(self.config.max_iterations):
            print(f"\n{'='*50}")
            print(f"Starting Iteration {iteration + 1}/{self.config.max_iterations}")
            print(f"{'='*50}")
            
            # Load model (from checkpoint if not first iteration)
            model, tokenizer = self.load_model_and_tokenizer(checkpoint_path)
            
            # Generate self-preferences
            train_prefs = self.generate_self_preferences(
                model, tokenizer, 
                base_dataset.select(range(self.config.samples_per_iteration)),
                iteration + 1
            )
            
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
            self_eval_accuracy = np.mean(list(cross_dataset_scores.values()))
            
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
                **{f"cross_dataset/{k}": v for k, v in cross_dataset_scores.items()}
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
    
    def save_results(self):
        """Save experiment results to file"""
        results = {
            "config": asdict(self.config),
            "metrics": [asdict(m) for m in self.iteration_metrics]
        }
        
        results_file = os.path.join(self.config.results_dir, "iteration_metrics.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_plots(self):
        """Generate visualization plots for the experiment"""
        iterations = [m.iteration for m in self.iteration_metrics]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training and Eval Loss
        ax1 = axes[0, 0]
        ax1.plot(iterations, [m.train_loss for m in self.iteration_metrics], 'b-', label='Train Loss')
        ax1.plot(iterations, [m.eval_loss for m in self.iteration_metrics], 'r-', label='Eval Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Self-Evaluation Accuracy
        ax2 = axes[0, 1]
        ax2.plot(iterations, [m.self_eval_accuracy for m in self.iteration_metrics], 'g-', marker='o')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Self-Evaluation Accuracy Over Iterations')
        ax2.grid(True)
        
        # Plot 3: Cross-Dataset Performance
        ax3 = axes[1, 0]
        for dataset_name in self.config.eval_datasets:
            scores = [m.cross_dataset_scores.get(dataset_name, 0) for m in self.iteration_metrics]
            ax3.plot(iterations, scores, marker='o', label=dataset_name)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Cross-Dataset Performance')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Preference Agreement and Diversity
        ax4 = axes[1, 1]
        ax4.plot(iterations, [m.preference_agreement for m in self.iteration_metrics], 'b-', label='Preference Agreement')
        ax4.plot(iterations, [m.response_diversity for m in self.iteration_metrics], 'r-', label='Response Diversity')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Score')
        ax4.set_title('Preference Stability and Response Diversity')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, 'iteration_metrics.png'), dpi=300)
        plt.close()
        
        # Create heatmap for cross-dataset scores
        if len(self.config.eval_datasets) > 1:
            scores_matrix = []
            for m in self.iteration_metrics:
                scores_matrix.append([m.cross_dataset_scores.get(d, 0) for d in self.config.eval_datasets])
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                np.array(scores_matrix).T,
                xticklabels=iterations,
                yticklabels=self.config.eval_datasets,
                cmap='RdYlGn',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Accuracy'}
            )
            plt.xlabel('Iteration')
            plt.ylabel('Dataset')
            plt.title('Cross-Dataset Performance Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.results_dir, 'cross_dataset_heatmap.png'), dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Iterative IPO Experiments")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--base_dataset", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--eval_datasets", nargs="+", default=["truthful_qa", "gsm8k", "hellaswag"])
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--samples_per_iteration", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/iterative_ipo")
    parser.add_argument("--results_dir", type=str, default="./results/iterative_ipo")
    parser.add_argument("--wandb_project", type=str, default="iterative-ipo-experiments")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        model_id=args.model_id,
        base_dataset=args.base_dataset,
        eval_datasets=args.eval_datasets,
        max_iterations=args.max_iterations,
        samples_per_iteration=args.samples_per_iteration,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        wandb_project=args.wandb_project
    )
    
    experiment = IterativeIPO(config)
    experiment.run_experiment()

if __name__ == "__main__":
    main()