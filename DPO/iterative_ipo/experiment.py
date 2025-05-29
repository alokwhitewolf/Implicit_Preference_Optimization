#!/usr/bin/env python3
"""
Main Iterative IPO experiment class (refactored)
Focus: Pure self-improvement dynamics without cross-dataset evaluation
"""

import os
import torch
import wandb
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional

from .core.config import ExperimentConfig, IterationMetrics
from .core.model_manager import ModelManager
from .core.data_manager import DataManager
from .training.preference_gen import PreferenceGenerator
from .training.sft_trainer import SFTTrainerWrapper
from .training.dpo_trainer import DPOTrainerWrapper
from .evaluation.self_evaluator import SelfEvaluator
from .evaluation.rewardbench_evaluator import RewardBenchIPOEvaluator
from .evaluation.fast_rewardbench_evaluator import FastRewardBenchIPOEvaluator
from .analysis.metrics import MetricsCalculator
from .analysis.reporting import ResultsReporter

class IterativeIPO:
    """Main class for iterative self-improvement experiments (refactored)"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.iteration_metrics: List[IterationMetrics] = []
        self.best_performance = 0.0
        self.best_iteration = 0
        
        # Initialize managers
        self.model_manager = ModelManager(config)
        self.data_manager = DataManager(config)
        self.preference_generator = PreferenceGenerator(config)
        self.sft_trainer = SFTTrainerWrapper(config)
        self.dpo_trainer = DPOTrainerWrapper(config)
        self.self_evaluator = SelfEvaluator(config)
        # Choose evaluator based on config
        if getattr(config, 'use_fast_rewardbench', True):
            self.rewardbench_evaluator = FastRewardBenchIPOEvaluator(config)
            print("🚀 Using fast RewardBench evaluator with batching")
        else:
            self.rewardbench_evaluator = RewardBenchIPOEvaluator(config)
            print("🐌 Using standard RewardBench evaluator")
        self.metrics_calculator = MetricsCalculator(config)
        self.results_reporter = ResultsReporter(config)
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for checkpoints and results"""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        # Create subdirectories for essential checkpoints only
        Path(os.path.join(self.config.checkpoint_dir, "current")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.config.checkpoint_dir, "best")).mkdir(parents=True, exist_ok=True)
    
    def run_experiment(self):
        """
        Run the full iterative IPO experiment with paper alignment
        
        Focus: Pure self-improvement dynamics (like ours.py)
        NO cross-dataset evaluation (that's a separate experiment)
        """
        # Initialize wandb with paper-specific tags
        wandb.init(
            project=self.config.wandb_project,
            config=asdict(self.config),
            tags=["iterative-ipo", "self-improvement", self.config.model_id.split("/")[-1]]
        )
        
        # Load base training dataset
        base_dataset = self.data_manager.load_base_dataset()
        
        previous_prefs = None
        checkpoint_path = None
        
        for iteration in range(self.config.max_iterations):
            print(f"\n{'='*50}")
            print(f"Starting Iteration {iteration + 1}/{self.config.max_iterations}")
            print(f"{'='*50}")
            
            # Clear CUDA cache before loading new model
            torch.cuda.empty_cache()
            
            # Step 1: Load model (base or from previous iteration)
            model, tokenizer = self.model_manager.load_model_and_tokenizer(checkpoint_path)
            
            # Step 2: SFT training (only for base models on first iteration)
            if iteration == 0:
                model = self._run_sft_training_if_needed(model, tokenizer, iteration + 1)
            
            # Step 3: Generate self-preferences from base dataset
            train_prefs = self.preference_generator.generate_self_preferences(
                model, tokenizer, 
                base_dataset.select(range(self.config.samples_per_iteration)),
                iteration + 1
            )
            
            # Step 4: Prepare DPO data
            train_dataset, eval_dataset = self.data_manager.prepare_dpo_data(train_prefs, iteration + 1)
            
            # Step 5: DPO training on self-generated preferences
            train_loss, eval_loss = self._train_iteration(
                model, tokenizer, train_dataset, eval_dataset, iteration + 1
            )
            
            # Step 6: Self-evaluation (category-specific accuracy from preferences)
            category_scores = self.self_evaluator.evaluate_categories(train_prefs)
            self_eval_accuracy = self.metrics_calculator.calculate_self_eval_accuracy(category_scores)
            
            # Step 7: RewardBench IPO evaluation (matches paper methodology)
            rewardbench_scores = {}
            if (self.config.enable_rewardbench_eval and 
                iteration % self.config.rewardbench_eval_frequency == 0):
                rewardbench_scores = self.rewardbench_evaluator.evaluate_rewardbench_ipo(
                    model, tokenizer, iteration + 1
                )
            
            # Step 8: Calculate additional metrics
            preference_agreement = self.metrics_calculator.calculate_preference_agreement(train_prefs, previous_prefs)
            response_diversity = self.metrics_calculator.calculate_response_diversity(train_prefs)
            
            # Record metrics
            metrics = IterationMetrics(
                iteration=iteration + 1,
                train_loss=train_loss,
                eval_loss=eval_loss,
                self_eval_accuracy=self_eval_accuracy,
                preference_agreement=preference_agreement,
                response_diversity=response_diversity,
                category_scores=category_scores,
                
                # RewardBench IPO scores (matches paper methodology)
                rewardbench_chat=rewardbench_scores.get('rewardbench_chat', 0.0),
                rewardbench_code=rewardbench_scores.get('rewardbench_code', 0.0),
                rewardbench_math=rewardbench_scores.get('rewardbench_math', 0.0),
                rewardbench_safety=rewardbench_scores.get('rewardbench_safety', 0.0),
                rewardbench_overall=rewardbench_scores.get('rewardbench_overall', 0.0),
                
                timestamp=datetime.now().isoformat()
            )
            self.iteration_metrics.append(metrics)
            
            # Log to wandb
            wandb_log = {
                "iteration": iteration + 1,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "self_eval_accuracy": self_eval_accuracy,
                "preference_agreement": preference_agreement,
                "response_diversity": response_diversity,
                **{f"category/{k}": v for k, v in category_scores.items()},
            }
            
            # Add RewardBench IPO scores if available
            if rewardbench_scores:
                wandb_log.update({
                    "rewardbench/chat": rewardbench_scores.get('rewardbench_chat', 0.0),
                    "rewardbench/code": rewardbench_scores.get('rewardbench_code', 0.0),
                    "rewardbench/math": rewardbench_scores.get('rewardbench_math', 0.0),
                    "rewardbench/safety": rewardbench_scores.get('rewardbench_safety', 0.0),
                    "rewardbench/overall": rewardbench_scores.get('rewardbench_overall', 0.0),
                })
            
            wandb.log(wandb_log)
            
            # Save results
            self.results_reporter.save_results(self.iteration_metrics, iteration + 1)
            
            # Handle selective model saving after evaluation
            # Use RewardBench overall performance if available, otherwise fall back to self-eval
            performance_metric = (
                rewardbench_scores.get('rewardbench_overall', 0.0) if rewardbench_scores 
                else self_eval_accuracy
            )
            
            if not getattr(self.config, 'save_all_checkpoints', True):
                self._handle_selective_saving(performance_metric, iteration + 1)
            
            # Update for next iteration
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration + 1}")
            previous_prefs = train_prefs
            
            # Check early stopping
            if self._should_stop_early():
                print(f"Stopping early at iteration {iteration + 1}")
                break
            
            # Clean up memory
            del model
            torch.cuda.empty_cache()
        
        # Generate final visualizations
        self.results_reporter.generate_plots(self.iteration_metrics)
        wandb.finish()
    
    def _run_sft_training_if_needed(self, model, tokenizer, iteration: int):
        """SFT training on Dolly-15k before DPO (for base models only)"""
        # Only run SFT for base models (not instruct models)
        if "Instruct" in self.config.model_id or "instruct" in self.config.model_id.lower():
            print("✓ Skipping SFT - using pre-trained instruct model")
            return model
        
        print(f"🔧 Running SFT training on Dolly-15k (Iteration {iteration})...")
        return self.sft_trainer.train(model, tokenizer, iteration)
    
    def _train_iteration(self, model, tokenizer, train_dataset, eval_dataset, iteration: int):
        """Train one iteration with paper-aligned configuration"""
        print(f"🚀 Training DPO iteration {iteration}...")
        return self.dpo_trainer.train(model, tokenizer, train_dataset, eval_dataset, iteration)
    
    def _should_stop_early(self) -> bool:
        """Enhanced early stopping with detailed degradation tracking for research"""
        if len(self.iteration_metrics) < 3:
            return False
        
        # Use RewardBench overall scores if available, otherwise self-eval
        recent_scores = []
        for m in self.iteration_metrics[-3:]:
            # Prioritize RewardBench overall as primary metric, fallback to self-eval
            if m.rewardbench_overall > 0:
                recent_scores.append(m.rewardbench_overall)
            else:
                recent_scores.append(m.self_eval_accuracy)
        
        # Check for consistent degradation
        degradation_count = 0
        for i in range(1, len(recent_scores)):
            if recent_scores[i] < recent_scores[i-1]:
                degradation_count += 1
        
        # Stop if degrading for 2+ consecutive iterations
        if degradation_count >= 2:
            metric_name = "RewardBench overall" if self.iteration_metrics[-1].rewardbench_overall > 0 else "self-eval accuracy"
            print(f"⚠️ {metric_name} degrading: {recent_scores}")
            return True
        
        return False
    
    def _handle_selective_saving(self, current_performance: float, iteration: int):
        """Handle selective model saving - only keep current and best models"""
        # Update best performance tracking
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.best_iteration = iteration
            
            # Save as best model
            best_path = os.path.join(self.config.checkpoint_dir, "best")
            current_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration}")
            
            print(f"🏆 New best performance: {current_performance:.4f} (iteration {iteration})")
            self.model_manager.copy_checkpoint(current_path, best_path)
        
        # Clean up old checkpoints (keep only current and best)
        if iteration > 1:
            old_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration-1}")
            if os.path.exists(old_path) and iteration-1 != self.best_iteration:
                print(f"🧹 Cleaning up iteration {iteration-1} checkpoint")
                self.model_manager.cleanup_checkpoint(old_path)