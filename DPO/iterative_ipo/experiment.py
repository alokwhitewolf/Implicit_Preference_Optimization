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
            # Ensure we don't exceed dataset size
            available_samples = len(base_dataset)
            samples_to_use = min(self.config.samples_per_iteration, available_samples)
            
            if samples_to_use < self.config.samples_per_iteration:
                print(f"⚠️ Dataset has only {available_samples} samples, using all available instead of {self.config.samples_per_iteration}")
            
            train_prefs = self.preference_generator.generate_self_preferences(
                model, tokenizer, 
                base_dataset.select(range(samples_to_use)),
                iteration + 1
            )
            
            # Step 4: Prepare DPO data
            train_dataset, eval_dataset = self.data_manager.prepare_dpo_data(train_prefs, iteration + 1)
            
            # Log dataset statistics to wandb
            dataset_stats = self._calculate_dataset_statistics(train_prefs, train_dataset, eval_dataset, iteration + 1)
            wandb.log({
                f"dataset_stats/iteration_{iteration + 1}/train_size": len(train_dataset),
                f"dataset_stats/iteration_{iteration + 1}/eval_size": len(eval_dataset),
                f"dataset_stats/iteration_{iteration + 1}/preference_pairs": len(train_prefs),
                **{f"dataset_stats/iteration_{iteration + 1}/{k}": v for k, v in dataset_stats.items()}
            })
            
            # Step 5: DPO training on self-generated preferences
            train_loss, eval_loss, training_metrics = self._train_iteration(
                model, tokenizer, train_dataset, eval_dataset, iteration + 1
            )
            
            # Log detailed training metrics
            if training_metrics:
                wandb.log({
                    f"training/iteration_{iteration + 1}/final_train_loss": train_loss,
                    f"training/iteration_{iteration + 1}/final_eval_loss": eval_loss,
                    **{f"training/iteration_{iteration + 1}/{k}": v for k, v in training_metrics.items()}
                })
            
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
            
            # Log to wandb with comprehensive metrics
            wandb_log = {
                # Core iteration metrics
                "iteration": iteration + 1,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "self_eval_accuracy": self_eval_accuracy,
                "preference_agreement": preference_agreement,
                "response_diversity": response_diversity,
                
                # Performance metrics
                "performance/train_loss": train_loss,
                "performance/eval_loss": eval_loss,
                "performance/loss_difference": abs(train_loss - eval_loss),
                "performance/self_eval_accuracy": self_eval_accuracy,
                
                # Preference quality metrics
                "preferences/agreement": preference_agreement,
                "preferences/diversity": response_diversity,
                "preferences/stability_score": preference_agreement * response_diversity,  # Combined metric
                
                # Category performance breakdown
                **{f"category/{k}": v for k, v in category_scores.items()},
                **{f"category_performance/{k}": v for k, v in category_scores.items()},
                
                # Iteration progress metrics
                "progress/iteration_pct": (iteration + 1) / self.config.max_iterations * 100,
                "progress/iterations_completed": iteration + 1,
                "progress/iterations_remaining": self.config.max_iterations - (iteration + 1),
            }
            
            # Add performance trends (if we have previous iterations)
            if len(self.iteration_metrics) > 0:
                prev_accuracy = self.iteration_metrics[-1].self_eval_accuracy if len(self.iteration_metrics) > 0 else 0
                accuracy_change = self_eval_accuracy - prev_accuracy
                wandb_log.update({
                    "trends/accuracy_change": accuracy_change,
                    "trends/accuracy_trend": "improving" if accuracy_change > 0.01 else "declining" if accuracy_change < -0.01 else "stable",
                    "trends/peak_accuracy_so_far": max([m.self_eval_accuracy for m in self.iteration_metrics] + [self_eval_accuracy]),
                })
            
            # Add RewardBench IPO scores if available
            if rewardbench_scores:
                wandb_log.update({
                    "rewardbench/chat": rewardbench_scores.get('rewardbench_chat', 0.0),
                    "rewardbench/code": rewardbench_scores.get('rewardbench_code', 0.0),
                    "rewardbench/math": rewardbench_scores.get('rewardbench_math', 0.0),
                    "rewardbench/safety": rewardbench_scores.get('rewardbench_safety', 0.0),
                    "rewardbench/overall": rewardbench_scores.get('rewardbench_overall', 0.0),
                    
                    # RewardBench performance metrics
                    "rewardbench_performance/overall": rewardbench_scores.get('rewardbench_overall', 0.0),
                    "rewardbench_performance/avg_category": sum([
                        rewardbench_scores.get('rewardbench_chat', 0.0),
                        rewardbench_scores.get('rewardbench_code', 0.0),
                        rewardbench_scores.get('rewardbench_math', 0.0),
                        rewardbench_scores.get('rewardbench_safety', 0.0)
                    ]) / 4,
                })
            
            # Add experiment metadata
            wandb_log.update({
                "experiment/model_id": self.config.model_id,
                "experiment/base_dataset": self.config.base_dataset,
                "experiment/samples_per_iteration": self.config.samples_per_iteration,
                "experiment/timestamp": datetime.now().isoformat(),
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
        
        # Log comprehensive experiment summary
        self._log_experiment_summary()
        
        wandb.finish()
    
    def _run_sft_training_if_needed(self, model, tokenizer, iteration: int):
        """SFT training on Dolly-15k before DPO (for base models only)"""
        # Only run SFT for base models (not instruct models)
        if "Instruct" in self.config.model_id or "instruct" in self.config.model_id.lower():
            print("✓ Skipping SFT - using pre-trained instruct model")
            return model
        
        print(f"🔧 Running SFT training on Dolly-15k (Iteration {iteration})...")
        return self.sft_trainer.train(model, tokenizer, iteration)
    
    def _calculate_dataset_statistics(self, train_prefs, train_dataset, eval_dataset, iteration: int):
        """Calculate comprehensive dataset statistics for logging"""
        import numpy as np
        from collections import Counter
        
        stats = {}
        
        # Preference generation statistics
        if len(train_prefs) > 0:
            # Category distribution
            categories = [ex['category'] for ex in train_prefs if 'category' in ex]
            category_dist = Counter(categories)
            for cat, count in category_dist.items():
                stats[f"category_distribution/{cat}"] = count
                stats[f"category_percentage/{cat}"] = count / len(categories) * 100 if categories else 0
            
            # Score statistics
            chosen_scores = [ex['chosen_score'] for ex in train_prefs if 'chosen_score' in ex]
            rejected_scores = [ex['rejected_score'] for ex in train_prefs if 'rejected_score' in ex]
            score_diffs = [ex['score_diff'] for ex in train_prefs if 'score_diff' in ex]
            
            if chosen_scores:
                stats['score_stats/chosen_mean'] = np.mean(chosen_scores)
                stats['score_stats/chosen_std'] = np.std(chosen_scores)
                stats['score_stats/chosen_min'] = np.min(chosen_scores)
                stats['score_stats/chosen_max'] = np.max(chosen_scores)
            
            if rejected_scores:
                stats['score_stats/rejected_mean'] = np.mean(rejected_scores)
                stats['score_stats/rejected_std'] = np.std(rejected_scores)
                stats['score_stats/rejected_min'] = np.min(rejected_scores)
                stats['score_stats/rejected_max'] = np.max(rejected_scores)
            
            if score_diffs:
                stats['score_stats/diff_mean'] = np.mean(score_diffs)
                stats['score_stats/diff_std'] = np.std(score_diffs)
                stats['score_stats/diff_min'] = np.min(score_diffs)
                stats['score_stats/diff_max'] = np.max(score_diffs)
                stats['score_stats/high_confidence_pairs'] = sum(1 for d in score_diffs if d > 0.3)
            
            # Text length statistics
            chosen_lengths = [len(ex['chosen'].split()) for ex in train_prefs if 'chosen' in ex]
            rejected_lengths = [len(ex['rejected'].split()) for ex in train_prefs if 'rejected' in ex]
            
            if chosen_lengths:
                stats['text_stats/chosen_length_mean'] = np.mean(chosen_lengths)
                stats['text_stats/chosen_length_std'] = np.std(chosen_lengths)
            
            if rejected_lengths:
                stats['text_stats/rejected_length_mean'] = np.mean(rejected_lengths)
                stats['text_stats/rejected_length_std'] = np.std(rejected_lengths)
        
        # Dataset quality metrics
        stats['quality/preference_pairs_generated'] = len(train_prefs)
        stats['quality/train_eval_split_ratio'] = len(train_dataset) / len(eval_dataset) if len(eval_dataset) > 0 else 0
        
        return stats
    
    def _train_iteration(self, model, tokenizer, train_dataset, eval_dataset, iteration: int):
        """Train one iteration and return detailed metrics"""
        # Enhanced training with detailed metrics collection
        train_loss, eval_loss, training_metrics = self.dpo_trainer.train_with_metrics(
            model, tokenizer, train_dataset, eval_dataset, iteration
        )
        return train_loss, eval_loss, training_metrics
    
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
    
    def _log_experiment_summary(self):
        """Log comprehensive experiment summary to wandb"""
        # Calculate final metrics
        final_metrics = self.metrics_calculator.calculate_final_metrics(self.iteration_metrics)
        
        # Log final metrics to wandb
        wandb.log({
            "final_metrics/best_performance": self.best_performance,
            "final_metrics/best_iteration": self.best_iteration,
            **{f"final_metrics/{k}": v for k, v in final_metrics.items()}
        })
        
        # Log trends and final statistics
        trends = self.metrics_calculator.calculate_trends(self.iteration_metrics)
        wandb.log({
            "trends/best_performance_trend": trends.get('best_performance_trend', 'stable'),
            "trends/best_iteration_trend": trends.get('best_iteration_trend', 'stable'),
            **{f"trends/{k}": v for k, v in trends.items()}
        })
        
        # Log final statistics
        stats = self.metrics_calculator.calculate_final_statistics(self.iteration_metrics)
        wandb.log({
            "final_statistics/best_performance": self.best_performance,
            "final_statistics/best_iteration": self.best_iteration,
            **{f"final_statistics/{k}": v for k, v in stats.items()}
        })