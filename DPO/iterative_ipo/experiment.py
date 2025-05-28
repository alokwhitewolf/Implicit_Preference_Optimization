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
from .evaluation.self_evaluator import SelfEvaluator
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
        self.self_evaluator = SelfEvaluator(config)
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
            
            # Step 7: Calculate additional metrics
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
                **{f"category/{k}": v for k, v in category_scores.items()},
            })
            
            # Save results
            self.results_reporter.save_results(self.iteration_metrics, iteration + 1)
            
            # Handle selective model saving after evaluation
            if not getattr(self.config, 'save_all_checkpoints', True):
                self._handle_selective_saving(self_eval_accuracy, iteration + 1)
            
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
        
        # TODO: Move SFT logic to SFTTrainerWrapper
        print(f"🔧 Running SFT training on Dolly-15k (Iteration {iteration})...")
        # [SFT training logic would go here]
        return model
    
    def _train_iteration(self, model, tokenizer, train_dataset, eval_dataset, iteration: int):
        """Train one iteration with paper-aligned configuration"""
        # TODO: Move DPO logic to DPOTrainerWrapper
        # [DPO training logic would go here]
        return 0.0, 0.0  # Placeholder
    
    def _should_stop_early(self) -> bool:
        """Enhanced early stopping with detailed degradation tracking for research"""
        # TODO: Move early stopping logic to MetricsCalculator
        return False  # Placeholder
    
    def _handle_selective_saving(self, current_performance: float, iteration: int):
        """Handle selective model saving - only keep current and best models"""
        # TODO: Move saving logic to ModelManager
        pass  # Placeholder