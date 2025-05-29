#!/usr/bin/env python3
"""
Main entry point for refactored Iterative IPO experiment
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import ExperimentConfig
from experiment import IterativeIPO

def main():
    """Run the refactored iterative IPO experiment"""
    
    # Configuration matching ours.py exactly
    config = ExperimentConfig(
        # Model configuration (matches ours.py)
        model_id="microsoft/Phi-3.5-mini-instruct",
        use_4bit=True,
        
        # Training hyperparameters (matches ours.py exactly)
        learning_rate=5e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=3,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        # LoRA configuration (matches ours.py exactly)
        lora_alpha=128,
        lora_dropout=0.05,
        lora_r=256,
        lora_target_modules="all-linear",
        
        # DPO configuration (matches ours.py exactly)
        dpo_beta=0.1,
        dpo_loss_type="sigmoid",
        
        # Experiment settings
        max_iterations=5,
        samples_per_iteration=1000,
        num_responses_per_instruction=4,
        save_all_checkpoints=False,  # Only save current and best
        
        # RewardBench IPO evaluation settings (matches paper)
        enable_rewardbench_eval=True,
        rewardbench_eval_frequency=1,  # Evaluate every iteration
        rewardbench_samples=100,  # Number of RewardBench samples
        
        # Paths
        base_dataset_path="databricks-dolly-15k",
        checkpoint_dir="./checkpoints_refactored_Phi-3.5-mini-instruct",
        results_dir="./results_refactored_Phi-3.5-mini-instruct",
        
        # Logging
        wandb_project="iterative-ipo-refactored",
        log_level="INFO"
    )
    
    print("🚀 Starting Refactored Iterative IPO Experiment")
    print(f"Model: {config.model_id}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Samples per iteration: {config.samples_per_iteration}")
    print(f"RewardBench IPO evaluation: {config.enable_rewardbench_eval}")
    if config.enable_rewardbench_eval:
        print(f"  Categories: Chat, Code, Math, Safety")
        print(f"  Samples: {config.rewardbench_samples}")
        print(f"  Frequency: every {config.rewardbench_eval_frequency} iteration(s)")
        print(f"  Method: P(Yes) probability extraction (IPO paper)")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Results dir: {config.results_dir}")
    
    # Create and run experiment
    experiment = IterativeIPO(config)
    experiment.run_experiment()
    
    print("✅ Experiment completed!")

if __name__ == "__main__":
    main()