#!/usr/bin/env python3
"""
Configuration classes for Iterative IPO experiments
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class IterationMetrics:
    """Metrics for a single iteration of IPO"""
    iteration: int
    train_loss: float
    eval_loss: float
    self_eval_accuracy: float
    preference_agreement: float
    response_diversity: float
    category_scores: Dict[str, float]
    timestamp: str

@dataclass
class ExperimentConfig:
    """Configuration for iterative IPO experiments"""
    # Core model and data
    model_id: str
    base_dataset: str
    
    # Training parameters
    max_iterations: int = 25
    samples_per_iteration: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 5
    performance_threshold: float = 0.005
    
    # Directories
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    wandb_project: str = "iterative-ipo"
    
    # Technical settings
    use_4bit: bool = True
    
    # Research settings
    save_all_checkpoints: bool = False
    track_degradation: bool = True
    plateau_detection_window: int = 3
    forced_iterations: Optional[int] = None
    
    # GPU optimization
    instruction_batch_size: int = 4
    eval_batch_size: int = 16
    
    # Multi-GPU configuration (future)
    use_multigpu: bool = False
    gpus: Optional[List[int]] = None
    parallel_mode: str = "ddp"
    global_batch_size: int = 24