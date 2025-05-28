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
    
    # External benchmark scores
    gsm8k_accuracy: float = 0.0
    truthful_qa_score: float = 0.0
    hellaswag_accuracy: float = 0.0
    
    timestamp: str = ""

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
    
    # External evaluation settings
    enable_external_eval: bool = True
    external_eval_frequency: int = 1  # Evaluate every N iterations
    external_eval_datasets: List[str] = None  # Default: ['gsm8k', 'truthful_qa', 'hellaswag']
    external_eval_samples: int = 100  # Number of samples per dataset
    
    # Multi-GPU configuration (future)
    use_multigpu: bool = False
    gpus: Optional[List[int]] = None
    parallel_mode: str = "ddp"
    global_batch_size: int = 24
    
    def __post_init__(self):
        """Set default values that depend on other fields"""
        if self.external_eval_datasets is None:
            self.external_eval_datasets = ['gsm8k', 'truthful_qa', 'hellaswag']