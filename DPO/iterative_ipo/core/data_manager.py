#!/usr/bin/env python3
"""
Dataset loading and preparation for Iterative IPO
"""

from datasets import load_dataset, Dataset
from typing import Dict, Tuple
from .config import ExperimentConfig

class DataManager:
    """Handles dataset loading and preparation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def load_base_dataset(self) -> Dataset:
        """Load base training dataset"""
        # For UltraFeedback-1k-Each, use split_1; for other datasets, use train split
        if "UltraFeedback-1k-Each" in self.config.base_dataset:
            dataset = load_dataset(self.config.base_dataset, split="split_1")
            print(f"✓ Using balanced UltraFeedback dataset (split_1) with 1k examples per category")
        else:
            dataset = load_dataset(self.config.base_dataset, split="train")
            print(f"✓ Using dataset: {self.config.base_dataset}")
        
        return dataset
    
    def prepare_dpo_data(self, preferences: Dataset, iteration: int) -> Tuple[Dataset, Dataset]:
        """Prepare DPO data in memory (no disk saving) for training"""
        # Convert Dataset to list for processing
        preferences_list = preferences.to_list() if hasattr(preferences, 'to_list') else list(preferences)
        
        # Transform to DPO format (prompt, chosen, rejected)
        transformed_data = []
        for entry in preferences_list:
            transformed = self._transform_entry(entry)
            transformed_data.append(transformed)
        
        # Split into train/test (90/10)
        train_size = int(0.9 * len(transformed_data))
        train_data = transformed_data[:train_size]
        test_data = transformed_data[train_size:]
        
        print(f"📊 Prepared DPO data: {len(train_data)} train, {len(test_data)} test examples")
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(test_data)
        
        return train_dataset, eval_dataset
    
    def _transform_entry(self, entry):
        """Transform preference pair to DPO format (from original ours.py)"""
        return {
            "prompt": f"[INST] {entry['instruction']} [/INST]",
            "chosen": f"{entry['chosen']}<|eot_id|>",
            "rejected": f"{entry['rejected']}<|eot_id|>"
        }