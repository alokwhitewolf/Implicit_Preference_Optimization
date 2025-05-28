#!/usr/bin/env python3
"""
Self-evaluation logic for category-specific performance
"""

import numpy as np
from datasets import Dataset
from typing import Dict

class SelfEvaluator:
    """Handles self-evaluation of generated preferences"""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_categories(self, train_prefs: Dataset) -> Dict[str, float]:
        """
        Evaluate category-specific performance based on preference consistency
        
        Note: This evaluates the consistency of the preference generation,
        not external task performance. For external evaluation, use cross_evaluator.py
        """
        category_scores = {}
        
        for category in ['code', 'math', 'chat', 'safety', 'reasoning']:
            cat_data = train_prefs.filter(lambda x: x['category'] == category)
            if len(cat_data) > 0:
                # Calculate accuracy based on chosen vs rejected scores
                cat_accuracy = np.mean([
                    1 if ex['chosen_score'] > ex['rejected_score'] else 0 
                    for ex in cat_data
                ])
                category_scores[category] = cat_accuracy
            else:
                category_scores[category] = 0.0
        
        return category_scores