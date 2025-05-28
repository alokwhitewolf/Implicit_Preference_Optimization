#!/usr/bin/env python3
"""
Metrics calculation for iterative IPO
"""

import numpy as np
from datasets import Dataset
from typing import Dict, List, Optional
from ..core.config import IterationMetrics

class MetricsCalculator:
    """Calculates various metrics for IPO analysis"""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_self_eval_accuracy(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall self-evaluation accuracy from category scores"""
        if not category_scores:
            return 0.0
        
        # Average across categories with non-zero scores
        valid_scores = [score for score in category_scores.values() if score > 0]
        return np.mean(valid_scores) if valid_scores else 0.0
    
    def calculate_preference_agreement(self, current_prefs: Dataset, previous_prefs: Optional[Dataset]) -> float:
        """Calculate agreement between current and previous preference rankings"""
        if previous_prefs is None or len(previous_prefs) == 0:
            return 1.0
        
        try:
            agreements = []
            # Convert datasets to lists for easier comparison
            current_list = current_prefs.to_list() if hasattr(current_prefs, 'to_list') else list(current_prefs)
            previous_list = previous_prefs.to_list() if hasattr(previous_prefs, 'to_list') else list(previous_prefs)
            
            # Limit to first 100 for performance
            current_sample = current_list[:100]
            
            for curr in current_sample:
                # Find matching instruction in previous preferences
                matching = [p for p in previous_list if p.get('instruction') == curr.get('instruction')]
                if matching:
                    prev = matching[0]
                    # Check if preference order is maintained (considering score differences)
                    curr_prefers_chosen = curr.get('chosen_score', 0) > curr.get('rejected_score', 0)
                    prev_prefers_chosen = prev.get('chosen_score', 0) > prev.get('rejected_score', 0)
                    agreements.append(float(curr_prefers_chosen == prev_prefers_chosen))
            
            return np.mean(agreements) if agreements else 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate preference agreement: {e}")
            return 0.0
    
    def calculate_response_diversity(self, dataset: Dataset) -> float:
        """Calculate diversity of generated responses"""
        from collections import Counter
        import string
        
        all_responses = []
        for example in dataset[:100]:
            # Handle both dict and string formats
            if isinstance(example, dict):
                all_responses.extend([example['chosen'], example['rejected']])
        
        # Calculate unique n-grams
        all_ngrams = []
        for response in all_responses:
            # Clean and tokenize
            words = response.lower().translate(str.maketrans('', '', string.punctuation)).split()
            # Generate 3-grams
            ngrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            all_ngrams.extend(ngrams)
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        # Also calculate average response length variance
        lengths = [len(r.split()) for r in all_responses]
        length_variance = np.var(lengths) if lengths else 0
        
        diversity_score = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        
        # Combine diversity metrics
        return diversity_score * (1 + min(length_variance / 1000, 1))  # Weighted by length variance
    
    def should_stop_early(self, iteration_metrics: List[IterationMetrics]) -> bool:
        """Enhanced early stopping with detailed degradation tracking for research"""
        # If forced iterations specified, ignore early stopping
        if self.config.forced_iterations and len(iteration_metrics) < self.config.forced_iterations:
            return False
            
        if len(iteration_metrics) < self.config.early_stopping_patience:
            return False
        
        # For research: Track but don't stop early unless explicitly degrading
        if not self.config.track_degradation:
            return False
        
        all_metrics = iteration_metrics
        
        # Plateau detection with configurable window
        window = self.config.plateau_detection_window
        if len(all_metrics) >= window:
            recent_scores = [m.self_eval_accuracy for m in all_metrics[-window:]]
            if np.std(recent_scores) < self.config.performance_threshold:
                print(f"Plateau detected: Performance variance {np.std(recent_scores):.4f} < threshold {self.config.performance_threshold}")
                # For research: continue to observe degradation
                if len(all_metrics) >= self.config.max_iterations * 0.8:  # Only stop if near max iterations
                    return True
        
        # Severe degradation detection (>20% drop from peak)
        if len(all_metrics) >= 3:
            peak_performance = max([m.self_eval_accuracy for m in all_metrics])
            current_performance = all_metrics[-1].self_eval_accuracy
            degradation_ratio = (peak_performance - current_performance) / peak_performance
            
            if degradation_ratio > 0.2:  # 20% degradation
                print(f"Severe degradation detected: {degradation_ratio:.1%} drop from peak {peak_performance:.3f}")
                return True
        
        # Catastrophic forgetting detection (enhanced)
        for category in ['code', 'math', 'chat']:
            if len(all_metrics) >= 3:
                category_scores = [m.category_scores.get(category, 0) for m in all_metrics]
                if any(s > 0 for s in category_scores):  # Only check if category has data
                    peak_score = max(category_scores)
                    current_score = category_scores[-1]
                    if peak_score > 0 and (peak_score - current_score) / peak_score > 0.5:  # 50% drop
                        print(f"Catastrophic forgetting in {category}: {current_score:.3f} vs peak {peak_score:.3f}")
                        return True
        
        return False