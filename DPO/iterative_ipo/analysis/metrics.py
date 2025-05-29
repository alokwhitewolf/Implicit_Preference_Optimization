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
    
    def calculate_final_metrics(self, iteration_metrics: List[IterationMetrics]) -> Dict[str, float]:
        """Calculate comprehensive final experiment metrics"""
        if not iteration_metrics:
            return {}
        
        # Performance metrics
        accuracies = [m.self_eval_accuracy for m in iteration_metrics]
        train_losses = [m.train_loss for m in iteration_metrics]
        eval_losses = [m.eval_loss for m in iteration_metrics]
        
        # Find peak performance
        peak_accuracy = max(accuracies)
        peak_iteration = accuracies.index(peak_accuracy) + 1
        final_accuracy = accuracies[-1]
        
        # Calculate degradation
        degradation = (peak_accuracy - final_accuracy) / peak_accuracy if peak_accuracy > 0 else 0
        
        # Stability metrics
        agreements = [m.preference_agreement for m in iteration_metrics]
        diversities = [m.response_diversity for m in iteration_metrics]
        
        # Category performance analysis
        category_trends = {}
        for category in ['code', 'math', 'chat', 'safety', 'reasoning']:
            cat_scores = [m.category_scores.get(category, 0) for m in iteration_metrics]
            if any(s > 0 for s in cat_scores):
                category_trends[f"category_{category}_peak"] = max(cat_scores)
                category_trends[f"category_{category}_final"] = cat_scores[-1]
                category_trends[f"category_{category}_degradation"] = (max(cat_scores) - cat_scores[-1]) / max(cat_scores) if max(cat_scores) > 0 else 0
        
        return {
            "peak_accuracy": peak_accuracy,
            "peak_iteration": peak_iteration,
            "final_accuracy": final_accuracy,
            "total_degradation": degradation,
            "iterations_after_peak": len(iteration_metrics) - peak_iteration,
            "avg_train_loss": sum(train_losses) / len(train_losses),
            "avg_eval_loss": sum(eval_losses) / len(eval_losses),
            "final_train_loss": train_losses[-1],
            "final_eval_loss": eval_losses[-1],
            "avg_preference_agreement": sum(agreements) / len(agreements),
            "avg_response_diversity": sum(diversities) / len(diversities),
            "final_preference_agreement": agreements[-1],
            "final_response_diversity": diversities[-1],
            **category_trends
        }
    
    def calculate_trends(self, iteration_metrics: List[IterationMetrics]) -> Dict[str, any]:
        """Calculate performance trends across iterations"""
        if len(iteration_metrics) < 2:
            return {}
        
        import numpy as np
        from scipy import stats
        
        iterations = list(range(1, len(iteration_metrics) + 1))
        accuracies = [m.self_eval_accuracy for m in iteration_metrics]
        agreements = [m.preference_agreement for m in iteration_metrics]
        diversities = [m.response_diversity for m in iteration_metrics]
        
        # Linear regression for trends
        accuracy_slope, _, accuracy_r, accuracy_p, _ = stats.linregress(iterations, accuracies)
        agreement_slope, _, agreement_r, agreement_p, _ = stats.linregress(iterations, agreements)
        diversity_slope, _, diversity_r, diversity_p, _ = stats.linregress(iterations, diversities)
        
        return {
            "accuracy_slope": accuracy_slope,
            "accuracy_trend": "improving" if accuracy_slope > 0.001 else "declining" if accuracy_slope < -0.001 else "stable",
            "accuracy_correlation": accuracy_r,
            "accuracy_significance": accuracy_p,
            "agreement_slope": agreement_slope,
            "agreement_trend": "improving" if agreement_slope > 0.001 else "declining" if agreement_slope < -0.001 else "stable",
            "agreement_correlation": agreement_r,
            "diversity_slope": diversity_slope,
            "diversity_trend": "improving" if diversity_slope > 0.001 else "declining" if diversity_slope < -0.001 else "stable",
            "diversity_correlation": diversity_r,
        }
    
    def calculate_final_statistics(self, iteration_metrics: List[IterationMetrics]) -> Dict[str, float]:
        """Calculate final experiment statistics"""
        if not iteration_metrics:
            return {}
        
        import numpy as np
        
        # Extract all metrics
        accuracies = [m.self_eval_accuracy for m in iteration_metrics]
        train_losses = [m.train_loss for m in iteration_metrics]
        eval_losses = [m.eval_loss for m in iteration_metrics]
        agreements = [m.preference_agreement for m in iteration_metrics]
        diversities = [m.response_diversity for m in iteration_metrics]
        
        # Calculate statistics
        return {
            "total_iterations": len(iteration_metrics),
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "accuracy_min": np.min(accuracies),
            "accuracy_max": np.max(accuracies),
            "accuracy_range": np.max(accuracies) - np.min(accuracies),
            "train_loss_mean": np.mean(train_losses),
            "train_loss_std": np.std(train_losses),
            "eval_loss_mean": np.mean(eval_losses),
            "eval_loss_std": np.std(eval_losses),
            "agreement_mean": np.mean(agreements),
            "agreement_std": np.std(agreements),
            "diversity_mean": np.mean(diversities),
            "diversity_std": np.std(diversities),
            "stability_score": np.mean(agreements) * np.mean(diversities),  # Combined stability metric
        }