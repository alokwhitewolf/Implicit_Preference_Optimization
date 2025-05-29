#!/usr/bin/env python3
"""
Results reporting and visualization for iterative IPO
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List
from datetime import datetime
from ..core.config import IterationMetrics

class ResultsReporter:
    """Handles saving results and generating visualizations"""
    
    def __init__(self, config):
        self.config = config
    
    def save_results(self, iteration_metrics: List[IterationMetrics], current_iteration: int):
        """Save experiment results with detailed analysis"""
        if not iteration_metrics:
            return
            
        # Calculate detailed performance analysis
        self_eval_scores = [m.self_eval_accuracy for m in iteration_metrics]
        peak_performance = max(self_eval_scores)
        peak_iteration = self_eval_scores.index(peak_performance) + 1
        final_performance = self_eval_scores[-1]
        
        # Calculate degradation patterns
        degradation_analysis = {
            "peak_performance": peak_performance,
            "peak_iteration": peak_iteration,
            "final_performance": final_performance,
            "total_degradation": (peak_performance - final_performance) / peak_performance if peak_performance > 0 else 0,
            "iterations_after_peak": len(iteration_metrics) - peak_iteration,
            "plateau_detected": False,
            "catastrophic_forgetting": {}
        }
        
        # Detect plateau
        if len(iteration_metrics) >= 5:
            last_5_scores = self_eval_scores[-5:]
            degradation_analysis["plateau_detected"] = np.std(last_5_scores) < self.config.performance_threshold
        
        # Category-wise degradation analysis
        for category in ['code', 'math', 'chat', 'safety', 'reasoning']:
            category_scores = [m.category_scores.get(category, 0) for m in iteration_metrics]
            if any(s > 0 for s in category_scores):
                cat_peak = max(category_scores)
                cat_final = category_scores[-1]
                degradation_analysis["catastrophic_forgetting"][category] = {
                    "peak_score": cat_peak,
                    "final_score": cat_final,
                    "degradation_ratio": (cat_peak - cat_final) / cat_peak if cat_peak > 0 else 0,
                    "peak_iteration": category_scores.index(cat_peak) + 1
                }
        
        results = {
            "config": self._convert_to_json_serializable(self.config.__dict__),
            "metrics": [self._convert_to_json_serializable(m.__dict__) for m in iteration_metrics],
            "model_family": self.config.model_id.split("/")[0],
            "degradation_analysis": degradation_analysis,
            "final_performance": {
                "iterations_completed": len(iteration_metrics),
                "peak_performance": peak_performance,
                "final_performance": final_performance,
                "performance_trajectory": self_eval_scores
            }
        }
        
        # Convert to JSON-serializable format
        results = self._convert_to_json_serializable(results)
        
        results_file = os.path.join(self.config.results_dir, "iteration_metrics.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Export to CSV for easy analysis
        self._export_metrics_to_csv(iteration_metrics)
    
    def generate_plots(self, iteration_metrics: List[IterationMetrics]):
        """Generate comprehensive visualizations"""
        if not iteration_metrics:
            return
            
        iterations = [m.iteration for m in iteration_metrics]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall performance trajectory
        ax1 = axes[0, 0]
        ax1.plot(iterations, [m.train_loss for m in iteration_metrics], 'b-', label='Train Loss')
        ax1.plot(iterations, [m.eval_loss for m in iteration_metrics], 'r-', label='Eval Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Self-evaluation accuracy
        ax2 = axes[0, 1]
        accuracies = [m.self_eval_accuracy for m in iteration_metrics]
        ax2.plot(iterations, accuracies, 'g-', marker='o', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Self-Evaluation Accuracy')
        ax2.grid(True)
        
        # 3. Category-specific performance
        ax3 = axes[1, 0]
        categories = ['code', 'math', 'chat', 'safety', 'reasoning']
        for category in categories:
            scores = [m.category_scores.get(category, 0) for m in iteration_metrics]
            if any(s > 0 for s in scores):
                ax3.plot(iterations, scores, marker='o', label=category)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Category Accuracy')
        ax3.set_title('Performance by Task Category')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Preference stability vs diversity
        ax4 = axes[1, 1]
        agreements = [m.preference_agreement for m in iteration_metrics]
        diversities = [m.response_diversity for m in iteration_metrics]
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(iterations, agreements, 'b-', label='Preference Agreement')
        line2 = ax4_twin.plot(iterations, diversities, 'r-', label='Response Diversity')
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Preference Agreement', color='b')
        ax4_twin.set_ylabel('Response Diversity', color='r')
        ax4.set_title('Stability vs Diversity Trade-off')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, 'iteration_metrics.png'), dpi=300)
        plt.close()
        
        print(f"📊 Plots saved to {self.config.results_dir}/iteration_metrics.png")
    
    def _export_metrics_to_csv(self, iteration_metrics: List[IterationMetrics]):
        """Export iteration metrics to CSV files for easy analysis and plotting"""
        if not iteration_metrics:
            return
        
        # Main iteration metrics CSV
        iteration_data = []
        for metric in iteration_metrics:
            row = {
                'iteration': metric.iteration,
                'train_loss': getattr(metric, 'train_loss', 0.0),
                'eval_loss': metric.eval_loss,
                'self_eval_accuracy': metric.self_eval_accuracy,
                'preference_agreement': metric.preference_agreement,
                'response_diversity': metric.response_diversity,
                'model_id': self.config.model_id,
                'experiment_type': 'self_improvement',
                'max_iterations': self.config.max_iterations,
                'timestamp': getattr(metric, 'timestamp', datetime.now().isoformat())
            }
            
            # Add category scores as separate columns
            for category, score in metric.category_scores.items():
                row[f'category_{category}'] = score
            
            iteration_data.append(row)
        
        # Save main metrics CSV
        df_metrics = pd.DataFrame(iteration_data)
        metrics_csv = os.path.join(self.config.results_dir, "iteration_metrics.csv")
        df_metrics.to_csv(metrics_csv, index=False)
        
        print(f"📊 CSV exported to: {metrics_csv}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable types to JSON-compatible types"""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj