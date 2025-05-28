#!/usr/bin/env python3
"""
Analyze performance degradation patterns in iterative IPO
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Tuple
import argparse

class DegradationAnalyzer:
    """Analyze performance degradation patterns in iterative self-improvement"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.metrics = self.load_metrics()
        
    def load_metrics(self) -> List[Dict]:
        """Load iteration metrics from results file"""
        metrics_file = self.results_dir / "iteration_metrics.json"
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return data['metrics']
    
    def find_peak_performance(self) -> Tuple[int, float]:
        """Find the iteration with peak performance"""
        accuracies = [m['self_eval_accuracy'] for m in self.metrics]
        peak_idx = np.argmax(accuracies)
        return self.metrics[peak_idx]['iteration'], accuracies[peak_idx]
    
    def calculate_degradation_rate(self) -> Dict[str, float]:
        """Calculate rate of performance degradation after peak"""
        peak_iter, peak_acc = self.find_peak_performance()
        
        # Get post-peak metrics
        post_peak_metrics = [m for m in self.metrics if m['iteration'] > peak_iter]
        
        if not post_peak_metrics:
            return {"rate": 0.0, "slope": 0.0, "r_squared": 0.0}
        
        iterations = [m['iteration'] for m in post_peak_metrics]
        accuracies = [m['self_eval_accuracy'] for m in post_peak_metrics]
        
        # Linear regression to find degradation slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(iterations, accuracies)
        
        return {
            "rate": abs(slope),  # degradation rate per iteration
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "peak_iteration": peak_iter,
            "peak_accuracy": peak_acc
        }
    
    def analyze_cross_dataset_transfer(self) -> pd.DataFrame:
        """Analyze how performance on different datasets changes over iterations"""
        transfer_data = []
        
        for i, metric in enumerate(self.metrics):
            for dataset, score in metric['cross_dataset_scores'].items():
                transfer_data.append({
                    'iteration': metric['iteration'],
                    'dataset': dataset,
                    'accuracy': score,
                    'normalized_accuracy': score / self.metrics[0]['cross_dataset_scores'].get(dataset, 1.0)
                })
        
        df = pd.DataFrame(transfer_data)
        
        # Calculate correlation between datasets
        pivot_df = df.pivot(index='iteration', columns='dataset', values='accuracy')
        correlation_matrix = pivot_df.corr()
        
        return df, correlation_matrix
    
    def analyze_preference_stability(self) -> Dict[str, float]:
        """Analyze stability of preferences across iterations"""
        agreements = [m['preference_agreement'] for m in self.metrics[1:]]  # Skip first iteration
        
        return {
            "mean_agreement": np.mean(agreements),
            "std_agreement": np.std(agreements),
            "min_agreement": np.min(agreements),
            "trend": stats.linregress(range(len(agreements)), agreements)[0]
        }
    
    def analyze_response_diversity(self) -> Dict[str, float]:
        """Analyze how response diversity changes over iterations"""
        diversities = [m['response_diversity'] for m in self.metrics]
        
        # Find point where diversity drops significantly
        diversity_drops = []
        for i in range(1, len(diversities)):
            drop = diversities[i-1] - diversities[i]
            if drop > 0.1:  # Significant drop threshold
                diversity_drops.append((self.metrics[i]['iteration'], drop))
        
        return {
            "initial_diversity": diversities[0],
            "final_diversity": diversities[-1],
            "total_decline": diversities[0] - diversities[-1],
            "significant_drops": diversity_drops,
            "diversity_trend": stats.linregress(range(len(diversities)), diversities)[0]
        }
    
    def detect_catastrophic_forgetting(self) -> List[Dict]:
        """Detect instances of catastrophic forgetting"""
        forgetting_events = []
        
        for i in range(1, len(self.metrics)):
            curr = self.metrics[i]
            prev = self.metrics[i-1]
            
            # Check for sudden drops in any dataset
            for dataset in curr['cross_dataset_scores']:
                if dataset in prev['cross_dataset_scores']:
                    curr_score = curr['cross_dataset_scores'][dataset]
                    prev_score = prev['cross_dataset_scores'][dataset]
                    
                    drop = prev_score - curr_score
                    if drop > 0.15:  # 15% drop threshold
                        forgetting_events.append({
                            'iteration': curr['iteration'],
                            'dataset': dataset,
                            'drop': drop,
                            'from_score': prev_score,
                            'to_score': curr_score
                        })
        
        return forgetting_events
    
    def calculate_performance_variance(self) -> Dict[str, float]:
        """Calculate variance in performance across datasets"""
        variances = []
        
        for metric in self.metrics:
            scores = list(metric['cross_dataset_scores'].values())
            variances.append(np.var(scores))
        
        return {
            "mean_variance": np.mean(variances),
            "variance_trend": stats.linregress(range(len(variances)), variances)[0],
            "max_variance": np.max(variances),
            "min_variance": np.min(variances)
        }
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("=" * 80)
        print("ITERATIVE IPO PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)
        
        # Peak performance
        peak_iter, peak_acc = self.find_peak_performance()
        print(f"\n1. PEAK PERFORMANCE")
        print(f"   - Achieved at iteration: {peak_iter}")
        print(f"   - Peak accuracy: {peak_acc:.4f}")
        
        # Degradation analysis
        degradation = self.calculate_degradation_rate()
        print(f"\n2. DEGRADATION ANALYSIS")
        print(f"   - Degradation rate: {degradation['rate']:.4f} per iteration")
        print(f"   - R-squared: {degradation['r_squared']:.4f}")
        print(f"   - Statistical significance (p-value): {degradation.get('p_value', 'N/A'):.4f}")
        
        # Cross-dataset transfer
        transfer_df, correlation = self.analyze_cross_dataset_transfer()
        print(f"\n3. CROSS-DATASET TRANSFER")
        print("   Dataset correlations:")
        print(correlation.round(3))
        
        # Preference stability
        pref_stability = self.analyze_preference_stability()
        print(f"\n4. PREFERENCE STABILITY")
        print(f"   - Mean agreement: {pref_stability['mean_agreement']:.4f}")
        print(f"   - Stability trend: {pref_stability['trend']:.4f}")
        
        # Response diversity
        diversity = self.analyze_response_diversity()
        print(f"\n5. RESPONSE DIVERSITY")
        print(f"   - Initial diversity: {diversity['initial_diversity']:.4f}")
        print(f"   - Final diversity: {diversity['final_diversity']:.4f}")
        print(f"   - Total decline: {diversity['total_decline']:.4f}")
        
        # Catastrophic forgetting
        forgetting = self.detect_catastrophic_forgetting()
        print(f"\n6. CATASTROPHIC FORGETTING EVENTS")
        if forgetting:
            for event in forgetting:
                print(f"   - Iteration {event['iteration']}: {event['dataset']} "
                      f"dropped {event['drop']:.3f} ({event['from_score']:.3f} → {event['to_score']:.3f})")
        else:
            print("   - No catastrophic forgetting detected")
        
        # Performance variance
        variance = self.calculate_performance_variance()
        print(f"\n7. PERFORMANCE VARIANCE ACROSS DATASETS")
        print(f"   - Mean variance: {variance['mean_variance']:.4f}")
        print(f"   - Variance trend: {variance['variance_trend']:.4f}")
        
        print("\n" + "=" * 80)
    
    def generate_detailed_plots(self):
        """Generate detailed analysis plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Performance trajectory with confidence bands
        ax1 = plt.subplot(3, 3, 1)
        iterations = [m['iteration'] for m in self.metrics]
        accuracies = [m['self_eval_accuracy'] for m in self.metrics]
        
        # Calculate rolling mean and std
        window = min(3, len(accuracies) // 2)
        rolling_mean = pd.Series(accuracies).rolling(window, center=True).mean()
        rolling_std = pd.Series(accuracies).rolling(window, center=True).std()
        
        ax1.plot(iterations, accuracies, 'b-', alpha=0.5, label='Actual')
        ax1.plot(iterations, rolling_mean, 'r-', linewidth=2, label='Rolling Mean')
        ax1.fill_between(iterations, 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.2, color='red')
        
        # Mark peak
        peak_iter, peak_acc = self.find_peak_performance()
        ax1.scatter([peak_iter], [peak_acc], color='green', s=100, marker='*', 
                   label=f'Peak (iter {peak_iter})', zorder=5)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Self-Eval Accuracy')
        ax1.set_title('Performance Trajectory with Confidence Bands')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cross-dataset performance divergence
        ax2 = plt.subplot(3, 3, 2)
        transfer_df, _ = self.analyze_cross_dataset_transfer()
        
        for dataset in transfer_df['dataset'].unique():
            dataset_data = transfer_df[transfer_df['dataset'] == dataset]
            ax2.plot(dataset_data['iteration'], 
                    dataset_data['normalized_accuracy'], 
                    marker='o', label=dataset)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Normalized Accuracy')
        ax2.set_title('Cross-Dataset Performance (Normalized)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Preference agreement vs diversity
        ax3 = plt.subplot(3, 3, 3)
        agreements = [m['preference_agreement'] for m in self.metrics]
        diversities = [m['response_diversity'] for m in self.metrics]
        
        ax3.scatter(diversities, agreements, c=iterations, cmap='viridis', s=50)
        ax3.set_xlabel('Response Diversity')
        ax3.set_ylabel('Preference Agreement')
        ax3.set_title('Preference Stability vs Response Diversity')
        cbar = plt.colorbar(ax3.scatter(diversities, agreements, c=iterations, cmap='viridis', s=50), ax=ax3)
        cbar.set_label('Iteration')
        ax3.grid(True, alpha=0.3)
        
        # 4. Loss curves
        ax4 = plt.subplot(3, 3, 4)
        train_losses = [m['train_loss'] for m in self.metrics]
        eval_losses = [m['eval_loss'] for m in self.metrics]
        
        ax4.plot(iterations, train_losses, 'b-', label='Train Loss')
        ax4.plot(iterations, eval_losses, 'r-', label='Eval Loss')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training and Evaluation Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance variance over time
        ax5 = plt.subplot(3, 3, 5)
        variances = []
        for metric in self.metrics:
            scores = list(metric['cross_dataset_scores'].values())
            variances.append(np.var(scores))
        
        ax5.plot(iterations, variances, 'g-', marker='o')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Performance Variance')
        ax5.set_title('Cross-Dataset Performance Variance')
        ax5.grid(True, alpha=0.3)
        
        # 6. Catastrophic forgetting visualization
        ax6 = plt.subplot(3, 3, 6)
        forgetting_events = self.detect_catastrophic_forgetting()
        
        # Plot all dataset performances
        for dataset in list(self.metrics[0]['cross_dataset_scores'].keys()):
            scores = [m['cross_dataset_scores'].get(dataset, 0) for m in self.metrics]
            ax6.plot(iterations, scores, alpha=0.5, label=dataset)
        
        # Mark forgetting events
        for event in forgetting_events:
            ax6.scatter([event['iteration']], [event['to_score']], 
                       color='red', s=100, marker='v', zorder=5)
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Catastrophic Forgetting Events')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Degradation rate analysis
        ax7 = plt.subplot(3, 3, 7)
        peak_iter, _ = self.find_peak_performance()
        post_peak_iters = [m['iteration'] for m in self.metrics if m['iteration'] > peak_iter]
        post_peak_accs = [m['self_eval_accuracy'] for m in self.metrics if m['iteration'] > peak_iter]
        
        if post_peak_iters:
            ax7.scatter(post_peak_iters, post_peak_accs, alpha=0.6)
            
            # Fit degradation line
            z = np.polyfit(post_peak_iters, post_peak_accs, 1)
            p = np.poly1d(z)
            ax7.plot(post_peak_iters, p(post_peak_iters), "r--", 
                    label=f'Degradation rate: {abs(z[0]):.4f}/iter')
            
            ax7.set_xlabel('Iteration (Post-Peak)')
            ax7.set_ylabel('Accuracy')
            ax7.set_title('Post-Peak Degradation Analysis')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Response diversity decay
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(iterations, diversities, 'purple', marker='o', linewidth=2)
        
        # Mark significant drops
        diversity_analysis = self.analyze_response_diversity()
        for iter_drop, drop_size in diversity_analysis['significant_drops']:
            ax8.axvline(x=iter_drop, color='red', linestyle='--', alpha=0.5)
        
        ax8.set_xlabel('Iteration')
        ax8.set_ylabel('Response Diversity')
        ax8.set_title('Response Diversity Decay')
        ax8.grid(True, alpha=0.3)
        
        # 9. Cross-dataset correlation heatmap
        ax9 = plt.subplot(3, 3, 9)
        _, correlation = self.analyze_cross_dataset_transfer()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax9)
        ax9.set_title('Cross-Dataset Performance Correlation')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed analysis plots saved to {self.results_dir / 'detailed_analysis.png'}")

def main():
    parser = argparse.ArgumentParser(description="Analyze IPO degradation patterns")
    parser.add_argument("--results_dir", type=str, required=True, 
                       help="Directory containing iteration_metrics.json")
    args = parser.parse_args()
    
    analyzer = DegradationAnalyzer(args.results_dir)
    analyzer.generate_comprehensive_report()
    analyzer.generate_detailed_plots()

if __name__ == "__main__":
    main()