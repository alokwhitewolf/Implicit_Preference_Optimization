#!/usr/bin/env python3
"""
Cross-Dataset Transfer Analysis for IPO Research
Tests how self-improvement on one dataset affects performance on others
"""

import os
import json
import argparse
from iterative_ipo import ExperimentConfig, IterativeIPO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class CrossDatasetAnalyzer:
    """Analyze cross-dataset transfer effects in iterative self-improvement"""
    
    def __init__(self, model_id: str, base_results_dir: str):
        self.model_id = model_id
        self.base_results_dir = base_results_dir
        self.results = {}
        
    def run_transfer_experiments(self, datasets: list, max_iterations: int = 15, force_iterations: bool = True, samples_per_iteration: int = 500):
        """Run experiments training on each dataset and evaluating on all others"""
        
        print(f"🔬 Running Cross-Dataset Transfer Analysis")
        print(f"📊 Training datasets: {datasets}")
        print(f"🎯 Each model will be evaluated on ALL datasets")
        print(f"⚙️ Max iterations: {max_iterations}")
        print(f"⚙️ Samples per iteration: {samples_per_iteration}")
        print(f"⚙️ Force completion: {force_iterations}")
        
        for train_dataset in datasets:
            print(f"\n{'='*50}")
            print(f"Training on: {train_dataset}")
            print(f"{'='*50}")
            
            # Configure experiment
            exp_name = f"train_{train_dataset}_eval_all"
            config = ExperimentConfig(
                model_id=self.model_id,
                base_dataset=train_dataset,
                eval_datasets=datasets,  # Evaluate on ALL datasets
                max_iterations=max_iterations,
                samples_per_iteration=samples_per_iteration,
                checkpoint_dir=f"{self.base_results_dir}/checkpoints/{exp_name}",
                results_dir=f"{self.base_results_dir}/results/{exp_name}",
                wandb_project="ipo-cross-dataset-transfer",
                
                # Research settings
                save_all_checkpoints=True,
                cross_dataset_eval=True,
                track_degradation=True,
                forced_iterations=max_iterations if force_iterations else None  # Configurable forcing
            )
            
            # Run experiment
            experiment = IterativeIPO(config)
            experiment.run_experiment()
            
            # Store results
            self.results[train_dataset] = {
                "config": config,
                "results_dir": config.results_dir
            }
            
        print(f"\n✅ All transfer experiments completed!")
        self.analyze_transfer_effects()
    
    def analyze_transfer_effects(self):
        """Analyze and visualize transfer effects across datasets"""
        print(f"\n📊 Analyzing Cross-Dataset Transfer Effects...")
        
        # Collect all results
        transfer_matrix = {}
        performance_trajectories = {}
        
        for train_dataset, exp_data in self.results.items():
            results_file = os.path.join(exp_data["results_dir"], "iteration_metrics.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                # Extract cross-dataset performance
                metrics = data["metrics"]
                for iteration_data in metrics:
                    iteration = iteration_data["iteration"]
                    cross_scores = iteration_data["cross_dataset_scores"]
                    
                    # Store transfer effects
                    if train_dataset not in transfer_matrix:
                        transfer_matrix[train_dataset] = {}
                        performance_trajectories[train_dataset] = {}
                    
                    for eval_dataset, score in cross_scores.items():
                        if eval_dataset not in transfer_matrix[train_dataset]:
                            transfer_matrix[train_dataset][eval_dataset] = []
                            performance_trajectories[train_dataset][eval_dataset] = []
                        
                        transfer_matrix[train_dataset][eval_dataset].append(score)
                        performance_trajectories[train_dataset][eval_dataset].append({
                            "iteration": iteration,
                            "score": score
                        })
        
        # Create visualizations
        self.create_transfer_visualizations(transfer_matrix, performance_trajectories)
        
        # Generate transfer analysis report
        self.generate_transfer_report(transfer_matrix, performance_trajectories)
    
    def create_transfer_visualizations(self, transfer_matrix, performance_trajectories):
        """Create comprehensive transfer visualization plots"""
        
        # 1. Transfer matrix heatmap (final performance)
        datasets = list(transfer_matrix.keys())
        final_scores = {}
        
        for train_ds in datasets:
            final_scores[train_ds] = {}
            for eval_ds in datasets:
                if eval_ds in transfer_matrix[train_ds]:
                    scores = transfer_matrix[train_ds][eval_ds]
                    final_scores[train_ds][eval_ds] = scores[-1] if scores else 0
                else:
                    final_scores[train_ds][eval_ds] = 0
        
        # Convert to DataFrame for heatmap
        df_final = pd.DataFrame(final_scores).T
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_final, annot=True, cmap='RdYlGn', center=0.5, 
                   fmt='.3f', square=True, linewidths=0.5)
        plt.title('Cross-Dataset Transfer Matrix (Final Performance)')
        plt.xlabel('Evaluation Dataset')
        plt.ylabel('Training Dataset')
        plt.tight_layout()
        plt.savefig(f"{self.base_results_dir}/transfer_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance trajectories
        fig, axes = plt.subplots(len(datasets), len(datasets), figsize=(15, 15))
        fig.suptitle('Performance Trajectories: Training vs Evaluation Datasets', fontsize=16)
        
        for i, train_ds in enumerate(datasets):
            for j, eval_ds in enumerate(datasets):
                ax = axes[i][j] if len(datasets) > 1 else axes
                
                if eval_ds in performance_trajectories[train_ds]:
                    traj = performance_trajectories[train_ds][eval_ds]
                    iterations = [p["iteration"] for p in traj]
                    scores = [p["score"] for p in traj]
                    
                    color = 'green' if train_ds == eval_ds else 'blue'
                    ax.plot(iterations, scores, marker='o', color=color, linewidth=2)
                    
                    # Highlight same-dataset (in-domain) vs cross-dataset (transfer)
                    if train_ds == eval_ds:
                        ax.set_facecolor('#f0f8f0')  # Light green for in-domain
                        ax.set_title(f'In-Domain\n{train_ds}', fontweight='bold')
                    else:
                        ax.set_title(f'Transfer\n{train_ds}→{eval_ds}')
                    
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Performance')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.base_results_dir}/performance_trajectories.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Transfer vs In-Domain comparison
        plt.figure(figsize=(12, 6))
        
        transfer_effects = []
        for train_ds in datasets:
            for eval_ds in datasets:
                if eval_ds in transfer_matrix[train_ds]:
                    scores = transfer_matrix[train_ds][eval_ds]
                    final_score = scores[-1] if scores else 0
                    
                    transfer_effects.append({
                        'train_dataset': train_ds,
                        'eval_dataset': eval_ds,
                        'final_score': final_score,
                        'transfer_type': 'In-Domain' if train_ds == eval_ds else 'Cross-Dataset'
                    })
        
        df_effects = pd.DataFrame(transfer_effects)
        
        # Box plot comparing in-domain vs cross-dataset performance
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df_effects, x='transfer_type', y='final_score')
        plt.title('In-Domain vs Cross-Dataset Performance')
        plt.ylabel('Final Performance Score')
        
        # Bar plot showing average transfer effects
        plt.subplot(1, 2, 2)
        avg_effects = df_effects.groupby(['train_dataset', 'transfer_type'])['final_score'].mean().reset_index()
        sns.barplot(data=avg_effects, x='train_dataset', y='final_score', hue='transfer_type')
        plt.title('Average Performance by Training Dataset')
        plt.xticks(rotation=45)
        plt.ylabel('Average Performance Score')
        
        plt.tight_layout()
        plt.savefig(f"{self.base_results_dir}/transfer_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Transfer visualizations saved to {self.base_results_dir}/")
    
    def generate_transfer_report(self, transfer_matrix, performance_trajectories):
        """Generate detailed transfer analysis report"""
        
        datasets = list(transfer_matrix.keys())
        report = {
            "experiment_summary": {
                "model_id": self.model_id,
                "datasets_tested": datasets,
                "total_experiments": len(datasets)
            },
            "transfer_analysis": {},
            "key_findings": []
        }
        
        # Calculate transfer metrics
        for train_ds in datasets:
            report["transfer_analysis"][train_ds] = {
                "in_domain_performance": None,
                "cross_dataset_performance": {},
                "positive_transfer": [],
                "negative_transfer": [],
                "transfer_summary": {}
            }
            
            # In-domain performance
            if train_ds in transfer_matrix[train_ds]:
                in_domain_scores = transfer_matrix[train_ds][train_ds]
                report["transfer_analysis"][train_ds]["in_domain_performance"] = {
                    "final_score": in_domain_scores[-1] if in_domain_scores else 0,
                    "peak_score": max(in_domain_scores) if in_domain_scores else 0,
                    "trajectory": in_domain_scores
                }
            
            # Cross-dataset performance
            for eval_ds in datasets:
                if eval_ds != train_ds and eval_ds in transfer_matrix[train_ds]:
                    cross_scores = transfer_matrix[train_ds][eval_ds]
                    final_cross = cross_scores[-1] if cross_scores else 0
                    peak_cross = max(cross_scores) if cross_scores else 0
                    
                    report["transfer_analysis"][train_ds]["cross_dataset_performance"][eval_ds] = {
                        "final_score": final_cross,
                        "peak_score": peak_cross,
                        "trajectory": cross_scores
                    }
                    
                    # Determine if transfer is positive or negative
                    # Compare to a baseline (could be initial performance or average)
                    if cross_scores and len(cross_scores) > 1:
                        initial_score = cross_scores[0]
                        if final_cross > initial_score:
                            report["transfer_analysis"][train_ds]["positive_transfer"].append(eval_ds)
                        else:
                            report["transfer_analysis"][train_ds]["negative_transfer"].append(eval_ds)
        
        # Generate key findings
        total_transfers = sum(len(data["cross_dataset_performance"]) for data in report["transfer_analysis"].values())
        positive_transfers = sum(len(data["positive_transfer"]) for data in report["transfer_analysis"].values())
        
        report["key_findings"] = [
            f"Tested {len(datasets)} datasets with {total_transfers} cross-dataset transfers",
            f"{positive_transfers}/{total_transfers} transfers showed positive effects",
            f"Negative transfer rate: {(total_transfers-positive_transfers)/total_transfers:.1%}"
        ]
        
        # Save report
        report_file = f"{self.base_results_dir}/transfer_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📋 Transfer Analysis Summary:")
        for finding in report["key_findings"]:
            print(f"   • {finding}")
        
        print(f"\n💾 Detailed report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Transfer Analysis for IPO")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--datasets", nargs="+", 
                       default=["databricks/databricks-dolly-15k", "tatsu-lab/alpaca", "truthful_qa"],
                       help="Datasets to test transfer between")
    parser.add_argument("--max_iterations", type=int, default=10, 
                       help="Maximum iterations per experiment")
    parser.add_argument("--samples_per_iteration", type=int, default=500,
                       help="Number of samples per iteration")
    parser.add_argument("--force_iterations", action="store_true", default=True,
                       help="Force completion of all iterations (ignore early stopping)")
    parser.add_argument("--results_dir", type=str, default="./results/cross_dataset_analysis")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CrossDatasetAnalyzer(args.model_id, args.results_dir)
    
    # Run transfer experiments with all configurable parameters
    analyzer.run_transfer_experiments(
        datasets=args.datasets, 
        max_iterations=args.max_iterations,
        force_iterations=args.force_iterations,
        samples_per_iteration=args.samples_per_iteration
    )

if __name__ == "__main__":
    main()