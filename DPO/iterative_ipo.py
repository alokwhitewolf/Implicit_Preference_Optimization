#!/usr/bin/env python3
"""
Iterative IPO: Refactored main entry point
Focus: Pure self-improvement dynamics (like ours.py)
"""

import argparse
from iterative_ipo import ExperimentConfig, IterativeIPO

def main():
    parser = argparse.ArgumentParser(description="Iterative IPO - Self-Improvement Research")
    
    # Basic model and dataset config
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--base_dataset", type=str, default="Ayush-Singh/UltraFeedback-1k-Each")
    
    # Training parameters
    parser.add_argument("--max_iterations", type=int, default=25, help="Maximum iterations")
    parser.add_argument("--samples_per_iteration", type=int, default=1000)
    parser.add_argument("--forced_iterations", type=int, default=None, help="Force exact number of iterations")
    
    # Output directories
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/self_improvement")
    parser.add_argument("--results_dir", type=str, default="./results/self_improvement")
    parser.add_argument("--wandb_project", type=str, default="ipo-self-improvement")
    
    # Technical parameters
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--instruction_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    
    # Research parameters
    parser.add_argument("--track_degradation", action="store_true", default=True)
    parser.add_argument("--save_all_checkpoints", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Create experiment name based on configuration
    model_name = args.model_id.split("/")[-1]
    exp_name = f"{model_name}_self_improvement_{args.max_iterations}iter"
    if args.forced_iterations:
        exp_name += f"_forced{args.forced_iterations}"
    
    config = ExperimentConfig(
        model_id=args.model_id,
        base_dataset=args.base_dataset,
        max_iterations=args.max_iterations,
        samples_per_iteration=args.samples_per_iteration,
        checkpoint_dir=f"{args.checkpoint_dir}/{exp_name}",
        results_dir=f"{args.results_dir}/{exp_name}",
        wandb_project=args.wandb_project,
        use_4bit=args.use_4bit,
        
        # Research-specific configs
        save_all_checkpoints=args.save_all_checkpoints,
        track_degradation=args.track_degradation,
        forced_iterations=args.forced_iterations,
        instruction_batch_size=args.instruction_batch_size,
        eval_batch_size=args.eval_batch_size
    )
    
    print(f"🔬 Starting Self-Improvement Experiment: {exp_name}")
    print(f"📊 Research Focus: How does iterative preference training affect model performance?")
    print(f"📝 Experiment will run for up to {args.max_iterations} iterations")
    if args.forced_iterations:
        print(f"   ⚠️ FORCED MODE: Will run exactly {args.forced_iterations} iterations")
    print(f"📋 Paper-Aligned Methodology (like ours.py):")
    print(f"   1. Generate preferences from {args.base_dataset}")
    print(f"   2. DPO training on self-generated preferences")
    print(f"   3. Repeat and track self-improvement dynamics")
    print(f"🚀 GPU Optimization:")
    print(f"   - Preference generation: {args.instruction_batch_size} instructions × 4 responses")
    print(f"💾 Results will be saved to: {config.results_dir}")
    print(f"🔬 For cross-dataset transfer analysis, use: cross_dataset_experiment.py")
    
    experiment = IterativeIPO(config)
    experiment.run_experiment()

if __name__ == "__main__":
    main()