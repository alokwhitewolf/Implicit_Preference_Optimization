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
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation and P(Yes) extraction")
    
    # Research parameters
    parser.add_argument("--track_degradation", action="store_true", default=True)
    parser.add_argument("--save_all_checkpoints", action="store_true", default=True)
    
    # RewardBench IPO evaluation parameters (matches paper)
    parser.add_argument("--enable_rewardbench_eval", action="store_true", default=True, help="Enable RewardBench IPO evaluation")
    parser.add_argument("--rewardbench_eval_frequency", type=int, default=1, help="Evaluate every N iterations")
    parser.add_argument("--rewardbench_samples", type=int, default=100, help="Number of RewardBench samples")
    
    # Performance optimization parameters
    parser.add_argument("--parallel_category_eval", action="store_true", default=True, help="Evaluate categories in parallel")
    parser.add_argument("--use_fast_rewardbench", action="store_true", default=True, help="Use optimized fast evaluator")
    
    # Legacy external evaluation (deprecated)
    parser.add_argument("--enable_external_eval", action="store_true", default=False, help="Enable legacy external evaluation")
    parser.add_argument("--external_eval_frequency", type=int, default=1, help="Evaluate every N iterations")
    parser.add_argument("--external_eval_samples", type=int, default=100, help="Samples per benchmark dataset")
    
    # Arguments that are ignored but accepted for compatibility
    parser.add_argument("--eval_datasets", nargs="*", help="Cross-dataset evaluation (ignored - use cross_dataset_experiment.py)")
    parser.add_argument("--experiment_type", type=str, help="Experiment type (ignored)")
    parser.add_argument("--cross_dataset_eval", action="store_true", help="Cross-dataset evaluation (ignored)")
    parser.add_argument("--plateau_window", type=int, help="Plateau detection window (ignored)")
    parser.add_argument("--use_qwen", action="store_true", help="Use Qwen model (ignored - specify model_id directly)")
    
    args = parser.parse_args()
    
    # Handle special model flags
    if args.use_qwen:
        args.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print("🔄 Using Qwen model: Qwen/Qwen2.5-1.5B-Instruct")
    
    # Warn about ignored arguments
    ignored_args = []
    if args.eval_datasets:
        ignored_args.append("--eval_datasets (use cross_dataset_experiment.py)")
    if args.experiment_type:
        ignored_args.append("--experiment_type")
    if args.cross_dataset_eval:
        ignored_args.append("--cross_dataset_eval (use cross_dataset_experiment.py)")
    if args.plateau_window:
        ignored_args.append("--plateau_window")
    
    if ignored_args:
        print("⚠️ NOTICE: The following arguments are ignored in self-improvement mode:")
        for arg in ignored_args:
            print(f"   • {arg}")
        print("💡 For cross-dataset evaluation, use: python cross_dataset_experiment.py")
        print("")
    
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
        
        # RewardBench IPO evaluation configs (matches paper)
        enable_rewardbench_eval=args.enable_rewardbench_eval,
        rewardbench_eval_frequency=args.rewardbench_eval_frequency,
        rewardbench_samples=args.rewardbench_samples,
        
        # Performance optimization configs
        parallel_category_eval=args.parallel_category_eval,
        use_fast_rewardbench=args.use_fast_rewardbench,
        eval_batch_size=args.eval_batch_size,
        
        # Legacy external evaluation configs
        enable_external_eval=args.enable_external_eval,
        external_eval_frequency=args.external_eval_frequency,
        external_eval_samples=args.external_eval_samples
    )
    
    print(f"🔬 Starting Self-Improvement Experiment: {exp_name}")
    print(f"📊 Research Focus: How does iterative preference training affect model performance?")
    print(f"📝 Experiment will run for up to {args.max_iterations} iterations")
    if args.forced_iterations:
        print(f"   ⚠️ FORCED MODE: Will run exactly {args.forced_iterations} iterations")
    print(f"📋 IPO Methodology (matches paper exactly):")
    print(f"   1. Generate preferences from {args.base_dataset}")
    print(f"   2. DPO training on self-generated preferences")
    print(f"   3. RewardBench IPO evaluation using P(Yes) extraction")
    print(f"   4. Track self-improvement trends across categories")
    print(f"🚀 GPU Optimization:")
    print(f"   - Preference generation: {args.instruction_batch_size} instructions × 4 responses")
    if args.enable_rewardbench_eval:
        print(f"📊 RewardBench IPO evaluation:")
        print(f"   - Categories: Chat, Code, Math, Safety")
        print(f"   - Frequency: every {args.rewardbench_eval_frequency} iteration(s)")
        print(f"   - Samples: {args.rewardbench_samples}")
        print(f"   - Method: P(Yes) probability extraction")
        if args.use_fast_rewardbench:
            print(f"   - 🚀 Fast mode: batch_size={args.eval_batch_size}, parallel={args.parallel_category_eval}")
        else:
            print(f"   - 🐌 Standard mode (slower but safer)")
    print(f"💾 Results will be saved to: {config.results_dir}")
    print(f"🔬 For cross-dataset transfer analysis, use: cross_dataset_experiment.py")
    
    experiment = IterativeIPO(config)
    experiment.run_experiment()

if __name__ == "__main__":
    main()