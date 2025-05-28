#!/bin/bash
# Run iterative IPO experiments with different configurations

# Experiment 1: Test iteration limits on different model sizes
echo "=== Experiment 1: Testing iteration limits ==="

# Small model (1B parameters)
python iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dataset "databricks/databricks-dolly-15k" \
    --eval_datasets "truthful_qa" "gsm8k" "hellaswag" \
    --max_iterations 15 \
    --samples_per_iteration 1000 \
    --checkpoint_dir "./checkpoints/iter_limit_1b" \
    --results_dir "./results/iter_limit_1b" \
    --wandb_project "ipo-iteration-limits" &

# Wait for GPU memory to be available
sleep 30

# Medium model (7B parameters) - if you have enough GPU memory
# python iterative_ipo.py \
#     --model_id "mistralai/Mistral-7B-Instruct-v0.1" \
#     --base_dataset "databricks/databricks-dolly-15k" \
#     --eval_datasets "truthful_qa" "gsm8k" "hellaswag" \
#     --max_iterations 10 \
#     --samples_per_iteration 500 \
#     --checkpoint_dir "./checkpoints/iter_limit_7b" \
#     --results_dir "./results/iter_limit_7b" \
#     --wandb_project "ipo-iteration-limits"

# Experiment 2: Cross-dataset transfer analysis
echo "=== Experiment 2: Cross-dataset transfer ==="

python run_cross_dataset_experiments.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --experiment_name "transfer_analysis_v1" \
    --datasets "dolly" "alpaca" "gsm8k" "truthful_qa" \
    --max_iterations 5

# Experiment 3: Different sample sizes per iteration
echo "=== Experiment 3: Sample size impact ==="

for samples in 100 500 1000 2000; do
    echo "Running with $samples samples per iteration..."
    python iterative_ipo.py \
        --model_id "meta-llama/Llama-3.2-1B-Instruct" \
        --base_dataset "databricks/databricks-dolly-15k" \
        --eval_datasets "truthful_qa" \
        --max_iterations 5 \
        --samples_per_iteration $samples \
        --checkpoint_dir "./checkpoints/sample_size_$samples" \
        --results_dir "./results/sample_size_$samples" \
        --wandb_project "ipo-sample-size"
done

# Analyze results
echo "=== Analyzing degradation patterns ==="

for exp_dir in ./results/iter_limit_*; do
    if [ -d "$exp_dir" ]; then
        echo "Analyzing $exp_dir..."
        python analyze_degradation.py --results_dir "$exp_dir"
    fi
done

echo "All experiments completed!"