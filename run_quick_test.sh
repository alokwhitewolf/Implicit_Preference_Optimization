#!/bin/bash
# Quick test script for IPO experiments (created for EC2 setup)

# This script is for a quick test run to verify everything is working
# For full experiments, use DPO/run_experiments.sh

echo "=== Quick IPO Test ==="
echo "This will run a small experiment to verify your setup"
echo ""

# Check if in conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: Not in a conda environment"
    echo "Run: conda activate ipo"
    exit 1
fi

# Export CUDA settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run quick test with small model
echo "Starting test experiment..."
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dataset "databricks/databricks-dolly-15k" \
    --eval_datasets "truthful_qa" \
    --max_iterations 3 \
    --samples_per_iteration 100 \
    --checkpoint_dir "./checkpoints/quick_test" \
    --results_dir "./results/quick_test" \
    --wandb_project "ipo-test"

echo ""
echo "Test complete! Check ./results/quick_test for results."
echo ""
echo "For full experiments, run:"
echo "  ./DPO/run_experiments.sh"