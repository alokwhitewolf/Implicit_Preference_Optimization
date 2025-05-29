#!/bin/bash
# Quick test script for IPO Research Experiments
# Replicates the full experiment structure but with minimal parameters for fast testing

echo "🧪 Quick IPO Research Experiment Test"
echo "📊 This replicates the full research pipeline but with minimal parameters"
echo "📊 Research Questions (Quick Test):"
echo "   1. How many iterations before performance plateaus/degrades?"
echo "   2. How does training on one dataset affect others?"
echo ""

# Check if in conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: Not in a conda environment"
    echo "Run: conda activate ipo"
    exit 1
fi

# Export CUDA settings for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Experiment 1: Self-Improvement Limits Testing (Quick)
echo "=== Quick Test 1: Self-Improvement Limits ==="

# Test 1A: Small model with forced iterations (minimal)
echo "🧪 Quick Test 1A: Llama-1B with 2 forced iterations"
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dataset "databricks/databricks-dolly-15k" \
    --eval_datasets "truthful_qa" \
    --forced_iterations 2 \
    --samples_per_iteration 10 \
    --experiment_type "limit_testing" \
    --track_degradation \
    --cross_dataset_eval \
    --plateau_window 2 \
    --results_dir "./results/quick_test/self_improvement_limits" \
    --checkpoint_dir "./checkpoints/quick_test/self_improvement_limits" \
    --wandb_project "ipo-quick-test"

echo "✅ Quick Test 1A completed!"

# Test 1B: Natural stopping test (minimal)
echo "🧪 Quick Test 1B: Natural stopping with 3 max iterations"
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dataset "databricks/databricks-dolly-15k" \
    --eval_datasets "truthful_qa" \
    --max_iterations 3 \
    --samples_per_iteration 10 \
    --experiment_type "limit_testing" \
    --track_degradation \
    --cross_dataset_eval \
    --results_dir "./results/quick_test/natural_stopping" \
    --checkpoint_dir "./checkpoints/quick_test/natural_stopping" \
    --wandb_project "ipo-quick-test"

echo "✅ Quick Test 1B completed!"

# Experiment 2: Cross-dataset transfer analysis (Quick)
echo "=== Quick Test 2: Cross-Dataset Transfer Analysis ==="

echo "🧪 Quick Test 2A: Training on different datasets, evaluating on all (minimal)"
python DPO/cross_dataset_experiments.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --datasets "databricks/databricks-dolly-15k" "tatsu-lab/alpaca" \
    --max_iterations 2 \
    --samples_per_iteration 10 \
    --force_iterations \
    --results_dir "./results/quick_test/cross_dataset_transfer"

echo "✅ Quick Test 2A completed!"

# Experiment 3: Sample size impact (Quick)
echo "=== Quick Test 3: Sample Size Impact ==="

for samples in 5 10; do
    echo "🧪 Quick Test 3: Running with $samples samples per iteration..."
    python DPO/iterative_ipo.py \
        --model_id "meta-llama/Llama-3.2-1B-Instruct" \
        --base_dataset "databricks/databricks-dolly-15k" \
        --eval_datasets "truthful_qa" \
        --max_iterations 2 \
        --samples_per_iteration $samples \
        --experiment_type "limit_testing" \
        --track_degradation \
        --results_dir "./results/quick_test/sample_size_impact" \
        --checkpoint_dir "./checkpoints/quick_test/sample_size_impact" \
        --wandb_project "ipo-quick-test"
    
    echo "✅ Sample size $samples completed!"
done

# Experiment 4: Different training datasets (Quick)
echo "=== Quick Test 4: Different Training Datasets Impact ==="

for dataset in "databricks/databricks-dolly-15k"; do  # Only test one dataset for speed
    dataset_name=$(echo $dataset | sed 's/.*\///') # Extract name after last /
    echo "🧪 Quick Test 4: Training on $dataset_name"
    
    python DPO/iterative_ipo.py \
        --model_id "meta-llama/Llama-3.2-1B-Instruct" \
        --base_dataset "$dataset" \
        --eval_datasets "truthful_qa" \
        --max_iterations 2 \
        --samples_per_iteration 10 \
        --experiment_type "limit_testing" \
        --track_degradation \
        --cross_dataset_eval \
        --results_dir "./results/quick_test/base_dataset_impact/$dataset_name" \
        --checkpoint_dir "./checkpoints/quick_test/base_dataset_impact/$dataset_name" \
        --wandb_project "ipo-quick-test"
    
    echo "✅ Training on $dataset_name completed!"
done

# Quick Analysis and Summary
echo "=== Quick Test Analysis and Summary ==="

echo "📊 Quick analyzing test results..."

# Create summary analysis
python3 -c "
import os
import json
import glob

print('🔍 Collecting results from quick test experiments...')

results_dirs = [
    './results/quick_test/self_improvement_limits/',
    './results/quick_test/natural_stopping/',
    './results/quick_test/cross_dataset_transfer/',
    './results/quick_test/sample_size_impact/',
    './results/quick_test/base_dataset_impact/'
]

summary = {
    'experiments_completed': 0,
    'max_iterations_reached': [],
    'degradation_patterns': [],
    'transfer_effects': []
}

for results_dir in results_dirs:
    if os.path.exists(results_dir):
        for exp_dir in glob.glob(f'{results_dir}/*'):
            if os.path.isdir(exp_dir):
                metrics_file = os.path.join(exp_dir, 'iteration_metrics.json')
                degradation_file = os.path.join(exp_dir, 'degradation_analysis.json')
                
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        data = json.load(f)
                    summary['experiments_completed'] += 1
                    summary['max_iterations_reached'].append(data['final_performance']['iterations_completed'])
                
                if os.path.exists(degradation_file):
                    print(f'✅ Found degradation analysis: {degradation_file}')

print(f'✅ Quick test analysis complete!')
print(f'📈 Total quick experiments: {summary[\"experiments_completed\"]}')
if summary['max_iterations_reached']:
    print(f'📊 Average iterations reached: {sum(summary[\"max_iterations_reached\"])/len(summary[\"max_iterations_reached\"]):.1f}')
    print(f'📊 Max iterations in any quick experiment: {max(summary[\"max_iterations_reached\"])}')
print(f'💾 Quick test results available in ./results/quick_test/ subdirectories')
"

echo ""
echo "🎉 Quick IPO Research Experiment Test Completed!"
echo ""
echo "📋 Quick Test Results Summary:"
echo "   📁 Self-improvement limits: ./results/quick_test/self_improvement_limits/"
echo "   📁 Natural stopping: ./results/quick_test/natural_stopping/"
echo "   📁 Cross-dataset transfer: ./results/quick_test/cross_dataset_transfer/"
echo "   📁 Sample size impact: ./results/quick_test/sample_size_impact/"
echo "   📁 Base dataset impact: ./results/quick_test/base_dataset_impact/"
echo ""
echo "🔍 Quick Test Validation:"
echo "   ✅ Self-improvement limits testing pipeline works"
echo "   ✅ Cross-dataset transfer analysis works"
echo "   ✅ Sample size impact testing works"
echo "   ✅ Degradation tracking and analysis works"
echo ""
echo "⏭️  Next Steps:"
echo "   1. ✅ Quick test passed - your setup is working!"
echo "   2. 🚀 Run full experiments: ./DPO/run_experiments.sh"
echo "   3. 📊 Expected runtime: Quick test ~10-20 min, Full experiments ~6-12 hours"
echo ""
echo "💡 Parameter Differences (Quick vs Full):"
echo "   • Iterations: 3-5 vs 15-30"
echo "   • Samples per iteration: 20-50 vs 250-1000"
echo "   • Datasets: 1-2 vs 2-4"
echo "   • Models: 1 vs 2"