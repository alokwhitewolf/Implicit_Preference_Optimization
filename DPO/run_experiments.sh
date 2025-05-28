#!/bin/bash
# Research Experiments: Testing Self-Improvement Limits and Cross-Dataset Transfer

echo "🔬 Starting IPO Self-Improvement Research Experiments"
echo "📊 Research Questions:"
echo "   1. How many iterations before performance plateaus/degrades?"
echo "   2. How does training on one dataset affect others?"
echo ""

# Experiment 1: Test iteration limits with different models
echo "=== Experiment 1: Self-Improvement Limits Testing ==="

# Test 1A: Small model with forced long run (to observe full degradation cycle)
echo "🧪 Test 1A: Llama-1B with 30 forced iterations"
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dataset "databricks/databricks-dolly-15k" \
    --eval_datasets "truthful_qa" "gsm8k" "hellaswag" \
    --forced_iterations 15 \
    --samples_per_iteration 500 \
    --instruction_batch_size 64 \
    --experiment_type "limit_testing" \
    --track_degradation \
    --cross_dataset_eval \
    --plateau_window 3

echo "✅ Test 1A completed!"

# Test 1B: Qwen model comparison (paper's top performer)
echo "🧪 Test 1B: Qwen-1.5B with natural stopping"
python DPO/iterative_ipo.py \
    --use_qwen \
    --base_dataset "databricks/databricks-dolly-15k" \
    --eval_datasets "truthful_qa" "gsm8k" "hellaswag" \
    --max_iterations 15 \
    --instruction_batch_size 64 \
    --samples_per_iteration 500 \
    --experiment_type "limit_testing" \
    --track_degradation \
    --cross_dataset_eval

echo "✅ Test 1B completed!"

# Experiment 2: Cross-dataset transfer analysis
echo "=== Experiment 2: Cross-Dataset Transfer Analysis ==="

echo "🧪 Test 2A: Training on different datasets, evaluating on all"
python DPO/cross_dataset_experiment.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --datasets "databricks/databricks-dolly-15k" "tatsu-lab/alpaca" \
    --max_iterations 12 \
    --results_dir "./results/cross_dataset_transfer"

echo "✅ Test 2A completed!"

# Experiment 3: Sample size impact on degradation patterns
echo "=== Experiment 3: Sample Size Impact on Self-Improvement ==="

for samples in 250 500 1000; do
    echo "🧪 Test 3: Running with $samples samples per iteration..."
    python DPO/iterative_ipo.py \
        --model_id "meta-llama/Llama-3.2-1B-Instruct" \
        --base_dataset "databricks/databricks-dolly-15k" \
        --eval_datasets "truthful_qa" "gsm8k" \
        --max_iterations 15 \
        --samples_per_iteration $samples \
        --experiment_type "limit_testing" \
        --track_degradation \
        --results_dir "./results/sample_size_impact" \
        --checkpoint_dir "./checkpoints/sample_size_impact"
    
    echo "✅ Sample size $samples completed!"
done

# Experiment 4: Different datasets as training base
echo "=== Experiment 4: Different Training Datasets Impact ==="

for dataset in "databricks/databricks-dolly-15k" "tatsu-lab/alpaca"; do
    dataset_name=$(echo $dataset | sed 's/.*\///') # Extract name after last /
    echo "🧪 Test 4: Training on $dataset_name"
    
    python DPO/iterative_ipo.py \
        --model_id "meta-llama/Llama-3.2-1B-Instruct" \
        --base_dataset "$dataset" \
        --eval_datasets "truthful_qa" "gsm8k" "hellaswag" \
        --max_iterations 15 \
        --instruction_batch_size 64 \
        --samples_per_iteration 500 \
        --experiment_type "limit_testing" \
        --track_degradation \
        --cross_dataset_eval \
        --results_dir "./results/base_dataset_impact/$dataset_name" \
        --checkpoint_dir "./checkpoints/base_dataset_impact/$dataset_name"
    
    echo "✅ Training on $dataset_name completed!"
done

# Analysis and Summary
echo "=== Final Analysis and Summary ==="

echo "📊 Analyzing all experimental results..."

# Create summary analysis script call
python3 -c "
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

print('🔍 Collecting results from all experiments...')

results_dirs = [
    './results/self_improvement_limits/',
    './results/cross_dataset_transfer/',
    './results/sample_size_impact/',
    './results/base_dataset_impact/'
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
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        data = json.load(f)
                    summary['experiments_completed'] += 1
                    summary['max_iterations_reached'].append(data['final_performance']['iterations_completed'])

print(f'✅ Analysis complete!')
print(f'📈 Total experiments: {summary[\"experiments_completed\"]}')
if summary['max_iterations_reached']:
    print(f'📊 Average iterations reached: {sum(summary[\"max_iterations_reached\"])/len(summary[\"max_iterations_reached\"]):.1f}')
    print(f'📊 Max iterations in any experiment: {max(summary[\"max_iterations_reached\"])}')
print(f'💾 Detailed results available in ./results/ subdirectories')
"

echo ""
echo "🎉 All IPO Self-Improvement Research Experiments Completed!"
echo ""
echo "📋 Summary of Results:"
echo "   📁 Self-improvement limits: ./results/self_improvement_limits/"
echo "   📁 Cross-dataset transfer: ./results/cross_dataset_transfer/"
echo "   📁 Sample size impact: ./results/sample_size_impact/"
echo "   📁 Base dataset impact: ./results/base_dataset_impact/"
echo ""
echo "🔍 Key Questions Answered:"
echo "   ✅ How many iterations before plateau/degradation?"
echo "   ✅ How does training dataset choice affect cross-dataset performance?"
echo "   ✅ What's the impact of sample size on self-improvement limits?"
echo "   ✅ Do different models show different degradation patterns?"
echo ""
echo "📊 Next steps: Analyze the results in ./results/ directories"