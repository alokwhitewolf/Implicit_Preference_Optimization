#!/bin/bash
# IPO Self-Improvement: Paper-Aligned Setup with Iterative Training

echo "🔬 Starting IPO Self-Improvement Experiments"
echo "📋 Paper-Aligned Setup:"
echo "   - 4,000 prompts from UltraFeedback (matches IPO paper)"
echo "   - Generate preferences → DPO training → RewardBench evaluation"
echo "   - Repeat for 15 epochs until performance saturates or degrades"
echo "📊 Research Question: How many epochs before RewardBench performance plateaus/degrades?"
echo ""

# Experiment 1: Llama-1B-Instruct for 15 epochs
echo "🧪 Experiment 1: Llama-1B-Instruct with 15 epochs (4k prompts each)"
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dataset "Ayush-Singh/UltraFeedback-1k-Each" \
    --forced_iterations 15 \
    --samples_per_iteration 4000 \
    --instruction_batch_size 64 \
    --enable_rewardbench_eval \
    --rewardbench_samples 500 \
    --use_fast_rewardbench \
    --eval_batch_size 32 \
    --parallel_category_eval \
    --track_degradation \
    --results_dir "./results/llama_4k_15epochs" \
    --checkpoint_dir "./checkpoints/llama_4k_15epochs" \
    --wandb_project "ipo-paper-aligned"

echo "✅ Llama experiment completed!"

# Experiment 2: Qwen-1.5B-Instruct for 15 epochs
echo "🧪 Experiment 2: Qwen-1.5B-Instruct with 15 epochs (4k prompts each)"
python DPO/iterative_ipo.py \
    --use_qwen \
    --base_dataset "Ayush-Singh/UltraFeedback-1k-Each" \
    --forced_iterations 15 \
    --samples_per_iteration 4000 \
    --instruction_batch_size 64 \
    --enable_rewardbench_eval \
    --rewardbench_samples 500 \
    --use_fast_rewardbench \
    --eval_batch_size 32 \
    --parallel_category_eval \
    --track_degradation \
    --results_dir "./results/qwen_4k_15epochs" \
    --checkpoint_dir "./checkpoints/qwen_4k_15epochs" \
    --wandb_project "ipo-paper-aligned"

echo "✅ Qwen experiment completed!"

echo ""
echo "🎉 All IPO Paper-Aligned Experiments Completed!"
echo "📊 Results:"
echo "   📁 Llama results: ./results/llama_4k_15epochs/"
echo "   📁 Qwen results: ./results/qwen_4k_15epochs/"
echo ""
echo "🔍 Analysis:"
echo "   - Compare RewardBench category trends (Chat/Code/Math/Safety)"
echo "   - Identify saturation/degradation points for each model"
echo "   - Total training data per model: 4k × 15 = 60k preference pairs"