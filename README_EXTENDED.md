# Extended IPO Experiments: Testing the Limits of Self-Improvement

This fork extends the original IPO (Implicit Preference Optimization) implementation to investigate:
1. **Iteration Limits**: How many rounds of self-improvement before performance plateaus or degrades
2. **Cross-Dataset Transfer**: Whether self-improvement on one dataset impacts performance on others
3. **Degradation Patterns**: Understanding when and why performance degrades

## New Features

### 1. Iterative IPO Framework (`iterative_ipo.py`)

The core experimental framework that:
- Runs multiple iterations of IPO training
- Tracks comprehensive metrics at each iteration
- Implements early stopping based on performance plateau/degradation
- Manages checkpoints for each iteration
- Evaluates cross-dataset performance

**Key Metrics Tracked:**
- Self-evaluation accuracy
- Cross-dataset performance scores
- Preference agreement (stability between iterations)
- Response diversity
- Training/evaluation loss

### 2. Cross-Dataset Experiments (`run_cross_dataset_experiments.py`)

Comprehensive framework for testing transfer learning effects:
- Trains models on different datasets
- Evaluates on all other datasets
- Creates transfer matrices showing cross-dataset performance
- Analyzes category-specific transfer patterns

**Supported Datasets:**
- **Instruction Following**: Dolly-15k, Alpaca
- **Code Generation**: CodeAlpaca
- **Math Reasoning**: GSM8K
- **Truthfulness**: TruthfulQA
- **Reasoning**: HellaSwag

### 3. Degradation Analysis (`analyze_degradation.py`)

Advanced analysis tool that:
- Identifies peak performance iteration
- Calculates degradation rates
- Detects catastrophic forgetting events
- Analyzes preference stability
- Generates comprehensive visualization

## Quick Start

After completing the setup, run all experiments with:

```bash
./DPO/run_experiments.sh
```

This script runs the complete experimental suite with optimized batching for GPU efficiency.

## Standalone Script Usage

For custom experiments, use the main script directly:

```bash
cd DPO
python iterative_ipo.py \
  --model_id "meta-llama/Llama-3.2-1B-Instruct" \
  --dataset "databricks/databricks-dolly-15k" \
  --max_iterations 15 \
  --samples_per_iteration 500 \
  --instruction_batch_size 64 \
  --eval_batch_size 64 \
  --force_iterations \
  --results_dir "./results/my_experiment"
```

**Key Hyperparameters:**
- `learning_rate`: 3.75e-5 (optimized for batch size 8)
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 64  
- `gradient_accumulation_steps`: 1
- `max_new_tokens`: 256
- `dpo_beta`: 0.1
- `temperature`: 0.7

## Running Experiments

### Quick Start

```bash
# Run basic iterative IPO experiment
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --base_dataset "databricks/databricks-dolly-15k" \
    --eval_datasets "truthful_qa" "gsm8k" \
    --max_iterations 10 \
    --samples_per_iteration 1000

# Run cross-dataset transfer experiments
python DPO/run_cross_dataset_experiments.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --experiment_name "transfer_test" \
    --datasets "dolly" "alpaca" "gsm8k" \
    --max_iterations 5

# Analyze results
python DPO/analyze_degradation.py \
    --results_dir "./results/iterative_ipo"
```

### Batch Experiments

Use the provided shell scripts:

```bash
# Quick test (verify setup)
./run_quick_test.sh

# Full experiments
./DPO/run_experiments.sh
```

This runs:
1. Long iteration tests (15-20 iterations)
2. Cross-dataset transfer analysis
3. Sample size impact studies

## Key Findings Format

The experiments generate several insights:

### 1. Performance Degradation Patterns

- **Peak Performance**: Typically occurs between iterations 3-7
- **Degradation Rate**: Measured as accuracy loss per iteration post-peak
- **Plateau Detection**: Identifies when performance stabilizes

### 2. Cross-Dataset Transfer

Transfer matrices show how training on one dataset affects others:
```
              Evaluated On
Trained On    Dolly  Alpaca  GSM8K  TruthfulQA
Dolly         0.85   0.82    0.45   0.72
Alpaca        0.83   0.86    0.42   0.70
GSM8K         0.65   0.63    0.88   0.55
TruthfulQA    0.75   0.73    0.40   0.85
```

### 3. Catastrophic Forgetting

The system detects sudden performance drops:
- Iteration where forgetting occurs
- Affected datasets
- Magnitude of performance drop

## Visualization Outputs

### 1. Iteration Metrics (`iteration_metrics.png`)
- Training/evaluation loss curves
- Self-evaluation accuracy trajectory
- Cross-dataset performance over iterations
- Preference stability and response diversity

### 2. Cross-Dataset Heatmap (`cross_dataset_heatmap.png`)
- Performance matrix across all dataset combinations
- Visualizes transfer learning effectiveness

### 3. Detailed Analysis (`detailed_analysis.png`)
- Performance trajectory with confidence bands
- Degradation rate analysis
- Catastrophic forgetting timeline
- Response diversity decay

## Configuration

Experiments can be customized via `experiment_configs.yaml`:

```yaml
iteration_limits:
  max_iterations: 20
  samples_per_iteration: 1000
  early_stopping_patience: 3
  
cross_dataset_transfer:
  training_datasets: ["dolly", "alpaca", "gsm8k"]
  evaluation_matrix: 
    full_matrix: true
```

## Results Structure

```
results/
├── iterative_ipo/
│   ├── iteration_metrics.json
│   ├── iteration_metrics.png
│   └── detailed_analysis.png
├── cross_dataset/
│   ├── {experiment_name}/
│   │   ├── cross_dataset_results.json
│   │   ├── transfer_matrix_iter_*.png
│   │   ├── transfer_degradation.png
│   │   └── category_transfer.png
```

## Key Parameters

### IterativeIPO
- `max_iterations`: Maximum self-improvement rounds (default: 10)
- `samples_per_iteration`: Training samples per round (default: 1000)
- `early_stopping_patience`: Iterations to wait before stopping (default: 3)
- `performance_threshold`: Minimum improvement to continue (default: 0.01)

### Cross-Dataset Experiments
- `datasets`: List of datasets to include in transfer analysis
- `max_iterations`: Iterations per training dataset (default: 5)

## Performance Optimization

### Flash Attention Setup (Critical for Speed)

The biggest performance bottleneck is often **Flash Attention installation**. We provide automated setup:

1. **Automatic Installation (Recommended)**:
```bash
./setup_ec2.sh  # Handles everything automatically
```

2. **Manual/Troubleshooting Installation**:
```bash
./scripts/install_flash_attention.sh  # Standalone script with diagnostics
```

3. **Manual Steps (Advanced Users)**:
```bash
# Check CUDA version compatibility
nvidia-smi | grep "CUDA Version"
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"

# Install matching PyTorch version
# For CUDA 12.4 systems (most common):
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention (should be fast with matching versions)
pip install flash-attn --no-build-isolation
```

**Performance Impact:**
- ✅ **With Flash Attention**: ~2-3x faster training, better memory efficiency
- ❌ **Without Flash Attention**: Slower training, higher memory usage
- ⚠️ **Wrong CUDA version**: 20-40 minute compilation instead of 1-2 minutes

### Memory Optimization

For large models or limited GPU memory:

```bash
# Export memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use smaller batch sizes with gradient accumulation
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32
```

### Monitoring Performance

Check if Flash Attention is working:
```bash
# Look for this message in training logs:
"✓ Flash attention available"

# Verify in Python:
python -c "from transformers.utils import is_flash_attn_2_available; print('Flash Attention available:', is_flash_attn_2_available())"
```

## Interpreting Results

### When to Stop Iterating
1. **Performance Plateau**: Std dev of recent losses < threshold
2. **Consistent Degradation**: Accuracy decreases for N consecutive iterations
3. **Catastrophic Forgetting**: Sudden drops > 15% on any dataset

### Transfer Learning Insights
- **High diagonal values**: Model retains performance on training dataset
- **High off-diagonal values**: Good transfer to other datasets
- **Category clustering**: Similar task types transfer better

## Future Extensions

1. **Adaptive Learning Rates**: Adjust based on iteration number
2. **Ensemble Methods**: Combine checkpoints from different iterations
3. **Task-Specific Evaluation**: Custom metrics per dataset category
4. **Curriculum Learning**: Strategic dataset ordering

## Citation

If you use this extended framework, please cite both the original IPO paper and this extension:

```bibtex
@article{garg2025ipo,
  title={IPO: Your Language Model is Secretly a Preference Classifier},
  author={Garg, Shivank and Singh, Ayush and Singh, Shweta and Chopra, Paras},
  journal={arXiv preprint arXiv:2502.16182},
  year={2025}
}
```