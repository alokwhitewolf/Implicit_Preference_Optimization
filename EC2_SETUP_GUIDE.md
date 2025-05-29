# EC2 Setup Guide for IPO Experiments

## Quick Start (5 minutes)

### 1. Launch EC2 Instance
```bash
# Recommended: g5.xlarge
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 20.04)
# This AMI has NVIDIA drivers pre-installed!
```

### 2. Connect and Run Setup
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Download and run setup script
wget https://raw.githubusercontent.com/alokwhitewolf/Implicit_Preference_Optimization/iterative-ipo-experiments/setup_ec2.sh
chmod +x setup_ec2.sh
./setup_ec2.sh
```

### 3. Authenticate Services
```bash
conda activate ipo

# Required: Hugging Face (for model downloads)
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens

# Optional: Weights & Biases (for experiment tracking)
wandb login
# Enter your W&B API key from https://wandb.ai/authorize
```

### 4. Test Setup
```bash
python test_setup.py
# Should show CUDA available and successful model loading
```

### 5. Run Experiment
```bash
# Quick test run
./run_experiment.sh

# Or full experiment in tmux (recommended)
tmux new -s ipo
python DPO/iterative_ipo.py --model_id "meta-llama/Llama-3.2-1B-Instruct" --max_iterations 10
# Press Ctrl+B, then D to detach
```

## Detailed Prerequisites

### Required Accounts:
1. **Hugging Face** - For downloading models
   - Sign up at https://huggingface.co
   - Create access token at https://huggingface.co/settings/tokens
   - Some models (Llama) require accepting license agreement

2. **Weights & Biases** (Optional) - For experiment tracking
   - Sign up at https://wandb.ai
   - Get API key from https://wandb.ai/authorize

### EC2 Instance Requirements:
- **Instance Type**: g5.xlarge (minimum) or g5.2xlarge (recommended)
- **Storage**: 100GB EBS volume (gp3)
- **AMI**: Deep Learning AMI with NVIDIA drivers OR Ubuntu 20.04/22.04
- **Security Group**: Allow SSH (port 22) from your IP

## Manual Installation (if setup script fails)

### 1. Install NVIDIA Drivers (if not using DL AMI)
```bash
# Check if already installed
nvidia-smi

# If not, install
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot  # Required after driver installation
```

### 2. Install Conda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Clone Repository
```bash
git clone https://github.com/alokwhitewolf/Implicit_Preference_Optimization.git
cd Implicit_Preference_Optimization
git checkout iterative-ipo-experiments
```

### 4. Create Environment
```bash
conda create -n ipo python=3.10 -y
conda activate ipo
```

### 5. Install PyTorch with CUDA
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

### 6. Install Dependencies
```bash
pip install transformers==4.47.0 datasets==3.2.0 accelerate==1.2.1 \
            peft==0.14.0 bitsandbytes==0.45.0 trl==0.13.0 \
            wandb matplotlib seaborn tqdm scipy scikit-learn
```

### 7. Install Flash Attention (Critical for Performance)
```bash
# First, check your CUDA version
nvidia-smi  # Look for "CUDA Version: X.Y"

# Install PyTorch with matching CUDA version
# For CUDA 12.4:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Then install Flash Attention (should use pre-compiled wheels):
pip install flash-attn --no-build-isolation

# Verify installation:
python -c "from transformers.utils import is_flash_attn_2_available; print('Flash Attention available:', is_flash_attn_2_available())"
```

## Common Issues & Solutions

### 1. CUDA Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 1
--gradient_accumulation_steps 32

# Enable more aggressive memory saving
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 2. Model Download Fails
```bash
# Check HF authentication
huggingface-cli whoami

# For Llama models, accept license at:
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

### 3. Slow Training
```bash
# Install flash attention (optional)
pip install flash-attn --no-build-isolation

# Use smaller model or reduce samples
--samples_per_iteration 500
```

### 4. Flash Attention Installation Issues (CRITICAL)

**Quick Fix:** Use the automated script:
```bash
./scripts/install_flash_attention.sh
```

**Manual Diagnosis:**
```bash
# Problem: Flash Attention compiles from source (takes 20-40 minutes)
# Solution: Match CUDA versions exactly

# Step 1: Check system CUDA version
nvidia-smi | grep "CUDA Version"
nvcc --version

# Step 2: Check PyTorch CUDA version
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"

# Step 3: If they don't match, reinstall PyTorch
# For CUDA 12.4 systems:
pip uninstall torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8 systems:
pip uninstall torch torchvision torchaudio  
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Step 4: Install Flash Attention (should be fast now)
pip install flash-attn --no-build-isolation

# Verify success:
python -c "from transformers.utils import is_flash_attn_2_available; print('Flash Attention available:', is_flash_attn_2_available())"
```
**Note**: The `scripts/install_flash_attention.sh` script automates these steps with better error handling.

### 5. Instance Runs Out of Disk
```bash
# Clean up checkpoints between runs
rm -rf checkpoints/iteration_*

# Use larger EBS volume (150-200GB)
```

## Cost Optimization

### Use Spot Instances
1. Request spot instance when launching
2. Set "Persistent request" and "Stop" interruption behavior
3. Save 70% on costs

### Use Checkpointing
```bash
# Your experiments auto-checkpoint every iteration
# Can resume from any checkpoint if interrupted
```

### Monitor Usage
```bash
# Check GPU usage
nvidia-smi -l 1

# Check disk usage
df -h

# Monitor training progress
tail -f logs/training.log
```

## Running Production Experiments

### 1. Use tmux for Long Runs
```bash
tmux new -s experiment_name
# Run your experiment
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t experiment_name
```

### 2. Set Up Logging
```bash
python DPO/iterative_ipo.py ... 2>&1 | tee logs/experiment_$(date +%Y%m%d_%H%M%S).log
```

### 3. Monitor Remotely
```bash
# Set up W&B for remote monitoring
wandb login
# View results at https://wandb.ai/your-username/iterative-ipo
```

## Ready-to-Run Commands

### Small Model Test (2-3 hours, ~$1)
```bash
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --max_iterations 5 \
    --samples_per_iteration 500
```

### Full Experiment (20-30 hours, ~$10-15)
```bash
python DPO/iterative_ipo.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --max_iterations 15 \
    --samples_per_iteration 1000 \
    --eval_datasets "truthful_qa" "gsm8k" "hellaswag"
```

### Cross-Dataset Analysis (50-75 hours, ~$20-30)
```bash
python DPO/run_cross_dataset_experiments.py \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --experiment_name "cross_dataset_v1" \
    --datasets "dolly" "alpaca" "gsm8k" "truthful_qa" \
    --max_iterations 5
```

## You're Ready! 🚀

Once setup is complete, you can immediately start experiments. The setup handles all dependencies, CUDA configuration, and environment setup automatically.