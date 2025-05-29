#!/bin/bash
# EC2 Setup Script for IPO Experiments
# Run this after launching your EC2 instance

set -e  # Exit on error

echo "=== IPO Experiment EC2 Setup Script ==="
echo "This script will install all prerequisites and set up the environment"
echo ""

# Update system
echo "1. Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Check if NVIDIA drivers are installed
echo "2. Checking NVIDIA drivers..."
if ! nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt install -y nvidia-driver-535 nvidia-utils-535
    echo "NVIDIA drivers installed. You may need to reboot."
else
    echo "NVIDIA drivers already installed:"
    nvidia-smi
fi

# Install system dependencies
echo "3. Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    htop \
    tmux \
    build-essential

# Install Miniconda if not present
echo "4. Setting up Conda..."
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    export PATH="$HOME/miniconda3/bin:$PATH"
else
    echo "Conda already installed"
fi

# Clone repository or navigate into it
echo "5. Ensuring IPO repository is set up..."
REPO_NAME="Implicit_Preference_Optimization"
REPO_URL="https://github.com/alokwhitewolf/Implicit_Preference_Optimization.git"
BRANCH_NAME="iterative-ipo-experiments"

# Check if the current directory is the repo and has a .git folder
if [ "$(basename "$PWD")" = "$REPO_NAME" ] && [ -d ".git" ]; then
    echo "Currently inside the '$REPO_NAME' repository."
    echo "Fetching latest changes and checking out '$BRANCH_NAME'..."
    git fetch --all # Fetch all remote branches
    git checkout "$BRANCH_NAME" # Switch to the desired branch
    git pull origin "$BRANCH_NAME" # Pull latest changes from origin for this branch
# Check if the repo exists as a subdirectory and has a .git folder
elif [ -d "$REPO_NAME/.git" ]; then
    echo "'$REPO_NAME' directory exists. Navigating into it."
    cd "$REPO_NAME"
    echo "Fetching latest changes and checking out '$BRANCH_NAME'..."
    git fetch --all
    git checkout "$BRANCH_NAME"
    git pull origin "$BRANCH_NAME"
# If neither of the above, clone the repo
else
    echo "'$REPO_NAME' directory not found. Cloning repository..."
    git clone "$REPO_URL" "$REPO_NAME" # Clone the repo into a directory named $REPO_NAME
    cd "$REPO_NAME" # Change directory into the newly cloned repo
    echo "Checking out '$BRANCH_NAME'..."
    git checkout "$BRANCH_NAME" # Checkout the specified branch
fi

# Create conda environment
echo "6. Creating conda environment..."
if ! conda env list | grep -q "ipo"; then
    # Create environment with Python 3.10 (compatible with most packages)
    conda create -n ipo python=3.10 -y
else
    echo "Conda environment 'ipo' already exists"
fi

# Activate environment
echo "7. Activating environment and installing packages..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ipo

# Install PyTorch with CUDA support
echo "8. Installing PyTorch with CUDA..."
# Check system CUDA version and install matching PyTorch
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
echo "Detected CUDA version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "12.4" ]]; then
    echo "Installing PyTorch for CUDA 12.4..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
elif [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12.2" ]] || [[ "$CUDA_VERSION" == "12.3" ]]; then
    echo "Installing PyTorch for CUDA 12.1..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
else
    echo "Unsupported CUDA version $CUDA_VERSION, defaulting to CUDA 12.4..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

# Install requirements from environment.yaml packages
echo "9. Installing required packages..."
pip install \
    transformers==4.47.0 \
    datasets==3.2.0 \
    accelerate==1.2.1 \
    peft==0.14.0 \
    bitsandbytes==0.45.0 \
    trl==0.13.0 \
    wandb \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    pandas \
    numpy

# Install flash-attention for faster inference (optional but recommended)
echo "10. Installing flash-attention with CUDA version matching..."
echo "Using standalone installation script for better error handling..."

# Use the standalone script for Flash Attention installation
if [ -f "scripts/install_flash_attention.sh" ]; then
    echo "Running Flash Attention installation script..."
    chmod +x scripts/install_flash_attention.sh
    ./scripts/install_flash_attention.sh --non-interactive
else
    echo "Flash Attention script not found, falling back to simple installation..."
    if pip install flash-attn --no-build-isolation; then
        echo "✓ Flash attention successfully installed."
    else
        echo "⚠️ Flash attention installation failed. Run ./scripts/install_flash_attention.sh manually later."
    fi
fi

# Set up Hugging Face CLI
echo "11. Setting up Hugging Face..."
pip install huggingface-hub
echo "Please run 'huggingface-cli login' to authenticate with your HF token"

# Create necessary directories
echo "12. Creating project directories..."
mkdir -p checkpoints results logs

# Set up wandb (optional)
echo "13. Setting up Weights & Biases (optional)..."
echo "Run 'wandb login' if you want to use W&B for experiment tracking"

# Create a simple test script
echo "14. Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import peft
import trl
import bitsandbytes as bnb

print("Testing environment setup...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"TRL version: {trl.__version__}")
print(f"BitsAndBytes version: {bnb.__version__}")

print("\nTrying to load a small model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("✓ Model loading works!")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

print("\nSetup test complete!")
EOF

chmod +x test_setup.py

# Note: Quick start script already exists in repo
echo "15. Quick start script available..."
echo "    - ./run_quick_test.sh (for quick 3-iteration test)"
echo "    - ./DPO/run_experiments.sh (for full experiments)"

# Final instructions
echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate ipo"
echo "2. Test the setup: python test_setup.py"
echo "3. Login to Hugging Face: huggingface-cli login"
echo "4. (Optional) Login to W&B: wandb login"
echo "5. Run a test experiment: ./run_quick_test.sh"
echo ""
echo "For production runs, use tmux or screen:"
echo "  tmux new -s ipo"
echo "  ./DPO/run_experiments.sh"
echo "  (Ctrl+B, D to detach)"
echo ""

# Check if reboot is needed for NVIDIA drivers
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  IMPORTANT: Reboot required for NVIDIA drivers!"
    echo "  Run: sudo reboot"
    echo "  Then run this script again after reboot"
fi