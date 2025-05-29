#!/bin/bash
# Flash Attention Installation Script with CUDA Version Matching
# This script ensures fast installation by matching CUDA versions

set -e

# Check for non-interactive mode
NON_INTERACTIVE=false
if [[ "$1" == "--non-interactive" ]]; then
    NON_INTERACTIVE=true
fi

echo "=== Flash Attention Quick Installation ==="
echo "This script will install Flash Attention with proper CUDA version matching"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
echo "🔍 Detected system CUDA version: $CUDA_VERSION"

# Check if Python/conda environment is active
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please activate your conda environment first."
    exit 1
fi

# Check current PyTorch CUDA version
PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "not_installed")
echo "🔍 Current PyTorch CUDA version: $PYTORCH_CUDA"

# Determine if we need to reinstall PyTorch
NEEDS_PYTORCH_REINSTALL=false
if [[ "$PYTORCH_CUDA" != "$CUDA_VERSION" ]]; then
    echo "⚠️  CUDA version mismatch detected!"
    echo "   System CUDA: $CUDA_VERSION"
    echo "   PyTorch CUDA: $PYTORCH_CUDA"
    echo "   This will cause Flash Attention to compile from source (slow)."
    echo ""
    if [[ "$NON_INTERACTIVE" == false ]]; then
        read -p "Do you want to reinstall PyTorch with matching CUDA version? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            NEEDS_PYTORCH_REINSTALL=true
        fi
    else
        NEEDS_PYTORCH_REINSTALL=true
    fi
fi

# Install PyTorch with matching CUDA version
if [[ "$NEEDS_PYTORCH_REINSTALL" == true ]]; then
    echo "📦 Installing PyTorch with CUDA $CUDA_VERSION..."
    
    if [[ "$CUDA_VERSION" == "12.4" ]]; then
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    elif [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12.2" ]] || [[ "$CUDA_VERSION" == "12.3" ]]; then
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11.8" ]]; then
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠️  Unsupported CUDA version $CUDA_VERSION"
        echo "   Supported versions: 11.8, 12.1, 12.2, 12.3, 12.4"
        echo "   Defaulting to CUDA 12.4..."
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    fi
    
    echo "✅ PyTorch installation complete"
fi

# Install Flash Attention
echo "⚡ Installing Flash Attention..."
echo "   With matching CUDA versions, this should use pre-compiled wheels and be fast..."

start_time=$(date +%s)

if pip install flash-attn --no-build-isolation; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [[ $duration -lt 300 ]]; then  # Less than 5 minutes
        echo "✅ Flash Attention installed successfully in ${duration}s using pre-compiled wheels!"
    else
        echo "⚠️  Flash Attention installed in ${duration}s (compiled from source)"
        echo "   This suggests CUDA version mismatch. Check versions manually."
    fi
else
    echo "❌ Flash Attention installation failed"
    exit 1
fi

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    import flash_attn
    print(f'Flash Attention version: {flash_attn.__version__}')
    
    from transformers.utils import is_flash_attn_2_available
    fa2_available = is_flash_attn_2_available()
    print(f'Flash Attention 2 available: {fa2_available}')
    
    if fa2_available:
        print('🚀 SUCCESS: Flash Attention is properly configured!')
    else:
        print('⚠️  Flash Attention installed but not detected by Transformers')
        
except ImportError as e:
    print(f'❌ Flash Attention import failed: {e}')
"

echo ""
echo "✅ Flash Attention installation complete!"
echo ""
echo "💡 Tips:"
echo "   - Your IPO training should now be significantly faster"
echo "   - Look for '✓ Flash attention available' message when training starts"
echo "   - If you see compilation messages in future installs, run this script again" 