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
    print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"TRL version: {trl.__version__}")
print(f"BitsAndBytes version: {bnb.__version__}")

# Test Flash Attention
print("\nTesting Flash Attention...")
try:
    import flash_attn
    print(f"✓ Flash Attention installed: {flash_attn.__version__}")
    
    from transformers.utils import is_flash_attn_2_available
    fa2_available = is_flash_attn_2_available()
    print(f"✓ Transformers detects Flash Attention 2: {fa2_available}")
    
    if fa2_available:
        print("🚀 Flash Attention is properly configured for maximum performance!")
    else:
        print("⚠️ Flash Attention detected but not available to Transformers")
        
except ImportError:
    print("❌ Flash Attention not installed")
    print("   Install with: pip install flash-attn --no-build-isolation")
    print("   Make sure PyTorch CUDA version matches system CUDA version!")

print("\nTrying to load a small model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("✓ Model loading works!")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

print("\nSetup test complete!")
