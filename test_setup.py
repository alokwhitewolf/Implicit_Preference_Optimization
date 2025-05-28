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
