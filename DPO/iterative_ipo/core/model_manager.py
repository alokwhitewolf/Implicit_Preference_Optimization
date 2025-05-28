#!/usr/bin/env python3
"""
Model loading and PEFT management for Iterative IPO
"""

import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login, HfFolder
from peft import prepare_model_for_kbit_training
from .config import ExperimentConfig

class ModelManager:
    """Handles model loading, PEFT configuration, and GPU management"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def authenticate_huggingface(self):
        """Authenticate with Hugging Face using stored token"""
        try:
            token = HfFolder.get_token()
            if token:
                login(token=token)
                print("✓ Successfully authenticated with Hugging Face using stored token")
            else:
                print("Warning: No stored Hugging Face token found")
                print("Run 'huggingface-cli login' first")
        except Exception as e:
            print(f"Warning: Could not authenticate with Hugging Face: {e}")
            print("Make sure you've run 'huggingface-cli login' first")
    
    def load_model_and_tokenizer(self, checkpoint_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer with paper-aligned configuration"""
        # Authenticate with Hugging Face first
        self.authenticate_huggingface()
        
        model_path = checkpoint_path or self.config.model_id
        
        # Quantization config from paper
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Check if flash-attention is available
        try:
            import flash_attn
            attn_implementation = "flash_attention_2" if torch.cuda.is_available() else None
            print("✓ Flash attention available")
        except ImportError:
            attn_implementation = None
            print("⚠️ Flash attention not available, using standard attention")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,  # Match ours.py - important for PEFT compatibility
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            max_memory={0: "35GB"},  # Force using more GPU memory
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Handle tokenizer configuration
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Important for batch generation
        
        # Prepare model for training if using quantization
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)
            
        return model, tokenizer