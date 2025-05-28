#!/usr/bin/env python3
"""
SFT training wrapper for iterative IPO
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model

class SFTTrainerWrapper:
    """Handles SFT training on Dolly-15k before DPO (for base models only)"""
    
    def __init__(self, config):
        self.config = config
    
    def run_sft_training(self, model, tokenizer, iteration: int):
        """SFT training on Dolly-15k before DPO (for base models only) - Paper Section 4.3"""
        # Only run SFT for base models (not instruct models)
        if "Instruct" in self.config.model_id or "instruct" in self.config.model_id.lower():
            print("✓ Skipping SFT - using pre-trained instruct model")
            return model
        
        print(f"🔧 Running SFT training on Dolly-15k (Iteration {iteration})...")
        
        # Load Dolly-15k dataset for SFT (as specified in paper)
        dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        print(f"✓ Loaded Dolly-15k with {len(dolly_dataset)} examples")
        
        # Format dataset for SFT training
        def format_dolly_for_sft(example):
            if "mistral" in self.config.model_id.lower():
                formatted_text = f"[INST] {example['instruction']} [/INST] {example['response']}"
            else:
                # Use chat template for other models
                messages = [
                    {"role": "user", "content": example['instruction']},
                    {"role": "assistant", "content": example['response']}
                ]
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            return {"text": formatted_text}
        
        sft_dataset = dolly_dataset.map(format_dolly_for_sft)
        
        # Split into train/eval
        sft_split = sft_dataset.train_test_split(test_size=0.1, seed=42)
        
        # SFT configuration (from paper appendix)
        sft_checkpoint_path = os.path.join(self.config.checkpoint_dir, f"sft_iteration_{iteration}")
        
        # LoRA configuration for SFT - match ours.py exactly
        sft_peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # SFT training arguments - match ours.py configuration
        sft_args = TrainingArguments(
            output_dir=sft_checkpoint_path,
            num_train_epochs=3,  # Keep original 3 epochs for SFT
            per_device_train_batch_size=6,  # Match ours.py batch size
            per_device_eval_batch_size=3,  # Match ours.py batch size
            learning_rate=5e-5,  # Match ours.py learning rate
            optim="adamw_torch_fused",  # Match ours.py optimizer
            max_grad_norm=0.3,  # Match ours.py
            lr_scheduler_type="cosine",  # Match ours.py
            logging_steps=25,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            gradient_checkpointing=True,
            warmup_ratio=0.1,
            bf16=True,
            remove_unused_columns=False,
            report_to="wandb" if hasattr(self, 'wandb_enabled') and self.wandb_enabled else "none",
            run_name=f"sft_iteration_{iteration}",
        )
        
        # Apply LoRA to model - check if model already has PEFT adapters to avoid double application
        if hasattr(model, 'peft_config') and model.peft_config:
            print("✓ Model already has PEFT adapters, skipping get_peft_model")
            # Model already has PEFT adapters from previous iteration
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            else:
                print("✓ Model has PEFT adapters but print_trainable_parameters not available")
        else:
            print("✓ Applying PEFT adapters to model")
            model = get_peft_model(model, sft_peft_config)
            model.print_trainable_parameters()
        
        # Create SFT trainer
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=sft_split["train"],
            eval_dataset=sft_split["test"],
            tokenizer=tokenizer,
            max_seq_length=1024,
        )
        
        # Train
        print("🚀 Starting SFT training...")
        sft_trainer.train()
        
        # Save SFT model
        sft_trainer.save_model(sft_checkpoint_path)
        tokenizer.save_pretrained(sft_checkpoint_path)
        
        print(f"✓ SFT training completed. Model saved to {sft_checkpoint_path}")
        
        # Return the SFT-trained model (still has LoRA adapters)
        return model