#!/usr/bin/env python3
"""
DPO training wrapper for iterative IPO
"""

import os
import torch
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType, get_peft_model

class DPOTrainerWrapper:
    """Handles DPO training with PEFT management"""
    
    def __init__(self, config):
        self.config = config
    
    def train(self, model, tokenizer, train_dataset, eval_dataset, iteration: int):
        """Train one iteration with paper-aligned configuration"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"iteration_{iteration}")
        
        # LoRA configuration - match ours.py exactly
        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # DPO Training configuration - match ours.py exactly
        training_args = DPOConfig(
            output_dir=checkpoint_path,
            num_train_epochs=1,  # Match ours.py default
            dataloader_num_workers=4,  # Match ours.py
            per_device_train_batch_size=6,  # Match ours.py
            per_device_eval_batch_size=3,  # Match ours.py
            gradient_accumulation_steps=1,  # Match ours.py
            gradient_checkpointing=True,  # Match ours.py
            optim="adamw_torch_fused",  # Match ours.py
            learning_rate=5e-5,  # Match ours.py
            max_grad_norm=0.3,  # Match ours.py
            warmup_ratio=0.1,  # Match ours.py
            lr_scheduler_type="cosine",  # Match ours.py
            logging_steps=25,  # Match ours.py
            save_steps=500,  # Match ours.py
            save_total_limit=2,  # Match ours.py
            eval_strategy="steps",  # Match ours.py
            eval_steps=300,  # Match ours.py
            bf16=True,  # Match ours.py
            push_to_hub=False,  # Match ours.py
            beta=0.1,  # Match ours.py DPO beta
            max_length=1024,  # Keep DPO-specific params
            max_completion_length=256,
            max_prompt_length=512,
            report_to="wandb" if hasattr(self, 'wandb_enabled') and self.wandb_enabled else "none",
            run_name=f"ipo_iteration_{iteration}",
        )
        
        # Apply LoRA - check if model already has PEFT adapters to avoid double application
        if hasattr(model, 'peft_config') and model.peft_config:
            print("✓ Model already has PEFT adapters, skipping get_peft_model")
            # Model already has PEFT adapters from previous iteration
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            else:
                print("✓ Model has PEFT adapters but print_trainable_parameters not available")
        else:
            print("✓ Applying PEFT adapters to model")
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # The dataset is already in DPO format from prepare_dpo_data()
        # No need to reformat - it already has prompt, chosen, rejected keys
        print(f"✓ DPO dataset format: {train_dataset.column_names}")
        print(f"✓ Training on {len(train_dataset)} preference pairs")
        
        # Create trainer
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        
        # Train
        train_result = trainer.train()
        
        # Clear cache before evaluation
        torch.cuda.empty_cache()
        
        # Get metrics with memory management
        try:
            # Try evaluation with memory safety
            if hasattr(self.config, 'skip_dpo_eval') and self.config.skip_dpo_eval:
                print("⚠️ Skipping DPO evaluation due to memory constraints")
                eval_results = {"eval_loss": 0.0}
            else:
                eval_results = trainer.evaluate()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ Evaluation OOM - skipping evaluation this iteration")
                torch.cuda.empty_cache()
                eval_results = {"eval_loss": 0.0}
            else:
                raise e
        
        # Save model and tokenizer only if selective saving is enabled
        if getattr(self.config, 'save_all_checkpoints', True):
            # Original behavior - save all checkpoints
            trainer.save_model(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
        else:
            # Selective saving - only save current model temporarily for evaluation
            # We'll determine if it's the best later and save accordingly
            temp_checkpoint_path = os.path.join(self.config.checkpoint_dir, "temp_current")
            trainer.save_model(temp_checkpoint_path)
            tokenizer.save_pretrained(temp_checkpoint_path)
        
        # Clear trainer memory immediately after saving
        del trainer
        torch.cuda.empty_cache()
        
        return train_result.training_loss, eval_results.get('eval_loss', 0)