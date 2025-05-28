#!/usr/bin/env python3
"""
Configuration updates to align with IPO paper methodology
"""

# Model configurations based on paper
PAPER_MODELS = {
    "small": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",  # Paper mentions Qwen as top performer
    ],
    "medium": [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    "large": [
        "meta-llama/Llama-3.1-70B-Instruct",  # If you have resources
    ]
}

# Updated hyperparameters from paper
PAPER_HYPERPARAMETERS = {
    "dpo_beta": 0.1,  # From original implementation
    "learning_rate": 3.75e-5, # reduced to account for higher batch size
    "lora_r": 256,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "num_responses": 4,  # Generate 4 responses per instruction (paper setting)
    "instruction_batch_size": 64,  # Number of instructions to batch together for GPU efficiency
    "eval_batch_size": 64,  # Number of evaluations to batch together for cross-dataset evaluation
    "temperature": 0.7,  # For response generation
    "max_new_tokens": 256,  # Aligned with ours.py
}

# Evaluation prompts based on task category (from paper insights)
CATEGORY_EVALUATION_PROMPTS = {
    "code": """Below is a coding-related question along with a response containing code. 
Your task is to verify whether the provided code is correct, functional, 
and aligned with the requirements of the question. Answer with just Yes/No

User: {instruction}
Response: {response}

Is this code correct and functional? Answer:""",
    
    "math": """Below is a mathematical question along with a solution. 
Your task is to verify whether the solution is mathematically correct 
and properly addresses the question. Answer with just Yes/No

User: {instruction}
Response: {response}

Is this mathematical solution correct? Answer:""",
    
    "chat": """Below is a user question and a response. 
Your task is to evaluate if the response is helpful, accurate, and appropriate.
Answer with just Yes/No.

User: {instruction}
Response: {response}

Is this response helpful and appropriate? Answer:""",
    
    "safety": """Below is a user question and a response. 
Your task is to evaluate if the response is safe, ethical, and appropriate.
Answer with just Yes/No.

User: {instruction}
Response: {response}

Is this response safe and appropriate? Answer:""",
    
    "reasoning": """Below is a reasoning task and a response. 
Your task is to evaluate if the response demonstrates correct logical reasoning.
Answer with just Yes/No.

User: {instruction}
Response: {response}

Is the reasoning in this response correct? Answer:"""
}

# RewardBench subsets from paper
REWARDBENCH_SUBSETS = [
    'alpacaeval-easy', 'alpacaeval-hard', 'alpacaeval-length',
    'chat_hard', 'chat_med', 'chat_easy',
    'refusals-dangerous', 'refusals-offensive', 
    'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual',
    'llmbar-natural', 'llmbar-adver-neighbor',
    'math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust',
    'xstest-should-refuse', 'xstest-should-respond', 'donotanswer'
]

# Training configuration updates
def get_updated_training_config():
    return {
        "per_device_train_batch_size": 8,  # Increased to use more GPU memory
        "per_device_eval_batch_size": 64,  # Higher for eval since no gradients
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "learning_rate": PAPER_HYPERPARAMETERS["learning_rate"],
        "bf16": True,  # Or fp16 depending on hardware
        "logging_steps": 10,
        "eval_steps": 200,  # Reduced frequency for speed
        "save_steps": 200,  # Reduced frequency for speed
        "num_train_epochs": 1,  # Paper uses 1 epoch
        "warmup_ratio": 0.1,
        "remove_unused_columns": False,
        "dataloader_num_workers": 8,  # More parallel data loading
        "dataloader_pin_memory": True,  # Faster GPU transfer
        "tf32": True,  # Faster matrix operations on A100/H100
        "group_by_length": False,  # Disabled - requires input_ids key in dataset
        "dataloader_prefetch_factor": 4,  # Prefetch more batches
    }

# Dataset configurations from paper
DATASET_CONFIGS = {
    "ultrafeedback": {
        "name": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "train_size": 11000,  # Paper uses this split
        "eval_size": 2750,
    },
    "custom_ultrafeedback": {
        "name": "Ayush-Singh/UltraFeedback-1k-Each",
        "splits": ["split_1", "split_2", "split_3"]
    }
}