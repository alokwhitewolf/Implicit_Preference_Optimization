import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import login
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_choices(choices):
    formatted_choices = "Choices:\n"
    formatted_choices += "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])  
    return formatted_choices

def format_instruction(row):
    choices_formatted = format_choices(row['choices'])
    instruction = f"""Answer the following multiple-choice question correctly
### Question: {row['question']}
{choices_formatted}
### Instruction: Choose the correct answer from the given options."""
    return instruction


def generate_single_response(instruction):
    instruction = f"[INST] {instruction} [/INST]"
    input_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(device)
    torch.manual_seed(42)
    outputs = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(instruction):].strip()
    return response

def process_dataset(dataset):
    df = pd.DataFrame(dataset)
    df['formatted_input'] = df.apply(format_instruction, axis=1)
    model_responses = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        response = generate_single_response(row['formatted_input'])
        model_responses.append(response)

    df['model_response'] = model_responses
    new_dataset = Dataset.from_pandas(df)
    return new_dataset

set_seed(42)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
login("key")
base_model = AutoModelForCausalLM.from_pretrained("base_model_id",torch_dtype = torch.float16)
peft_model_id = "peft_model_id"
model = PeftModel.from_pretrained(base_model, peft_model_id)

device = "cuda"
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("peft_model_id")

dataset_name = "dataset_name"
dataset = load_dataset(dataset_name)['train']

processed_dataset = process_dataset(dataset)
# Push to hub
processed_dataset.push_to_hub(
    "Shwetasingh123/MMLU_subset_llama_self",  
    token="key"  
)
