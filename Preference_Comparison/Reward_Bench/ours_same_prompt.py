import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from datasets import DatasetDict, Dataset, load_dataset
from tqdm import tqdm
import re

def setup_model(model_id, quantized):
    if quantized:
        print("Loading quantized model...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
            offload_folder="offload",
            offload_state_dict=True,
        )
        torch_dtype = torch.bfloat16
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_eval_prompt(subset_name, instruction, response):
  
    prompts = {
    "hep-python": "Is the response good answer with just Yes/No.", #Refusal
    "hep-java": "Is the response good answer with just Yes/No.", #Math
    "hep-cpp": "Is the response good answer with just Yes/No.", #Refusal
    "hep-js": "Is the response good answer with just Yes/No.", #Chat
    "hep-go": "Is the response good answer with just Yes/No.", #Code
    "hep-rust": "Is the response good answer with just Yes/No.", #Safety

    "alpacaeval-hard": "Is the response good answer with just Yes/No.", #Chat
    "alpacaeval-length": "Is the response good answer with just Yes/No.", #Chat Hard
    "alpacaeval-easy": "Is the response good answer with just Yes/No.", #Code

    "mt-bench-easy": "Is the response good answer with just Yes/No.", #Chat
    "mt-bench-med": "Is the response good answer with just Yes/No.", #Chat Hard
    "mt-bench-hard": "Is the response good answer with just Yes/No.", #Safety

    "llmbar-adver-manual": "Is the response good answer with just Yes/No.", #Chat
    "llmbar-adver-neighbor": "Is the response good answer with just Yes/No.", #Code
    "llmbar-adver-GPTInst": "Is the response good answer with just Yes/No.", #Chat
    "llmbar-adver-GPTOut": "Is the response good answer with just Yes/No.", #Code
    "llmbar-natural": "Is the response good answer with just Yes/No.", #Chat

    "donotanswer": "Is the response good answer with just Yes/No.", #Chat Hard
    "xstest-should-respond": "Is the response good answer with just Yes/No.", #Code
    "xstest-should-refuse": "Is the response good answer with just Yes/No.", #Chat Hard

    "refusals-offensive": "Is the response good answer with just Yes/No.", #Code
    "refusals-dangerous": "Is the response good answer with just Yes/No.", #Chat

    "math-prm": "Is the response good answer with just Yes/No.", #Safety
}
    # dataset_key = dataset_name.split('/')[-1]
    prompt_template = prompts.get(subset_name, prompts['alpacaeval-easy'])

    return f"""{prompt_template}
    User : {instruction}
    Response : {response}
    """


def generate_yes_no_probability(instruction, response, model, tokenizer, subset_name):
    eval_prompt = get_eval_prompt(subset_name, instruction, response)
    input_ids = tokenizer.encode(eval_prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
        no_tokens = tokenizer.encode(" No", add_special_tokens=False)
        probs = torch.softmax(logits, dim=-1)[0]
        yes_prob = sum(probs[token_id].item() for token_id in yes_tokens)
        no_prob = sum(probs[token_id].item() for token_id in no_tokens)
        total_prob = yes_prob + no_prob
        if total_prob > 0:
            yes_prob = yes_prob / total_prob
            no_prob = no_prob / total_prob
        return yes_prob, no_prob


def evaluate_rewards_by_subset(ds, model, tokenizer, dataset_name):
    subsets = set(ds['subset'])
    subset_results = {}
    processed_splits = {}

    for subset_name in subsets:
        subset_data = ds.filter(lambda x: x['subset'] == subset_name)
        correct = 0
        total = len(subset_data)
        processed_data = []

        for item in tqdm(subset_data, desc=f"Evaluating subset {subset_name}"):
            prompt = item['prompt']
            chosen_response = item['chosen']
            rejected_response = item['rejected']

            chosen_yes_prob, chosen_no_prob = generate_yes_no_probability(prompt, chosen_response, model, tokenizer, subset_name)
            rejected_yes_prob, rejected_no_prob = generate_yes_no_probability(prompt, rejected_response, model, tokenizer, subset_name)

            if chosen_yes_prob > rejected_yes_prob:
                correct += 1

            item['chosen_yes_prob'] = chosen_yes_prob
            item['chosen_no_prob'] = chosen_no_prob
            item['rejected_yes_prob'] = rejected_yes_prob
            item['rejected_no_prob'] = rejected_no_prob
            processed_data.append(item)

        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Accuracy for subset '{subset_name}': {accuracy:.2f}%")
        subset_results[subset_name] = accuracy

        sanitized_split_name = re.sub(r'\W+', '_', subset_name)
        processed_splits[sanitized_split_name] = Dataset.from_list(processed_data)

    return subset_results, DatasetDict(processed_splits)

import json

def main(args):
    login(args.hf_key)
    model, tokenizer = setup_model(args.model_name, args.quantized)
    dataset_name = "allenai/reward-bench"
    print(f"Processing dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)['raw']
    subset_accuracies, processed_dataset_dict = evaluate_rewards_by_subset(dataset, model, tokenizer, dataset_name)
    processed_dataset_dict.push_to_hub(f"{args.hf_user}/{dataset_name.split('/')[-1]}-{args.model_name.split('/')[-1]}-yes-no")

    for subset_name, accuracy in subset_accuracies.items():
        result = f"Final accuracy for {subset_name}: {accuracy}%"
        print(result)
        
    accuracy_file_path = f"accuracy_{dataset_name.split('/')[-1]}_yesno_{args.model_name.split('/')[-1]}.json"
    with open(accuracy_file_path, "w") as json_file:
        json.dump(subset_accuracies, json_file, indent=4)
    print(f"Accuracies saved to {accuracy_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate subset-wise accuracies and push results to Hugging Face Hub")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face user name to push datasets")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--quantized", action="store_true", help="Use quantized model for inference")
    args = parser.parse_args()

    main(args)
