import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from datasets import DatasetDict, Dataset, load_dataset
from tqdm import tqdm
import re
from typing import Dict, List

def load_prompts(json_path: str) -> Dict[str, List[str]]:
    """Load category-wise prompts from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def setup_model(model_id, quantized):
    # [Previous setup_model implementation remains the same]
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
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def get_eval_prompt(subset_name: str, instruction: str, response: str, prompt_template: str) -> str:
    """Generate evaluation prompt using the provided template."""
    return f"""{prompt_template}
    User : {instruction}
    Response : {response}
    """

def generate_yes_no_probability(instruction, response, model, tokenizer, subset_name, prompt_template):
    eval_prompt = get_eval_prompt(subset_name, instruction, response, prompt_template)
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

def evaluate_rewards_by_subset(ds, model, tokenizer, dataset_name, prompts_dict):
    subsets = set(ds['subset'])
    subset_results = {}
    prompt_results = {}
    processed_splits = {}

    for subset_name in subsets:
        subset_data = ds.filter(lambda x: x['subset'] == subset_name)
        prompt_templates = prompts_dict.get(subset_name, [prompts_dict['default']])
        
        subset_accuracies = []
        processed_data_all = []

        for prompt_idx, prompt_template in enumerate(prompt_templates):
            correct = 0
            total = len(subset_data)
            processed_data = []

            for item in tqdm(subset_data, desc=f"Evaluating subset {subset_name} with prompt {prompt_idx + 1}"):
                prompt = item['prompt']
                chosen_response = item['chosen']
                rejected_response = item['rejected']

                chosen_yes_prob, chosen_no_prob = generate_yes_no_probability(
                    prompt, chosen_response, model, tokenizer, subset_name, prompt_template
                )
                rejected_yes_prob, rejected_no_prob = generate_yes_no_probability(
                    prompt, rejected_response, model, tokenizer, subset_name, prompt_template
                )

                if chosen_yes_prob > rejected_yes_prob:
                    correct += 1

                item_with_probs = item.copy()
                item_with_probs.update({
                    'prompt_template': prompt_template,
                    'prompt_index': prompt_idx,
                    'chosen_yes_prob': chosen_yes_prob,
                    'chosen_no_prob': chosen_no_prob,
                    'rejected_yes_prob': rejected_yes_prob,
                    'rejected_no_prob': rejected_no_prob
                })
                processed_data.append(item_with_probs)

            accuracy = (correct / total) * 100 if total > 0 else 0
            print(f"Accuarcy for prompt {prompt_idx} of subset {subset_name} is {accuracy}.")
            subset_accuracies.append(accuracy)
            processed_data_all.extend(processed_data)
            
            prompt_results[f"{subset_name}_prompt_{prompt_idx + 1}"] = {
                'accuracy': accuracy,
                'prompt_template': prompt_template
            }

        # Calculate average accuracy for the subset
        subset_results[subset_name] = {
            'average_accuracy': sum(subset_accuracies) / len(subset_accuracies),
            'individual_accuracies': subset_accuracies
        }

        sanitized_split_name = re.sub(r'\W+', '_', subset_name)
        processed_splits[sanitized_split_name] = Dataset.from_list(processed_data_all)

    return subset_results, prompt_results, DatasetDict(processed_splits)

def main(args):
    # Load prompts from JSON file
    prompts_dict = load_prompts(args.prompts_file)
    
    login(args.hf_key)
    model, tokenizer = setup_model(args.model_name, args.quantized)
    dataset_name = "allenai/reward-bench"
    print(f"Processing dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)['raw']
    
    subset_accuracies, prompt_accuracies, processed_dataset_dict = evaluate_rewards_by_subset(
        dataset, model, tokenizer, dataset_name, prompts_dict
    )
    
    # Push processed dataset to Hub
    # processed_dataset_dict.push_to_hub(
    #     f"{args.hf_user}/{dataset_name.split('/')[-1]}-{args.model_name.split('/')[-1]}-yes-no-multi-prompt"
    # )

    # Save both subset-level and prompt-level results
    results = {
        'subset_accuracies': subset_accuracies,
        'prompt_accuracies': prompt_accuracies
    }
    
    results_file_path = f"accuracy_{dataset_name.split('/')[-1]}_yesno_{args.model_name.split('/')[-1]}_multi_prompt.json"
    with open(results_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    print("\nSubset-level Results:")
    for subset_name, results in subset_accuracies.items():
        print(f"\n{subset_name}:")
        print(f"Average accuracy: {results['average_accuracy']:.2f}%")
        print("Individual prompt accuracies:")
        for idx, acc in enumerate(results['individual_accuracies']):
            print(f"Prompt {idx + 1}: {acc:.2f}%")

    print(f"\nDetailed results saved to {results_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate subset-wise accuracies with multiple prompts")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face user name to push datasets")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--quantized", action="store_true", help="Use quantized model for inference")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to JSON file containing prompts")
    args = parser.parse_args()

    main(args)
