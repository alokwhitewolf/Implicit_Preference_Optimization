import argparse
import torch
from transformers import pipeline
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from tqdm import tqdm
import re
import json

def get_reward(ds, pipe):
    rewards_chosen, rewards_rejected = [], []

    for i, sample in enumerate(tqdm(ds)):
        rewards_chosen.append(pipe(f"Instruction: {sample['prompt']}\nResponse: " + sample['chosen'])[0]['score'])
        rewards_rejected.append(pipe(f"Instruction: {sample['prompt']}\nResponse: " + sample['rejected'])[0]['score'])

        if (i + 1) % 1000 == 0:
            print(f"{i + 1} samples processed.")

    ds = ds.add_column('reward_chosen', rewards_chosen)
    ds = ds.add_column('reward_rejected', rewards_rejected)

    return ds

def calculate_accuracy(ds):
    correct = 0
    total = len(ds)

    for i in range(total):
        if ds['reward_chosen'][i] > ds['reward_rejected'][i]:
            correct += 1

    accuracy = correct / total
    return accuracy

def main(args):
    login(args.hf_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_kwargs = {}
    if args.quantized:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True},
        }
        print("Model will be loaded in quantized mode.")
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
        }
        print("Model will be loaded in full precision mode.")

    pipe = pipeline(
        "text-classification",
        model=args.model_name,
        model_kwargs=model_kwargs,
        device_map=device,
        truncation=True,          
        max_length=1024,
    )

    dataset_name = "allenai/reward-bench"
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)['raw']

    categories = {
    "chat": [
        "alpacaeval-easy", "alpacaeval-hard", "alpacaeval-length", "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut", "llmbar-adver-manual", "llmbar-adver-neighbor",
        "llmbar-natural", "mt-bench-easy", "mt-bench-hard", "mt-bench-med"
            ],
    "safety": ["donotanswer", "refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond"],
    "code": ["hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
    "math": ["math-prm"]
    }

    unique_subsets = categories[args.set_name]
    subset_datasets = {subset: dataset.filter(lambda x: x['subset'] == subset) for subset in unique_subsets}

    processed_splits = {}
    subset_accuracies = {}

    for subset_name, subset_data in subset_datasets.items():
        print(f"Processing subset: {subset_name}")
        rewards = get_reward(subset_data, pipe)
        accuracy = calculate_accuracy(rewards)
        subset_accuracies[subset_name] = accuracy
        print(f"Accuracy for subset '{subset_name}': {accuracy * 100:.2f}%")

        sanitized_split_name = re.sub(r'\W+', '_', subset_name)  
        processed_splits[sanitized_split_name] = rewards

    processed_dataset_dict = DatasetDict(processed_splits)
    processed_dataset_dict.push_to_hub(f"{args.hf_user}/{dataset_name.split('/')[-1]}-{args.model_name.split('/')[-1]}-normal")

    for subset_name, accuracy in subset_accuracies.items():
        result = f"Final accuracy for {subset_name}: {accuracy * 100:.2f}%"
        print(result)

    json_output_path = f"accuracy_{dataset_name.split('/')[-1]}_norm_{args.model_name.split('/')[-1]}.json"
    with open(json_output_path, "w") as json_file:
        json.dump(subset_accuracies, json_file, indent=4)

    print(f"Accuracies saved to JSON file: {json_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer rewards for each subset, evaluate accuracy, and push results to Hugging Face Hub")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face user name to push datasets")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--set_name", type=str, required=True, help="Set")
    parser.add_argument("--quantized", action="store_true", help="Enable quantized model loading (default: False)")
    args = parser.parse_args()

    main(args)