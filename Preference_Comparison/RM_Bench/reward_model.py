import argparse
import torch
from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import re
import json

def get_reward(ds, pipe):
    reward_chosen_1, reward_rejected_1 = [], []
    reward_chosen_2, reward_rejected_2 = [], []
    reward_chosen_3, reward_rejected_3 = [], []

    for i, sample in enumerate(tqdm(ds)):
        reward_chosen_1.append(pipe(f"Instruction: {sample['prompt']}\nResponse: "+sample['chosen_1'])[0]['score'])
        reward_rejected_1.append(pipe(f"Instruction: {sample['prompt']}\nResponse: "+sample['rejected_1'])[0]['score'])
        reward_chosen_2.append(pipe(f"Instruction: {sample['prompt']}\nResponse: "+sample['chosen_2'])[0]['score'])
        reward_rejected_2.append(pipe(f"Instruction: {sample['prompt']}\nResponse: "+sample['rejected_2'])[0]['score'])
        reward_chosen_3.append(pipe(f"Instruction: {sample['prompt']}\nResponse: "+sample['chosen_3'])[0]['score'])
        reward_rejected_3.append(pipe(f"Instruction: {sample['prompt']}\nResponse: "+sample['rejected_3'])[0]['score'])

        if (i + 1) % 1000 == 0:
            print(f"{i + 1} samples inferred.")

    ds = ds.add_column('reward_chosen_1', reward_chosen_1)
    ds = ds.add_column('reward_rejected_1', reward_rejected_1)
    ds = ds.add_column('reward_chosen_2', reward_chosen_2)
    ds = ds.add_column('reward_rejected_2', reward_rejected_2)
    ds = ds.add_column('reward_chosen_3', reward_chosen_3)
    ds = ds.add_column('reward_rejected_3', reward_rejected_3)

    return ds

def calculate_accuracy(ds):
    levels = [1, 2, 3]
    accuracies = {}

    for level in levels:
        correct = 0
        total = len(ds)
        for i in range(total):
            chosen_score = ds[f'reward_chosen_{level}'][i]
            rejected_score = ds[f'reward_rejected_{level}'][i]
            if chosen_score > rejected_score:
                correct += 1
        accuracies[f'level_{level}_accuracy'] = correct / total

    for level, acc in accuracies.items():
        print(f"Accuracy for {level}: {acc * 100:.2f}%")

    return accuracies

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

    datasets = [
        "Ayush-Singh/RM-Bench-chat",
        "Ayush-Singh/RM-Bench-code",
        "Ayush-Singh/RM-Bench-math",
        "Ayush-Singh/RM-Bench-safety-response",
        "Ayush-Singh/RM-Bench-safety-refuse",
    ]

    accuracies_dict = {}

    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)['train']
        rewards = get_reward(dataset, pipe)
        accuracies = calculate_accuracy(rewards)
        accuracies_dict[dataset_name] = accuracies

        name = re.search(r'/([^/]+)$', dataset_name).group(1)
        rewards.push_to_hub(f"{args.hf_user}/{name}-{args.model_name.split('/')[-1]}-normal")
    
    json_output_path = f"accuracy_{dataset_name.split('/')[-1]}_norm_{args.model_name.split('/')[-1]}.json"
    with open(json_output_path, "w") as f:
        json.dump(accuracies_dict, f, indent=4)
    print(f"Accuracies have been saved to {json_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer rewards using a pre-trained model and push results to Hugging Face Hub")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face user name to push datasets")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--quantized", action="store_true", help="Enable quantized model loading (default: False)")
    args = parser.parse_args()

    main(args)
