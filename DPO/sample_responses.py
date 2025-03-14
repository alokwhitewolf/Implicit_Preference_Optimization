import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm
from huggingface_hub import login

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_responses(model, tokenizer, prompt, num_responses=4, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    responses = []
    for _ in range(num_responses):
        output = model.generate(**inputs, 
                                max_new_tokens=256,
                                num_return_sequences=1,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        responses.append(response_text)
    return responses

torch.manual_seed(42)
def main(dataset_name, split_name, model_name, hf_api, username, save_dataset_name):
    login(hf_api)
    dataset = load_dataset(dataset_name, split=split_name)
    model, tokenizer = load_model(model_name)
    
    new_data = {"prompt": [], "responses": []}
    for example in tqdm(dataset):
        prompt = example["instruction"]
        responses = generate_responses(model, tokenizer, prompt)
        new_data["prompt"].append(prompt)
        new_data["responses"].append(responses)
    
    new_dataset = Dataset.from_dict(new_data)
    save_path = f"{username}/{save_dataset_name}"
    new_dataset.push_to_hub(save_path)
    print(f"Dataset saved successfully to {save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Name of the Hugging Face dataset")
    parser.add_argument("--split_name", type=str, help="Dataset split to use (e.g., 'train', 'test')")
    parser.add_argument("--model_name", type=str, help="Model name to use from Hugging Face")
    parser.add_argument("--hf_api", type=str, help="Enter HF API key")
    parser.add_argument("--username", type=str, help="Your Hugging Face username")
    parser.add_argument("--save_dataset_name", type=str, help="Name to save the dataset")

    args = parser.parse_args()
    
    main(args.dataset_name, args.split_name, args.model_name, args.hf_api, args.username, args.save_dataset_name)
