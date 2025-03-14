import argparse
import os
import json
import re
from tqdm import tqdm
from datasets import DatasetDict, Dataset, load_dataset
from openai import OpenAI
from typing import List, Dict, Any
from huggingface_hub import login

login("key")

def setup_gpt_client(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> OpenAI:
    """Set up the OpenAI client for GPT API."""
    return OpenAI(base_url=base_url, api_key=api_key)

def get_comparative_prompt(instruction: str, response1: str, response2: str) -> str:
    """Create a prompt for comparing two responses."""
    comparative_prompt = """Compare the following two responses to the user's question and determine which one is better. 
    Consider factors such as:
    - Relevance to the question
    - Completeness of the answer
    - Clarity and coherence
    - Accuracy and helpfulness
    - Overall quality

    User Question: {instruction}

    Response 1:
    {response1}

    Response 2:
    {response2}

    Which response is better? Answer with either "Response 1" or "Response 2" followed by a brief explanation. 
    Format your answer as:
    Better response: [Response 1/Response 2]
    Explanation: [Your explanation]"""
    
    return comparative_prompt.format(
        instruction=instruction,
        response1=response1,
        response2=response2
    )

def extract_better_response(text: str) -> int:
    pattern = r"Better response:\s*(Response\s*[12])"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        response_text = match.group(1).lower()
        return 0 if "1" in response_text else 1
    return -1  # Return -1 if no clear choice could be extracted

def evaluate_responses(ds, client: OpenAI, dataset_name: str, model: str = "openai/chatgpt-4o-latest"):
    """Evaluate response comparisons for each subset using the GPT API."""
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

            comparative_prompt = get_comparative_prompt(prompt, chosen_response, rejected_response)
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": comparative_prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                comparative_text = completion.choices[0].message.content
                better_response_idx = extract_better_response(comparative_text)
                
                if better_response_idx == 0:
                    correct += 1
                
                item['comparative_evaluation'] = comparative_text
                item['better_response_idx'] = better_response_idx
                item['model_agrees_with_human'] = better_response_idx == 0
                processed_data.append(item)

            except Exception as e:
                print(f"Error processing prompt: {str(e)}")
                continue

        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Accuracy for subset '{subset_name}': {accuracy:.2f}%")
        subset_results[subset_name] = accuracy

        sanitized_split_name = re.sub(r'\W+', '_', subset_name)
        processed_splits[sanitized_split_name] = Dataset.from_list(processed_data)

    return subset_results, DatasetDict(processed_splits)

def main(args):
    """Main function to evaluate subsets and push results to Hugging Face Hub."""
    client = setup_gpt_client(api_key=args.api_key)
    dataset_name = "allenai/reward-bench"
    print(f"Processing dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)['raw']
    subset_accuracies, processed_dataset_dict = evaluate_responses(dataset, client, dataset_name, model=args.model)
    processed_dataset_dict.push_to_hub(f"{args.hf_user}/{dataset_name.split('/')[-1]}-{args.model.split('/')[-1]}-comparison")

    for subset_name, accuracy in subset_accuracies.items():
        result = f"Final accuracy for {subset_name}: {accuracy}%"
        print(result)

    accuracy_file_path = f"accuracy_{dataset_name.split('/')[-1]}_comparison_{args.model.split('/')[-1]}.json"
    with open(accuracy_file_path, "w") as json_file:
        json.dump(subset_accuracies, json_file, indent=4)
    print(f"Accuracies saved to {accuracy_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate subset-wise comparisons using GPT API and push results to Hugging Face Hub")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face user name to push datasets")
    parser.add_argument("--model", type=str, default="openai/chatgpt-4o-latest", help="GPT model to use for evaluation")
    args = parser.parse_args()

    main(args)

