from datasets import load_dataset, concatenate_datasets
import numpy as np
from huggingface_hub import HfApi
from tqdm import tqdm
from collections import Counter

def sample_and_upload_dataset(
    dataset_name,
    num_samples=500,
    new_dataset_name=None,
    organization=None,
    private=True,
    token=None
):
    all_configs = ["default"]
    
    print(f"Loading {len(all_configs)} configurations...")
    combined_datasets = []
    for config in tqdm(all_configs, desc="Loading configurations"):
        dataset = load_dataset(dataset_name, config)
        dataset['test'] = dataset['test'].add_column(
            'source_config', [config] * len(dataset['test'])
        )
        combined_datasets.append(dataset['test'])
    combined_datasets = []
    for config in tqdm(all_configs, desc="Loading configurations"):
        dataset = load_dataset(dataset_name, config)
        dataset['test'] = dataset['test'].add_column(
            'source_config', [config] * len(dataset['test'])
        )
        combined_datasets.append(dataset['test'])
    combined_dataset = concatenate_datasets(combined_datasets)
    print(f"Total examples in combined dataset: {len(combined_dataset)}")
    
    # Get random indices for sampling
    total_size = len(combined_dataset)
    if num_samples > total_size:
        raise ValueError(f"Requested samples ({num_samples}) exceeds dataset size ({total_size})")
    
    random_indices = np.random.choice(total_size, num_samples, replace=False)
    
    # Create new sampled dataset
    sampled_dataset = combined_dataset.select(random_indices)
    
    # Count distribution using Counter
    config_distribution = Counter(sampled_dataset['source_config'])
    print("\nDistribution of samples across configurations:")
    for config, count in config_distribution.items():
        print(f"{config}: {count} samples")
    
    # Prepare the new dataset for upload
    if new_dataset_name is None:
        new_dataset_name = f"{dataset_name.split('/')[-1]}_sampled_{num_samples}"
    
    # Push to the Huggingface Hub
    if organization:
        repo_id = f"{organization}/{new_dataset_name}"
    else:
        repo_id = new_dataset_name
        
    sampled_dataset.push_to_hub(
        repo_id,
        private=private,
        token=token
    )
    
    print(f"\nSuccessfully created and uploaded dataset: {repo_id}")
    return sampled_dataset

# Example usage
if __name__ == "__main__":
    # Replace with your values
    dataset_name = "Dataset_name"
    your_token = "Key"  
    
    sampled_data = sample_and_upload_dataset(
        dataset_name=dataset_name,
        num_samples=500,
        token=your_token,
        new_dataset_name = "Dataset_name",
        private=False
    )