from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import random
from huggingface_hub import login
import argparse
import gc
import os
import time
import re
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from trl import DPOTrainer,DPOConfig

MISTRAL_FIXED_CHAT_TEMPLATE = "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

MISTRAL_FINAL_TURN_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only assistant roles are supported when modified template for the final turn!') }}{% endif %}{% endfor %}"

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

huggingface_key = "hf_key"
wandb_key = "wandb_key"

os.environ["HF_API_TOKEN"] = huggingface_key
os.environ["WANDB_API_KEY"] = wandb_key
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def generate_responses(instruction,model, tokenizer, num_responses=4):
    responses = []
    input_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(device)
    
    for i in range(num_responses):
        
        outputs = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=256,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded[len(instruction):].strip()
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0].strip()
        responses.append(response)
        #print(i)
        #print(response)
        #print("======================================================")
    #print("--------------------------end------------------------------------------------")
    
    return responses

def get_eval_prompt(instruction, response):
    scoring_prompt = """Review the user's question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective,
addressing the user's question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.

User: {instruction}
<response>{response}</response>

After examining the user's instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Score: " """

    return scoring_prompt.format(instruction=instruction, response=response)

def extract_score(text):
    patterns = [
        r'Score\s*[:"]*\s*(\d+)',  # Matches formats like `Score: 5`, `Score "5"`, or `Score 5`
        r'"\s*Score\s*[:"]*\s*(\d+)',  # Matches formats where "Score" appears in quotes
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 5)  # Ensure score is between 1 and 5
    
    return 0

def generate_reward(instruction, response, model, tokenizer,category):
    eval_prompt = get_eval_prompt(instruction, response)
    input_ids = tokenizer.encode(eval_prompt, return_tensors="pt", max_length=2048, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=0.1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    score = extract_score(generated_text)
    return score, generated_text

results = []

def transform_entry(entry):
    chosen_prompt, chosen_response = entry["chosen"].split(" Response:", 1)
    chosen_prompt = chosen_prompt.replace("Assistant:", "").strip()
    rejected_prompt, rejected_response = entry["rejected"].split(" Response:", 1)
    # Clean responses and add EOT token
    return {
        "prompt": f"[INST] {chosen_prompt} [/INST]",
        "chosen": f"{chosen_response.strip().split('<|eot_id|>')[0]}<|eot_id|>",
        "rejected": f"{rejected_response.strip().split('<|eot_id|>')[0]}<|eot_id|>"
    }

def extract_final_assistant_message(messages):
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    raise ValueError("No assistant message found")

# this version is more general and useful for other tasks
def extract_final_assistant_message_index(messages):
    for i, message in enumerate(reversed(messages)):
        if message["role"] == "assistant":
            return len(messages) - i - 1
    raise ValueError("No assistant message found")

def test_extract_final_assistant_message_index():
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you."},
        {"role": "user", "content": "That's good to hear."},
        {"role": "assistant", "content": "Yes, it is."}
    ]

    # print(extract_final_assistant_message_index(messages))
    assert extract_final_assistant_message_index(messages) == 3
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you."},
        {"role": "user", "content": "That's good to hear."},
    ]
    # print(extract_final_assistant_message_index(messages))
    assert extract_final_assistant_message_index(messages) == 1

    try:
        extract_final_assistant_message_index([])
        assert False, "No exception raised"
    except ValueError:
        pass


def maybe_insert_system_message(messages, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    if messages[0]["role"] == "system":
        if "system" not in chat_template:
            raise ValueError("Model uses system messages, but system message found in conversation")
        return

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template:
        messages.insert(0, {"role": "system", "content": default_system_message})


def create_triplets(example, tokenizer, final_turn_chat_template=None, default_system_message=DEFAULT_SYSTEM_MESSAGE):

    chosen_messages = example["chosen"]
    rejected_messages = example["rejected"]
    chosen_index = extract_final_assistant_message_index(chosen_messages)
    rejected_index = extract_final_assistant_message_index(rejected_messages)
    if chosen_messages[:chosen_index] != rejected_messages[:rejected_index]:
        raise ValueError("More than one assistant message is different between chosen and rejected responses. This is not expected."
                         "\nMust be able to extract a single assistant message that is different and keep the rest for the prompt.")
    prompt_messages = chosen_messages[:chosen_index]
    chosen_messages = [chosen_messages[chosen_index]]
    rejected_messages =[rejected_messages[rejected_index]]

    maybe_insert_system_message(prompt_messages, tokenizer, default_system_message=default_system_message)

    if final_turn_chat_template is None:
        final_turn_chat_template = tokenizer.chat_template
    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
        "chosen": tokenizer.apply_chat_template(chosen_messages, chat_template=final_turn_chat_template, tokenize=False),
        "rejected": tokenizer.apply_chat_template(rejected_messages, chat_template=final_turn_chat_template, tokenize=False),
    }

def generate(split_1, model, tokenizer):
    for item in tqdm(split_1):
        instruction = item["instruction"]
        category = item["category"]
        responses = generate_responses(instruction,model, tokenizer, num_responses=4)
        scored_responses = []
        for response in responses:
            reward = generate_reward(
                instruction, 
                response, 
                model, 
                tokenizer, 
                category
            )
            scored_responses.append((response, reward))
        scored_responses.sort(key=lambda x: x[1], reverse=True)
        # print(scored_responses)
        best_response = scored_responses[0][0]
        worst_response = scored_responses[-1][0]
        pair = {
            "chosen": f"Assistant: {instruction}\n Response:{best_response}",
            "rejected": f"Assistant:{instruction}\n Response:{worst_response}"
        }
       # print(pair['chosen'],end="\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
       #
       #  print(pair['rejected'],end="\n--------------------------------------------------------")
        results.append(pair)
        # print(results)
    with open(f'{args.mapped_path}', 'w') as f:
        json.dump(results, f, indent=2)
    with open(f"{args.mapped_path}", "r") as f:
        data = json.load(f)
    transformed_data = [transform_entry(entry) for entry in data]
    with open(f"{args.train_path}", "w") as f:
        json.dump(transformed_data, f, indent=2)

def parse_arguments(input_args=None):

    DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    parser = argparse.ArgumentParser(description="DPO Alignment Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="The model ID for the tokenizer and model")
    parser.add_argument("--dataset_id", type=str, default="argilla/ultrafeedback-binarized-preferences-cleaned",
                        help="The dataset ID for the training data (must comply with DPO expected format, see code and comments if changing)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="The number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The learning rate for training")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="The DPO beta factor for loss, controls divergence from the reference model, higher is less divergence")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="The output directory for saving the trained model")
    parser.add_argument("--run_name", type=str, default=f"dpo_{DEFAULT_MODEL_ID.replace('/', '-')}_{time.strftime('%Y%m%d%H%M')}",
                        help="The name of the training run")
    parser.add_argument("--merge_adapters", action="store_true", help="Merge the adapters and save the model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to the Hugging Face model hub")
    parser.add_argument("--hub_repo_id", type=str, default=f"dpo-ultrafeedback-{DEFAULT_MODEL_ID.replace('/', '-')}",
                        help="The Hugging Face model hub repository ID")
    parser.add_argument("--wandb_project", type=str, default="dpo-ultrafeedback",
                        help="The Weights & Biases project name")
    parser.add_argument("--dataset_name", type=str, default="Ayush-Singh/UltraFeedback-1k-Each",
                        help="Name of the dataset to use")
    parser.add_argument("--split_name", type=str, default="split_1",
                        help="Name of the dataset split to use")
    parser.add_argument("--train_path", type=str, default="/content/sample_data/responses_dpo.json",
                        help="Path to save/load the training dataset")
    parser.add_argument("--mapped_path", type=str, default="/content/sample_data/responses.json",
                        help="Path to save/load the mapped responses")
    parser.add_argument("--just_sample", action="store_true",
                        help="If set, the model will be loaded in float16")
    args = parser.parse_args(input_args)
    return args

def train(args: argparse.Namespace):
    login("key")
    model_id = args.model_id
    dataset_id = args.dataset_id
    run_name = args.run_name
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if args.just_sample:
        print("Loading model in float16...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        print("Loading model in default precision...")
        model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_dataset(f"{args.dataset_name}", split=f"{args.split_name}")
#    split_1 = dataset["split_1"]
    # split_1 = split_1.select(range(min(5, len(dataset))))
    generate(dataset, model, tokenizer)
    if args.just_sample:
        exit()

    output_dir = os.path.join(args.output_dir, run_name)
    resume_from_checkpoint = output_dir if os.path.exists(output_dir) else None
    print("Model ID:", model_id)
    print("Dataset ID:", dataset_id)
    print("Run name:", run_name)
    print("Output directory:", output_dir)
    print("Resume from checkpoint:", "True" if resume_from_checkpoint else "False")



    if os.environ.get('WANDB_PROJECT', None) is None:
        os.environ['WANDB_PROJECT'] = "dpo-ultrafeedback"

    wandb.init(
        project=os.environ['WANDB_PROJECT'],
        name=run_name,
        group=f"{model_id.replace('/', '-')}",
        resume="allow",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    tokenizer.truncation_side = "left" 
    #tokenizer.chat_template = MISTRAL_FIXED_CHAT_TEMPLATE
    if os.path.exists(args.train_path) and os.path.exists(args.train_path):
        print("Loading datasets from disk...")
        train_dataset = load_dataset("json", data_files=args.train_path, split="train")
        eval_dataset = load_dataset("json", data_files=args.train_path, split="train")
    else:
        dataset = load_dataset(dataset_id, split="train").shuffle().select(range(13750))

        # map triplet creation function to our splits of the dataset
        dataset = dataset.map(
            create_triplets,
            remove_columns=dataset.features,
            fn_kwargs={
                "tokenizer": tokenizer,
                "final_turn_chat_template": MISTRAL_FINAL_TURN_CHAT_TEMPLATE,
            },
        )

        # split 11,000 and 2,750 for training and validation assuming 13,750 examples
        dataset = dataset.train_test_split(test_size=int(0.2 * len(dataset)))

        # save to disk as json
        dataset["train"].to_json("train_dataset.json", orient="records")
        dataset["test"].to_json("test_dataset.json", orient="records")
        # this is a bit wasteful, but load from disk to make sure it's the same behaviour
        # as if we had loaded from disk in the first place
        train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
        eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

    print("\nMapped dataset formatting for training, validation with chat_templating for prompt, chosen, rejected")
    print(train_dataset[0]["prompt"])
    print(train_dataset[0]["chosen"])
    print(train_dataset[0]["rejected"])
    print(f"\n\nTraining dataset: {train_dataset}\nValidation dataset: {eval_dataset}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.float16,
    )
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer_args =DPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=4,
        per_device_train_batch_size=8,  # lower to 1 to fit on 16 or even 24GB GPU if necessary
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,  # increase if per_device_train_batch_size is lowered
        gradient_checkpointing=True,  # save memory
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=700,
        fp16= True,
       # bf16=True,
       # tf32=True,  # mixed precision, needed for accumulation
        push_to_hub=False,  # we'll do this manually
        report_to="wandb",
        run_name=run_name,
    )

    dpo_args = {
        "beta": args.dpo_beta,  # DPO beta factor for loss, controls divergence from the reference model, higher is less divergence
        "loss_type": "sigmoid"
    }

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        peft_config=peft_config,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    print("\n\nTraining...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

    print("\n\nTraining complete\n")
    tokenizer.push_to_hub(args.hub_repo_id)

    if args.push_to_hub:
        print("Pushing to Hugging Face model hub...")
        # push to hub
        model.push_to_hub(args.hub_repo_id)
        tokenizer.push_to_hub(args.hub_repo_id)

if __name__ == "__main__":
    args = parse_arguments()
    train(args)
    print("DPO training and model saving complete")
    exit(0)
