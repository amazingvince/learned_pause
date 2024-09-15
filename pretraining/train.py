from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, interleave_datasets
import torch
import random
import os
from torch.utils.data import IterableDataset

os.environ["WANDB_PROJECT"] = "learned_pause_token"

# Define the <pause> token
PAUSE_TOKEN = "<pause>"
seed = 42
model_name = "BEE-spoke-data/smol_llama-220M-GQA"

# Modify tokenizer to include the <pause> token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

PAUSE_TOKEN_ID = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)

print(f"pause token id: {PAUSE_TOKEN_ID}")

# Save the modified tokenizer
tokenizer_save_path = "./modified_tokenizer"
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Modified tokenizer saved to: {tokenizer_save_path}")

# Load or create your dataset
ds_fw = load_dataset(
    "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", streaming=True,
    split="train").shuffle(seed=seed)
ds_python = load_dataset(
    "BEE-spoke-data/smollm-corpus-python", streaming=True, split="train"
).shuffle(seed=seed)
ds_cosmo = load_dataset(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True,
    split="train").shuffle(seed=seed)

def keep_only_text(dataset):
    return dataset.remove_columns([col for col in dataset.column_names if col != 'text'])

ds_fw = keep_only_text(ds_fw)
ds_python = keep_only_text(ds_python)
ds_cosmo = keep_only_text(ds_cosmo)

print("Dataset column names:")
print("ds_fw:", ds_fw.column_names)
print("ds_python:", ds_python.column_names)
print("ds_cosmo:", ds_cosmo.column_names)

ds_set = [ds_fw, ds_python, ds_cosmo]
probabilities = [0.5, 0.2, 0.3]
dataset = interleave_datasets(ds_set, probabilities=probabilities, stopping_strategy="first_exhausted")

# print("Interleaved dataset features:")
# print(next(iter(dataset)))

def insert_pause_tokens(text):
    tokens = tokenizer.tokenize(text)
    # print(f"tokens: {tokens}")
    total_length = len(tokens)
    for pause in range(random.randint(0, total_length//20)):
        pause_position = random.randint(0, total_length)
        for i in range(random.randint(1, 5)):
            tokens.insert(pause_position, PAUSE_TOKEN)
        # tokens.insert(pause_position, PAUSE_TOKEN)
            total_length += 1
    
    return tokenizer.convert_tokens_to_string(tokens)

def custom_data_collator(features):
    texts = [f for f in features if isinstance(f, str)]
    
    if not texts:
        raise ValueError("No valid text found in features.")
    
    modified_texts = [insert_pause_tokens(text) for text in texts]
    batch = tokenizer(modified_texts, padding=True, truncation=True, return_tensors="pt", max_length=2048)
    
    # Ensure input_ids are long integers, but keep them on CPU
    # batch["input_ids"] = batch["input_ids"].to(torch.long)
    
    # # Create labels, ignoring loss for <pause> tokens
    labels = batch["input_ids"].clone()
    # pause_token_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)
    # labels[labels == PAUSE_TOKEN_ID] = -100  # -100 is ignored in loss calculation
    
    # Shift labels
    labels = torch.roll(labels, -1, dims=1)
    labels[:, -1] = -100
    batch["labels"] = labels
    
    return batch



# Custom IterableDataset wrapper
class CustomIterableDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            if 'text' in item and isinstance(item['text'], str):
                yield item['text']
            else:
                print(f"Skipping item: {item}")  # Debug print

# Load your model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))

# # Ensure model parameters have consistent dtype
# dtype = torch.float32
# model.to(dtype)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    # gradient_accumulation_steps=4,
    save_steps=1000,
    save_total_limit=2,
    report_to="wandb",
    max_steps=100000,  # Adjust this value based on your needs
    logging_steps=10,
    evaluation_strategy="no",
    bf16=True,  # Disable mixed precision training
    dataloader_pin_memory=True, 
)

# Wrap the dataset
wrapped_dataset = CustomIterableDataset(dataset)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=wrapped_dataset,
    data_collator=custom_data_collator,
)

# Enable TensorFloat32 if available
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("Enabling TensorFloat32")
    torch.set_float32_matmul_precision('high')

# Start training
trainer.train()