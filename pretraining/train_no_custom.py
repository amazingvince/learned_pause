from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from datasets import load_dataset, interleave_datasets
import torch
import random
import os

os.environ["WANDB_PROJECT"] = "learned_pause_token"

# Define the <pause> token
PAUSE_TOKEN = "<pause>"
seed = 42
model_name = "BEE-spoke-data/smol_llama-220M-GQA"

# Modify tokenizer to include the <pause> token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

# Modify tokenizer to include the <pause> token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

PAUSE_TOKEN_ID = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)
print(f"pause token id: {PAUSE_TOKEN_ID}")


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

def insert_pause_tokens(example, num_pauses=5):
    text = example['text']
    tokens = tokenizer.tokenize(text)
    total_length = len(tokens)
    pause_positions = sorted(random.sample(range(total_length), min(num_pauses, total_length)))
    for pos in reversed(pause_positions):
        tokens.insert(pos, PAUSE_TOKEN)
    example['text'] = tokenizer.convert_tokens_to_string(tokens)
    return example

# Apply the insert_pause_tokens function to the dataset
dataset = dataset.map(insert_pause_tokens)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=2048)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Load your model
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             attn_implementation="flash_attention_2",
                                             )
model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    # gradient_accumulation_steps=4,
    save_steps=5000,
    save_total_limit=2,
    report_to="wandb",
    max_steps=100000,  # Adjust this value based on your needs
    logging_steps=100,
    evaluation_strategy="no",
    bf16=True,
    dataloader_pin_memory=True,
    
)

# Initialize DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt", max_length=2048)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Enable TensorFloat32 if available
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("Enabling TensorFloat32")
    torch.set_float32_matmul_precision('high')

# Start training
trainer.train()