from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm

# Initialize wandb
wandb.init(project="llama_pause_policy", config={
    "model_name": "/home/vincent/Documents/learned_pause/pretraining/results/checkpoint-3000",
    "dataset_name": "pszemraj/infinity-instruct-7m-T2T_en",  # Replace with your dataset name
    "max_pauses": 5,
    "base_reward": 1.0,
    "penalty_factor": 0.2,
    "large_penalty": -5.0,
    "max_length": 512,
    "batch_size": 1,
    "accumulation_steps": 1,
    "num_epochs": 5,
    "learning_rate": 1e-5,
    "subset_size": 10000,
    "warmup_steps": 500,
})

# Constants from wandb config
CONFIG = wandb.config

# Initialize accelerator
accelerator = Accelerator()
set_seed(42)

# Initialize tokenizer and add special tokens
tokenizer = LlamaTokenizer.from_pretrained("/home/vincent/Documents/learned_pause/pretraining/modified_tokenizer")

class LLaMAPausePolicy(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(model_name)
        self.pause_token_id = tokenizer.convert_tokens_to_ids('<pause>')

    def forward(self, input_ids, attention_mask=None):
        return self.llama(input_ids, attention_mask=attention_mask, output_hidden_states=True)


def tokenize_function(examples):
    instructions = tokenizer(examples['instruction'], max_length=CONFIG.max_length, truncation=True)
    responses = tokenizer(examples['response'], max_length=CONFIG.max_length, truncation=True)
    return {
        'instruction_ids': instructions['input_ids'],
        'instruction_mask': instructions['attention_mask'],
        'response_ids': responses['input_ids'],
        'response_mask': responses['attention_mask']
    }

# Update the data collator
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        instruction_ids = [f['instruction_ids'] for f in features]
        instruction_mask = [f['instruction_mask'] for f in features]
        response_ids = [f['response_ids'] for f in features]
        response_mask = [f['response_mask'] for f in features]

        instruction_batch = self.tokenizer.pad(
            {'input_ids': instruction_ids, 'attention_mask': instruction_mask},
            padding=True,
            return_tensors='pt'
        )
        response_batch = self.tokenizer.pad(
            {'input_ids': response_ids, 'attention_mask': response_mask},
            padding=True,
            return_tensors='pt'
        )

        return {
            'instruction_ids': instruction_batch['input_ids'],
            'instruction_mask': instruction_batch['attention_mask'],
            'response_ids': response_batch['input_ids'],
            'response_mask': response_batch['attention_mask']
        }

# Load and prepare the dataset
full_dataset = load_dataset(CONFIG.dataset_name)['train'].select(range(CONFIG.subset_size*5))
tokenized_datasets = full_dataset.map(tokenize_function, batched=True, remove_columns=full_dataset.column_names)

data_collator = CustomDataCollator(tokenizer)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(CONFIG.subset_size))
val_dataset = tokenized_datasets["test"].shuffle(seed=6969).select(range(CONFIG.subset_size // 10))

train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, collate_fn=data_collator)

def select_action(policy, input_ids, attention_mask):
    outputs = policy(input_ids, attention_mask)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    dist = Categorical(probs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs, dist.entropy()

def compute_returns(rewards, gamma=0.99):
    returns = torch.zeros_like(rewards)
    R = torch.zeros(rewards.size(0), device=rewards.device)
    for t in reversed(range(rewards.size(1))):
        R = rewards[:, t] + gamma * R
        returns[:, t] = R
    return returns

def train_model(policy, optimizer, scheduler, train_dataloader, val_dataloader, num_epochs):
    policy, optimizer, train_dataloader, scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, scheduler
    )

    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0
        total_reward = 0
        batch_count = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            instruction_ids = batch['instruction_ids']
            instruction_mask = batch['instruction_mask']
            response_ids = batch['response_ids']
            response_mask = batch['response_mask']
            batch_size, resp_len = response_ids.shape

            log_probs_list = []
            rewards_list = []
            entropies_list = []
            pause_counts = torch.zeros(batch_size, device=response_ids.device)
            done = torch.zeros(batch_size, dtype=torch.bool, device=response_ids.device)

            # Process instruction first
            with torch.no_grad():
                instruction_output = policy(instruction_ids, attention_mask=instruction_mask)
                # Use the last token's hidden state from the output
                instruction_hidden_state = instruction_output.hidden_states[-1][:, -1, :]

            for t in range(resp_len - 1):
                # Combine instruction hidden state with response
                combined_input_ids = torch.cat([instruction_ids, response_ids[:, :t+1]], dim=1)
                combined_attention_mask = torch.cat([instruction_mask, response_mask[:, :t+1]], dim=1)
                
                actions, log_probs, entropies = select_action(policy, combined_input_ids, combined_attention_mask)

                correct_actions = response_ids[:, t+1]
                is_correct = (actions == correct_actions).float()
                is_pause = (actions == policy.pause_token_id).float()
                pause_counts += is_pause

                reward = CONFIG.base_reward * is_correct - CONFIG.penalty_factor * pause_counts
                reward = torch.where(pause_counts >= CONFIG.max_pauses, CONFIG.large_penalty, reward)
                reward = torch.where(done, torch.zeros_like(reward), reward)

                done = done | (pause_counts >= CONFIG.max_pauses)

                log_probs_list.append(log_probs)
                rewards_list.append(reward)
                entropies_list.append(entropies)

                if done.all():
                    break

            log_probs = torch.stack(log_probs_list, dim=1)
            rewards = torch.stack(rewards_list, dim=1)
            entropies = torch.stack(entropies_list, dim=1)

            returns = compute_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            policy_loss = -(log_probs * returns).mean()
            entropy_loss = -0.01 * entropies.mean()
            loss = policy_loss + entropy_loss

            accelerator.backward(loss)

            if (batch_count + 1) % CONFIG.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += accelerator.gather(loss).item() * batch_size
            total_reward += accelerator.gather(rewards.sum()).item()
            batch_count += 1

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'reward': f"{rewards.sum().item() / batch_size:.4f}"
            })

        avg_loss = total_loss / len(train_dataloader.dataset)
        avg_reward = total_reward / len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_reward": avg_reward,
            "learning_rate": scheduler.get_last_lr()[0],
        })

        val_reward, val_pauses = evaluate_model(policy, val_dataloader)
        wandb.log({
            "val_reward": val_reward,
            "val_pauses": val_pauses,
        })

def evaluate_model(policy, dataloader):
    policy.eval()
    total_reward = 0
    total_pauses = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size, seq_len = input_ids.shape

        pause_counts = torch.zeros(batch_size, device=input_ids.device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        with torch.no_grad():
            for t in range(seq_len - 1):
                actions, _, _ = select_action(policy, input_ids[:, :t+1], attention_mask[:, :t+1])

                correct_actions = input_ids[:, t+1]
                is_correct = (actions == correct_actions).float()
                is_pause = (actions == policy.pause_token_id).float()
                pause_counts += is_pause

                reward = CONFIG.base_reward * is_correct - CONFIG.penalty_factor * pause_counts
                reward = torch.where(pause_counts >= CONFIG.max_pauses, CONFIG.large_penalty, reward)
                reward = torch.where(done, torch.zeros_like(reward), reward)

                done = done | (pause_counts >= CONFIG.max_pauses)

                total_reward += accelerator.gather(reward.sum()).item()
                total_pauses += accelerator.gather(pause_counts.sum()).item()

                if done.all():
                    break

    avg_reward = total_reward / len(dataloader.dataset)
    avg_pauses = total_pauses / len(dataloader.dataset)
    print(f"Evaluation - Avg Reward: {avg_reward:.4f}, Avg Pauses: {avg_pauses:.2f}")
    return avg_reward, avg_pauses

# Initialize the model
policy = LLaMAPausePolicy(CONFIG.model_name)

# Initialize optimizer and scheduler
optimizer = optim.AdamW(policy.parameters(), lr=CONFIG.learning_rate)
total_steps = len(train_dataloader) * CONFIG.num_epochs // CONFIG.accumulation_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CONFIG.warmup_steps, num_training_steps=total_steps)

# Start training
train_model(policy, optimizer, scheduler, train_dataloader, val_dataloader, CONFIG.num_epochs)

# Save the fine-tuned model and tokenizer
unwrapped_model = accelerator.unwrap_model(policy)
unwrapped_model.llama.save_pretrained("llama_pause_policy")
tokenizer.save_pretrained("llama_pause_policy")

# Finish the wandb run
wandb.finish()