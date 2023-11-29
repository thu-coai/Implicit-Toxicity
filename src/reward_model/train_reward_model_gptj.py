import os

import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments
import json


def create_comparison_dataset(path, split):
    with open(f"{path}/{split}.json", 'r') as f:
        dataset = json.load(f)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        chosen_summary = sample['win_response']
        rejected_summary = sample['lose_response']
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = f"Query: {sample['context']}\nResponse: {sample['win_response']}"
        pair["rejected"] = f"Query: {sample['context']}\nResponse: {sample['lose_response']}"
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        print(pairs[0])
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                chosen + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                rejected + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            # 判断一下 tokenize截断后也不能一样
            if torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                continue
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":
    tokenizer_path = 'TODO'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    print('tokenizer pad token = ', tokenizer.pad_token)
    
    output_dir = 'TODO'

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=1,
        half_precision_backend=True,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_gpt_j.json",
        load_best_model_at_end=True,
    )

    # Initialize the reward model from the policy model
    model = GPTRewardModel("TODO", tokenizer_path=tokenizer_path)

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.layers
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = 'TODO'
    train_pairs = create_comparison_dataset(data_path, "train")
    val_pairs = create_comparison_dataset(data_path, "test")

    # Make pairwise datasets for training
    max_length = 512
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_dir)