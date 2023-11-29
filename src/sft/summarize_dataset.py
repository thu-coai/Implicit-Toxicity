import json
import copy

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

        
class SafetyDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        self.post_list = []
        with open(f"{train_path}/{split}.json", 'r') as f:
            dataset = json.load(f)
        for sample in dataset:
            self.post_list.append(f"Query: {sample['context']}\nResponse: {sample['implicit_toxic'] if 'implicit_toxic' in sample else sample['response']}{tokenizer.eos_token}")
        print(self.post_list[0])
        # if "valid" in train_path:
        #     self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = encodings_dict['input_ids']
        labels = copy.deepcopy(input_ids)
        input_ids = torch.tensor(input_ids)
        attn_masks = torch.tensor(encodings_dict["attention_mask"])
        labels = torch.tensor(labels)
        labels[labels==self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels,
        }