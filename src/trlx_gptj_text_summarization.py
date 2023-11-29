import os
import pathlib
import json
import requests
from typing import List
import numpy as np
import random

import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer
import warnings

import trlx
from trlx.data.configs import TRLConfig


def calc_distinct_k(texts, k):
    d = {}
    tot = 0
    for sen in texts:
        words = sen.split()
        for i in range(0, len(words)-k):
            key = tuple(words[i:i+k])
            d[key] = 1
            tot += 1
    if tot > 0:
        dist = len(d) / tot
    else:
        warnings.warn('the distinct is invalid')
        dist = 0.
    return dist

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

if __name__ == "__main__":
    set_seed(42)
    
    def get_scores(samples):
        # call api
        url = 'http://localhost:8115/reward'
        resp = requests.post(url, data=json.dumps(samples))
        scores = resp.json()
        
        url = 'http://localhost:8119/attack_reward'
        resp = requests.post(url, data=json.dumps(samples))
        attack_scores = resp.json()
        
        alpha = 1
        beta = 5
        
        assert len(scores) == len(attack_scores)
        
        scores = [i * alpha - j * beta for i, j in zip(scores, attack_scores)]
        scores = torch.tensor(scores, dtype=torch.float)
        return scores

    def reward_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        if '<sep>' in samples[0]:
            samples = [f'Query: {i.split("<sep>")[0].strip()}\nResponse: {i.split("<sep>")[1].strip()}' for i in samples] 
        scores = get_scores(samples)
        return scores
    
    def metric_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        
        url = 'http://localhost:8119/attack_reward'
        resp = requests.post(url, data=json.dumps(samples))
        attack_scores = resp.json()
        
        response_list = []
        for i in samples:
            tmp = i.split('\n')
            response = tmp[1][len("Response:"):].strip()
            response_list.append(response)
        
        metric = {"attack score": attack_scores}
  
        for k in range(1, 5):
            dist = calc_distinct_k(response_list, k)
            metric[f'dist-{k}'] = dist
        return metric
        

    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_summ_llama.yml")
    config = TRLConfig.load_yaml(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    print('pad token = ', tokenizer.pad_token)
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    data_path = 'TODO'
   
    with open(f"{data_path}/train.json", 'r') as f:
        train_prompts = []
        for sample in json.load(f):
            item = f"Query: {sample['context']}\nResponse:"
            if item not in train_prompts:
                train_prompts.append(item)

    with open(f"{data_path}/test.json", 'r') as f:
        val_prompts = []
        for sample in json.load(f):
            item = f"Query: {sample['context']}\nResponse:"
            if item not in val_prompts:
                val_prompts.append(item)
                
    print('train size = ', len(train_prompts))
    print('val size = ', len(val_prompts))

    trainer = trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )
