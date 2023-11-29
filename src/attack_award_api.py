import json
import argparse
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pytorch_lightning as pl
import pathlib
from trlx.data.configs import TRLConfig
import re

from flask import Flask, request, jsonify, make_response


app = Flask(__name__)



REWARD_CHECKPOINT_PATH = "TODO"

class Bert(pl.LightningModule):
    def __init__(self, load_dir=None, lr=None, weight_decay=None, warm_up=None, num_labels=None, model_config=None):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up = warm_up
        self.max_step = None
        self.dev_dataset_names = None

        self.save_hyperparameters()
        
        if load_dir is None: # load pretrain parameter and config
            self.model = AutoModelForSequenceClassification.from_pretrained(model_config, num_labels=num_labels)
        else: # only load config
            config = AutoConfig.from_pretrained(model_config, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_config(config)

        self.model.resize_token_embeddings(50266)

        print('num_labels = ', self.model.num_labels)

    def forward(self, **kargs):
        return self.model(**kargs)


model_config = 'TODO'
load_dir = 'TODO'

rw_tokenizer = AutoTokenizer.from_pretrained(model_config)
rw_tokenizer.pad_token = rw_tokenizer.pad_token

rw_model = Bert.load_from_checkpoint(load_dir, model_config=model_config, load_dir=load_dir, num_labels=2)
# rw_model.half()
rw_model.eval()
rw_device = torch.device("cuda:{}".format(0))  # set reward model device
rw_model.to(rw_device)
print('reward model loaded!')

def get_scores(samples: List[str]):
    scores_list = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            context_list = []
            response_list = []
            for i in sub_samples:
                tmp = re.split("Response:", i)
                context = tmp[0][len("Query:"):].strip()
                response = tmp[1].strip()
                context_list.append(context)
                response_list.append(response)
            encodings_dict = rw_tokenizer(
                context_list,
                response_list,
                truncation="longest_first",
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            with torch.no_grad():
                outputs = rw_model(input_ids=input_ids, attention_mask=attn_masks)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().detach()
                probs = probs[:, 1] # Unsafety probs
            scores_list.append(probs)
    scores = torch.cat(scores_list, dim=0)
    return scores

 
@app.route('/attack_reward', methods=['POST'])
def get_reward():
    data = json.loads(request.data)
    samples = data
    scores = get_scores(samples)
    # print('samples = ', samples)
    # print('scores = ', scores)
    scores = scores.detach().cpu().tolist()
    return make_response(jsonify(scores))
 


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8119)