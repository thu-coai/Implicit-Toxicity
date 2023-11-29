import json
import argparse
from typing import List
import torch
from reward_model.reward_model import GPTRewardModel
from transformers import AutoTokenizer
import pathlib
from trlx.data.configs import TRLConfig
from tqdm import tqdm

from flask import Flask, request, jsonify, make_response


app = Flask(__name__)


SFT_MODEL_PATH='TODO'
REWARD_CHECKPOINT_PATH="TODO"


tokenizer_path = 'TODO'


config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_summ_llama.yml")
config = TRLConfig.load_yaml(config_path)

rw_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if rw_tokenizer.pad_token is None:
    rw_tokenizer.pad_token = rw_tokenizer.unk_token
    print('set pad token to unk token: ', rw_tokenizer.pad_token)
rw_model = GPTRewardModel(SFT_MODEL_PATH, tokenizer_path)
rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
rw_model.half()
rw_model.eval()
rw_device = torch.device("cuda:{}".format(0))
rw_model.to(rw_device)
print('reward model loaded!')

def get_scores(samples: List[str]):
    scores_list = []
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size)):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [chosen + rw_tokenizer.eos_token for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(scores_list, dim=0)
    return scores

 
@app.route('/reward', methods=['POST'])
def get_reward():
    data = json.loads(request.data)
    samples = data
    scores = get_scores(samples)
    scores = scores.detach().cpu().tolist()
    return make_response(jsonify(scores))
 


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8115)