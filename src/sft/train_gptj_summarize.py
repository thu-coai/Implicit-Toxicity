import random
import argparse

import evaluate
import numpy as np
import torch
from summarize_dataset import SafetyDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def get_args():
    parser = argparse.ArgumentParser()
    
    # for distributed launcher
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--output_dir", type=str,)
    parser.add_argument("--data_path", type=str)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    
    parser.add_argument("--eval_step", type=int, default=100)

    
    return parser.parse_args()


def sft_collator(instances):
    input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    
    
if __name__ == "__main__":
    
    args = get_args()
    
    pretrained_model = args.model_config
    output_dir = args.output_dir
    train_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation
    learning_rate = args.lr
    eval_batch_size = 16
    eval_steps = args.eval_step
    max_input_length = 512
    save_steps = args.eval_step
    num_train_epochs = 20
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("TODO")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    print('pad token = ', tokenizer.pad_token)

    # Set up the datasets
    data_path = args.data_path
    train_dataset = SafetyDataset(
        data_path,
        tokenizer,
        "train",
        max_length=max_input_length,
    )
    dev_dataset = SafetyDataset(
        data_path,
        tokenizer,
        "test",
        max_length=max_input_length,
    )
    
    print(train_dataset[0])
    # print(dev_dataset[0])
    
    model = AutoModelForCausalLM.from_pretrained(pretrained_model, use_cache=False)
    model.resize_token_embeddings(len(tokenizer)) # FIXME: 可以不需要
    model.config.end_token_id = tokenizer.eos_token_id
    
    assert model.config.pad_token_id is not None

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=50,
        deepspeed="./ds_config_gptj.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # compute_metrics=compute_metrics,
        data_collator=sft_collator,
        # data_collator=default_data_collator,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
