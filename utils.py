import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
import json
import math

def get_preprocessed_dataset(tokenizer, dataset, chat_template, max_length):
    def apply_prompt_template(sample):
        return {
            'text': chat_template.format(prompt=sample['prompts'], response=sample['response'])
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenized_dataset(text):
        input_text = text['text']
        tokenized_output = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
        tokenized_output['labels'] = tokenized_output['input_ids'].copy()
        return tokenized_output

    return dataset.map(tokenized_dataset, batched=True, remove_columns=['text'])

def collect_gradient(model_name, lora_adapter_path, tokenizer, tokenized_tr, tokenized_val):
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
    quantization_config = None
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
    
    collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
    train_dataloader_stochastic = DataLoader(tokenized_tr, 
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              batch_size=1)
    val_dataloader_stochastic = DataLoader(tokenized_val, 
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              batch_size=1)

    model.eval()
    tr_grad_dict = {}
    for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
        model.zero_grad()
        batch['labels'] = batch['input_ids']
        batch.to('cuda')
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
            
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        tr_grad_dict[step] = grad_dict
        del grad_dict
            
    val_grad_dict = {}
    for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
        model.zero_grad()
        batch['labels'] = batch['input_ids']
        batch.to('cuda')
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
            
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        val_grad_dict[step] = grad_dict    
        del grad_dict
            
    return tr_grad_dict, val_grad_dict

