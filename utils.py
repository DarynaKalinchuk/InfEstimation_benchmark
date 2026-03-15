import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib.pyplot as plt
import os
import json

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

def gradient_influence_methods(tr_grad_dict, val_grad_dict, hvp_cal='gradient_match', lambda_const_param = 10, n_iteration = 10, alpha_const = 1.):
    
    lambda_const_param = int(lambda_const_param)
    n_iteration = int(n_iteration)
    alpha_const = float(alpha_const)

    hvp_dict = defaultdict(dict)
    IF_dict = defaultdict(dict)
    n_train = len(tr_grad_dict.keys())

    def calculate_lambda_const(tr_grad_dict, weight_name):
        S = torch.zeros(len(tr_grad_dict.keys()))
        for tr_id in tr_grad_dict:
            tmp_grad = tr_grad_dict[tr_id][weight_name]
            S[tr_id] = torch.mean(tmp_grad**2)

        return torch.mean(S) / lambda_const_param

    if hvp_cal == 'Original':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)

                AAt_matrix = torch.zeros(torch.outer(tr_grad_dict[0][weight_name].reshape(-1), 
                                                     tr_grad_dict[0][weight_name].reshape(-1)).shape)
                for tr_id in tr_grad_dict:
                    tmp_mat = torch.outer(tr_grad_dict[tr_id][weight_name].reshape(-1), 
                                          tr_grad_dict[tr_id][weight_name].reshape(-1))
                    AAt_matrix += tmp_mat

                L, V = torch.linalg.eig(AAt_matrix)
                L, V = L.float(), V.float()
                hvp = val_grad_dict[val_id][weight_name].reshape(-1) @ V
                hvp = (hvp / (lambda_const + L / n_train)) @ V.T
                hvp_dict[val_id][weight_name] = hvp.reshape(len(tr_grad_dict[0][weight_name]), -1)
                del tmp_mat, AAt_matrix, V

    elif hvp_cal == 'DataInf':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)

                hvp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
                for tr_id in tr_grad_dict:
                    tmp_grad = tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (val_grad_dict[val_id][weight_name] - C_tmp * tmp_grad) / (n_train * lambda_const)
                
                hvp_dict[val_id][weight_name] = hvp

    elif hvp_cal == 'LiSSA':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)

                running_hvp = val_grad_dict[val_id][weight_name]
                for _ in range(n_iteration):
                    hvp_tmp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
                    for tr_id in tr_grad_dict:
                        tmp_grad = tr_grad_dict[tr_id][weight_name]
                        hvp_tmp += (torch.sum(tmp_grad * running_hvp) * tmp_grad - lambda_const * running_hvp) / n_train / 1e3
                    
                    running_hvp = val_grad_dict[val_id][weight_name] + running_hvp - alpha_const * hvp_tmp

                hvp_dict[val_id][weight_name] = running_hvp

    elif hvp_cal == 'gradient_match':
        hvp_dict = val_grad_dict.copy()
    else:
        raise Exception("hvp calculation options: [Original, DataInf, LiSSA, gradient_match]")

    for tr_id in tr_grad_dict:
        for val_id in val_grad_dict:
            if_tmp_value = 0
            for weight_name in val_grad_dict[0]:
                if_tmp_value += torch.sum(hvp_dict[val_id][weight_name] * tr_grad_dict[tr_id][weight_name])

            IF_dict[tr_id][val_id] = -if_tmp_value

    print("End of influence estimation.")
    return pd.DataFrame(IF_dict, dtype=float)

def check_acc_cov(influence, train_dataset, validation_dataset, dataset_name='', model='', influence_est=''):
    acc = 0
    cov = 0
    cov_cnt = int(len(train_dataset) / len(set(train_dataset['variation'])))
    
    val_to_train_scores = {}

    for i in range(len(influence)):
        array = -(influence.loc[i].to_numpy())
        indices = np.argpartition(array, -cov_cnt)[-cov_cnt:]
        topk_indices = indices[np.argsort(array[indices])[::-1]]

        if train_dataset['variation'][int(topk_indices[0])] == validation_dataset['variation'][i]:
            acc += 1

        for ele in topk_indices:
            if train_dataset['variation'][int(ele)] == validation_dataset['variation'][i]:
                cov += 1

        #dictionary for this validation sample
        sample_scores = {int(ele): float(array[int(ele)]) for ele in topk_indices}
        sorted_sample_scores = dict(sorted(sample_scores.items(), key=lambda x: x[1], reverse=True))
        val_to_train_scores[int(i)] = sorted_sample_scores

    acc_rate = acc / len(influence)
    cov_rate = cov / (len(influence) * cov_cnt)
    print("Acc:", acc_rate, '\nCover:', cov_rate)

    # Plotting
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.bar(['Accuracy', 'Coverage'], [acc_rate, cov_rate], color=["#DFB5ED", "#AF9CEE"])
    plt.ylim(0, 1)
    plt.ylabel('Rate')
    plt.title(f'{dataset_name} | {model} | {influence_est}')
    
    # Saving
    filename = f"{dataset_name}_{model}_{influence_est}_acc_cov.png".replace(" ", "_")
    plt.savefig(plot_dir + '/' + filename, bbox_inches='tight')
    plt.show()

    #Dictionary saving
    dict_filename = f"{dataset_name}_{model}_{influence_est}_val_train_scores.json".replace(" ", "_")
    with open(plot_dir + '/' + dict_filename, 'w') as f:
        json.dump(val_to_train_scores, f, indent=2)



def random_method(tr_grad_dict, val_grad_dict, distribution = "normal"):
    n_train = len(tr_grad_dict.keys())
    n_val = len(val_grad_dict.keys())

    if distribution == "normal":
        random_matrix = torch.randn(n_val, n_train)  # standard normal
    elif distribution == "uniform":
        random_matrix = torch.rand(n_val, n_train)  # uniform in [0,1)
    else:
        raise ValueError("distribution must be 'normal' or 'uniform'")

    IF_df = pd.DataFrame(
        random_matrix.numpy(),
        index=list(val_grad_dict.keys()),
        columns=list(tr_grad_dict.keys()),
        dtype=float
    )

    return IF_df



def influence_estimation(tr_grad_dict, val_grad_dict, hvp_cal='gradient_match', needed_args=None):
    if needed_args is None:
        needed_args = {}

    print(f"Calculating influence with {hvp_cal}.")
    print(f"All params: {needed_args}")


    if hvp_cal == "random":
        
        influence_df = random_method(tr_grad_dict, val_grad_dict, **needed_args)
    else:
        # gradient influence function
        influence_df = gradient_influence_methods(
            tr_grad_dict,
            val_grad_dict,
            hvp_cal=hvp_cal,
            **needed_args
        )

    return influence_df
