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


def gradient_influence_methods(tr_grad_dict, val_grad_dict, hvp_cal='GradDot', lambda_const_param = "10", n_iteration = "10", alpha_const = "1."):
    
    lambda_const_param = float(lambda_const_param)
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
                        hvp_tmp += (torch.sum(tmp_grad * running_hvp) * tmp_grad + lambda_const * running_hvp) / n_train / 1e3
                    
                    running_hvp = val_grad_dict[val_id][weight_name] + running_hvp - alpha_const * hvp_tmp

                hvp_dict[val_id][weight_name] = running_hvp

    
    elif hvp_cal == 'GradDot' or hvp_cal == "GradCos":
        hvp_dict = val_grad_dict.copy()
    else:
        raise Exception("Invalid hvp calculation option.")

    for tr_id in tr_grad_dict:
        for val_id in val_grad_dict:
            if_tmp_value = 0
            norm_tr = 0
            norm_val_hvp = 0
            for weight_name in val_grad_dict[0]:

                g_val_hvp = hvp_dict[val_id][weight_name]
                g_tr = tr_grad_dict[tr_id][weight_name]
                if_tmp_value += torch.sum(g_val_hvp * g_tr)

                # for normalization
                norm_val_hvp += torch.sum(g_val_hvp * g_val_hvp)
                norm_tr += torch.sum(g_tr * g_tr)

            if hvp_cal == "GradCos":
                cos_sim = if_tmp_value / (torch.sqrt(norm_tr) * torch.sqrt(norm_val_hvp) + 1e-12)
                IF_dict[tr_id][val_id] = -cos_sim
            else:
                IF_dict[tr_id][val_id] = -if_tmp_value

    print("End of influence estimation.")
    return pd.DataFrame(IF_dict, dtype=float)



def check_acc_cov(influence, train_dataset, validation_dataset, metrics_path):
    acc = 0
    cov = 0

    #700/2 in backdoor case
    cov_cnt = int(len(train_dataset) / len(set(train_dataset['variation'])))
    

    for i in range(len(influence)):
        #i-th row, so influence for i-th test sample across train, flip of sign
        # => the more positive the more influential now
        array = -(influence.loc[i].to_numpy())

        # take #cov_cnt largest values
        indices = np.argpartition(array, -cov_cnt)[-cov_cnt:]

        # Indices of the top cov_cnt training samples,
        # sorted by (-influence) from largest to smallest => first ones are more influential
        topk_indices = indices[np.argsort(array[indices])[::-1]]

        # does the sample ranked as most influential have the same variation label as the test sample?
        if train_dataset['variation'][int(topk_indices[0])] == validation_dataset['variation'][i]:
            acc += 1

        # how many of the #cov_cnt samples with largest influence values have the same variation 
        # as this test sample? 
        for ele in topk_indices:
            if train_dataset['variation'][int(ele)] == validation_dataset['variation'][i]:
                cov += 1


    acc_rate = acc / len(influence)
    cov_rate = cov / (len(influence) * cov_cnt)
    print("Acc:", acc_rate, '\nCover:', cov_rate)

    metrics = {
        "accuracy": acc_rate,
        "coverage": cov_rate
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)



def plot_all_acc_cov(results_dir="results", figsize_per_plot=(4, 4)):
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    files = [
        f for f in os.listdir(results_dir)
        if f.endswith("_acc_cov.json") and os.path.isfile(os.path.join(results_dir, f))
    ]

    if not files:
        raise ValueError(f"No *_acc_cov.json files found in {results_dir}")

    files = sorted(files)
    n = len(files)

    cols = math.ceil(math.sqrt(n)) 
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
    )

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax, filename in zip(axes, files):
        filepath = os.path.join(results_dir, filename)

        with open(filepath, "r") as f:
            metrics = json.load(f)

        acc = metrics.get("accuracy", 0.0)
        cov = metrics.get("coverage", 0.0)

        bars = ax.bar(["Accuracy", "Coverage"], [acc, cov])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.bar_label(bars, fmt="%.4f", padding=3)
        title = filename.replace("_acc_cov.json", "").replace("_", " ")
        ax.set_title(title, fontsize=10)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()


    plt.savefig(results_dir + "/acc_cov_plots.png", bbox_inches="tight")




def gradient_influence_estimation(tr_grad_dict, val_grad_dict, hvp_cal='GradDot', needed_args=None):
    if needed_args is None:
        needed_args = {}

    print(f"Calculating influence with {hvp_cal}.")
    print(f"All params: {needed_args}")

    
    # gradient influence function
    influence_df = gradient_influence_methods(
        tr_grad_dict,
        val_grad_dict,
        hvp_cal=hvp_cal,
        **needed_args
    )

    return influence_df



def similarity_influence_estimation(test_vec, train_vecs, hvp_cal = "rep_cos_sim"):
    rep_cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    rep_dot_sim = lambda a, b: np.dot(a, b)
    rep_euc_sim = lambda a, b: -np.linalg.norm(a - b)**2

    sim_fn = locals().get(hvp_cal)
    if sim_fn is None:
        raise ValueError(f"Unknown similarity type: {hvp_cal}")
    
    sim = []
    for i in range(len(train_vecs)):
        sim.append(sim_fn(test_vec, train_vecs[i]))
    return np.array(sim)

