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



import json
import numpy as np
from collections import defaultdict


def check_acc_cov(influence, train_dataset, validation_dataset, metrics_path):
    has_subvariation = (
        "subvariation" in train_dataset.column_names
        and "subvariation" in validation_dataset.column_names
    )

    # 700/2 in backdoor case
    cov_cnt = int(len(train_dataset) / len(set(train_dataset["variation"])))
    n = len(influence)

    overall = {
        "variation": {"acc": 0, "cov": 0},
    }

    if has_subvariation:
        overall["subvariation"] = {"acc": 0, "cov": 0}

    per_variation = defaultdict(lambda: {"count": 0, "acc": 0, "cov": 0})
    per_subvariation = (
        defaultdict(lambda: {"count": 0, "acc": 0, "cov": 0})
        if has_subvariation
        else None
    )

    for i in range(n):
        array = -(influence.loc[i].to_numpy())

        indices = np.argpartition(array, -cov_cnt)[-cov_cnt:]
        topk_indices = indices[np.argsort(array[indices])[::-1]]

        val_variation = validation_dataset["variation"][i]
        per_variation[val_variation]["count"] += 1

        if has_subvariation:
            val_subvariation = validation_dataset["subvariation"][i]
            per_subvariation[val_subvariation]["count"] += 1

        top1_idx = int(topk_indices[0])

        if train_dataset["variation"][top1_idx] == val_variation:
            overall["variation"]["acc"] += 1
            per_variation[val_variation]["acc"] += 1

        if has_subvariation:
            if train_dataset["subvariation"][top1_idx] == val_subvariation:
                overall["subvariation"]["acc"] += 1
                per_subvariation[val_subvariation]["acc"] += 1

        var_cov_hits = 0
        subvar_cov_hits = 0

        for ele in topk_indices:
            ele = int(ele)

            if train_dataset["variation"][ele] == val_variation:
                overall["variation"]["cov"] += 1
                var_cov_hits += 1

            if has_subvariation:
                if train_dataset["subvariation"][ele] == val_subvariation:
                    overall["subvariation"]["cov"] += 1
                    subvar_cov_hits += 1

        per_variation[val_variation]["cov"] += var_cov_hits

        if has_subvariation:
            per_subvariation[val_subvariation]["cov"] += subvar_cov_hits

    metrics = {
        "overall": {
            "variation": {
                "accuracy": overall["variation"]["acc"] / n,
                "coverage": overall["variation"]["cov"] / (n * cov_cnt),
            }
        },
        "per_variation": {},
    }

    if has_subvariation:
        metrics["overall"]["subvariation"] = {
            "accuracy": overall["subvariation"]["acc"] / n,
            "coverage": overall["subvariation"]["cov"] / (n * cov_cnt),
        }
        metrics["per_subvariation"] = {}

    for var, vals in per_variation.items():
        count = vals["count"]
        metrics["per_variation"][str(var)] = {
            "num_samples": count,
            "accuracy": vals["acc"] / count if count > 0 else 0.0,
            "coverage": vals["cov"] / (count * cov_cnt) if count > 0 else 0.0,
        }

    if has_subvariation:
        for subvar, vals in per_subvariation.items():
            count = vals["count"]
            metrics["per_subvariation"][str(subvar)] = {
                "num_samples": count,
                "accuracy": vals["acc"] / count if count > 0 else 0.0,
                "coverage": vals["cov"] / (count * cov_cnt) if count > 0 else 0.0,
            }

    print("Variation Acc:", metrics["overall"]["variation"]["accuracy"])
    print("Variation Cover:", metrics["overall"]["variation"]["coverage"])

    if has_subvariation:
        print("Subvariation Acc:", metrics["overall"]["subvariation"]["accuracy"])
        print("Subvariation Cover:", metrics["overall"]["subvariation"]["coverage"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics



def plot_all_acc_cov_oneplot(results_dir="results", figsize_per_subplot=(8, 6)):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(results_dir, f))
    ]

    if not files:
        raise ValueError(f"No .json files found in {results_dir}")

    files = sorted(files)
    plot_data = []
    max_rows = 0

    for filename in files:
        filepath = os.path.join(results_dir, filename)

        with open(filepath, "r") as f:
            metrics = json.load(f)

        row_labels = []
        values = []

        overall = metrics.get("overall", {})

        # -------------------------
        # 1) overall variation first
        # -------------------------
        if "variation" in overall:
            row_labels.append("overall variation")
            values.append([
                overall["variation"].get("accuracy", 0.0),
                overall["variation"].get("coverage", 0.0),
            ])

        # -------------------------
        # 2) all variations
        # -------------------------
        per_variation = metrics.get("per_variation", {})
        for key in sorted(per_variation.keys(), key=str):
            val = per_variation[key]
            row_labels.append(f"variation: {key}")
            values.append([
                val.get("accuracy", 0.0),
                val.get("coverage", 0.0),
            ])

        # -------------------------
        # 3) overall subvariation
        # -------------------------
        if "subvariation" in overall:
            row_labels.append("overall subvariation")
            values.append([
                overall["subvariation"].get("accuracy", 0.0),
                overall["subvariation"].get("coverage", 0.0),
            ])

        # -------------------------
        # 4) all subvariations
        # -------------------------
        per_subvariation = metrics.get("per_subvariation", {})
        for key in sorted(per_subvariation.keys(), key=str):
            val = per_subvariation[key]
            row_labels.append(f"subvariation: {key}")
            values.append([
                val.get("accuracy", 0.0),
                val.get("coverage", 0.0),
            ])

        if values:
            values = np.array(values)
            plot_data.append((filename, row_labels, values))
            max_rows = max(max_rows, len(row_labels))

    if not plot_data:
        raise ValueError(f"No plottable metrics found in JSON files in {results_dir}")

    n = len(plot_data)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig_width = figsize_per_subplot[0] * cols
    fig_height = max(figsize_per_subplot[1] * rows, 0.35 * max_rows * rows + 1.2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    im = None

    for ax, (filename, row_labels, values) in zip(axes, plot_data):
        im = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Accuracy", "Coverage"])
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        title = filename.replace(".json", "").replace("_", " ")
        ax.set_title(title, fontsize=10)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j] * 100:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    for ax in axes[len(plot_data):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes[:len(plot_data)],
            orientation="horizontal",
            fraction=0.04,
            pad=0.08
        )
        cbar.set_label("Rate")
        ticks = np.linspace(0, 1, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t * 100:.0f}%" for t in ticks])

    plt.savefig(os.path.join(results_dir, "acc_cov_all_in_one.png"), bbox_inches="tight")
    plt.close(fig)




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


def random_influence_estimation(dataset, metrics_path):
    train_var = dataset["train"]["variation"]
    eval_var = dataset["test"]["variation"]
    N = len(train_var)

    has_subvariation = (
        "subvariation" in dataset["train"].column_names
        and "subvariation" in dataset["test"].column_names
    )

    # counts in train set
    var_counts = {}
    for v in train_var:
        var_counts[v] = var_counts.get(v, 0) + 1

    if has_subvariation:
        train_subvar = dataset["train"]["subvariation"]
        eval_subvar = dataset["test"]["subvariation"]

        subvar_counts = {}
        for s in train_subvar:
            subvar_counts[s] = subvar_counts.get(s, 0) + 1

    # overall expected metrics
    var_values = [var_counts[v] / N for v in eval_var]
    var_acc_rate = np.mean(var_values)
    var_cov_rate = np.mean(var_values)

    metrics = {
        "overall": {
            "variation": {
                "accuracy": float(var_acc_rate),
                "coverage": float(var_cov_rate),
            }
        },
        "per_variation": {},
    }

    # per-variation expected metrics
    eval_var_counts = {}
    for v in eval_var:
        eval_var_counts[v] = eval_var_counts.get(v, 0) + 1

    for v, count in eval_var_counts.items():
        p = var_counts.get(v, 0) / N
        metrics["per_variation"][str(v)] = {
            "num_samples": count,
            "accuracy": float(p),
            "coverage": float(p),
        }

    if has_subvariation:
        subvar_values = [subvar_counts[s] / N for s in eval_subvar]
        subvar_acc_rate = np.mean(subvar_values)
        subvar_cov_rate = np.mean(subvar_values)

        metrics["overall"]["subvariation"] = {
            "accuracy": float(subvar_acc_rate),
            "coverage": float(subvar_cov_rate),
        }

        metrics["per_subvariation"] = {}

        eval_subvar_counts = {}
        for s in eval_subvar:
            eval_subvar_counts[s] = eval_subvar_counts.get(s, 0) + 1

        for s, count in eval_subvar_counts.items():
            p = subvar_counts.get(s, 0) / N
            metrics["per_subvariation"][str(s)] = {
                "num_samples": count,
                "accuracy": float(p),
                "coverage": float(p),
            }

    print("Variation Acc:", metrics["overall"]["variation"]["accuracy"])
    print("Variation Cover:", metrics["overall"]["variation"]["coverage"])

    if has_subvariation:
        print("Subvariation Acc:", metrics["overall"]["subvariation"]["accuracy"])
        print("Subvariation Cover:", metrics["overall"]["subvariation"]["coverage"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
