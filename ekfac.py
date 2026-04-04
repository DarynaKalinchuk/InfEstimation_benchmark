"""
Script to compute token-wise, differential influence scores for an LLM
on the provided training and query datasets.
"""
import argparse
import logging
import os
import numpy as np
from typing import Dict

import torch
from datetime import timedelta
from transformers import default_data_collator
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

import kronfluence
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import ScoreArguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from datetime import timedelta

from kronfluence.arguments import ScoreArguments, FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments


from utils import *
from utilsekfac import construct_hf_model
from utilsekfac import LanguageModelingTask

BATCH_TYPE = Dict[str, torch.Tensor]



model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset = "backdoor"
epochs = 1
max_length = 128
output_dir="results/EKFAC"
use_half_precision = False
use_compile = False
query_gradient_rank = -1
save_id = True

save_path = f"finetuned_model/TinyLlama/{dataset}_{epochs}"


model = AutoModelForCausalLM.from_pretrained(
    save_path,
    dtype=torch.float32,
)
model.config.use_cache = False

        

config = {
    "model": {
        "family": "tinyllama",
        "num_layers": model.config.num_hidden_layers,
    }
}

task = LanguageModelingTask(config)
model = kronfluence.prepare_model(model, task)


kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
accelerator = Accelerator(kwargs_handlers=[kwargs])
model = accelerator.prepare_model(model)

if use_compile:
    model = torch.compile(model)

analyzer = Analyzer(
    analysis_name="if_results",
    model=model,
    task=task,
    profile=False,
    output_dir=output_dir
)


# Configure parameters for DataLoader.
dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
analyzer.set_dataloader_kwargs(dataloader_kwargs)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk("datasets/" + dataset)

#debug
dataset["train"] = dataset["train"].select(range(5))
dataset["test"] = dataset["test"].select(range(2))
#debugend

chat_template = f"[INST] {{prompt}} [/INST] {{response}}"

tokenized_tr = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=max_length)    
tokenized_val = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=max_length)


factor_strategy = "diagonal"

factor_args = FactorArguments(strategy=factor_strategy)
if use_half_precision:
    factor_args = all_low_precision_factor_arguments(
        strategy=factor_strategy,
        dtype=torch.float16
    )
    factor_strategy += "_half"
if use_compile:
    factor_strategy += "_compile"

factor_args.covariance_max_examples = 1
factor_args.lambda_max_examples = 1

analyzer.fit_all_factors(
    factors_name=factor_strategy,
    dataset=tokenized_tr,
    per_device_batch_size=1,
    factor_args=factor_args,
    overwrite_output_dir=True,
)



##### Compute pairwise scores. #####
score_args = ScoreArguments()
scores_name = "default_scores"


if use_half_precision:
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    scores_name += "_half"

if use_compile:
    scores_name += "_compile"

rank = query_gradient_rank if query_gradient_rank != -1 else None
if rank is not None:
    score_args.query_gradient_low_rank = rank
    score_args.query_gradient_accumulation_steps = 10
    scores_name += f"_qlr{rank}"

if save_id:
    scores_name += f"_{save_id}"


score_args.compute_per_token_scores = False
score_args.aggregate_query_gradients = False

analyzer.compute_pairwise_scores(
    scores_name=scores_name,
    score_args=score_args,
    factors_name=factor_strategy,   
    query_dataset=tokenized_val,
    train_dataset=tokenized_tr,
    per_device_query_batch_size=1,
    per_device_train_batch_size=1,
    overwrite_output_dir=True,
)



scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
print(f"Scores shape: {scores.shape}")
print(f"Saved to: {scores_name}")


