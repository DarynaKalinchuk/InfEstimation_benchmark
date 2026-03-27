from transformers import AutoTokenizer
from hooks_Llama import *
from ekfac_utils import *
import pickle
import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

hooker = MLPHookController.LLaMA(model)

### checking if it works #####

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

for name, cache in hooker._caches.items():
    print(name)
    print("inputs:", cache.inputs is not None)
    print("hiddens:", cache.hiddens is not None)
    print("activates:", cache.activates is not None)
    print("outputs:", cache.outputs is not None)
    print("weights:", cache.weights is not None)
    print("----")
    break

for name, cache in hooker._caches.items():
    print(name)
    print("inputs shape:", cache.inputs.shape)
    print("hiddens shape:", cache.hiddens.shape)
    print("activates shape:", cache.activates.shape)
    print("outputs shape:", cache.outputs.shape)
    print("weights shape:", cache.weights.shape)
    break


inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model(**inputs)
logits = outputs.logits

target_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
logits[:, -1, target_id].backward()

for name, cache in hooker._caches.items():
    print(name, cache.hiddens.grad is not None)
    break



estimator = CovarianceEstimator()


texts = ["The capital of France is Paris."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

outputs = model(**inputs)
logits = outputs.logits

target_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
loss = -logits[:, -1, target_id].mean()
loss.backward()

layer_states = hooker.collect_states()
mask = inputs["attention_mask"]

estimator.update_cov(layer_states, mask)

model.zero_grad()

estimator.calculate_eigenvalues_and_vectors()

print(estimator.layer_svds.keys())

outputs = model(**inputs)
logits = outputs.logits

# predict last token from previous position
target_id = inputs["input_ids"][:, -1]

loss = -logits[:, -2, :].gather(1, target_id.unsqueeze(1)).mean()
loss.backward()

layer_states = hooker.collect_states()
mask = inputs["attention_mask"]

# OOM
estimator.update_lambdas(layer_states, mask)

model.zero_grad()

estimator.save_to_disk("ekfac_stats")

influence_estimator = InfluenceEstimator.load_from_disk("ekfac_stats")


import torch
import torch.nn.functional as F

# -----------------------------------------
# helper: last-token logit loss
# -----------------------------------------
def last_token_logit_loss(logits, target_ids):
    # logits: [B, T, V]
    # target_ids: [B]
    return -logits[:, -1, :].gather(1, target_ids.unsqueeze(1)).mean()


# =========================================
# QUERY: compute H^{-1} g_query
# =========================================
query_texts = ["The capital of France is"]
query_inputs = tokenizer(
    query_texts,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(model.device)

outputs = model(**query_inputs)
query_labels = torch.tensor(
    [tokenizer.encode(" Paris", add_special_tokens=False)[0]],
    device=model.device
)

loss = last_token_logit_loss(outputs.logits, query_labels)
loss.backward()

def hook_name_to_layer_name(name):
    layer_idx = int(name.split(".")[2]) + 1
    return f"Layer{layer_idx}"

layer_grad_query = {}
for name, cache in hooker._caches.items():
    if cache.weights is not None and cache.weights.grad is not None:
        est_name = hook_name_to_layer_name(name)
        layer_grad_query[est_name] = cache.weights.grad.detach().clone()


print("query grad keys:", layer_grad_query.keys())
print("svd keys:", influence_estimator.layer_svds.keys())


layer_hvps_query = influence_estimator.calculate_hvp(layer_grad_query)
model.zero_grad()


# =========================================
# TRAIN: compute grad of one train example
# =========================================
train_texts = ["The capital of France is Paris."]
train_inputs = tokenizer(
    train_texts,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(model.device)

outputs = model(**train_inputs)

# same target style for consistency:
# influence of this train point on predicting " Paris"
train_labels = torch.tensor(
    [tokenizer.encode(" Paris", add_special_tokens=False)[0]],
    device=model.device
)

loss = last_token_logit_loss(outputs.logits, train_labels)
loss.backward()

layer_grad_train = {}
for name, cache in hooker._caches.items():
    if cache.weights is not None and cache.weights.grad is not None:
        est_name = hook_name_to_layer_name(name)
        grad = cache.weights.grad.detach().clone()
        layer_grad_train[est_name] = (None, grad)   # for your current estimator

total_infl = influence_estimator.calculate_total_influence(
    layer_hvps_query,
    layer_grad_train
)

print("Total influence:", total_infl)
model.zero_grad()

