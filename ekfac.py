# Author: Yaochen Zhu (uqp4qh@virginia.edu) and Xuansheng Wu (wuxsmail@163.com) 
# Last Modify: 2024-01-11
# Description: Computing the EK-FAC approximated influence function over a corpus.
import os
import re
import json
import tqdm
import nltk
import random
import argparse
import os
from datasets import load_from_disk
import pickle as pkl
from hooks_Llama import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from hooks_Llama import *
from ekfac_utils import *


seed = 12345
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



if __name__ == "__main__":

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


    root = os.getcwd()
    dataset_name = "backdoor_sample"
    result_folder = os.path.join("results/EKFAC", dataset_name)

    dataset = load_from_disk("datasets/" + dataset_name)

    train_corpus = list(dataset["train"])
    train_texts = [
    sample["prompts"].strip() + " " + sample["response"].strip()
    for sample in train_corpus
    ]

    test_corpus = list(dataset["test"])

    generator = Generator(model_name, device="cuda")


    hooker = MLPHookController.LLaMA(generator._model)


    inf_root = os.path.join(root, result_folder, model_name)

    try:
        inf_estimator = InfluenceEstimator.load_from_disk(inf_root)
    except Exception:
        ekfac_calculate_SVD_Lambda(corpus = train_texts, 
                                                   hooker = hooker,
                                                   generator = generator, 
                                                   save_path = inf_root,
                                                   batch_size_cov=2,
                                                   batch_size_lambda=1)
        
        inf_estimator = InfluenceEstimator.load_from_disk(inf_root)
    
    

    num_train = len(train_texts)
    num_test = len(test_corpus)
    influence_matrix = torch.zeros((num_test, num_train), dtype=torch.float32)
    
    
    # Loop over all the queries
    for i, sample in enumerate(test_corpus):

        query = sample["prompts"].strip()
        generated = generator.generate([query])[0]

        query, completion, _ = generator.prepare4explain([query], [generated])
        #cutoff at 1024
        completion = [completion[0][:1024 - len(query[0])]]
        
        # Forward propagation
        ids = torch.tensor([generator._tokenizer.convert_tokens_to_ids(query[0] + completion[0])])
        probs = torch.softmax(generator._model(input_ids=ids.to(generator._model.device)).logits, dim=-1)
        out_mask = torch.tensor([[0] * len(query[0]) + [1] * len(completion[0])]).to(generator._model.device)
        
        # Calculte the log probability
        query_loss = compute_LM_loss(ids, out_mask, probs)[0]
        query_loss.backward()
        
        # Backward propagation
        with torch.no_grad():
            ### Remove all the weightsn
            query_grads = hooker.collect_weight_grads()
            # use .copy, otherwise zero_grad will remove the grad
            query_grads = {layer:grad.clone() for layer, (_, grad) in query_grads.items()}
        zero_grad(generator._model)
        
        # Calculate the HVP for the query
        query_hvps = inf_estimator.calculate_hvp(query_grads)
        
        bar = tqdm.tqdm(total=num_train, desc=f"Influence for Test Sample {i}")
        
        for j in range(num_train):
            # Forward propagation
            train_text = train_texts[j]
            inputs, outputs = generator.forward([train_text])
            losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
            for loss in losses:
                loss.backward(retain_graph=True)
            
            # Backward propagation
            with torch.no_grad():
                grads = hooker.collect_weight_grads()
                inf = inf_estimator.calculate_total_influence(query_hvps, grads)
                influence_matrix[i, j] = inf.detach().cpu().item()
            zero_grad(generator._model)
            bar.update(1)

    save_root = os.path.join(root, result_folder, model_name)
    os.makedirs(save_root, exist_ok=True)

    print("Saving influence matrix...")
    print(f"Shape: {tuple(influence_matrix.shape)}")  # (num_test, num_train)
    torch.save(influence_matrix, os.path.join(save_root, "influence_matrix.pt"))
    print(f"Saved to: {os.path.join(save_root, 'influence_matrix.pt')}")

        