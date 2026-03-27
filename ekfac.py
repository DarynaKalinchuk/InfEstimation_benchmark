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

import pickle as pkl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from hooks_Llama import *
from ekfac_utils import *
from ekfac_utils2 import *
from ekfac_utils3 import *


seed = 12345
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str,
    #     help="specify the name of the model")
    # parser.add_argument("--check_point", type=str,
    #     help="specify the checkpoint for EKFAC")
    # args = parser.parse_args()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    check_point = "180"

    root = os.getcwd()

    batch_size = 1
    corpus = CorpusSearchIndex("datasets/corpus.txt")
    generator = Generator(model_name, device="cuda")



    hooker = MLPHookController.LLaMA(generator._model)

    estimator = CovarianceEstimator()
    bar_cov = tqdm.tqdm(total=len(corpus), desc="EstimatingCovariance")
    device = generator._model.device
    
    ### Estimating S and A
    for i, texts in enumerate(batchit(corpus, batch_size)):
        texts = [_ for _ in texts if len(_.strip()) > 0]
        if len(texts) == 0:
            continue
        zero_grad(generator._model)
        inputs, outputs = generator.forward(texts)
        losses = compute_pseudo_loss(inputs["attention_mask"],
                                   outputs.logits)
        for loss in losses:
            loss.backward(retain_graph=True)
        with torch.no_grad():
            estimator.update_cov(hooker.collect_states(),
                                 inputs["attention_mask"].to(generator._device))
        bar_cov.update(len(texts))

        if i>=500:
            break
    
    ### Calculating the SVD decomposition of S and A
    estimator.calculate_eigenvalues_and_vectors()

    ### Estimating Lambda
    batch_size = 1 
    bar_lambda = tqdm.tqdm(total=len(corpus), desc="EstimatingLambda")
    for i, texts in enumerate(batchit(corpus, batch_size)):
        texts = [_ for _ in texts if len(_.strip()) > 0]
        if len(texts) == 0:
            continue
        zero_grad(generator._model)
        inputs, outputs = generator.forward(texts)
        losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
        for loss in losses:
            loss.backward(retain_graph=True)
        with torch.no_grad():
            estimator.update_lambdas(hooker.collect_states(),
                                     inputs["attention_mask"].to(generator._device))
        bar_lambda.update(len(texts))
        
        if i>=500:
            break

    save_root = os.path.join(root, "EKFAC_RESULT", model_name)
    os.makedirs(save_root, exist_ok=True)
    estimator.save_to_disk(save_root)