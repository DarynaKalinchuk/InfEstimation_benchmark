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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str,
    #     help="specify the name of the model")
    # parser.add_argument("--check_point", type=str,
    #     help="specify the checkpoint for EKFAC")
    # args = parser.parse_args()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


    root = os.getcwd()
    result_folder = "EKFAC_RESULT"
    inf_root = os.path.join(root, result_folder, model_name)
    inf_estimator = InfluenceEstimator.load_from_disk(inf_root)
    
    dataset = load_from_disk("datasets/" + "backdoor")
    corpus = list(dataset["train"]["prompts"]) #?

    generator = Generator(model_name, device="cuda")


    hooker = MLPHookController.LLaMA(generator._model)

    num_samples = len(corpus)
    topks = [k for k in [1, 5, 10, 20, 50, 100] if k <= num_samples]
    num_negs = [k for k in [1, 2, 3] if k <= num_samples - 1]
    num_neg = min(max(num_negs), num_samples - 1)
    recalls = [[[] for _ in topks] for _ in num_negs]

    num_samples = len(corpus)
    
    # Loop over all the queries
    for i, sample in enumerate(corpus):
        sample = sentence_tokenize(sample)
        query, completion = u" ".join(sample[:3]).strip(), u" ".join(sample[3:]).strip()
        completion = generator.generate([query])[0]
        
        # Prepare the input to the LLM
        query, completion, full_text = generator.prepare4explain([query], [completion])
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
        
        influences = []

        sample_idxes = get_sample_indices(num_samples, num_neg, i)
        bar = tqdm.tqdm(total=len(sample_idxes), desc="Influence for Query=%d" % i)
        
        save_root = os.path.join(root, result_folder, model_name, "samples")
        os.makedirs(save_root, exist_ok=True)

        with open(os.path.join(save_root, f"{i}.pkl"), "wb") as f:
            pkl.dump(sample_idxes, f)

        for j in sample_idxes:
            # Forward propagation
            sample = corpus[j]
            inputs, outputs = generator.forward([sample])
            losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
            for loss in losses:
                loss.backward(retain_graph=True)
            
            # Backward propagation
            with torch.no_grad():
                grads = hooker.collect_weight_grads()
                inf = inf_estimator.calculate_total_influence(query_hvps, grads)
                influences.append((j, float(inf.cpu().numpy())))
            zero_grad(generator._model)
            bar.update(1)

        for l, cur_num_neg  in enumerate(num_negs):
            results_file = os.path.join(root, result_folder, model_name, f"results_{cur_num_neg }.txt")
            
            pred_rank = [_[0] for _ in sorted(influences[:cur_num_neg ], key=lambda x: x[1], reverse=True)]
            for topk, hit in zip(topks, recalls[l]):
                hit.append(1.0 if i in pred_rank[:topk] else 0.0)

            if i % 5 == 0:
                info = "Cases=%d | %s" % (i, u" | ".join("top-%d=%.4f" % (k, sum(hit) / len(hit)) for k, hit in zip(topks, recalls[l])))
                print(f"neg_{cur_num_neg }: " + info)

                ### This is a new run
                if i == 0:
                    with open(results_file, "w") as f:
                        f.write(f"{i}: " + info + "\n")

                    if cur_num_neg  == 100:
                        print(influences)
                        inf_path = os.path.join(root, result_folder, model_name, f"inf_{i}.pkl")
                        with open(inf_path, "wb") as f:
                            pkl.dump(influences, f)

                ### Otherwise we append
                else:
                    with open(results_file, "a") as f:
                        f.write(f"{i}: " + info + "\n")