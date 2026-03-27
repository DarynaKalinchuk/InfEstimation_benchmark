# Authors: Yaochen Zhu and Xuansheng Wu
# Emails: uqp4qh@virginia.edu, wuxsmail@163.com
# Implement the EK-FAC approximated influence calculation for LLMs

import os
import pickle
import random
import re
import torch
import functools
import string
import tqdm
import numpy as np
import transformers as trf


class CovarianceEstimator():
    '''
        This class estimates the uncentered covariances of input
        and pre-activation grad with running average, calculates
        the SVD decomposition of the estimated cov matrices, and 
        save them to the disk via pickle.
    '''
    def __init__(self):
        # The covariance matrices for each layer
        self.layer_covs = {}  
        
        # The number of samples for each layer
        self.num_samples_A = {}
        self.num_samples_S = {}
        self.num_samples_lambda = {}
        
        # The eigenvalues and eigenvectors for each layer
        self.layer_svds = {}
        
        # The estimated eigenvectors
        self.layer_lambdas = {}

    def update_cov(self, layer_states, mask):
        for layer_name, (a_prev, ds_cur) in layer_states.items():
            # Initialize the cov estimations
            if layer_name not in self.layer_covs:
                a_hidden_size = a_prev.size(-1)
                ds_hidden_size = ds_cur.size(-1)
                self.layer_covs[layer_name] = {
                    'A': torch.zeros((a_hidden_size, a_hidden_size)),
                    'S': torch.zeros((ds_hidden_size, ds_hidden_size))
                }
                self.num_samples_A[layer_name] = 0
                self.num_samples_S[layer_name] = 0

            # Reshape a_prev and ds_cur to (num_samples, hidden_size)
            batch_size = a_prev.size(0)
            num_steps = a_prev.size(1)
            total_samples = int(mask.sum())

            ### Update the uncentered covariance for A ###                        
            # Apply the mask to a_prev and ds_cur
            mask = mask.reshape(-1, 1)
            a_prev_reshaped = a_prev.reshape(-1, a_prev.size(-1)) # (bs * ts, dim)
            masked_a_prev = a_prev_reshaped * mask.to(a_prev_reshaped.device)

            # Calculate the uncentered covariance matrices for A and S
            batch_cov_A = torch.matmul(masked_a_prev.transpose(0, 1), masked_a_prev) 
            batch_cov_A /= total_samples
            
            self.num_samples_A[layer_name] += total_samples
                        
            # Update the running covariance matrices for A and S
            if self.num_samples_A[layer_name] == total_samples:
                self.layer_covs[layer_name]['A'] = batch_cov_A
            else:
                old_weight = (self.num_samples_A[layer_name] - total_samples) / self.num_samples_A[layer_name]
                new_weight = total_samples / self.num_samples_A[layer_name]
                self.layer_covs[layer_name]['A'] = old_weight * self.layer_covs[layer_name]['A'] + new_weight * batch_cov_A         
            
            ### Update the uncentered covariance for S ###
            ds_cur_reshaped = ds_cur.view(-1, ds_cur.size(-1))
            
            if not torch.isnan(ds_cur_reshaped).any():
                masked_ds_cur = ds_cur_reshaped * mask.to(ds_cur_reshaped.device)

                batch_cov_S = torch.matmul(masked_ds_cur.transpose(0, 1), masked_ds_cur)
                batch_cov_S /= total_samples
                
                self.num_samples_S[layer_name] += total_samples

                # Update the running covariance matrices for A and S
                if self.num_samples_S[layer_name] == total_samples:
                    self.layer_covs[layer_name]['S'] = batch_cov_S
                else:
                    old_weight = (self.num_samples_S[layer_name] - total_samples) / self.num_samples_S[layer_name]
                    new_weight = total_samples / self.num_samples_S[layer_name]
                    self.layer_covs[layer_name]['S'] = old_weight * self.layer_covs[layer_name]['S'] + new_weight * batch_cov_S
            else:
                print(f"ignore layer: {layer_name} for grads")
 
                
    def update_lambdas(self, layer_states, mask):
        # Assuming a_prev, ds_cur, and mask are PyTorch tensors with the specified shapes
        # a_prev: (batch_size, num_steps, in_size)
        # ds_cur: (batch_size, num_steps, out_size)
        # mask: (batch_size, num_steps)
        for layer_name, (a_prev, ds_cur) in layer_states.items():
            # Initialize the lambda estimations
            if layer_name not in self.layer_lambdas:
                a_hidden_size = a_prev.size(-1)
                ds_hidden_size = ds_cur.size(-1)
                self.layer_lambdas[layer_name] = torch.zeros((ds_cur.size(-1), a_prev.size(-1)))
                self.num_samples_lambda[layer_name] = 0
            
            # Obtain the kronecker product between Q_S and Q_A
            # The result has shape (in_size * out_size, in_size * out_size)
            Q_A = self.layer_svds[layer_name]["Q_A"]
            Q_S = self.layer_svds[layer_name]["Q_S"]
            
            # Obtain info regarding the data
            batch_size = a_prev.size(0)
            timesteps = a_prev.size(1)
            
            # Apply the mask
            a_prev_masked = a_prev * mask.unsqueeze(-1).to(a_prev.device)
            ds_cur_masked = ds_cur * mask.unsqueeze(-1).to(ds_cur.device)

            # Perform batched matrix multiplication to get the outer product
            # Reshape ds_cur_masked to (batch_size, num_steps, out_size, 1)
            # Reshape a_prev_masked to (batch_size, num_steps, 1, in_size)
            # batch_dtheta_steps: (batch_size, num_steps, out_size, in_size)
            # batch_dtheta = (ds_cur_masked.unsqueeze(-1) @ a_prev_masked.unsqueeze(2)).sum(axis=1)
            
            batch_dtheta = torch.zeros(batch_size, ds_cur_masked.shape[-1], 
                                       a_prev_masked.shape[-1], device=ds_cur.device)
            
            for bs in range(batch_size):
                for ts in range(timesteps):
                    batch_dtheta[bs] += ds_cur_masked[bs, ts].unsqueeze(1) @ a_prev_masked[bs, ts].unsqueeze(0)
            
            # Calculate the estimation (inefficient)
            # kron_basis = torch.kron(Q_A, Q_S) (memory OOD)
            # batch_dtheta = torch.sum(batch_dtheta_steps, dim=1).reshape(batch_size, -1)
            # batch_lambda = (torch.square(batch_dtheta @ kron_basis.T)).mean(axis=0)
           
            # https://math.stackexchange.com/questions/1879933/vector-multiplication-with-multiple-kronecker-products
            batch_lambda = torch.square(Q_S @ batch_dtheta @ Q_A.T).mean(axis=0)
            
            # Update the count
            self.num_samples_lambda[layer_name] += batch_size

            # Update the running covariance matrices for A and S
            if self.num_samples_lambda[layer_name] == batch_size:
                self.layer_lambdas[layer_name] = batch_lambda
            else:
                old_weight = (self.num_samples_lambda[layer_name] - batch_size) / self.num_samples_lambda[layer_name]
                new_weight = batch_size / self.num_samples_lambda[layer_name]
                self.layer_lambdas[layer_name] = old_weight * self.layer_lambdas[layer_name] + new_weight * batch_lambda    
       
    def get_running_covariance(self, layer_name):
        return self.layer_covs.get(layer_name, {'A': None, 'S': None})

    def get_running_lambda(self, layer_name):
        return self.layer_lambdas.get(layer_name, 0)

    def get_num_samples_A(self, layer_name):
        return self.num_samples_A.get(layer_name, 0)

    def get_num_samples_S(self, layer_name):
        return self.num_samples_S.get(layer_name, 0)

    def get_num_samples_lambda(self, layer_name):
        return self.num_samples_lambda.get(layer_name, 0)

    def calculate_eigenvalues_and_vectors(self):
        for layer_name, cov_matrices in self.layer_covs.items():
            eigenvalues_S, eigenvectors_S = torch.linalg.eigh(cov_matrices['S'], UPLO='U')
            eigenvalues_A, eigenvectors_A = torch.linalg.eigh(cov_matrices['A'], UPLO='U')

            self.layer_svds[layer_name] = {
                'Q_S': eigenvectors_S,
                'Q_A': eigenvectors_A,
            }

    def get_eigenvalues_and_vectors(self, layer_name):
        return self.layer_svds.get(layer_name, None)
    
    def save_to_disk(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        svds_file = os.path.join(dir, "layer_svds.pkl")
        with open(svds_file, "wb") as f:
            pickle.dump(self.layer_svds, f)
        lambdas_file = os.path.join(dir, "layer_lambdas.pkl")
        with open(lambdas_file, "wb") as f:
            pickle.dump(self.layer_lambdas, f)
            

class InfluenceEstimator():
    '''
        Give the layer_svds calculated by the EKFACCovarianceEstimator
        given the layerwise gradient of the query of interest, EKFACInfluenceEstimator
        first calculate the HVP between the approximate Hessian and grad.
        Given the layerwise gradient of a training sample, we also 
        calculate the layerwise influence as well as the total influences.
    '''
    def __init__(self, layer_svds, layer_lambdas, lambda_value=None):
        self.layer_svds = layer_svds
        self.layer_lambdas = layer_lambdas
        self.lambda_value = lambda_value

    @classmethod
    def load_from_disk(cls, dir, lambda_value=None):
        svd_path = os.path.join(dir, "layer_svds.pkl")
        assert os.path.exists(svd_path)
        with open(svd_path, "rb") as f:
            svds = pickle.load(f)
        lambda_path = os.path.join(dir, "layer_lambdas.pkl")
        assert os.path.exists(lambda_path)
        with open(lambda_path, "rb") as f:
            lambdas = pickle.load(f)
        return cls(svds, lambdas, lambda_value)
        
    def calculate_hvp(self, layer_grad_query):
        layer_hvps = {}
        for layer_name, grad in layer_grad_query.items():
            svd_data = self.layer_svds[layer_name]
            
            # Get the SVDs of the two matrices
            Q_S = svd_data['Q_S']
            Q_A = svd_data['Q_A']
            
            in_shape = Q_A.shape[-1]
            out_shape = Q_S.shape[-1]
            
            # Get the Lambdas
            Lambda = self.layer_lambdas[layer_name]

            # Calculate (\mathbf{G}+\lambda \mathbf{I})^{-1} \mathbf{v} using the provided formula
            if not self.lambda_value:
                lambda_value = Lambda.mean()*0.1
            else:
                lambda_value = self.lambda_value
            
            Lambda = Lambda + lambda_value

            hvp = Q_S.T @ ((Q_S @ grad @ Q_A.T) / Lambda) @ Q_A
            layer_hvps[layer_name] = hvp.reshape(-1)
        return layer_hvps

    def calculate_layerwise_influence(self, layer_hvps_query, layer_grad_train):
        layer_influence = {}
        for layer_name, hvp in layer_hvps_query.items():
            grad_train = layer_grad_train[layer_name][1]
            # Reshape hvp and grad_train into vectors
            grad_train_vector = grad_train.reshape(-1)
            # Calculate the inner product
            influence = torch.dot(hvp, grad_train_vector)
            layer_influence[layer_name] = influence
        return layer_influence

    def calculate_total_influence(self, layer_hvps_query, layer_grad_train):
        layer_influence = self.calculate_layerwise_influence(layer_hvps_query, layer_grad_train)
        total_influence = sum(_.cpu() for _ in layer_influence.values())
        return total_influence



def compute_LM_loss(ids, masks, probs):
    bs, ts = ids.shape
    probs = probs[:, :-1, :].reshape(bs * (ts - 1), -1)
    probs = probs[torch.arange(bs * (ts - 1)), ids[:, 1:].flatten()].reshape(bs, ts - 1)
    return -(masks[:, :-1] * torch.log2(probs)).sum(axis=1) #/ (1e-9 + masks.sum(axis=1))


def compute_pseudo_loss(masks, logits):
    bs, ts = masks.shape
    ids = logits.argmax(dim=-1) # assuming that the pseudo labels are greedy-search generated    
    probs = torch.softmax(logits, -1).reshape(bs * ts, -1)
    probs = probs[torch.arange(bs * ts), ids.flatten()].reshape(bs, ts)
    return -(masks * torch.log2(probs)).sum(axis=1)


def get_sample_indices(num_samples, num_neg, i):
    indices = list(range(num_samples))
    indices.remove(i)
    neg_indices = random.sample(indices, num_neg)
    return [i] + neg_indices


def sentence_tokenize(s):
    return re.split(r'(?<=[^A-Z].[.!?]) +(?=[A-Z])', s)




def zero_grad(*obj):
    if len(obj) > 1:
        for subobj in obj:
            zero_grad(subobj)
    elif hasattr(obj[0], "parameters"):
        for subobj in obj[0].parameters():
            zero_grad(subobj)
    elif obj[0].grad is not None:
        obj[0].grad.data.zero_()


def format_llama_weight(layer, wtype):
    assert wtype in {"q", "k", "v", 'o', 'gate', 'down', 'up', 'down', 'norm'}
    if wtype == "norm":
        pattern = "input_layernorm"
    elif len(wtype) == 1:
        pattern = "self_attn.%s_proj" % wtype
    else:
        pattern = "mlp.%s_proj" % wtype
    return "model.layers.%s.%s.weight" % (layer, pattern)


class Generator:
    def __init__(self, model, device="cuda:0", **params):
        super().__init__()
        self._name = model
        self._device = device
        self._params = {}
        self.parameters = params
        self.build()      

    def build(self):
        print("Initializing LLM: %s" % self._name)
        maps = None if self._device == "cpu" else "auto"
        self._tokenizer = trf.AutoTokenizer.from_pretrained(self._name, use_fast=False, padding_side="right", cache_dir="./cache")
        self._model = trf.AutoModelForCausalLM.from_pretrained(self._name, cache_dir="./cache", device_map=maps).float()
        self._out_embed = self._model.get_output_embeddings().weight.data.detach()
        self._inp_embed = self._model.get_input_embeddings()
        if not self._tokenizer.eos_token:
            self._tokenizer.eos_token = "</s>"
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._tokenizer.eos_token_id  
        self._config = self._model.config
        self._headsize = self._config.hidden_size // self._config.num_attention_heads

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._params.copy()

    @parameters.setter
    def parameters(self, params):
        for key, val in self._params:
            if key not in params:
                params[key] = val
        self._params = {"min_length": params.get("minlen", 1),
                        "max_length": params.get("maxlen", 50),
                        "temperature": params.get("temperature", 0.0),
                        "top_p": params.get("top_p", 0.000),
                        "num_return_sequences": params.get("ngen", 1),
                        "penalty_alpha": params.get("penalty", 0.),
                        "do_sample": False}
    
    def pretrain(self, data_file, output_dir, **config):
        from datasets import load_dataset
        maxlen = config.get("max_length", 1024)
        def tokenize(content):
            outputs = self._tokenizer(content["text"], truncation=True,
                                     max_length=maxlen,
                                     return_overflowing_tokens=True,
                                     return_length=True)
            input_batch = []
            for length, ids in zip(outputs["length"], outputs["input_ids"]):
                if length <= maxlen:
                    input_batch.append(ids)
            return {"input_ids": input_batch}
        dataset = load_dataset("csv", data_files=[data_file], delimiter='\t', column_names=['text'])
        
        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names)
        args = trf.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.get("batch_size", 1),
            per_device_eval_batch_size=config.get("batch_size", 1),
            evaluation_strategy="steps",
            eval_steps=config.get("eval_steps", 1000),
            logging_steps=config.get("eval_steps", 1000),
            save_steps=config.get("eval_steps", 1000),
            gradient_accumulation_steps=config.get("gradient_accumulation", 4),
            num_train_epochs=config.get("epochs", 3),
            weight_decay=config.get("weight_decay", 1e-7),
            warmup_steps=config.get("warmup", 1000),
            lr_scheduler_type=config.get("scheduler", "cosine"),
            learning_rate=config.get("learn_rate", 1e-5),
            fp16=config.get("fp16", True),
            )
        self._model.train()
        print("Pre-training info:", dataset["train"])
        trainer = trf.Trainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=args,
            data_collator=trf.DataCollatorForLanguageModeling(self._tokenizer, mlm=False),
            train_dataset=dataset["train"],
            eval_dataset=dataset["train"].select(range(100))
            )
        trainer.train()
        trainer.save_model()

    def get_inputs(self, texts):
        inputs = self._tokenizer(texts, padding=True, max_length=1024,
                                 truncation=True, return_tensors="pt")
        for key in list(inputs.keys()):
            if key not in ["input_ids", "attention_mask"]:
                del inputs[key]
            else:
                inputs[key] = inputs[key].to(self._device)
        return inputs
    
    def tokenize(self, text):
        return self._tokenizer.tokenize(text.strip())

    def prepare4generate(self, input_texts):
        inputs = self.get_inputs(input_texts) 
        batch_size, seq_len = inputs['input_ids'].shape
        
        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = seq_len - inputs['attention_mask'].sum(dim=-1)
        for idx in range(batch_size):
            inputs['input_ids'][idx] = inputs['input_ids'][idx].roll(shifts[idx].item())

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()} | self._params
        if inputs['min_length'] is not None:
            inputs['min_length'] = inputs['min_length'] + seq_len
        if inputs['max_length'] is not None:
            inputs['max_length'] = min(self._model.config.max_position_embeddings,
                                       inputs['max_length'] + seq_len)
        return inputs, seq_len

    def generate(self, texts):
        with torch.no_grad():
            self._model.eval()
            inputs, seq_len = self.prepare4generate(texts)
            output_ids = self._model.generate(**inputs)[:, seq_len:]
            return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def forward(self, texts):
        inputs = self.get_inputs(texts)
        return inputs, self._model(**inputs)

    def prepare4explain(self, inps, refs):
        if isinstance(inps, str):
            inps = [inps]
        if isinstance(refs, str):
            refs = [refs]
        assert isinstance(inps, list) and isinstance(refs, list)
        assert len(inps) == len(refs)
        texts = [i.strip() + " " + r.strip() \
                 for i, r in zip(inps, refs)]
        new_inps, new_refs = [], []
        for inp, ref, txt in zip(inps, refs, texts): 
            inp = self.tokenize(inp)
            ref = self.tokenize(ref)
            txt = self.tokenize(txt)
            assert len(txt) - (len(inp) + len(ref)) <= 1, str(txt) + " | " + str(inp) + " | " + str(ref)
            # the insert blank may be splitted into multiple tokens
            ref = txt[len(inp):]
            new_inps.append(inp)
            new_refs.append(ref)
        return new_inps, new_refs, texts
        
    def input_explain(self, inps, refs, L=10, b=7, p=4, eps=1e-7):
        self._model.eval()
        inps, refs, texts = self.prepare4explain(inps, refs) 
        ids = self.get_inputs(texts)["input_ids"]
        embs = self._inp_embed(ids.to(self._device)).detach().requires_grad_()
        
        # LLaMA family may always automatically append a prefix to the begining
        # you need to set bias=1 for GPT family! 
        # For other family, like T5, OPT, bloomz, you may need to revise this code manually.
        bias = 1 if "gpt" in self._name else 0

        expls, attrs, confs = [], [], []
        probs = torch.softmax(self._model(inputs_embeds=embs)["logits"], -1)
        for i, (inp, ref) in enumerate(zip(inps, refs)): 
            ref = torch.tensor(self._tokenizer.convert_tokens_to_ids(ref)).long()
            obj = probs[i, torch.arange(len(inp) - bias, len(inp) + len(ref)- bias), ref]  
            confs.append(obj.cpu().detach().numpy())
            grad = []
            for j in range(len(ref)): 
                zero_grad(self._model, embs)
                obj[j].backward(retain_graph=True)
                grad.append(embs.grad.data[i, 1 - bias:1 + len(inp) - bias].detach().cpu())

            if len(grad) == 0:
                expls.append(np.array([]))
                attrs.append(np.array([]))
                continue
            
            with torch.no_grad():
                # importance
                emb = embs[i, 1 - bias:1 + len(inp) - bias].unsqueeze(0).cpu()
                grad = torch.stack(grad, 0).cpu()
                expl = (grad * emb).sum(axis=-1).T
                expls.append(expl.numpy())

                # sparsify and normalize
                zeros = torch.zeros_like(expl)
                expl = torch.maximum(zeros, expl)
                expl = expl / (expl.max(axis=0, keepdims=True).values + eps)
                expl = torch.ceil(expl * L)
                expl = torch.where(expl <= b, zeros, expl)
                #expls.append(expl.numpy())

                # word attribution with density
                l1 = expl.sum(axis=-1)
                lp = (expl ** p).sum(axis=-1) ** (1. / p) + eps
                attrs.append((l1 / lp).numpy())
        return inps, refs, expls, attrs, confs

    @torch.no_grad()
    def _get_embeds(self, words, batch_size=1024):
        def encode_batch(ids, Hi, Ho):
            M, maxlen = [], max(map(len, ids))
            for _ in ids:
                M.append([1] * len(_) + [0] * (maxlen - len(_)))
                _.extend([self._tokenizer.eos_token_id] * (maxlen - len(_)))
            M = torch.tensor(M).float().unsqueeze(-1).cpu()
            ids = torch.tensor(ids).long()
            Ho.append((Eo[ids] * M).sum(axis=1) / (1e-9 + M.sum(axis=1)))
            Hi.append((Ei[ids] * M).sum(axis=1) / (1e-9 + M.sum(axis=1)))
        
        Ei = self._inp_embed.weight.cpu().float()
        Eo = self._out_embed.cpu().float()
        Hi, Ho, batchE = [], [], []
        for word in words:
            tokens = self._tokenizer.tokenize(" " + word)
            if tokens[0] in (u'Ġ', u'▁'):
                tokens = tokens[1:]
            batchE.append(self._tokenizer.convert_tokens_to_ids(tokens))
            if len(batchE) == batch_size:
                encode_batch(batchE, Hi, Ho)
                batchE.clear()
        if len(batchE) > 0:
            encode_batch(batchE, Hi, Ho)
        return torch.cat(Hi, axis=0), torch.cat(Ho, axis=0)

    def _get_weights(self, layer, wtype):
        assert isinstance(layer, int) and layer >= 0
        assert wtype in {"qk", "vo", "down"}

        @functools.cache
        def get_weight(l, w):
            name = format_llama_weight(l, w)
            weight = self._model.get_parameter(name).detach()
            if w == "down":
                return weight.T
            if w == "o":
                return weight.reshape(self._config.num_attention_heads,
                        self._config.hidden_size // self._config.num_attention_heads,
                        self._config.hidden_size)
            norm = self._model.get_parameter(format_llama_weight(layer, "norm")).detach()
            weight = weight * norm.unsqueeze(1)
            return weight.reshape(self._config.hidden_size,
                    self._config.num_attention_heads,
                    self._config.hidden_size // self._config.num_attention_heads,
                    ).permute(1, 0, 2)
        if wtype in {"qk", "vo"}:
            return get_weight(layer, wtype[0]), get_weight(layer, wtype[1])
        return get_weight(layer, wtype)
            




def batchit(X, bs=1, droplast=False):
    batch = []
    for x in X:
        batch.append(x)
        if len(batch) == bs:
            yield batch
            batch.clear()
    if not droplast and len(batch) > 0:
        yield batch