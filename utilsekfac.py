
"""
Utilities for loading models.
"""
import os
import yaml
import logging
import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


@torch.no_grad()
def replace_conv1d_modules(model: nn.Module) -> int:
    changed_modules = 0
    for name, module in model.named_children():
        # Recurse into child modules
        if len(list(module.children())) > 0:
            changed_modules += replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            changed_modules += 1
            new_module = nn.Linear(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)

    return changed_modules


def construct_hf_model(
    registry_config: dict, 
    checkpoint_dir: str = None,
) -> PreTrainedModel:
    """
    Constructs a HuggingFace model with Kronfluence compatability.

    Args:
        config (dict): Configuration dictionary containing model and tokenizer information.
        checkpoint_dir (str): Path to a checkpoint directory to load.
    """
    model_kwargs = registry_config['model']['kwargs']

    if checkpoint_dir:
        if not os.path.isdir(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist.")

        logging.info(f"Loading model from checkpoint: {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
            trust_remote_code=True,
            **model_kwargs,
        )
    elif not registry_config['model']['pretrained']:
        logging.info(f"Loading {registry_config['model']['name']} from scratch")
        model_config = AutoConfig.from_pretrained(
            registry_config['model']['path'],
            trust_remote_code=True,
        )
        model_config.vocab_size = registry_config['tokenizer']['vocab_size']
        model = AutoModelForCausalLM.from_config(
            config=model_config,
            trust_remote_code=True,
            **model_kwargs,
        )
    else:
        logging.info(f"Loading pretrained {registry_config['model']['name']} from HuggingFace hub")
        model = AutoModelForCausalLM.from_pretrained(
            registry_config['model']['path'],
            trust_remote_code=True,
            **model_kwargs,
        )

    # check if vocab size is the same
    if model.config.vocab_size != registry_config['tokenizer']['vocab_size']:
        raise ValueError(f"Vocab size mismatch: actual = {model.config.vocab_size} vs expected = {registry_config['tokenizer']['vocab_size']}")
    
    # Replace Conv1D modules with Linear modules for Kronfluence compatability
    num_changed_modules = replace_conv1d_modules(model)
    if num_changed_modules > 0:
        logging.info(f"Replaced {num_changed_modules} Conv1D modules with Linear modules for Kronfluence compatability")

    return model




"""
Required set-up for influence function computation.
Prepares models for running EK-FAC and defines the language modeling task
(in our case, cross-entropy training and query loss).
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]

class LanguageModelingTask(Task):
    def __init__(
        self, 
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()
        if not sample:
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            summed_loss = F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        num_layers = self.config['model']['num_layers']

        if self.config['model']['family'] == "pythia":

            for i in range(num_layers):
                    total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_h_to_4h")
                    total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_4h_to_h")

                    total_modules.append(f"gpt_neox.layers.{i}.attention.query_key_value")
                    total_modules.append(f"gpt_neox.layers.{i}.attention.dense")   

        elif self.config['model']['family'] in ["llama-3", "tinyllama", "llama"]:

            for i in range(num_layers):
                    total_modules.append(f"model.layers.{i}.mlp.gate_proj")
                    total_modules.append(f"model.layers.{i}.mlp.up_proj")
                    total_modules.append(f"model.layers.{i}.mlp.down_proj")

                    total_modules.append(f"model.layers.{i}.self_attn.q_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.k_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.v_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.o_proj")
        else:
            raise NotImplementedError(
                f"Model family {self.config['model']['family']} not supported for influence tracking."
                " Please add the required modules to the `get_influence_tracked_modules` method in `utils/tasks.py`."
            )

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]