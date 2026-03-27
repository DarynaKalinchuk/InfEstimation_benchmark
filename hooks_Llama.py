# Author: Xuansheng Wu (wuxsmail@163.com)
# Last Modify: 2024-01-11
# Description: hooks to collect intermediate outcomes of LLaMA MLP.

from dataclasses import dataclass
from functools import wraps

import transformers.models.llama.modeling_llama as llama
import torch as tc


KEY = "__hook_cache"


@dataclass
class _Cache:
    LayerID: int

    @property
    def named_parameters(self):
        params = {}
        for name in sorted(self.__dataclass_fields__):
            if name != "LayerID":
                params[name] = getattr(self, name)
        return params

    def clear(self):
        for name in self.named_parameters:
            setattr(self, name, None)

    def zero_grad(self):
        for _, var in self.named_parameters.items():
            if var is not None and var.grad is not None:
                var.grad = None

    def retain_grad(self):
        for _, var in self.named_parameters.items():
            if isinstance(var, tc.Tensor) and var.requires_grad:
                var.retain_grad()

    def check_type(self):
        for name, var in self.named_parameters.items():
            assert isinstance(var, tc.Tensor), f"variable `{name}` is not torch.Tensor!"


@dataclass
class MLPCache(_Cache):
    """Given MLP: y = down_proj(act(gate_proj(x)) * up_proj(x))"""
    inputs: tc.Tensor = None
    hiddens: tc.Tensor = None
    activates: tc.Tensor = None
    outputs: tc.Tensor = None
    weights: tc.Tensor = None

    def collect_states(self):
        states = (self.inputs, self.hiddens, self.activates, self.outputs)
        for each in states:
            assert isinstance(each, tc.Tensor)
        return states


class HookWrapper:
    @staticmethod
    def manual(forward_func):
        @wraps(forward_func)
        def cached_forward(self, *args, **kwargs):
            if not hasattr(self, KEY):
                return forward_func(self, *args, **kwargs)
            cache = getattr(self, KEY)
            cache.clear()
            outputs = forward_func(self, *args, **kwargs)
            cache.check_type()
            cache.retain_grad()
            return outputs
        return cached_forward


class _HookController:
    def __init__(self, model, target_block, cache_type):
        self._model = model
        self._target = target_block
        self._caches = {}
        self._modules = []

        for name, layer in self:
            layer_cache = cache_type(len(self._caches))
            setattr(layer, KEY, layer_cache)
            self._modules.append(name)
            self._caches[name] = layer_cache

        print(
            "Target Block: %s | Cache Type: %s | Numbers: %d"
            % (target_block, cache_type, len(self._caches))
        )

    def __iter__(self):
        for name, layer in self._model.named_modules():
            if isinstance(layer, self._target):
                yield name, layer


class MLPHookController(_HookController):
    def __init__(self, model):
        super().__init__(model, llama.LlamaMLP, MLPCache)

    def collect_states(self):
        states = {}
        for l, (name, _) in enumerate(self, 1):
            layer_states = self._caches[name].collect_states()
            states[f"Layer{l}"] = (layer_states[0], layer_states[1].grad)
        return states

    def collect_weight_grads(self):
        grads = {}
        for l, (name, _) in enumerate(self, 1):
            weight = self._caches[name].weights
            grads[f"Layer{l}"] = (weight, weight.grad)
        return grads

    @classmethod
    def LLaMA(cls, model):
        return cls(model)


@HookWrapper.manual
def custom_LlamaMLP(self, x):
    if not hasattr(self, KEY):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    cache = getattr(self, KEY)
    cache.weights = self.gate_proj.weight
    cache.inputs = x
    cache.hiddens = self.gate_proj(x)
    cache.activates = self.act_fn(cache.hiddens) * self.up_proj(x)
    cache.outputs = self.down_proj(cache.activates)
    return cache.outputs


llama.LlamaMLP.forward = custom_LlamaMLP