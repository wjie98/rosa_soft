import torch
import torch.nn as nn
import torch.nn.functional as F

import re
import math
from torch import Tensor
from typing import *

from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging



__all__ = [
    "SuffixAttention",
    "SuffixCache",
]


logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    bsz, n_kvs, seq_len, dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kvs, n_rep, seq_len, dim)
    return hidden_states.reshape(bsz, n_kvs * n_rep, seq_len, dim)


class SuffixAttention(nn.Module):
    def __init__(self,
            layer_idx: int,
            dim: int,
            num_heads: int,
            num_kv_heads: int,
            num_qk_bits: int = 8,
            num_v_bits: int = 8,
            bias: bool = False,
    ):
        super().__init__()

        assert num_heads % num_kv_heads == 0

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_qk_bits = num_qk_bits
        self.num_v_bits = num_v_bits

        self.n_rep = num_heads // num_kv_heads

        self.window_size = 256 // self.num_qk_bits
        assert self.num_qk_bits * self.window_size % 16 == 0

        self.q_proj = nn.Linear(dim, num_heads * num_qk_bits, bias=bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * num_qk_bits, bias=bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * num_v_bits, bias=bias)
        self.o_proj = nn.Linear(num_heads * num_v_bits, dim, bias=bias)

        self.v_emb0 = nn.Parameter(torch.full((num_heads * num_v_bits,), -1e-5))
        self.v_emb1 = nn.Parameter(torch.full((num_heads * num_v_bits,), +1e-5))

        self.register_buffer("tau", torch.ones((1,)), persistent=False)
    
    @classmethod
    def update_tau_(cls, module: nn.Module, tau: float):
        for m in module.modules():
            if isinstance(m, cls):
                m.get_buffer("tau").fill_(tau)
    
    @classmethod
    def patch_attention(
            cls,
            model: nn.Module,
            dim: int,
            num_heads: int,
            num_kv_heads: int,
            num_qk_bits: int = 8,
            num_v_bits: int = 8,
            bias: bool = False,
            module_names: List[str] = ["self_attn", "attn", "attention"],
    ):
        layer_container_names = ["layers", "h", "blocks", "block"]
        layers = None
        for name in layer_container_names:
            if hasattr(model, name):
                layers = getattr(model, name)
                break

            elif hasattr(model, "model") and hasattr(getattr(model, "model"), name):
                layers = getattr(getattr(model, "model"), name)
                break
        
        if layers is None:
            raise ValueError("Could not find decoder layers in the model.")

        attention_modules = []
        for layer in layers:
            for name in module_names:
                if hasattr(layer, name):
                    attention_modules.append(getattr(layer, name))
                    break
            else:
                logger.warning(f"Could not find an attention module in layer: {layer}")
        
        if not attention_modules:
            raise ValueError("No attention modules found to patch.")
        
        logger.info(f"Found {len(attention_modules)} attention modules to patch.")

        attention_class_forward = {}
        for layer_idx, attn_module in enumerate(attention_modules):
            sufa_attn = cls(
                layer_idx = layer_idx,
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                num_qk_bits=num_qk_bits,
                num_v_bits=num_v_bits,
                bias=bias,
            )

            setattr(attn_module, "sufa_attn", sufa_attn)

            attn_cls = type(attn_module)
            attention_class_forward[attn_cls] = attn_cls.forward
        
        def wrap_forward(fwd):
            def forward(self, *args, **kwargs):
                a, *others = fwd(self, *args, **kwargs)
                b = self.sufa_attn(*args, **kwargs)
                return a + b, *others
            return forward
                    
        for attn_cls, attn_forward in attention_class_forward.items():
            attn_cls.forward = wrap_forward(attn_forward)

        logger.info("Successfully patched all attention modules.")

        return model
    
    @classmethod
    def freeze_base_model_and_enable_patch_params(cls, model: nn.Module):
        for name, p in model.named_parameters():
            if re.search(r"\.sufa_attn\.", name):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        return model
    
    @classmethod
    def state_dict_for_patch(cls, model: nn.Module) -> Dict[str, Any]:
        state = {}
        for name, p in model.state_dict().items():
            if re.search(r"\.rosa_attn\.", name):
                state[name] = p
        return state

    def _attn_bias(self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        bsz, seq_len, _ = hidden_states.size()

        attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril_(-1)
        if attention_mask is not None:
            attn_mask = attn_mask & attention_mask
        
        r = torch.arange(0, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_bias = r.view(1, -1) / (r.view(-1, 1) + 1)
        attn_bias = attn_bias.expand_as(attn_mask).clone()
        attn_bias[~attn_mask] = -torch.inf
        return attn_bias
    
    def forward(
            self,
            hidden_states: Tensor,
            attention_mask = None,
            past_key_values = None,
            **kwargs
    ):
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, -1, self.num_v_bits)

        tau = self.get_buffer("tau")
        
        query_states = torch.tanh(query_states / tau).transpose(1, 2)
        key_states = torch.tanh(key_states / tau).transpose(1, 2)
        value_states = torch.sigmoid(value_states / tau).transpose(1, 2)

        if past_key_values is not None:
            if not hasattr(past_key_values, "sufa_cache"):
                setattr(past_key_values, "sufa_cache", SufaCache(window_size=self.window_size))
            
            cache: SufaCache = getattr(past_key_values, "sufa_cache")
            query_states, key_states, value_states = cache.update(
                query_states, key_states, value_states,
                layer_idx=self.layer_idx,
            )
        else:
            query_states = F.pad(query_states, (0, 0, self.window_size - 1, 0))
            key_states = F.pad(key_states, (0, 0, self.window_size - 1, 0))
            value_states = F.pad(value_states, (0, 0, -1, 1)) # recall next token

        qk_dim = self.window_size * self.num_qk_bits
        
        query_states = query_states.unfold(-2, self.window_size, 1).transpose(-2, -1).reshape(bsz, -1, seq_len, qk_dim)
        key_states = key_states.unfold(-2, self.window_size, 1).transpose(-2, -1).reshape(bsz, -1, seq_len, qk_dim)

        scale = 1.0 / math.sqrt(qk_dim) / tau.item()
        attn_bias = self._attn_bias(hidden_states, attention_mask) / tau
        
        output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=scale,
            attn_mask=attn_bias, enable_gqa=True,
        )

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = output * self.v_emb1 + (1 - output) * self.v_emb0
        output = self.o_proj(output)
        return output


class SufaCache:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.layers: Dict[int, Tuple[Tensor, Tensor, Tensor]] = {}
    
    def update(self,
            query_states: Tensor,
            key_states: Tensor,
            value_states: Tensor,
            layer_idx: int,
    ):
        
        query_cache, key_cache, value_cache = self.layers.get(layer_idx, (None, None, None))
        
        if query_cache is None:
            query_states = F.pad(query_states, (0, 0, self.window_size - 1, 0))
        else:
            query_states = torch.cat([query_cache, query_states], dim=-2)

        if key_cache is None:
            key_states = F.pad(key_states, (0, 0, self.window_size - 1, 0))
        else:
            key_states = torch.cat([key_cache, key_states], dim=-2)
        
        if value_cache is not None:
            value_states = torch.cat([value_cache, value_states], dim=-2)
        
        query_cache = query_states.detach()[:, :, -(self.window_size - 1), :]
        key_cache = key_states.detach()
        value_cache = value_states.detach()

        self.layers[layer_idx] = (query_cache, key_cache, value_cache)

        value_states = F.pad(value_states, (0, 0, -1, 1))
        return query_states, key_states, value_states


if __name__ == "__main__":
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(num_hidden_layers=2, hidden_size=512, intermediate_size=1024)
    config.use_cache = False

    model = Qwen3ForCausalLM(config)
    model = SuffixAttention.patch_attention(
        model,
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
    )

    model = SuffixAttention.freeze_base_model_and_enable_patch_params(model)
    for name, _ in SuffixAttention.state_dict_for_patch(model).items():
        print(name)

    x = torch.randint(0, 2, (3, 10))
    y = model(x)
    print(y)
