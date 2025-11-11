import torch
import torch.nn as nn
import torch.nn.functional as F

import torch._dynamo as dynamo

import math
from torch import Tensor
from typing import *

from transformers.utils import logging
logger = logging.get_logger(__name__)


__all__ = [
    "SuffixAttention",
    "SufaCache",
]



from .base import RosaConfig, RosaBase


class SuffixAttention(RosaBase):
    def __init__(
            self,
            config: RosaConfig,
            layer_idx: int,
    ):
        super().__init__(config=config)

        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_qk_bits = config.num_query_key_bits
        self.num_v_bits = config.num_value_bits

        self.n_rep = self.num_heads // self.num_kv_heads

        self.window_size = config.max_attention_head_size // self.num_qk_bits
        assert self.num_qk_bits * self.window_size % 16 == 0

        bias = config.bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.num_qk_bits, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_qk_bits, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_v_bits, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.num_v_bits, config.hidden_size, bias=bias)

        self.v_emb0 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), -1e-5))
        self.v_emb1 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), +1e-5))

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

        # key_states = self.repeat_kv(key_states, self.n_rep)
        # value_states = self.repeat_kv(value_states, self.n_rep)

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

        with dynamo.config.patch(capture_scalar_outputs=True):
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
    
    @classmethod
    def apply_adapter_to_modules_(
        cls,
        module: nn.Module,
        config: RosaConfig,
        target_modules: str = r".*self_attn",
        autocast_dtype: bool = True,
    ):
        adapter_factory = lambda i, m: cls(config, layer_idx=i)
        super().apply_adapter_to_modules_(
            module=module,
            adapter_factory=adapter_factory,
            adapter_name="sufa_attn",
            target_modules=target_modules,
            autocast_dtype=autocast_dtype,
        )


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
