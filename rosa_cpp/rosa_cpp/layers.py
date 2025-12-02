import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Dict, Callable

from transformers.cache_utils import Cache

from . import RosaContext, rosa_att_ops, rosa_gss_ops



class RosaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        
        self.layer_idx = layer_idx

        hidden_size = getattr(config, "hidden_size")
        assert hidden_size % 64 == 0

        self.num_qk_bits = getattr(config, "rosa_num_query_key_bits", 8)
        self.num_v_bits = getattr(config, "rosa_num_value_bits", 8)

        self.num_heads = getattr(config, "rosa_num_heads", hidden_size // 64)
        self.num_kv_heads = getattr(config, "rosa_num_key_value_heads", hidden_size // 64)

        self.n_rep = self.num_heads // self.num_kv_heads

        self.tau = getattr(config, "rosa_tau", 1.0)
        self.proxy_type = getattr(config, "rosa_proxy_type", "gss")
        self.proxy_att_head_dim = getattr(config, "rosa_proxy_att_head_dim", 128)
        self.proxy_gss_num_samples = getattr(config, "rosa_proxy_gss_num_samples", 64)

        bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.num_qk_bits, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.num_qk_bits, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.num_v_bits, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.num_v_bits, hidden_size, bias=bias)

        self.v_emb0 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), -1e-5))
        self.v_emb1 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), +1e-5))
        self.o_gate = nn.Linear(hidden_size, hidden_size, bias=bias)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        attention_states: Optional[Tensor] = None,
        past_key_values = None,
    ) -> Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states: Tensor = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.num_qk_bits).transpose(1, 2)

        key_states: Tensor = self.k_proj(hidden_states)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, self.num_qk_bits).transpose(1, 2)

        value_states: Tensor = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, self.num_v_bits).transpose(1, 2)

        if past_key_values is None:
            if self.proxy_type == "gss":
                output = rosa_gss_ops(
                    query_states, key_states, value_states, 0,
                    num_samples=self.proxy_gss_num_samples,
                    tau=self.tau,
                    training=self.training,
                    async_op=False,
                )

            output = rosa_att_ops(
                query_states, key_states, value_states, 0,
                attn_mask=attention_mask,
                suffix_head_dim=self.proxy_att_head_dim,
                tau=self.tau,
                training=self.training,
            )
        else:
            if not hasattr(past_key_values, "_rosa_cache"):
                setattr(past_key_values, "_rosa_cache", RosaCache())
            cache: RosaCache = getattr(past_key_values, "_rosa_cache")
            output = cache.update(query_states, key_states, value_states, layer_idx=self.layer_idx)
        
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.v_emb1 * output + self.v_emb0 * (1 - output)
        output = self.o_proj(output)
        
        gate = torch.sigmoid(self.o_gate(hidden_states))
        return output * gate


class RosaCache:
    def __init__(self):
        self.layers: Dict[int, RosaContext] = {}

    def update(self, query: Tensor, key: Tensor, value: Tensor, layer_idx: int):
        if layer_idx not in self.layers:
            self.layers[layer_idx] = RosaContext()
        
        return self.layers[layer_idx].update(query, key, value)
