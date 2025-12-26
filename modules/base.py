import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

from rosa_cpp import RosaContext, RosaWork, rosa_bits_ops, RosaBitsWork



class RosaBase(nn.Module):
    def __init__(self):
        super().__init__()

    def init_rosa(self, config, layer_idx):
        self.rosa_layer_idx = layer_idx

        hidden_size = getattr(config, "hidden_size")
        assert hidden_size % 64 == 0

        self.rosa_num_qk_bits = getattr(config, "rosa_num_query_key_bits", 8)
        self.rosa_num_v_bits = getattr(config, "rosa_num_value_bits", None)

        self.rosa_num_heads = getattr(config, "rosa_num_heads", hidden_size // 8)
        self.rosa_num_kv_heads = getattr(config, "rosa_num_key_value_heads", None)

        self.rosa_suffix_window = getattr(config, "rosa_suffix_window", 8)
        self.rosa_suffix_factor = getattr(config, "rosa_suffix_factor", None)

        if self.rosa_num_v_bits is None:
            self.rosa_num_v_bits = self.rosa_num_qk_bits

        if self.rosa_num_kv_heads is None:
            self.rosa_num_kv_heads = self.rosa_num_heads

        bias = getattr(config, "attention_bias", False)

        self.rosa_q_proj = nn.Linear(hidden_size, self.rosa_num_heads * self.rosa_num_qk_bits, bias=bias)
        self.rosa_k_proj = nn.Linear(hidden_size, self.rosa_num_kv_heads * self.rosa_num_qk_bits, bias=bias)
        self.rosa_v_proj = nn.Linear(hidden_size, self.rosa_num_kv_heads * self.rosa_num_v_bits, bias=bias)
        self.rosa_o_proj = nn.Linear(self.rosa_num_heads * self.rosa_num_v_bits, hidden_size, bias=bias)

        self.rosa_v_emb0 = nn.Parameter(torch.zeros(self.rosa_num_heads * self.rosa_num_v_bits))
        self.rosa_v_emb1 = nn.Parameter(torch.zeros(self.rosa_num_heads * self.rosa_num_v_bits))
        # self.rosa_o_gate = nn.Parameter(torch.zeros(hidden_size))

        # scale = -0.02
        # self.rosa_q_proj.weight.data.uniform_(scale, -scale)
        # self.rosa_k_proj.weight.data.uniform_(scale, -scale)
        # self.rosa_v_proj.weight.data.uniform_(scale, -scale)
        # self.rosa_q_proj.weight.data.copy_(self.rosa_k_proj.weight.data)
        self.rosa_o_proj.weight.data.zero_()

        self.rosa_v_emb0.data.fill_(-1e-5)
        self.rosa_v_emb1.data.fill_(+1e-5)
        # self.rosa_o_gate.data.zero_()
    
    def rosa_dispatch(self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            past_key_values = None,
    ) -> Tuple[Union[RosaBitsWork, RosaWork], Tensor]:
        bsz, seq_len, _ = hidden_states.size()

        query_states: Tensor = self.rosa_q_proj(hidden_states)
        query_states = query_states.view(bsz, seq_len, self.rosa_num_heads, self.rosa_num_qk_bits).transpose(1, 2)

        key_states: Tensor = self.rosa_k_proj(hidden_states)
        key_states = key_states.view(bsz, seq_len, self.rosa_num_kv_heads, self.rosa_num_qk_bits).transpose(1, 2)

        value_states: Tensor = self.rosa_v_proj(hidden_states)
        value_states = value_states.view(bsz, seq_len, self.rosa_num_kv_heads, self.rosa_num_v_bits).transpose(1, 2)

        if past_key_values is None:
            work = rosa_bits_ops(
                query_states, key_states, value_states,
                suffix_window=self.rosa_suffix_window,
                suffix_factor=self.rosa_suffix_factor,
                attention_mask=attention_mask,
                async_op=True,
            )
            states = (work, hidden_states)
        else:
            if not hasattr(past_key_values, "_rosa_cache"):
                setattr(past_key_values, "_rosa_cache", RosaCache())
            cache: RosaCache = getattr(past_key_values, "_rosa_cache")

            work = cache.get_rosa_sam(layer_idx=self.rosa_layer_idx).update(
                query_states, key_states, value_states, 0,
                async_op=True,
            )
            states = (work, hidden_states)
        
        return states
    
    def rosa_combine(
            self,
            states: Tuple[Union[RosaBitsWork, RosaWork], Tensor],
            inject_states: Optional[Tensor] = None,
    ) -> Tensor:
        work, hidden_states = states
        output = work.wait()

        bsz, _, seq_len, _ = output.size()

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.rosa_v_emb1 * output + self.rosa_v_emb0 * (1 - output)
        output = self.rosa_o_proj(output)
        
        # gate = torch.sigmoid(self.rosa_o_gate)
        if inject_states is not None:
            # output = output * gate + inject_states * (1 - gate)
            output = output + inject_states
        return output


class RosaCache:
    def __init__(self):
        self.layers: Dict[int, RosaContext] = {}
    
    def get_rosa_sam(self, layer_idx: int):
        if layer_idx not in self.layers:
            self.layers[layer_idx] = RosaContext()
        return self.layers[layer_idx]

    def update(self, query: Tensor, key: Tensor, value: Tensor, layer_idx: int):
        return self.get_rosa_sam(layer_idx).update(query, key, value)
