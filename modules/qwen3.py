import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch import Tensor
from typing import *

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
)

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

from rosa_soft import (
    RosaContext, RosaWork,
    rosa_bits_ops, RosaBitsWork,
    rosa_scan_ops, RosaScanWork,
)

logger = logging.get_logger(__name__)


__all__ = [
    "RosaQwen3Attention",
    "RosaQwen3DecoderLayer",
]

##################################################################################

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

        self.rosa_schmitt_trigger = 0.1

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
        self.rosa_o_gate = nn.Parameter(torch.zeros(hidden_size))

        scale = -0.02 / getattr(config, "num_hidden_layers")
        self.rosa_q_proj.weight.data.uniform_(scale, -scale)
        self.rosa_k_proj.weight.data.copy_(self.rosa_q_proj.weight.data)

        s0 = self.rosa_v_proj.weight.size(0)
        s1 = self.rosa_v_proj.weight.size(1)
        scale = max(1.0, math.sqrt(s0 / s1))
        nn.init.orthogonal_(self.rosa_v_proj.weight.data, gain=scale)
        nn.init.zeros_(self.rosa_o_proj.weight.data)

        self.rosa_v_emb0.data.fill_(-1e-5)
        self.rosa_v_emb1.data.fill_(+1e-5)
        self.rosa_o_gate.data.zero_()
    
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
            # work = rosa_bits_ops(
            #     query_states, key_states, value_states,
            #     suffix_window=self.rosa_suffix_window,
            #     suffix_factor=self.rosa_suffix_factor,
            #     attention_mask=attention_mask,
            #     schmitt_trigger=self.rosa_schmitt_trigger,
            #     async_op=True,
            # )
            work = rosa_scan_ops(
                query_states, key_states, value_states,
                suffix_window=self.rosa_suffix_window,
                suffix_factor=self.rosa_suffix_factor,
                schmitt_trigger=self.rosa_schmitt_trigger,
                async_op=True,
            )
            states = (work, hidden_states)
        else:
            if not hasattr(past_key_values, "_rosa_cache"):
                setattr(past_key_values, "_rosa_cache", RosaCache())
            cache: RosaCache = getattr(past_key_values, "_rosa_cache")

            work = cache.get_rosa_sam(layer_idx=self.rosa_layer_idx).update(
                query_states, key_states, value_states, 0,
                schmitt_trigger=self.rosa_schmitt_trigger, async_op=True,
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
        
        gate = torch.sigmoid(self.rosa_o_gate)
        if inject_states is not None:
            output = output * gate + inject_states * (1 - gate)
            # output = output + inject_states
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


##################################################################################


class RosaQwen3Attention(Qwen3Attention, RosaBase):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        
        self.init_rosa(config=config, layer_idx=layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        rosa_states = self.rosa_dispatch(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        attn_output, attn_weights = Qwen3Attention.forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        attn_output = self.rosa_combine(
            states=rosa_states,
            inject_states=attn_output,
        )

        return attn_output, attn_weights
    
    @classmethod
    def patch_model(cls, model: nn.Module):
        for name, m in model.named_modules():
            if not isinstance(m, Qwen3Attention):
                continue
            elif hasattr(m, "rosa_q_proj"):
                logger.info(f"Skip {name} since it's already patched.")
                continue

            t = cls(config=m.config, layer_idx=m.layer_idx)
            t.load_state_dict(m.state_dict(), strict=False)

            for p in m.parameters():
                t = t.to(dtype=p.dtype, device=p.device)
                break

            model.set_submodule(name, t)
        return model


class RosaQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rosa_attn = RosaBase()
        self.rosa_attn.init_rosa(config, layer_idx)
        self.rosa_norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        
        hidden_states = self.rosa_attn.rosa_combine(
            states=self.rosa_attn.rosa_dispatch(
                hidden_states=self.rosa_norm(hidden_states),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            ),
            inject_states=hidden_states,
        )

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    @classmethod
    def patch_model(cls, model: nn.Module):
        for name, m in model.named_modules():
            if not isinstance(m, Qwen3DecoderLayer):
                continue
            elif hasattr(m, "rosa_attn"):
                logger.info(f"Skip {name} since it's already patched.")
                continue

            config = m.self_attn.config
            layer_idx = m.self_attn.layer_idx

            t = cls(config=config, layer_idx=layer_idx)
            t.load_state_dict(m.state_dict(), strict=False)

            for p in m.parameters():
                t = t.to(dtype=p.dtype, device=p.device)
                break

            model.set_submodule(name, t)
        return model
    