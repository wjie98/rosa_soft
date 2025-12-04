import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional


from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3Attention,
)

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)

from .base import RosaBase


class RosaQwen3Attention(Qwen3Attention, RosaBase):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)

        hidden_size = config.hidden_size
        bias = config.attention_bias

        self.o_gate = nn.Linear(hidden_size, hidden_size, bias=bias)

        config.rosa_num_query_key_bits = 8
        config.rosa_num_value_bits = 64
        config.rosa_num_heads = config.num_attention_heads * 2
        config.rosa_num_key_value_heads = config.rosa_num_heads
        
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
        attn_output, attn_weights = Qwen3Attention.forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        attn_output = attn_output * self.o_gate(hidden_states)

        attn_output = attn_output + RosaBase.forward(
            self,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
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

