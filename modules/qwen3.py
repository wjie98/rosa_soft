import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional


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

logger = logging.get_logger(__name__)

from .base import RosaBase


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
        rosa_states = self.rosa_norm(hidden_states)
        rosa_states = self.rosa_attn.rosa_dispatch(
            hidden_states=rosa_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        
        # Self Attention
        residual = hidden_states
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
        
        hidden_states = self.rosa_attn.rosa_combine(
            states=rosa_states,
            inject_states=hidden_states,
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
    