import torch
import torch.nn as nn
import torch.nn.functional as F

import re
import math
import types

from torch import Tensor
from typing import *

import logging

logger = logging.getLogger(__name__)


__all__ = [
    "RosaConfig",
    "RosaBase",
    "calculate_tau_decay",
    "calculate_distillation_loss",
]



class RosaConfig:
    def __init__(
            self,
            hidden_size: int = 512,
            num_heads: int = 8,
            num_key_value_heads: int = 2,
            num_query_key_bits: int = 8,
            num_value_bits: int = 8,
            bias: bool = False,
            bits_tau: float = 1.0,
            attn_tau: float = 1.0,
            proxy_type: Literal["rosa", "sufa"] = "rosa",
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_query_key_bits = num_query_key_bits
        self.num_value_bits = num_value_bits
        self.bias = bias
        self.bits_tau = bits_tau
        self.attn_tau = attn_tau
        self.proxy_type = proxy_type


class RosaBase(nn.Module):
    def __init__(self, config: RosaConfig) -> None:
        super().__init__()
        
        assert config.num_heads % config.num_key_value_heads == 0
        self.config = config

        self.register_buffer("rosa_ops_alpha", torch.ones((1,)), persistent=False)
    
    @staticmethod
    def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
        bsz, n_kvs, seq_len, dim = hidden_states.size()
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kvs, n_rep, seq_len, dim)
        return hidden_states.reshape(bsz, n_kvs * n_rep, seq_len, dim)
    
    def get_rosa_ops_alpha(self) -> Tensor:
        return self.get_buffer("rosa_ops_alpha").clamp(0.0, 1.0)
    
    @classmethod
    def set_rosa_ops_alpha(cls, module: nn.Module, alpha: float = 1.0):
        for m in module.modules():
            if isinstance(m, cls):
                m.get_buffer("rosa_ops_alpha").fill_(alpha)
    
    @classmethod
    def apply_adapter_to_modules_(
        cls,
        module: nn.Module,
        config: RosaConfig,
        target_modules: str = r".*self_attn",
        autocast_dtype: bool = True,
    ):
        adapter_name = getattr(cls, "adapter_name", "rosa_attn")
        adapter_factory = lambda i, m: cls(config, layer_idx=i)
        cls._apply_adapter_to_modules_impl(
            module=module,
            adapter_factory=adapter_factory,
            adapter_name=adapter_name,
            target_modules=target_modules,
            autocast_dtype=autocast_dtype,
        )
    
    @classmethod
    def _apply_adapter_to_modules_impl(
        cls,
        module: nn.Module,
        adapter_factory: Callable[[int, nn.Module], nn.Module],
        adapter_name: str = "rosa_attn",
        target_modules: str = r".*self_attn",
        autocast_dtype: bool = True,
    ):
        def apply_forward(m: nn.Module, adapter_name: str):
            setattr(m, f"{adapter_name}_original_forward", m.forward)

            def forward(self, *args, **kwargs):
                original_forward = getattr(self, f"{adapter_name}_original_forward")
                adapter: RosaBase = getattr(self, adapter_name)

                a, *others = original_forward(*args, **kwargs)
                b, *_ = adapter(*args, **kwargs)
                return a + b, *others
            
            m.forward = types.MethodType(forward, m)

        count = 0
        for name, m in module.named_modules():
            if not re.fullmatch(target_modules, name):
                continue
            elif hasattr(m, adapter_name):
                continue

            layer_idx = getattr(m, "layer_idx", count)
            adapter = adapter_factory(layer_idx, m)

            for p in m.parameters():
                dtype = p.dtype if autocast_dtype else None
                adapter = adapter.to(device=p.device, dtype=dtype)
                break

            setattr(m, adapter_name, adapter)

            apply_forward(m, adapter_name)

            count += 1
        
        if count == 0:
            raise ValueError("No target modules found to patch.")
        
        logger.info(f"Found {count} attention modules to patch.")

        return module
    
    @classmethod
    def apply_trainable_parameters_(cls, module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

        for m in module.modules():
            if not isinstance(m, cls):
                continue
            for p in m.parameters():
                p.requires_grad_(True)


def calculate_tau_decay(step: int, total_steps: int, initial_tau: float = 1.0, final_tau: float = 0.0, decay_type: str = "cosine"):
    progress = step / total_steps
    if decay_type == "linear":
        tau =  initial_tau - progress * (initial_tau - final_tau)
    elif decay_type == "cosine":
        cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
        tau = final_tau + cosine_progress * (initial_tau - final_tau)
    elif decay_type == "exponential":
        decay_rate = (final_tau / initial_tau) ** (1 / total_steps)
        tau = initial_tau * (decay_rate ** step)
    else:
        raise ValueError(f"Unsupported decay_type: {decay_type}")
    
    return max(min(initial_tau, tau), final_tau)


def calculate_distillation_loss(
    loss_ce: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    if alpha == 0.0:
        return loss_ce
    
    loss_distill = F.kl_div(
        input=F.log_softmax(student_logits / temperature, dim=-1).flatten(0, -2),
        target=F.softmax(teacher_logits / temperature, dim=-1).flatten(0, -2),
        reduction="batchmean",
    ) * (temperature ** 2)

    if alpha == 1.0:
        return loss_distill
    else:
        return alpha * loss_distill + (1 - alpha) * loss_ce
