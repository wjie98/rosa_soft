# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch._dynamo as dynamo

# import math
# from torch import Tensor
# from typing import *

# from transformers.utils import logging
# logger = logging.get_logger(__name__)


# __all__ = [
#     "SuffixConfig",
#     "SuffixAttention",
# ]



# from .base import RosaConfig, RosaBase


# class SuffixConfig(RosaConfig):
#     def __init__(
#             self,
#             hidden_size: int = 512,
#             num_heads: int = 8,
#             num_key_value_heads: int = 2,
#             num_query_key_bits: int = 8,
#             num_value_bits: int = 8,
#             bias: bool = False,
#             bits_norm_eps: float = 1e-6,
#             max_attention_head_size: int = 256,
#     ) -> None:
#         super().__init__(
#             hidden_size=hidden_size,
#             num_heads=num_heads,
#             num_key_value_heads=num_key_value_heads,
#             num_query_key_bits=num_query_key_bits,
#             num_value_bits=num_value_bits,
#             bias=bias,
#             bits_norm_eps=bits_norm_eps,
#         )

#         assert max_attention_head_size % 16 == 0, f"max_attention_head_size must be multiple of 16, got {max_attention_head_size}"
#         self.max_attention_head_size = max_attention_head_size


# class SuffixAttention(RosaBase):
#     def __init__(
#             self,
#             config: SuffixConfig,
#             layer_idx: int,
#     ):
#         super().__init__(config=config)
#         assert config.hidden_size % config.num_heads == 0

#         self.layer_idx = layer_idx
#         self.num_heads = config.num_heads
#         self.num_kv_heads = config.num_key_value_heads
#         self.num_qk_bits = config.num_query_key_bits
#         self.num_v_bits = config.num_value_bits

#         self.n_rep = self.num_heads // self.num_kv_heads

#         self.win_qk_size = config.max_attention_head_size // self.num_qk_bits

#         self.win_v_size = config.max_attention_head_size // self.num_v_bits

#         assert self.num_qk_bits * self.win_qk_size % 16 == 0
#         assert self.num_v_bits * self.win_v_size % 16 == 0

#         bias = config.bias
#         self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.num_qk_bits, bias=bias)
#         self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_qk_bits, bias=bias)
#         self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_v_bits, bias=bias)
#         self.o_proj = nn.Linear(self.num_heads * self.num_v_bits * self.win_v_size, config.hidden_size, bias=bias)

#     def _attn_bias(self,
#             hidden_states: Tensor,
#             attention_mask: Optional[Tensor] = None,
#             sharpness: float = 8192.0,
#     ):
#         bsz, seq_len, _ = hidden_states.size()

#         attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril_(-1)
#         if attention_mask is not None:
#             attn_mask = attn_mask & attention_mask
        
#         r = torch.arange(0, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
#         attn_bias = 1 - torch.log(r.view(-1, 1) - r.view(1, -1) + 1).clamp_min_(1.0) / math.log(sharpness)
#         attn_bias = attn_bias.clamp_min_(0.0).expand_as(attn_mask).clone()

#         attn_bias[~attn_mask] = -torch.inf
#         return attn_bias
    
#     def forward(
#             self,
#             hidden_states: Tensor,
#             attention_mask = None,
#             past_key_values = None,
#             **kwargs
#     ):
#         bsz, seq_len, _ = hidden_states.size()

#         qk_dim = self.win_qk_size * self.num_qk_bits
#         v_dim = self.win_v_size * self.num_v_bits

#         attn_tau = self.get_buffer("attn_tau")
#         bits_tau = self.get_buffer("bits_tau")

#         query_states: Tensor = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.num_qk_bits)
#         key_states: Tensor = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.num_qk_bits)
#         value_states: Tensor = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.num_v_bits)
        
#         # query_states = torch.tanh(self.norm_bits(query_states) / bits_tau).transpose(1, 2)
#         # key_states = torch.tanh(self.norm_bits(key_states) / bits_tau).transpose(1, 2)
#         # value_states = torch.tanh(self.norm_bits(value_states) / bits_tau).transpose(1, 2)

#         query_states = self.sign_bits(query_states).transpose(1, 2)
#         key_states = self.sign_bits(key_states).transpose(1, 2)
#         value_states = self.sign_bits(value_states).transpose(1, 2)

#         if past_key_values is not None:
#             raise NotImplementedError
#         else:
#             query_states = F.pad(query_states, (0, 0, self.win_qk_size - 1, 0), value=0.0)
#             key_states = F.pad(key_states, (0, 0, self.win_qk_size - 1, 0), value=0.0)
#             value_states = F.pad(value_states, (0, 0, self.win_v_size - 1, 0), value=0.0)

#             query_states = query_states.unfold(-2, self.win_qk_size, 1).transpose(-2, -1).reshape(bsz, self.num_heads, seq_len, qk_dim)
#             key_states = key_states.unfold(-2, self.win_qk_size, 1).transpose(-2, -1).reshape(bsz, self.num_kv_heads, seq_len, qk_dim)
#             value_states = value_states.unfold(-2, self.win_v_size, 1).transpose(-2, -1).reshape(bsz, self.num_kv_heads, seq_len, v_dim)

#             value_states = F.pad(value_states, (0, 0, -1, 1)) # recall next token

#             # attn_bias = self._attn_bias(hidden_states, attention_mask) / attn_tau
#             # with dynamo.config.patch(capture_scalar_outputs=True):
#             #     scale = 1.0 / math.sqrt(qk_dim) / attn_tau.item()
#             #     output = F.scaled_dot_product_attention(
#             #         query_states, key_states, value_states, scale=scale,
#             #         attn_mask=attn_bias, enable_gqa=True,
#             #     )

#             output = ArgmaxSTEFunction.apply(query_states, key_states, value_states)
        
#         output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
#         output = self.o_proj(output)
#         return output
    
#     @classmethod
#     def apply_adapter_to_modules_(
#         cls,
#         module: nn.Module,
#         config: RosaConfig,
#         target_modules: str = r".*self_attn",
#         autocast_dtype: bool = True,
#     ):
#         adapter_factory = lambda i, m: cls(config, layer_idx=i)
#         super().apply_adapter_to_modules_(
#             module=module,
#             adapter_factory=adapter_factory,
#             adapter_name="sufa_attn",
#             target_modules=target_modules,
#             autocast_dtype=autocast_dtype,
#         )


# class ArgmaxSTEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(
#             ctx,
#             query_states: Tensor,
#             key_states: Tensor,
#             value_states: Tensor,
#     ):
#         bsz, q_heads, seq_len, dim = query_states.size()
#         _, kv_heads, _, _ = key_states.size()

#         ctx.save_for_backward(query_states, key_states, value_states)

#         n_rep = q_heads // kv_heads
#         key_states = RosaBase.repeat_kv(key_states, n_rep)
#         value_states = RosaBase.repeat_kv(value_states, n_rep)

#         attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(dim)

#         attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=query_states.device).tril_(-1)
#         attn_scores = attn_scores.masked_fill(~attn_mask, -torch.inf)

#         max_val = torch.max(attn_scores, dim=-1, keepdim=True).values
#         indices = torch.arange(seq_len, device=attn_scores.device)
#         indices = torch.where(attn_scores == max_val, indices.view(1, 1, 1, -1), -1).max(dim=-1).values

#         attn_scores = F.one_hot(indices, num_classes=seq_len).to(attn_scores.dtype)
#         output = torch.matmul(attn_scores, value_states)
#         output[:, :, 0, :] = 0.0  # first token has no previous token to attend to
#         return output

#     @staticmethod
#     def backward(ctx, grad_output: Tensor):
#         query_states, key_states, value_states = ctx.saved_tensors
#         bsz, q_heads, seq_len, dim = query_states.size()
#         _, kv_heads, _, _ = key_states.size()

#         n_rep = q_heads // kv_heads

#         attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=query_states.device).tril_(-1)
#         with torch.enable_grad():
#             query_states.requires_grad_(True)
#             key_states.requires_grad_(True)
#             value_states.requires_grad_(True)
        
#             output_proxy = F.scaled_dot_product_attention(
#                 query_states, key_states, value_states,
#                 attn_mask=attn_mask, enable_gqa=True,
#             )

#             grads = torch.autograd.grad(
#                 outputs=output_proxy,
#                 inputs=(query_states, key_states, value_states),
#                 grad_outputs=grad_output,
#             )

#             grad_query, grad_key, grad_value = grads

#         return grad_query, grad_key, grad_value
    