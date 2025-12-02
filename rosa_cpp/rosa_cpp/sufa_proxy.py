import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext


__all__ = [
    "rosa_att_ops",
]

def rosa_att_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mismatch: int = 0,
        attn_mask: Optional[Tensor] = None,
        suffix_head_dim: int = 128,
        tau: float = 1.0,
        training: bool = False,
):
    cache = RosaContext()
    xq, xk, xv = cache.dispatch(query, key, value)

    if training:
        x_soft = sufa_bits_soft_ops(
            query, key, value, attn_mask=attn_mask,
            suffix_head_dim=suffix_head_dim, tau=tau,
        )

        x_hard = cache.combine(xq, xk, xv, mismatch)
        return x_soft + (x_hard - x_soft).detach()
    else:
        x_hard = cache.combine(xq, xk, xv, mismatch)
        return x_hard


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    bsz, n_kvs, seq_len, dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kvs, n_rep, seq_len, dim)
    return hidden_states.reshape(bsz, n_kvs * n_rep, seq_len, dim)


def sufa_bits_soft_ops(
        query: Tensor, key: Tensor, value: Tensor,
        attn_mask: Optional[Tensor] = None,
        suffix_head_dim: int = 128, tau: float = 1.0,
):
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    assert num_qk_bits * 2 <= suffix_head_dim, "num_qk_bits must be <= head_dim / 2"
    assert num_v_bits <= suffix_head_dim, "num_v_bits must be <= head_dim"

    xq = torch.tanh(query / tau)
    xk = torch.tanh(key / tau)
    xv = torch.sigmoid(value / tau)
    n_rep = num_heads // num_kv_heads

    if n_rep > 1:
        xk = repeat_kv(xk, n_rep)
        xv = repeat_kv(xv, n_rep)

    win_qk_size = suffix_head_dim // num_qk_bits

    xq = F.pad(xq, (0, 0, win_qk_size - 1, 0), value=0.0).unfold(-2, win_qk_size, 1) \
        .transpose(-1, -2).reshape(bsz, -1, seq_len, suffix_head_dim)
    
    # predict next token
    xk = F.pad(xk, (0, 0, win_qk_size, -1), value=0.0).unfold(-2, win_qk_size, 1) \
        .transpose(-1, -2).reshape(bsz, -1, seq_len, suffix_head_dim)

    scale = 1.0 / math.sqrt(suffix_head_dim) / tau
    output = F.scaled_dot_product_attention(
        xq, xk, xv, scale=scale,
        is_causal=True, attn_mask=attn_mask,
    )
    
    return output

