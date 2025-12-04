import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext


__all__ = [
    "rosa_gqs_ops",
]


def rosa_gqs_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mismatch: int = 0,
        attn_mask: Optional[Tensor] = None,
        look_ahead: int = 1,
        head_dim: int = 64,
        tau: float = 1.0,
        eps: float = 1e-6,
        training: bool = False,
):
    cache = RosaContext()
    xq, xk, xv = cache.dispatch(query, key, value)

    if training:
        x_hard, info = cache.inspect(xq, xk, xv, mismatch)
        x_soft = rosa_guided_query_shaping(
            query, key, value,
            length=info["length"],
            attn_mask=attn_mask,
            look_ahead=look_ahead,
            head_dim=head_dim,
            tau=tau, eps=eps,
        )
        return x_soft + (x_hard - x_soft).detach()
    else:
        x_hard = cache.combine(xq, xk, xv, mismatch)
        return x_hard


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    bsz, num_heads, seq_len, head_dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, -1, seq_len, head_dim)

def unfold_qk(hidden_states: Tensor, win_size: int, offset: int = 0):
    hidden_states = F.pad(hidden_states, (0, 0, win_size - 1 + offset, - offset))
    hidden_states = hidden_states.unfold(-2, win_size, 1).transpose(-2, -1)
    return hidden_states

def rosa_guided_query_shaping(
        query: Tensor, key: Tensor, value: Tensor, length: Tensor,
        attn_mask: Optional[Tensor] = None, look_ahead: int = 1,
        head_dim: int = 64, tau: float = 1.0, eps: float = 1e-6,
        noise_ratio: float = 0.1, boost_strength: float = 1.0,
):
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    assert num_qk_bits * 2 <= head_dim, f"num_qk_bits must be <= head_dim / 2, got {num_qk_bits} > {head_dim} / 2"
    assert num_v_bits <= head_dim, f"num_v_bits must be <= head_dim, got {num_v_bits} > {head_dim}"

    xq = F.normalize(query, p=2, dim=-1, eps=eps)
    xk = F.normalize(key, p=2, dim=-1, eps=eps)
    xv = F.normalize(value, p=2, dim=-1, eps=eps)

    xq = torch.tanh(xq / tau)
    xk = torch.tanh(xk / tau)
    xv = torch.sigmoid(xv / tau)
    
    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    win_qk_size = head_dim // num_qk_bits

    xq = unfold_qk(xq, win_qk_size, offset=0)
    xk = unfold_qk(xk, win_qk_size, offset=1) # predict next token

    with torch.no_grad():
        # Look-ahead Masking
        lenq = length.view(bsz, num_heads, seq_len)
        mask = torch.arange(win_qk_size, dtype=lenq.dtype, device=lenq.device)
        mask = win_qk_size - 1 - mask < lenq.unsqueeze(-1) + look_ahead

        # Entropy Boosting
        boost = torch.clamp_min(lenq.float(), win_qk_size)
        boost = boost.log1p_().div_(math.log1p(win_qk_size) + eps)

        # We want Gate to start decaying significantly when Var == noise_threshold_var
        # Let exp(-Var / gamma) = 0.9 (i.e., Gate = 0.1, closed state)
        # => -Var / gamma = ln(0.9) approx -0.105
        # => gamma = Var / 0.105 approx 9.5 * Var
        expected_energy = 1.0 / num_qk_bits
        noise_threshold_var = expected_energy * noise_ratio
        gamma = max(9.5 * noise_threshold_var, 1e-6)

    xq_vars = xq.var(dim=-2, unbiased=False).mean(dim=-1)
    entropy_gate = 1 - torch.exp(xq_vars.mul(-1.0 / gamma))
    boost = 1.0 + (boost - 1.0) * boost_strength * entropy_gate
    boost = boost.view(bsz, num_heads, seq_len, 1) * mask

    xq = xq * boost.view(bsz, num_heads, seq_len, win_qk_size, 1).type_as(xq)
    
    xq = xq.reshape(bsz, num_heads, seq_len, head_dim)
    xk = xk.reshape(bsz, num_heads, seq_len, head_dim)

    scale = 1.0 / math.sqrt(head_dim) / tau
    output = F.scaled_dot_product_attention(
        xq, xk, xv, scale=scale,
        is_causal=True, attn_mask=attn_mask,
    )
    
    return output

