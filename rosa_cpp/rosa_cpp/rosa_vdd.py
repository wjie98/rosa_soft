import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext


__all__ = [
    "rosa_vdd_ops",
]


def rosa_vdd_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        decay_factor: float = 0.45,
        tau: float = 1.0,
        norm: bool = False,
        training: bool = False,
        debug: bool = False,
):
    """
    Performs the ROSA operation using the VDD (Value Detach & Decay) mechanism.

    This operator implements a hybrid discrete-continuous attention mechanism designed to
    solve the "co-adaptation" problem in training discrete suffix automatons. It combines
    Hard ROSA (for precise structural retrieval) with Soft SUFA (for semantic search) using
    a novel gradient flow strategy.

    **Core Mechanism: VDD (Value Detach Decay)**
    The optimization landscape is decomposed into two decoupled tasks:
    1.  **Search (Q/K Optimization)**: Handled by a Soft Suffix Attention (SUFA) proxy. Crucially,
        the `value` tensor is **detached** in this branch. This prevents the model from learning
        a "blurry mean" value to minimize loss. Instead, gradients force Q and K to align geometrically
        to find the best existing V.
    2.  **Content (V Optimization)**: Handled by the Hard ROSA branch. The `value` tensor receives
        gradients *only* from the precise index selected by the Hard ROSA algorithm. This ensures
        V remains sharp and structurally aligned.

    Args:
        query (Tensor): Query tensor (B, H, T, D). Continuous representations.
        key (Tensor): Key tensor (B, H, T, D). Continuous representations.
        value (Tensor): Value tensor (B, H, T, D_v).
        attn_mask (Optional[Tensor]): Standard causal attention mask.
        head_dim (int): The dimension of the attention heads. Used to determine the window
            size for the geometric decay projection.
        decay_factor (float): The geometric decay rate (lambda) for the Suffix Attention proxy.
            A value < 0.5 (e.g., 0.45) enforces a strict hierarchy where matching the immediate
            suffix token is weighted higher than matching any number of distant past tokens.
        tau (float): Temperature scaling for the soft attention scores.
        norm (bool): If True, uses Spherical Optimization (F.normalize). If False, uses
            Hypercube Optimization (Clamp/Tanh). Defaults to True for better manifold properties.
        training (bool): If True, executes the VDD hybrid forward/backward pass. If False,
            executes only the efficient Hard ROSA inference pass.
        debug (bool): If True, performs consistency checks between hard and soft outputs.

    Returns:
        Tensor: The output tensor.
            - In inference: The exact result from the Discrete Suffix Automaton.
            - In training: A hybrid tensor where the value comes from Hard ROSA (for V updates)
              but the gradient direction for Q/K comes from Soft SUFA.
    """

    cache = RosaContext()
    xq, xk, xv = cache.dispatch(query, key, value)

    if training:
        x_hard, info = cache.inspect(xq, xk, xv, 0)
        x_soft = rosa_value_detach_decay(
            query, key, value,
            length=info["length"],
            endpos=info["endpos"],
            attn_mask=attn_mask,
            head_dim=head_dim,
            decay_factor=decay_factor,
            tau=tau, norm=norm,
        )
        
        if debug:
            assert torch.allclose(x_hard, x_soft)

        return x_soft + (x_hard - x_soft).detach() # ensure train and eval have same output
    else:
        x_hard = cache.combine(xq, xk, xv, 0)
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

def rosa_value_detach_decay(
        query: Tensor, key: Tensor, value: Tensor,
        length: Tensor, endpos: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        decay_factor: float = 0.45,
        tau: float = 1.0,
        eps: float = 1e-6,
        norm: bool = False,
):
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    assert num_qk_bits * 2 <= head_dim, f"num_qk_bits must be <= head_dim / 2, got {num_qk_bits} > {head_dim} / 2"
    assert num_v_bits <= head_dim, f"num_v_bits must be <= head_dim, got {num_v_bits} > {head_dim}"

    if norm:
        xq = F.normalize(query, p=2, dim=-1, eps=eps)
        xk = F.normalize(key, p=2, dim=-1, eps=eps)
        xv = F.normalize(value, p=2, dim=-1, eps=eps)
    else:
        xq = query.clamp(-6, 6)
        xk = key.clamp(-6, 6)
        xv = value.clamp(-6, 6)

        xq = xq + (torch.where(xq > 0, 1.0, -1.0) - xq).detach()
        xk = xk + (torch.where(xk > 0, 1.0, -1.0) - xk).detach()
        xv = xv + (torch.where(xv > 0, 1.0, -1.0) - xv).detach() # [-1, 1] is better
    
    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    win_qk_size = head_dim // num_qk_bits

    xq = unfold_qk(xq, win_qk_size, offset=0)
    xk = unfold_qk(xk, win_qk_size, offset=1) # predict next token

    with torch.no_grad():
        lenq = length.view(bsz, num_heads, seq_len)
        epos = endpos.view(bsz, num_heads, seq_len)

        inds = torch.arange(win_qk_size, dtype=lenq.dtype, device=lenq.device)
        inds = win_qk_size - 1 - inds

        qk_w_sqrt = torch.pow(decay_factor, inds).sqrt_()

    xq = xq * qk_w_sqrt.view(-1, 1).type_as(xq)
    xk = xk * qk_w_sqrt.view(-1, 1).type_as(xk)

    xq = xq.reshape(bsz, num_heads, seq_len, head_dim)
    xk = xk.reshape(bsz, num_heads, seq_len, head_dim)

    scale = 1.0 / tau if norm else 1.0 / num_qk_bits / tau
    boost = lenq.type_as(xq).log1p_().add_(1)
    xq = xq * boost.view(bsz, num_heads, seq_len, 1)

    xo = F.scaled_dot_product_attention(
        xq, xk, xv.detach(), scale=scale,
        is_causal=True, attn_mask=attn_mask,
    )

    index_m = epos.greater_equal(0).view(bsz, num_heads, seq_len, 1).type_as(xk)
    index_v = epos.add(1).view(bsz, num_heads, seq_len, 1).expand_as(xv)
    tv = xv.gather(dim=-2, index=index_v) * index_m # mask out the padding tokens

    output = xo + tv + (torch.where(tv > 0, 1.0, 0.0) - xo - tv).detach()

    return output

