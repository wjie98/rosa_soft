import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext


__all__ = [
    "rosa_vds_ops",
]


def rosa_vds_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        win_size: int = 0,
        tau: float = 1.0,
        norm: bool = False,
):
    params = ROSA_VDS_Params(
        attn_mask=attn_mask,
        head_dim=head_dim,
        win_size=win_size,
        tau=tau, norm=norm,
    )
    return ROSA_VDS_Function.apply(query, key, value, params)


class ROSA_VDS_Params:
    def __init__(self,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        win_size: int = 0,
        tau: float = 1.0,
        norm: bool = False,
    ):
        if isinstance(attn_mask, Tensor):
            self.attn_mask = attn_mask.detach()
        else:
            self.attn_mask = None

        self.head_dim = head_dim
        self.win_size = win_size
        self.tau = tau
        self.norm = norm
        self.eps = 1e-6
        self.info: Dict[str, Tensor] = {}


class ROSA_VDS_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: ROSA_VDS_Params):
        x_hard, info = RosaContext().update(query, key, value, 0, inspect=True)
        params.info.update(info)

        ctx.save_for_backward(
            query.detach(),
            key.detach(),
            value.detach(),
        )
        ctx.saved_params = params
        return x_hard

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        params: ROSA_VDS_Params = ctx.saved_params

        with torch.enable_grad():
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

            x_soft = rosa_value_detach_sampling(
                query, key, value,
                length=params.info["length"],
                endpos=params.info["endpos"],
                attn_mask=params.attn_mask,
                head_dim=params.head_dim,
                win_size=params.win_size,
                tau=params.tau,
                eps=params.eps,
                norm=params.norm,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                outputs=x_soft,
                inputs=(query, key, value),
                grad_outputs=grad_output,
                retain_graph=False,
                only_inputs=True,
            )
        return grad_query, grad_key, grad_value, None


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    bsz, num_heads, seq_len, head_dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, -1, seq_len, head_dim)


def unfold_qk(xq: Tensor, xk: Tensor, win_size: int, head_dim: int):
    bsz, num_heads, seq_len, num_bits = xq.size()

    assert head_dim % num_bits == 0, f"head_dim must be divisible by num_bits, got {head_dim} % {num_bits} != 0"
    tok_size = head_dim // num_bits

    if win_size <= tok_size:
        xq = F.pad(xq, (0, 0, tok_size - 1, 0))
        xq = xq.unfold(-2, tok_size, 1).transpose(-2, -1)

        xk = F.pad(xk, (0, 0, tok_size, -1)) # predict next token
        xk = xk.unfold(-2, tok_size, 1).transpose(-2, -1)
    else:
        cols = torch.randperm(win_size, device=xq.device)[:tok_size].sort().values
        rows = torch.arange(0, seq_len, device=xk.device)
        inds_q = (rows[:, None] - cols[None, :]).clamp_min_(0)
        inds_k = (rows[:, None] - cols[None, :] - 1).clamp_min_(0) # predict next token

        xq = xq.index_select(-2, inds_q.view(-1)).view(bsz, num_heads, seq_len, tok_size, num_bits)
        xk = xk.index_select(-2, inds_k.view(-1)).view(bsz, num_heads, seq_len, tok_size, num_bits)
    
    return xq, xk


def gather_v(xv: Tensor, endpos: Tensor):
    bsz, num_heads, seq_len, num_bits = xv.size()
    with torch.no_grad():
        epos = endpos.view(bsz, num_heads, seq_len)
    
    mask = (epos >= 0).view(bsz, num_heads, seq_len, 1).type_as(xv)
    inds = (epos + 1).view(bsz, num_heads, seq_len, 1).expand_as(xv)
    return xv.gather(-2, inds) * mask # mask out the padding tokens


def rosa_value_detach_sampling(
        query: Tensor, key: Tensor, value: Tensor,
        length: Tensor, endpos: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        win_size: int = 0,
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

    xq, xk = unfold_qk(xq, xk, win_size=win_size, head_dim=head_dim)

    xq = xq.reshape(bsz, num_heads, seq_len, head_dim)
    xk = xk.reshape(bsz, num_heads, seq_len, head_dim)

    scale = 1.0 / math.sqrt(head_dim) / tau
    xo = F.scaled_dot_product_attention(
        xq, xk, xv.detach(), scale=scale,
        is_causal=True, attn_mask=attn_mask,
    )

    tv = gather_v(xv, endpos)
    return xo + tv

