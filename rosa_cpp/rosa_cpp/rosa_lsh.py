import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext


__all__ = [
    "rosa_lsh_ops",
]


def rosa_lsh_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        win_size: int = 128,
        tau: float = 1.0,
        norm: bool = False,
):
    params = ROSA_LSH_Params(
        attn_mask=attn_mask,
        head_dim=head_dim,
        win_size=win_size,
        tau=tau, norm=norm,
    )
    return ROSA_LSH_Function.apply(query, key, value, params)


class ROSA_LSH_Params:
    def __init__(self,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        win_size: int = 128,
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


class ROSA_LSH_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: ROSA_LSH_Params):
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
        params: ROSA_LSH_Params = ctx.saved_params

        with torch.enable_grad():
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

            x_soft = rosa_value_detach_hashing(
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


def hash_kernel_1d(hidden_states: Tensor, channels: int, win_size: int):
    dtype, device = hidden_states.dtype, hidden_states.device

    decays = (1 - 1.0 / channels) ** torch.arange(channels, dtype=dtype, device=device)
    decays = decays * (1 - 1.0 / win_size)
    decays = decays.view(-1, 1) ** (win_size - 1 - torch.arange(win_size, dtype=dtype, device=device))
    
    kernel = torch.randn(channels, win_size, dtype=dtype, device=device)
    kernel = kernel * decays
    kernel = F.normalize(kernel, p=2, dim=-1, eps=1e-6)
    kernel = kernel.view(channels, 1, win_size)
    return kernel


def hash_kernel_2d(hidden_states: Tensor, channels: int, win_size: int):
    dtype, device = hidden_states.dtype, hidden_states.device
    bsz, num_heads, seq_len, num_bits = hidden_states.size()

    decays = (1 - 1.0 / channels) ** torch.arange(channels, dtype=dtype, device=device)
    decays = decays * (1 - 1.0 / win_size)
    decays = decays.view(-1, 1) ** (win_size - 1 - torch.arange(win_size, dtype=dtype, device=device))
    decays = decays.view(channels, win_size, 1)
    
    kernel = torch.randn(channels, win_size, num_bits, dtype=dtype, device=device)
    kernel = kernel * decays
    kernel = F.normalize(kernel, p=2, dim=-1, eps=1e-6)
    kernel = kernel.view(channels, 1, win_size, num_bits)
    return kernel


def hash_qk(hidden_states: Tensor, kernel: Tensor, offset: int = 0):
    bsz, num_heads, seq_len, num_bits = hidden_states.size()

    if kernel.dim() == 3: # 1D
        channels, _ , win_size = kernel.size()

        hidden_states = hidden_states.permute(0, 1, 3, 2).reshape(-1, 1, seq_len)
        hidden_states = F.pad(hidden_states, (win_size - 1 + offset, - offset))

        hidden_states = F.conv1d(hidden_states, kernel)
    
        hidden_states = hidden_states.reshape(bsz, num_heads, num_bits, channels, seq_len)
        hidden_states = hidden_states.permute(0, 1, 4, 3, 2)

    elif kernel.dim() == 4: # 2D
        channels, _, win_size, num_bits = kernel.size()

        hidden_states = F.pad(hidden_states, (0, 0, win_size - 1 + offset, - offset))
        hidden_states = hidden_states.reshape(bsz * num_heads, 1, -1, num_bits)
        
        hidden_states = F.conv2d(hidden_states, kernel)
        
        hidden_states = hidden_states.reshape(bsz, num_heads, channels, seq_len, num_bits)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4)
    else:
        raise ValueError(f"Invalid kernel dimension: {kernel.size()}")
    
    return hidden_states


def unfold_qk(hidden_states: Tensor, win_size: int, offset: int = 0):
    hidden_states = F.pad(hidden_states, (0, 0, win_size - 1 + offset, - offset))
    hidden_states = hidden_states.unfold(-2, win_size, 1).transpose(-2, -1)
    return hidden_states


def gather_v(xv: Tensor, endpos: Tensor):
    bsz, num_heads, seq_len, num_bits = xv.size()
    with torch.no_grad():
        epos = endpos.view(bsz, num_heads, seq_len)
    
    mask = (epos >= 0).view(bsz, num_heads, seq_len, 1).type_as(xv)
    inds = (epos + 1).view(bsz, num_heads, seq_len, 1).expand_as(xv)
    return xv.gather(-2, inds) * mask # mask out the padding tokens


def rosa_value_detach_hashing(
        query: Tensor, key: Tensor, value: Tensor,
        length: Tensor, endpos: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_dim: int = 64,
        win_size: int = 128,
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

    channels = head_dim // num_qk_bits
    win_size = max(win_size, channels)
    if channels < win_size:
        kernel = hash_kernel_1d(xq, channels=channels, win_size=win_size)
        xq = hash_qk(xq, kernel, offset=0)
        xk = hash_qk(xk, kernel, offset=1) # predict next token
    else:
        xq = unfold_qk(xq, win_size=channels, offset=0)
        xk = unfold_qk(xk, win_size=channels, offset=1) # predict next token
        
    xq = xq.reshape(bsz, num_heads, seq_len, head_dim)
    xk = xk.reshape(bsz, num_heads, seq_len, head_dim)

    scale = 1.0 / math.sqrt(head_dim) / tau
    xo = F.scaled_dot_product_attention(
        xq, xk, xv.detach(), scale=scale,
        is_causal=True, attn_mask=attn_mask,
    )

    tv = gather_v(xv, endpos)
    return xo + tv

