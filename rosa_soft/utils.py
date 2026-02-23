import torch
import torch.nn.functional as F

from torch import Tensor
from typing import *


@torch.no_grad()
def quantize(x: Tensor, schmitt_trigger: float = 0.0) -> Tensor:
    assert x.is_floating_point()
    *C, seq_len, num_bits = x.size()
    if num_bits <= 8:
        dtype = torch.uint8
    elif num_bits <= 16:
        dtype = torch.int16
    elif num_bits <= 32:
        dtype = torch.int32
    else:
        dtype = torch.int64

    r = torch.arange(num_bits, device=x.device)
    m = ((torch.abs(x) >= schmitt_trigger).to(dtype) << r).sum(dim=-1)
    x = ((x > 0).to(dtype) << r).sum(dim=-1)
    return x, m


@torch.no_grad()
def dequantize(x: Tensor, m: Optional[Tensor], v: Union[Tensor, int]) -> Tensor:
    assert not x.is_floating_point()
    
    if isinstance(v, Tensor):
        num_bits = v.size(-1)
    else:
        num_bits = int(v)

    r = torch.arange(num_bits, device=x.device)
    x = (x.unsqueeze(-1) >> r) & 1
    x = torch.where(x != 0, 1.0, -1.0)

    if isinstance(m, Tensor):
        assert not m.is_floating_point()
        m = (m.unsqueeze(-1) >> r) & 1
        x = torch.where(m != 0, x, 0.0)

    if isinstance(v, Tensor):
        x = x.type_as(v)
    else:
        x = x.float()

    return x


class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, eps: float):
        ctx.save_for_backward(x)
        ctx.saved_eps = eps
        return torch.where(x > 0, 1.0, -1.0)
    
    @staticmethod
    def backward(ctx, g: Tensor):
        x: Tensor = ctx.saved_tensors[0]
        eps: float = ctx.saved_eps

        # mag = x.abs().add_(eps)
        mag = x.abs().clamp_min_(eps)
        return g / mag, None


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


def decay_qk(xq: Tensor, xk: Tensor, decay_factor: Optional[float] = None):
    bsz, num_heads, seq_len, win_size, num_bits = xq.size()
    
    if win_size > 1:
        inds = torch.arange(win_size, device=xq.device)
        inds = win_size - 1 - inds

        if decay_factor is None: # dynamic
            wd = max(0.1, 1.0 / win_size) ** (1.0 / win_size)
        else:
            wd = max(0.1, min(decay_factor, 0.99))
        
        qk_w = torch.pow(wd, inds)
        qk_w = qk_w / qk_w.sum()
        qk_w_sqrt = torch.sqrt(qk_w) # [win_size]

        xq = xq * qk_w_sqrt.view(-1, 1).type_as(xq)
        xk = xk * qk_w_sqrt.view(-1, 1).type_as(xk)
    return xq, xk


def gather_x(x: Tensor, endpos: Tensor, offset: int = 1):
    bsz, num_heads, seq_len, num_bits = x.size()
    with torch.no_grad():
        epos = endpos.view(bsz, num_heads, seq_len)
    
    mask = (epos >= 0).view(bsz, num_heads, seq_len, 1).type_as(x)
    inds = (epos + offset).clamp(0, seq_len - 1)
    inds = inds.view(bsz, num_heads, seq_len, 1).expand_as(x)

    return x.gather(-2, inds) * mask # mask out the padding tokens
