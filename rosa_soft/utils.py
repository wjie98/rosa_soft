import torch
import torch.nn.functional as F

from torch import Tensor
from typing import *


def quantize(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        quant_mode: str = "tanh",
        quant_scale: Optional[float] = None,
):
    if quant_scale is None:
        quant_scale = 1.0
    
    assert quant_scale >= 1.0, "quant_scale must be >= 1.0"
    
    if quant_scale != 1.0:
        query = query * quant_scale
        key = key * quant_scale
        value = value * quant_scale
        
    if quant_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quant_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    elif quant_mode == "cubic":
        xq = torch.tanh(query.pow(3) + 0.1 * query)
        xk = torch.tanh(key.pow(3) + 0.1 * key)
        xv = torch.tanh(value.pow(3) + 0.1 * value)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}, expected 'tanh', 'soft' or 'cubic'")

    xq = (torch.where(query > 0, 1.0, -1.0) - query).detach() + query
    xk = (torch.where(key > 0, 1.0, -1.0) - key).detach() + key
    xv = (torch.where(value > 0, 1.0, -1.0) - value).detach() + value


    # xq = query
    # xk = key
    # xv = value

    # xq = F.normalize(xq.float(), dim=-1, p=2).type_as(xq) * xq.size(-1)
    # xk = F.normalize(xk.float(), dim=-1, p=2).type_as(xk) * xk.size(-1)
    # xv = F.normalize(xv.float(), dim=-1, p=2).type_as(xv) * xv.size(-1)

    return xq, xk, xv

# def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
#     bsz, num_heads, seq_len, head_dim = hidden_states.size()
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_heads, n_rep, seq_len, head_dim)
#     return hidden_states.reshape(bsz, -1, seq_len, head_dim)


# def unfold_qk(hidden_states: Tensor, win_size: int, offset: int = 0):
#     hidden_states = F.pad(hidden_states, (0, 0, win_size - 1 + offset, - offset))
#     hidden_states = hidden_states.unfold(-2, win_size, 1).transpose(-2, -1)
#     return hidden_states


# def decay_qk(xq: Tensor, xk: Tensor, decay_factor: Optional[float] = None):
#     bsz, num_heads, seq_len, win_size, num_bits = xq.size()
    
#     if win_size > 1:
#         inds = torch.arange(win_size, device=xq.device)
#         inds = win_size - 1 - inds

#         if decay_factor is None: # dynamic
#             wd = max(0.1, 1.0 / win_size) ** (1.0 / win_size)
#         else:
#             wd = max(0.1, min(decay_factor, 0.99))
        
#         qk_w = torch.pow(wd, inds)
#         qk_w = qk_w / qk_w.sum()
#         qk_w_sqrt = torch.sqrt(qk_w) # [win_size]

#         xq = xq * qk_w_sqrt.view(-1, 1).type_as(xq)
#         xk = xk * qk_w_sqrt.view(-1, 1).type_as(xk)
#     return xq, xk


# def gather_x(x: Tensor, endpos: Tensor, offset: int = 1):
#     bsz, num_heads, seq_len, num_bits = x.size()
#     with torch.no_grad():
#         epos = endpos.view(bsz, num_heads, seq_len)
    
#     mask = (epos >= 0).view(bsz, num_heads, seq_len, 1).type_as(x)
#     inds = (epos + offset).clamp(0, seq_len - 1)
#     inds = inds.view(bsz, num_heads, seq_len, 1).expand_as(x)

#     return x.gather(-2, inds) * mask # mask out the padding tokens
