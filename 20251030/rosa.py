import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *


def _rosa_qkv_scan_score(x: Tensor, tau: float = 0.0) -> Tensor:
    B, H, _, T = x.size()
    r = torch.arange(T, dtype=torch.long, device=x.device)

    m = r.view(-1, 1) < r.view(1, -1)
    x = x.masked_fill(m.view(1, 1, T, T), 0)

    c = (r.view(1, 1, -1, 1) + r.view(1, 1, 1, -1) + 1) % T
    x = x.gather(-1, c.repeat(B, H, 1, 1))

    if tau <= 0.0:
        # dp[i, j] = a[i, j] ? dp[i-1, j-1] + 1 : 0
        x = x.long()
        a = x.cumsum(dim=-2)
        x = a - torch.where(x == 0, a, 0).cummax(dim=-2).values
    else:
        # dp[i, j] = cumsum(a[i, j]) - cummax(cumsum(a[i, j]) * (1 - a[i, j]))
        x = x.float()
        a = x.cumsum(dim=-2)
        x = a - (a * (1 - x)).cummax(dim=-2).values

    c = (r.view(1, 1, 1, -1) - r.view(1, 1, -1, 1) + T - 1) % T
    x = x.gather(-1, c.repeat(B, H, 1, 1))
    return x


def _rosa_qkv_attn_score(x: Tensor, tau: float = 0.0) -> Tensor:
    T = x.size(-1)
    r = torch.arange(T, dtype=torch.long, device=x.device)
    m = r.view(-1, 1) < r.view(1, -1)

    if tau <= 0.0:
        x = (x << 32) | r
        x = x.masked_fill(m.view(1, 1, T, T), -1)
        x = x.argmax(dim=-1)
        return F.one_hot(x, T)
    else:
        p = r.to(x.dtype)
        p = p.view(1, -1) / (p.view(-1, 1) + 1)
        x = x + p # add positional encoding, use the last one for "no-match"
        x = x.masked_fill(m.view(1, 1, T, T), -torch.inf)
        return F.softmax(x / tau, dim=-1)

def _rosa_qkv_core(iq_sm: Tensor, ik_sm: Tensor, iv_sm: Tensor, xv: Tensor, tau: float = 0.0) -> Tensor:
    a = iq_sm.permute(0, 2, 1, 3) @ ik_sm.permute(0, 2, 3, 1) # (B, H, T, T)
    a = _rosa_qkv_scan_score(a, tau=tau)                # (B, H, T, T)
    a = _rosa_qkv_attn_score(a, tau=tau).type_as(iv_sm) # (B, H, T, T)
    a = a @ iv_sm.permute(0, 2, 1, 3)                   # (B, H, T, V)
    x = a.type_as(xv) @ xv[None, ...]                   # (B, H, T, C)
    return x.permute(0, 2, 1, 3)                        # (B, T, H, C)

def rosa_qkv_ops(iq: Tensor, ik: Tensor, iv: Tensor, xv: Tensor, tau: float = 0.0):
    if tau <= 0.0:
        iq_sm = F.one_hot(iq.argmax(dim=-1), iq.size(-1)).to(torch.float32) # (B, T, H, V)
        ik_sm = F.one_hot(ik.argmax(dim=-1), ik.size(-1)).to(torch.float32) # (B, T, H, V)
        iv_sm = F.one_hot(iv.argmax(dim=-1), iv.size(-1)).to(torch.float32) # (B, T, H, V)
    else:
        iq_sm = torch.softmax(iq / tau, dim=-1).to(torch.float32) # (B, T, H, V)
        ik_sm = torch.softmax(ik / tau, dim=-1).to(torch.float32) # (B, T, H, V)
        iv_sm = torch.softmax(iv / tau, dim=-1).to(torch.float32) # (B, T, H, V)
    
    xv = xv.to(torch.float32)
    xo = _rosa_qkv_core(iq_sm, ik_sm, iv_sm, xv, tau=tau) # (B, T, H, C)
    return xo.type_as(xv)


class ROSA_QKV_LAYER(nn.Module):
    def __init__(self, dims: int, num_heads: int, head_dims: int, bias: bool = False, tau: float = 1e-1) -> None:
        super().__init__()
        assert dims % num_heads == 0

        self.num_heads = num_heads
        self.head_dims = head_dims

        self.wq = nn.Linear(dims, num_heads * head_dims, bias=bias)
        self.wk = nn.Linear(dims, num_heads * head_dims, bias=bias)
        self.wv = nn.Linear(dims, num_heads * head_dims, bias=bias)
        self.wo = nn.Parameter(torch.zeros(num_heads * head_dims, dims // num_heads)) # (H, V, D)
        self.tau = tau

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.size()
        iq = self.wq.forward(x).view(B, T, self.num_heads, self.head_dims) # (B, T, H, C)
        ik = self.wk.forward(x).view(B, T, self.num_heads, self.head_dims) # (B, T, H, C)
        iv = self.wv.forward(x).view(B, T, self.num_heads, self.head_dims) # (B, T, H, C)
        xv = self.wo.view(self.num_heads, self.head_dims, -1)              # (H, V, D / H)

        xo = rosa_qkv_ops(iq, ik, iv, xv, tau=self.tau) # (B, T, H, D / H)
        return xo.reshape(B, T, -1)


if __name__ == "__main__":
    B, T, H, V, C = 64, 8, 8, 4, 64

    iq = torch.randn(B, T, H, V).cuda()
    ik = torch.randn(B, T, H, V).cuda()
    iv = torch.randn(B, T, H, V).cuda()
    xv = torch.randn(H, V, C).cuda()

    hard = rosa_qkv_ops(iq, ik, iv, xv, 0.0)
    for tau in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0]:
        soft = rosa_qkv_ops(iq, ik, iv, xv, tau)
        p = (soft == hard).count_nonzero() / soft.numel()
        print(f"tau={tau}: \t{p:.2%}")
    
    net = ROSA_QKV_LAYER(64, 8, 2).cuda().bfloat16()
    x = torch.randn(B, T, 64).cuda().bfloat16().requires_grad_()
    net(x).sum().backward()
    print(x.grad.size())


