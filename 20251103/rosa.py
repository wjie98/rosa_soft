import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *



def _rosa_diag_to_cols(x: Tensor) -> Tensor:
    *B, _, T = x.size()
    r = torch.arange(T, dtype=torch.long, device=x.device)
    c = (r.view(-1, 1) + r.view(1, -1) + 1) % T
    x = x.gather(-1, c.expand(*B, T, T))
    return x

def _rosa_cols_to_diag(x: Tensor) -> Tensor:
    *B, _, T = x.size()
    r = torch.arange(T, dtype=torch.long, device=x.device)
    c = (r.view(1, -1) - r.view(-1, 1) + T - 1) % T
    x = x.gather(-1, c.expand(*B, T, T))
    return x

def _rosa_qkv_scan(x: Tensor) -> Tensor:
    x = _rosa_diag_to_cols(x)
    a = x.cumsum(dim=-2)
    x = a - (a * (1 - x)).cummax(dim=-2).values
    x = _rosa_cols_to_diag(x)
    return x

def _rosa_qkv_attn_hard(q: Tensor, k: Tensor, v: Tensor, bits_mode: bool):
    B, H, T, C = q.size()
    if bits_mode:
        c = torch.arange(C, device=q.device)
        q = torch.sum((q > 0).long() << c, dim=-1)
        k = torch.sum((k > 0).long() << c, dim=-1)
        v = (v > 0).long()
    else:
        q = q.argmax(dim=-1)
        k = k.argmax(dim=-1)
        v = v.argmax(dim=-1)

    x = (q.view(B, H, T, 1) == k.view(B, H, 1, T)).long().tril_(diagonal=-1)
    a = _rosa_qkv_scan(x)

    r = torch.arange(1, T + 1, dtype=a.dtype, device=a.device)
    r = ((a << 32) | r).tril_(diagonal=-1).argmax(dim=-1).add_(1).clamp_max_(T - 1) # (B, H, T)

    if bits_mode:
        r = r.view(B, H, T, 1).expand(*v.size())
        r = v.gather(-2, r) * x.max(dim=-1, keepdim=True).values
    else:
        r = v.gather(-1, r) * x.max(dim=-1).values
    return r

def _rosa_qkv_attn_soft(q: Tensor, k: Tensor, v: Tensor, bits_mode: bool, tau: Tensor):
    B, H, T, C = q.size()
    if bits_mode:
        q = torch.tanh(q / tau)
        k = torch.tanh(k / tau)
        v = torch.sigmoid(v / tau)

        x = q @ k.transpose(-1, -2) - (C - 1)
        x = torch.sigmoid(x / tau)
    else:
        q = torch.softmax(q / tau, dim=-1)
        k = torch.softmax(k / tau, dim=-1)
        v = torch.softmax(v / tau, dim=-1)

        x = q @ k.transpose(-1, -2)

    x = torch.tril(x, diagonal=-1)
    a = _rosa_qkv_scan(x)

    a = F.pad(a, (1, -1), value=0.0)

    r = torch.arange(T, dtype=a.dtype, device=a.device)
    a = a + r.view(1, -1) / (r.view(-1, 1) + 1)

    m = torch.ones(T, T, dtype=torch.bool, device=a.device).triu_(diagonal=1)
    a = (a / tau).masked_fill(m, -torch.inf)
    r = torch.softmax(a, dim=-1)
    r = r @ v
    
    g = x.max(dim=-1, keepdim=True).values
    if bits_mode:
        r = r * g
    else:
        z = torch.zeros(1, dtype=torch.long, device=r.device)
        r = r * g + (1 - g) * F.one_hot(z, r.size(-1)).type_as(r)
    return r

@torch.compile
def rosa_qkv_ops(
    q: Tensor, k: Tensor, v: Tensor,
    e0: Tensor, e1: Tensor | None = None,
    tau: Tensor | None = None,
) -> Tensor:
    B, H, T, _ = q.size()

    bits_mode = e1 is not None
    if bits_mode:
        _, C = e0.size()
        assert e0.size(0) == H
        assert v.size(-1) == C
        e0 = e0.view(1, H, 1, C)
        e1 = e1.view(1, H, 1, C)
    else:
        _, V, C = e0.size()
        assert e0.size(0) == H
        assert v.size(-1) == V
        e0 = e0.view(1, H, V, C)

    if tau is None:
        r = _rosa_qkv_attn_hard(q, k, v, bits_mode=bits_mode)
        if bits_mode:
            r = r.type_as(e0)
            o = r * e1 + (1 - r) * e0
        else:
            r = r.view(B, H, T, 1).expand(B, H, T, C)
            o = e0.expand(B, H, V, C).gather(-2, r)
    else:
        r = _rosa_qkv_attn_soft(q, k, v, bits_mode=bits_mode, tau=tau)
        if bits_mode:
            r = r.type_as(e0)
            o = r * e1 + (1 - r) * e0
        else:
            o = r.view(B, H, T, V) @ e0
    return o


class RosaAttention(nn.Module):
    def __init__(self,
        dims: int,
        num_heads: int,
        bits_mode: bool = True,
        bits_size: int = 8,
        bias: bool = False,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        assert dims % num_heads == 0

        self.num_heads = num_heads
        self.head_dims = dims // num_heads
        self.bits_mode = bits_mode
        self.bits_size = bits_size

        if self.bits_mode:
            self.wq = nn.Linear(dims, self.num_heads * self.bits_size, bias=bias)
            self.wk = nn.Linear(dims, self.num_heads * self.bits_size, bias=bias)
            self.wv = nn.Linear(dims, self.num_heads * self.head_dims, bias=bias)
            self.wo = nn.Linear(self.num_heads * self.head_dims, dims, bias=bias)

            self.e0 = nn.Parameter(torch.full((self.num_heads, self.bits_size), -1e-5))
            self.e1 = nn.Parameter(torch.full((self.num_heads, self.bits_size), +1e-5))
        else:
            self.wq = nn.Linear(dims, self.num_heads * self.head_dims, bias=bias)
            self.wk = nn.Linear(dims, self.num_heads * self.head_dims, bias=bias)
            self.wv = nn.Linear(dims, self.num_heads * self.head_dims, bias=bias)
            self.wo = nn.Linear(self.num_heads * self.head_dims, dims, bias=bias)

            self.e0 = nn.Parameter(torch.zeros(self.num_heads, self.head_dims, self.head_dims))
            self.e1 = None
        self.tau = tau

    def forward(self, x: Tensor, tau: float | None = None) -> Tensor:
        B, T, _ = x.size()
        xq = self.wq.forward(x).view(B, T, self.num_heads, -1).permute(0, 2, 1, 3)
        xk = self.wk.forward(x).view(B, T, self.num_heads, -1).permute(0, 2, 1, 3)
        xv = self.wv.forward(x).view(B, T, self.num_heads, -1).permute(0, 2, 1, 3)

        tau = self.tau if tau is None else tau
        tau = torch.full((1,), tau, dtype=x.dtype, device=x.device)
        xo = rosa_qkv_ops(xq, xk, xv, e0=self.e0, e1=self.e1, tau=tau)

        xo = xo.permute(0, 2, 1, 3).reshape(B, T, -1)
        xo = self.wo.forward(xo)
        return xo


if __name__ == "__main__":
    B, T, H, C, V = 4, 8, 2, 4, 5

    def samx_qkv_slow(qqq, kkk, vvv): # slow, only for reference
        """from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py
        """
        n=len(qqq); y=[-1]*n; s=2*n+1; t=[None]*s; f=[-1]*s; m=[0]*s; r=[-1]*s; t[0]={}; g=0; u=1; w=h=0; assert n==len(kkk)==len(vvv)
        for i,(q,k) in enumerate(zip(qqq,kkk)):
            p,x=w,h
            while p!=-1 and q not in t[p]: x=m[p] if x>m[p] else x; p=f[p]
            p,x=(t[p][q],x+1) if p!=-1 else (0,0); v=p
            while f[v]!=-1 and m[f[v]]>=x: v=f[v]
            while v!=-1 and (m[v]<=0 or r[v]<0): v=f[v]
            y[i]=vvv[r[v]+1] if v!=-1 else -1; w,h=p,x; j=u; u+=1; t[j]={}; m[j]=m[g]+1; p=g
            while p!=-1 and k not in t[p]: t[p][k]=j; p=f[p]
            if p==-1: f[j]=0
            else:
                d=t[p][k]
                if m[p]+1==m[d]: f[j]=d
                else:
                    b=u; u+=1; t[b]=t[d].copy(); m[b]=m[p]+1; f[b]=f[d]; r[b]=r[d]; f[d]=f[j]=b
                    while p!=-1 and t[p][k]==d: t[p][k]=b; p=f[p]
            v=g=j
            while v!=-1 and r[v]<i: r[v]=i; v=f[v]
        return [max(0,y) for y in y] # use "0" for both "no-match" and matched "0"

    def test_rosa_qkv_attn_hard(q, k, v, bits_mode = False):
        if bits_mode:
            q = torch.tensor(q).long().view(1, 1, -1, 1) >> torch.arange(16)
            k = torch.tensor(k).long().view(1, 1, -1, 1) >> torch.arange(16)
            v = torch.tensor(v).long().view(1, 1, -1, 1) >> torch.arange(16)

            q = q & 1
            k = k & 1
            v = v & 1

            x = _rosa_qkv_attn_hard(q, k, v, bits_mode=True)
            x = x << torch.arange(16)
            x = x.sum(dim=-1)
        else:
            q = torch.tensor(q).long().view(1, 1, -1)
            k = torch.tensor(k).long().view(1, 1, -1)
            v = torch.tensor(v).long().view(1, 1, -1)
            
            q = F.one_hot(q, 16)
            k = F.one_hot(k, 16)
            v = F.one_hot(v, 16)

            x = _rosa_qkv_attn_hard(q, k, v, bits_mode=False)

        return x.flatten().tolist()
    
    try:    
        for _ in range(10):
            q = torch.randint(0, 2, size=(8,)).tolist()
            k = torch.randint(0, 2, size=(8,)).tolist()
            v = torch.randint(0, 2, size=(8,)).tolist()

            o1 = torch.tensor(samx_qkv_slow(q, k, v))
            o2 = torch.tensor(test_rosa_qkv_attn_hard(q, k, v, bits_mode=False))
            o3 = torch.tensor(test_rosa_qkv_attn_hard(q, k, v, bits_mode=True))

            assert (o1 == o2).all()
            assert (o1 == o3).all()

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()

    try:
        net = RosaAttention(64, 8).cuda().bfloat16()
        x = torch.randn(B, T, 64).cuda().bfloat16().requires_grad_()
        net(x).sum().backward()
        assert not x.grad.isnan().any()
        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()
    
    print("bits_mode = True")
    q = torch.randn(B, H, T, C).cuda()
    k = torch.randn(B, H, T, C).cuda()
    v = torch.randn(B, H, T, C).cuda()

    e0 = torch.randn(H, C).cuda()
    e1 = torch.randn(H, C).cuda()

    hard = rosa_qkv_ops(q, k, v, e0, e1)
    for tau in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        _tau = torch.full((1,), tau, dtype=q.dtype, device=q.device)
        soft = rosa_qkv_ops(q, k, v, e0, e1, tau=_tau)
        p = (soft == hard).count_nonzero() / soft.numel()
        print(f"tau={tau}: \t{p:.2%}")
    print()
    
    print("bits_mode = False")
    q = torch.randn(B, H, T, C).cuda()
    k = torch.randn(B, H, T, C).cuda()
    v = torch.randn(B, H, T, V).cuda()

    e0 = torch.randn(H, V, C).cuda()

    hard = rosa_qkv_ops(q, k, v, e0)
    for tau in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        _tau = torch.full((1,), tau, dtype=q.dtype, device=q.device)
        soft = rosa_qkv_ops(q, k, v, e0, tau=_tau)
        p = (soft == hard).count_nonzero() / soft.numel()
        print(f"tau={tau}: \t{p:.2%}")
    print()

