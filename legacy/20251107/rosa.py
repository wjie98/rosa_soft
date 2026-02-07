import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *


__all__ = [
    "rosa_qkv_ops",
    "RapidOnlineSuffixAutomaton",
]


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

def _rosa_qkv_attn_hard(q: Tensor, k: Tensor, v: Tensor, bits_mode: bool, attn_mask: Tensor | None):
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
    if attn_mask is not None:
        x = x.masked_fill_(~attn_mask, 0)

    a = _rosa_qkv_scan(x)
    a = F.pad(a, (1, -1), value=0)

    r = torch.arange(1, T + 1, dtype=a.dtype, device=a.device)
    r = ((a << 32) | r).argmax(dim=-1) # (B, H, T)

    g = x.max(dim=-1).values
    if bits_mode:
        r = r.view(B, H, T, 1).expand(*v.size())
        r = v.gather(-2, r) * g[..., None]
    else:
        r = v.gather(-1, r) * g
    return r

def _rosa_qkv_attn_soft(q: Tensor, k: Tensor, v: Tensor, bits_mode: bool, attn_mask: Tensor | None, tau: Tensor | float):
    B, H, T, C = q.size()
    if bits_mode:
        q = torch.tanh(q / tau)
        k = torch.tanh(k / tau)
        v = torch.sigmoid(v / tau)

        x = q @ k.transpose(-1, -2) - (C - 1) * (1 - tau)
        x = torch.sigmoid(x / tau)
    else:
        q = torch.softmax(q / tau, dim=-1)
        k = torch.softmax(k / tau, dim=-1)
        v = torch.softmax(v / tau, dim=-1)

        x = q @ k.transpose(-1, -2)

    x = torch.tril(x, diagonal=-1)
    if attn_mask is not None:
        x = x.masked_fill(~attn_mask, 0.0)
        
    a = _rosa_qkv_scan(x)
    a = F.pad(a, (1, -1), value=0.0)

    r = torch.arange(T, dtype=a.dtype, device=a.device)
    a = a + r.view(1, -1) / (r.view(-1, 1) + 1)

    m = torch.ones(T, T, dtype=torch.bool, device=a.device).triu_(diagonal=1)
    a = a.masked_fill(m, -torch.inf)
    
    r = torch.softmax(a.float() / tau, dim=-1).type_as(a)
    r = r @ v
    
    g = x.max(dim=-1, keepdim=True).values
    if bits_mode:
        r = r * g
    else:
        z = torch.zeros(1, dtype=torch.long, device=r.device)
        u = F.one_hot(z, r.size(-1)).type_as(r)
        r = r * g + (1 - g) * u
    return r

def rosa_qkv_ops(
    q: Tensor, k: Tensor, v: Tensor,
    e0: Tensor, e1: Tensor | None = None,
    attn_mask: Tensor | None = None,
    tau: Tensor | float | None = None,
) -> Tensor:
    """
    Performs the ROSA (Rapid Online Suffix Automaton) QKV operation.

    This function implements a differentiable, DP-based version of ROSA-style
    sequence matching. It supports two main modes for token representation
    ("categorical" and "bits") and two execution modes ("hard" and "soft").

    Args:
        q (Tensor): Query tensor of shape `(B, H, T, C)`. Represents query vectors
            or logits at each time step.
        k (Tensor): Key tensor of shape `(B, H, T, C)`. Represents key vectors
            or logits at each time step.
        v (Tensor): Value tensor. Its shape and meaning depend on the mode:
            - In categorical mode, it's a tensor of one-hot vectors or logits
              of shape `(B, H, T, V)`, where V is the vocabulary size.
            - In bits mode, it's a tensor of bit vectors or logits of shape
              `(B, H, T, C_out)`, where C is the number of value bits.
        e0 (Tensor): Primary embedding tensor. Its role depends on the mode:
            - In categorical mode (`e1=None`), this is the main embedding table
              of shape `(B, H, V, C_out)`, mapping token indices/distributions
              to output embeddings.
            - In bits mode (`e1` is provided), this is the embedding for "bit 0",
              typically of shape `(B, H, T, C_out)`.
        e1 (Tensor | None, optional): Secondary embedding tensor. If provided,
            activates "bits mode". This is the embedding for "bit 1", with a
            shape compatible with `e0`. Defaults to None.
        attn_mask (Tensor | None, optional): An attention mask of shape
            `(B, H, T, T)`. Boolean tensor where `False` values indicate
            positions to be masked out from matching. Defaults to None.
        tau (Tensor | float | None, optional): The temperature parameter. If None,
            the function operates in "hard" mode (using argmax/thresholding).
            If a float or tensor, it operates in "soft" (differentiable) mode.
            Defaults to None.

    Returns:
        Tensor: The output tensor of shape `(B, H, T, C_out)`, where `C_out` is
            the dimensionality of the output embeddings.

    Modes of Operation:
    1.  **Categorical Mode** (`e1` is `None`):
        - `q`, `k`, `v` are treated as logits over a vocabulary of size `V`.
        - `e0` is an embedding table of shape `(..., V, C_out)`.
        - **Hard (`tau=None`)**: `argmax` is used on `q`, `k`, `v` to get discrete
          token IDs. The operator finds the longest matching historical key sequence
          for each query and returns the value token ID that followed the match.
          This ID is used to `gather` the final embedding from `e0`.
        - **Soft (`tau` is provided)**: `softmax` is applied to `q`, `k`, `v`. The
          operator produces a probability distribution over the vocabulary `V`, which
          is then matrix-multiplied with `e0` to get a weighted average embedding.

    2.  **Bits Mode** (`e1` is not `None`):
        - `q`, `k`, `v` are treated as logits for `C` independent bits.
        - `e0` and `e1` are embeddings for bit "0" and bit "1" respectively.
        - **Hard (`tau=None`)**: A threshold (`> 0`) is used to get discrete bits.
          The operator returns a tensor of result bits `r` (0s and 1s). The final
          output is an interpolation: `r * e1 + (1 - r) * e0`.
        - **Soft (`tau` is provided)**: `sigmoid` or `tanh` is applied. The operator
          produces a soft probability for each bit, which is then used to
          interpolate between `e0` and `e1`.
    """

    B, H, T, _ = q.size()

    if e1 is None:
        bits_mode = False
        _, _, V, C = e0.size()
        assert e0.size(0) in {1, B}
        assert e0.size(1) in {1, H}
    else:
        bits_mode = True
        _, _, _, C = e0.size()
        assert e0.size(0) in {1, B}
        assert e0.size(1) in {1, H}
        assert e0.size(2) in {1, T}
        assert e0.size(3) == v.size(3)
    
    if tau is None:
        r = _rosa_qkv_attn_hard(q, k, v, bits_mode=bits_mode, attn_mask=attn_mask)
    else:
        r = _rosa_qkv_attn_soft(q, k, v, bits_mode=bits_mode, attn_mask=attn_mask, tau=tau)
    
    if bits_mode:
        r = r.type_as(e0)
        o = r * e1 + (1 - r) * e0
    elif tau is None:
        r = r.view(B, H, T, 1).expand(B, H, T, C)
        o = e0.expand(B, H, V, C).gather(-2, r)
    else:
        o = r.view(B, H, T, V) @ e0
    return o


class _RosaState:
    __slots__ = ("endpos", "length", "suffix_link", "transitions")

    def __init__(self):
        self.endpos = -1
        self.length = 0
        self.suffix_link: Optional[_RosaState] = None
        self.transitions: Dict[int, _RosaState] = {}


class RapidOnlineSuffixAutomaton:
    def __init__(self):
        self.query_states: List[int] = []
        self.key_states: List[int] = []
        self.value_states: List[int] = []

        self._root: _RosaState = _RosaState()
        self._last_query: _RosaState = self._root
        self._last_key: _RosaState = self._root
    
    def append(self, query: int, key: int, value: int, default: int = -1) -> int:
        i = len(self.value_states)

        self.query_states.append(query)
        self.key_states.append(key)
        self.value_states.append(value)

        r = _RosaState()
        r.length = self._last_key.length + 1

        p = self._last_key
        while (p is not None) and (key not in p.transitions):
            p.transitions[key] = r
            p = p.suffix_link
        
        if p is None:
            r.suffix_link = self._root
        else:
            q = p.transitions[key]

            if p.length + 1 == q.length:
                r.suffix_link = q
            else:
                u = _RosaState()
                u.endpos = q.endpos
                u.length = p.length + 1
                u.suffix_link = q.suffix_link
                u.transitions.update(q.transitions)

                q.suffix_link = u
                r.suffix_link = u

                while (p is not None) and (p.transitions.get(key) is q):
                    p.transitions[key] = u
                    p = p.suffix_link

        j = -1
        
        p = self._last_query
        while (p is not None) and (query not in p.transitions):
            p = p.suffix_link
        
        if p is None:
            self._last_query = self._root
        else:
            self._last_query = p.transitions[query]

            p = self._last_query
            while p is not None:
                if p.length > 0 and p.endpos >= 0:
                    j = p.endpos + 1
                    break
                p = p.suffix_link
        
        self._last_key = r
        while (r is not None) and (r.endpos < i):
            r.endpos = i
            r = r.suffix_link
        
        return self.value_states[j] if j >= 0 else default
    
    def extend(
            self,
            query_states: List[int],
            key_states: List[int],
            value_states: List[int],
            default: int = -1,
    ) -> List[int]:
        outs = []
        for q, k, v in zip(query_states, key_states, value_states):
            x = self.append(q, k, v, default)
            outs.append(x)
        return outs


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

            q = F.pad(q, (0, 0, 0, 10), value=0)
            k = F.pad(k, (0, 0, 0, 10), value=0)
            v = F.pad(v, (0, 0, 0, 10), value=0)

            m = torch.ones(q.size(2))
            m[-10:] = 0
            m = m.bool().view(1, 1, 1, -1)

            x = _rosa_qkv_attn_hard(q, k, v, bits_mode=True, attn_mask=m)[..., :-10, :]
            x = x << torch.arange(16)
            x = x.sum(dim=-1)
        else:
            q = torch.tensor(q).long().view(1, 1, -1)
            k = torch.tensor(k).long().view(1, 1, -1)
            v = torch.tensor(v).long().view(1, 1, -1)
            
            q = F.one_hot(q, 16)
            k = F.one_hot(k, 16)
            v = F.one_hot(v, 16)

            q = F.pad(q, (0, 0, 0, 10), value=0)
            k = F.pad(k, (0, 0, 0, 10), value=0)
            v = F.pad(v, (0, 0, 0, 10), value=0)

            m = torch.ones(q.size(2))
            m[-10:] = 0
            m = m.bool().view(1, 1, 1, -1)

            x = _rosa_qkv_attn_hard(q, k, v, bits_mode=False, attn_mask=m)[..., :-10]

        return x.flatten().tolist()
    
    try:    
        for _ in range(10):
            q = torch.randint(0, 2, size=(8,)).tolist()
            k = torch.randint(0, 2, size=(8,)).tolist()
            v = torch.randint(0, 2, size=(8,)).tolist()

            r = RapidOnlineSuffixAutomaton()

            o1 = torch.tensor(samx_qkv_slow(q, k, v))
            o2 = torch.tensor(r.extend(q, k, v, 0))
            o3 = torch.tensor(test_rosa_qkv_attn_hard(q, k, v, bits_mode=False))
            o4 = torch.tensor(test_rosa_qkv_attn_hard(q, k, v, bits_mode=True))

            assert (o1 == o2).all()
            assert (o1 == o3).all()
            assert (o1 == o4).all()

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()

    try:
        q = k = v = torch.randn(1, 8, 128, 64).requires_grad_()
        e0 = torch.randn(1, 8, 1, 64).requires_grad_()
        e1 = torch.randn(1, 8, 1, 64).requires_grad_()
        e2 = torch.randn(1, 8, 64, 32).requires_grad_()

        rosa_qkv_ops(q, k, v, e0, e1, tau=0.1).sum().backward()
        rosa_qkv_ops(q, k, v, e2, tau=0.1).sum().backward()

        assert not q.grad.isnan().any()
        assert not k.grad.isnan().any()
        assert not v.grad.isnan().any()

        assert not e0.grad.isnan().any()
        assert not e1.grad.isnan().any()
        assert not e2.grad.isnan().any()

        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()
    
    print("bits_mode = True")
    q = torch.randn(B, H, T, C).cuda()
    k = torch.randn(B, H, T, C).cuda()
    v = torch.randn(B, H, T, C).cuda()
    m = torch.randint(0, 2, (B, 1, T, T)).cuda().bool()

    e0 = torch.full((1, H, 1, C), -1).cuda()
    e1 = torch.full((1, H, 1, C), +1).cuda()

    hard = rosa_qkv_ops(q, k, v, e0, e1, attn_mask=m)
    for tau in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        _tau = torch.full((1,), tau, dtype=q.dtype, device=q.device)
        soft = rosa_qkv_ops(q, k, v, e0, e1, tau=_tau, attn_mask=m)
        p = (soft == hard).count_nonzero() / soft.numel()
        print(f"tau={tau}: \t{p:.2%}")
    print()
    
    print("bits_mode = False")
    q = torch.randn(B, H, T, C).cuda()
    k = torch.randn(B, H, T, C).cuda()
    v = torch.randn(B, H, T, V).cuda()
    m = torch.randint(0, 2, (B, 1, T, T)).cuda().bool()

    e0 = torch.randn(1, H, V, C).cuda()

    hard = rosa_qkv_ops(q, k, v, e0, attn_mask=m)
    for tau in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        _tau = torch.full((1,), tau, dtype=q.dtype, device=q.device)
        soft = rosa_qkv_ops(q, k, v, e0, tau=_tau, attn_mask=m)
        p = (soft == hard).count_nonzero() / soft.numel()
        print(f"tau={tau}: \t{p:.2%}")
    print()

