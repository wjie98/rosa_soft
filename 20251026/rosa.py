import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

import triton
import triton.language as tl



def load_torch_rosa():
    from torch.utils.cpp_extension import load
    load(
        name="torch_rosa",
        sources=["rosa.cpp"],
        extra_cflags=["-O3", "-fopenmp"],
        is_python_module=False,
        verbose=True,
    )

load_torch_rosa()


@triton.jit
def _rosa_dp_combine_op(g0, a0, g1, a1):
    g_ = g1 * g0
    a_ = g1 * a0 + a1
    return g_, a_

@triton.jit
def _rosa_scan_fwd_kernel(
    x_ptr, y_ptr, T,
    stride_x_b, stride_x_r, stride_x_c,
    stride_y_b, stride_y_r, stride_y_c,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)

    x_ptr_b = x_ptr + pid_b * stride_x_b
    y_ptr_b = y_ptr + pid_b * stride_y_b

    carry_g = 1.0
    carry_a = 0.0

    for block_start in range(0, T - pid_r, BLOCK_SIZE_T):
        off_c = block_start + tl.arange(0, BLOCK_SIZE_T)
        off_r = off_c + pid_r

        mask = off_r < T

        g = tl.load(x_ptr_b + off_r * stride_x_r + off_c * stride_x_c, mask=mask, other=1.0)
        a = tl.where(mask, g, 0.0)

        _g, _a = _rosa_dp_combine_op(carry_g, carry_a, g, a)
        _g = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _g, g)
        _a = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _a, a)

        _g, _a = tl.associative_scan((_g, _a), axis=0, combine_fn=_rosa_dp_combine_op)
        tl.store(y_ptr_b + off_r * stride_y_r + off_c * stride_y_c, _a, mask=mask)

        _g, _a = tl.reduce((g, a), axis=0, combine_fn=_rosa_dp_combine_op)
        carry_g, carry_a = _rosa_dp_combine_op(carry_g, carry_a, _g, _a)

@triton.jit
def _rosa_scan_bwd_kernel(
    x_ptr, y_ptr, g_ptr, o_ptr, T,
    stride_x_b, stride_x_r, stride_x_c,
    stride_y_b, stride_y_r, stride_y_c,
    stride_g_b, stride_g_r, stride_g_c,
    stride_o_b, stride_o_r, stride_o_c,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)

    x_ptr_b = x_ptr + pid_b * stride_x_b
    y_ptr_b = y_ptr + pid_b * stride_y_b
    g_ptr_b = g_ptr + pid_b * stride_g_b
    o_ptr_b = o_ptr + pid_b * stride_o_b

    carry_a = 1.0
    carry_b = 0.0

    M = tl.cdiv(T - pid_r, BLOCK_SIZE_T)
    for i in range(M):
        block_start = (M - 1 - i) * BLOCK_SIZE_T
        off_c = block_start + BLOCK_SIZE_T - 1 - tl.arange(0, BLOCK_SIZE_T)
        off_r = off_c + pid_r

        mask = off_r < T

        a = tl.load(x_ptr_b + (off_r + 1) * stride_x_r + (off_c + 1) * stride_x_c, mask=mask & (off_r < T - 1), other=1.0)
        b = tl.load(g_ptr_b + off_r * stride_g_r + off_c * stride_g_c, mask=mask, other=0.0)
        
        _a, _b = _rosa_dp_combine_op(carry_a, carry_b, a, b)
        _a = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _a, a)
        _b = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _b, b)

        _a, _b = tl.associative_scan((_a, _b), axis=0, combine_fn=_rosa_dp_combine_op)

        y = tl.load(y_ptr_b + (off_r - 1) * stride_y_r + (off_c - 1) * stride_y_c, mask=mask & (off_r > pid_r), other=0.0)
        tl.store(o_ptr_b + off_r * stride_o_r + off_c * stride_o_c, (y + 1) * _b, mask=mask)

        _a, _b = tl.reduce((a, b), axis=0, combine_fn=_rosa_dp_combine_op)
        carry_a, carry_b = _rosa_dp_combine_op(carry_a, carry_b, _a, _b)

def _rosa_scan_fwd_hard(x: Tensor) -> Tensor:
    B, _, T = x.size()
    r = torch.arange(T, dtype=torch.long, device=x.device)

    m = r.view(-1, 1) < r.view(1, -1)
    x = x.masked_fill(m.view(1, T, T), 0)

    c = (r.view(1, -1, 1) + r.view(1, 1, -1) + 1) % T
    x = x.gather(-1, c.repeat(B, 1, 1))

    a = x.cumsum(dim=-2)
    x = a - torch.where(x == 0, a, 0).cummax(dim=-2).values

    c = (r.view(1, 1, -1) - r.view(1, -1, 1) + T - 1) % T
    x = x.gather(-1, c.repeat(B, 1, 1))
    return x


class RosaScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        *B, _, T = x.size()
        x = x.contiguous().view(-1, T, T)

        if not x.is_floating_point():
            y = _rosa_scan_fwd_hard(x)
        else:
            y = torch.zeros_like(x)

            BLOCK_SIZE_T = triton.next_power_of_2(min(T, 1024))
            
            _rosa_scan_fwd_kernel[(x.size(0), T)](
                x, y, T,
                x.stride(0), x.stride(1), x.stride(2),
                y.stride(0), y.stride(1), y.stride(2),
                BLOCK_SIZE_T=BLOCK_SIZE_T,
            )

            ctx.save_for_backward(x, y)
        return y.view(*B, T, T)
    
    @staticmethod
    def backward(ctx, g: Tensor) -> Tensor:
        x, y = cast(Tuple[Tensor, ...], ctx.saved_tensors)

        *B, _, T = g.size()

        g = g.contiguous().view(-1, T, T)
        o = torch.zeros_like(x)

        BLOCK_SIZE_T = triton.next_power_of_2(min(T, 1024))

        _rosa_scan_bwd_kernel[(x.size(0), T)](
            x, y, g, o, T,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            g.stride(0), g.stride(1), g.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            BLOCK_SIZE_T=BLOCK_SIZE_T,
        )
        return o.view(*B, T, T)

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

    if tau <= 0.0:
        a = RosaScanFunction.apply(a.long()) # dp[i, j] = a[i, j] ? dp[i-1, j-1] + 1 : 0
    else:
        a = RosaScanFunction.apply(a)        # dp[i, j] = dp[i-1, j-1] * a[i, j] + a[i, j]

    a = _rosa_qkv_attn_score(a, tau=tau).type_as(iv_sm) # (B, H, T, T)
    a = a @ iv_sm.permute(0, 2, 1, 3)                   # (B, H, T, V)
    x = a.type_as(xv) @ xv[None, ...]                   # (B, H, T, C)
    return x.permute(0, 2, 1, 3)                        # (B, T, H, C)

def rosa_qkv_ops(iq: Tensor, ik: Tensor, iv: Tensor, xv: Tensor, tau: float = 0.0):
    if tau <= 0.0:
        iq_sm = F.one_hot(iq.argmax(dim=-1), iq.size(-1)).type_as(iq) # (B, T, H, V)
        ik_sm = F.one_hot(ik.argmax(dim=-1), ik.size(-1)).type_as(iq) # (B, T, H, V)
        iv_sm = F.one_hot(iv.argmax(dim=-1), iv.size(-1)).type_as(iq) # (B, T, H, V)
    else:
        iq_sm = torch.softmax(iq / tau, dim=-1) # (B, T, H, V)
        ik_sm = torch.softmax(ik / tau, dim=-1) # (B, T, H, V)
        iv_sm = torch.softmax(iv / tau, dim=-1) # (B, T, H, V)
    return _rosa_qkv_core(iq_sm, ik_sm, iv_sm, xv, tau=tau) # (B, T, H, C)


class ROSA_SEQ(nn.Module):
    def __init__(self, batch_size: int = 1, reserve_size: int = 0):
        super().__init__()

        self.objs = torch.zeros((batch_size,), dtype=torch.int64, device="cpu")
        self._reserve_size = reserve_size

        self.reset()
    
    def __del__(self):
        torch.ops.torch_rosa.rosa_seq_free(self.objs)
    
    def reset(self):
        torch.ops.torch_rosa.rosa_seq_init(self.objs, self._reserve_size)
        assert self.objs.count_nonzero() == self.objs.numel(), "Failed to initialize ROSA_SEQ context"
    
    def append(self, x: Tensor, v: Tensor, u: int) -> Tensor:
        assert x.dtype == torch.int64, "Input tensor must be of type int64"
        assert x.dim() == 1, "Input tensor must be 1D"
        assert x.size() == v.size(), "Input tensor sizes must match"
        assert x.numel() == self.objs.numel(), "Input tensor size must match batch size"
        return torch.ops.torch_rosa.rosa_seq_append(self.objs, x, v, u)
    
    def extend(self, x: Tensor, v: Tensor, u: int) -> Tensor:
        assert x.dtype == torch.int64, "Input tensor must be of type int64"
        assert x.dim() == 2, "Input tensor must be 2D"
        assert x.size() == v.size(), "Input tensor sizes must match"
        assert x.size(0) == self.objs.numel(), "Input tensor size must match batch size"
        return torch.ops.torch_rosa.rosa_seq_extend(self.objs, x, v, u)
    
    @staticmethod
    def run(x: Tensor, v: Tensor, u: int) -> Tensor:
        return torch.ops.torch_rosa.rosa_seq_run(x, v, u)


class ROSA_QKV(nn.Module):
    def __init__(self, batch_size: int = 1, reserve_size: int = 0):
        super().__init__()

        self.objs = torch.zeros((batch_size,), dtype=torch.int64, device="cpu")
        self._reserve_size = reserve_size
        self.reset()
    
    def __del__(self):
        torch.ops.torch_rosa.rosa_qkv_free(self.objs)
    
    def reset(self):
        torch.ops.torch_rosa.rosa_qkv_init(self.objs, self._reserve_size)
        assert self.objs.count_nonzero() == self.objs.numel(), "Failed to initialize ROSA_QKV context"
    
    def append(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        assert q.dtype == torch.int64, "Input tensor must be of type int64"
        assert q.dim() == 1, "Input tensor must be 1D"
        assert q.size() == k.size() and q.size() == v.size(), "Input tensor sizes must match"
        assert q.numel() == self.objs.numel(), "Input tensor size must match batch size"
        return torch.ops.torch_rosa.rosa_qkv_append(self.objs, q, k, v)
    
    def extend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        assert q.dtype == torch.int64, "Input tensor must be of type int64"
        assert q.dim() == 2, "Input tensor must be 2D"
        assert q.size() == k.size() and q.size() == v.size(), "Input tensor sizes must match"
        assert q.size(0) == self.objs.numel(), "Input tensor size must match batch size"
        return torch.ops.torch_rosa.rosa_qkv_extend(self.objs, q, k, v)
    
    @staticmethod
    def run(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return torch.ops.torch_rosa.rosa_qkv_run(q, k, v)


class ROSA_SEQ_Embedding(nn.Module):
    def __init__(self, vocab_size: int, C: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, C)
    
    def forward(self, x: Tensor):
        _x = x.to(dtype=torch.long, device="cpu")
        inds = ROSA_SEQ.run(_x, _x, -1).type_as(x)
        outs = self.emb(inds.clamp_min(0))
        outs = torch.where((inds >= 0)[..., None], outs, 0)
        return outs


class ROSA_QKV_LAYER(nn.Module):
    def __init__(self, dims: int, n_heads: int, bias: bool = False, tau: float = 1e-1) -> None:
        super().__init__()
        assert dims % n_heads == 0

        self.num_heads = n_heads
        self.head_dims = dims // self.num_heads

        self.wq = nn.Linear(dims, dims, bias=bias)
        self.wk = nn.Linear(dims, dims, bias=bias)
        self.wv = nn.Linear(dims, dims, bias=bias)
        self.wo = nn.Parameter(torch.zeros(dims, self.head_dims)) # (H, V, C)
        self.tau = tau

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.size()
        iq = self.wq.forward(x).view(B, T, self.num_heads, self.head_dims) # (B, T, H, C)
        ik = self.wk.forward(x).view(B, T, self.num_heads, self.head_dims) # (B, T, H, C)
        iv = self.wv.forward(x).view(B, T, self.num_heads, self.head_dims) # (B, T, H, C)
        xv = self.wo.view(self.num_heads, self.head_dims, self.head_dims)

        xo = rosa_qkv_ops(iq, ik, iv, xv, tau=self.tau) # (B, T, H, C)
        return xo.reshape(B, T, -1)


if __name__ == "__main__":
    B, T, H, V, C = 64, 8, 8, 4, 64

    iq = torch.randn(B, T, H, V).cuda()
    ik = torch.randn(B, T, H, V).cuda()
    iv = torch.randn(B, T, H, V).cuda()
    xv = torch.randn(H, V, C).cuda()

    hard = rosa_qkv_ops(iq, ik, iv, xv, 0.0)
    for tau in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0]:
        soft = rosa_qkv_ops(iq, ik, iv, xv, tau)
        p = (soft == hard).count_nonzero() / soft.numel()
        print(f"tau={tau}: \t{p:.2%}")
    
    net = ROSA_QKV_LAYER(64, 8).cuda()
    x = torch.randn(B, T, 64).cuda().requires_grad_()
    net(x).sum().backward()
    print(x.grad.size())


