import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

from concurrent.futures import ThreadPoolExecutor



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


class RosaEmbedding(nn.Module):
    def __init__(self, dims: int, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dims)
    
    def rosa_seq(self, x: Tensor) -> Tensor:
        t = x.to(dtype=torch.int64, device="cpu")
        t = torch.ops.torch_rosa.rosa_seq_fwd(t, t, -1)
        return t.type_as(x)
    
    def forward(self, x: Tensor):
        inds = self.rosa_seq(x)
        outs = self.emb(inds.clamp_min(0))
        outs = torch.where((inds >= 0)[..., None], outs, 0)
        return outs


class RosaSeqFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, v: Tensor, emb0: Tensor, emb1: Tensor, num_x_bits: int, num_v_bits: int, tau: float):
        B, T, _ = x.size()
        _x = (x > 0).view(B, T, -1, num_x_bits).permute(0, 2, 1, 3).to(torch.int64)
        _v = (v > 0).view(B, T, -1, num_v_bits).permute(0, 2, 1, 3).to(torch.int64)

        rx = torch.arange(num_x_bits, device=x.device)
        rv = torch.arange(num_v_bits, device=x.device)

        _x = torch.sum(_x << rx, dim=-1).to(dtype=torch.int64, device="cpu")
        _v = torch.sum(_v << rv, dim=-1).to(dtype=torch.int64, device="cpu")

        _y = torch.ops.torch_rosa.rosa_seq_fwd(_x.view(-1, T), _v.view(-1, T), 0).view(B, -1, T)
        
        m: Tensor = (_y.to(x.device)[..., None] >> rv) & 1
        m = m.permute(0, 2, 1, 3).bool().reshape(B, T, -1)

        x0 = x.view(B, T, -1, num_x_bits).permute(0, 2, 1, 3).to(dtype=torch.float32, device="cpu")
        e0 = emb0.to(dtype=torch.float32, device="cpu")
        e1 = emb1.to(dtype=torch.float32, device="cpu")
        ctx.save_for_backward(x0, e0, e1, _x, _v, _y, m)
        ctx.saved_tau = tau

        return torch.where(m, emb1, emb0)
    
    @staticmethod
    def backward(ctx, grad_y: Tensor):
        x0, e0, e1, _x, _v, _y, m = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        tau: float = ctx.saved_tau

        B, H, T = _y.size()
        dy = grad_y.view(B, T, H, -1).permute(0, 2, 1, 3).to(dtype=torch.float32, device="cpu")

        dx, dv = torch.ops.torch_rosa.rosa_seq_bwd(dy, x0, e0, e1, _x, _v, _y, 0, tau)

        dx = dx.type_as(grad_y).permute(0, 2, 1, 3).reshape(B, T, -1)
        dv = dv.type_as(grad_y).permute(0, 2, 1, 3).reshape(B, T, -1)

        demb0 = torch.where(m, 0, grad_y).sum(dim=(0, 1))
        demb1 = torch.where(m, grad_y, 0).sum(dim=(0, 1))

        return dx, dv, demb0, demb1, None, None, None


class RosaSeqLayer(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_bits: int, bias: bool = False, tau: float = 1e-1) -> None:
        super().__init__()

        assert dims % num_heads == 0

        self.num_heads = num_heads
        self.num_x_bits = num_bits
        self.num_v_bits = dims // num_heads

        self.wx = nn.Linear(dims, self.num_heads * self.num_x_bits, bias=bias)
        self.wv = nn.Linear(dims, self.num_heads * self.num_v_bits, bias=bias)

        self.emb0 = nn.Parameter(torch.full((dims,), -1e-5))
        self.emb1 = nn.Parameter(torch.full((dims,), +1e-5))

        self.tau = tau

    def forward(self, x: Tensor, tau: float | None = None) -> Tensor:
        tau = self.tau if tau is None else tau
        xx = self.wx(x)
        xv = self.wv(x)
        x = RosaSeqFunction.apply(xx, xv, self.emb0, self.emb1, self.num_x_bits, self.num_v_bits, tau)
        return x


class RosaQKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, emb0: Tensor, emb1: Tensor, num_x_bits: int, num_v_bits: int, tau: float):
        B, T, _ = q.size()

        _q = (q > 0).view(B, T, -1, num_x_bits).permute(0, 2, 1, 3).to(torch.int64)
        _k = (k > 0).view(B, T, -1, num_x_bits).permute(0, 2, 1, 3).to(torch.int64)
        _v = (v > 0).view(B, T, -1, num_v_bits).permute(0, 2, 1, 3).to(torch.int64)

        rx = torch.arange(num_x_bits, device=q.device)
        rv = torch.arange(num_v_bits, device=q.device)

        _q = torch.sum(_q << rx, dim=-1).to(dtype=torch.int64, device="cpu")
        _k = torch.sum(_k << rx, dim=-1).to(dtype=torch.int64, device="cpu")
        _v = torch.sum(_v << rv, dim=-1).to(dtype=torch.int64, device="cpu")

        _y = torch.ops.torch_rosa.rosa_qkv_fwd(_q.view(-1, T), _k.view(-1, T), _v.view(-1, T), 0).view(B, -1, T)

        m: Tensor = (_y.to(q.device)[..., None] >> rv) & 1
        m = m.permute(0, 2, 1, 3).bool().reshape(B, T, -1)

        q0 = q.view(B, T, -1, num_x_bits).permute(0, 2, 1, 3).to(dtype=torch.float32, device="cpu")
        k0 = k.view(B, T, -1, num_x_bits).permute(0, 2, 1, 3).to(dtype=torch.float32, device="cpu")
        e0 = emb0.to(dtype=torch.float32, device="cpu")
        e1 = emb1.to(dtype=torch.float32, device="cpu")

        ctx.save_for_backward(q0, k0, e0, e1, _q, _k, _v, _y, m)
        ctx.saved_tau = tau

        return torch.where(m, emb1, emb0)
    
    @staticmethod
    def backward(ctx, grad_y: Tensor):
        q0, k0, e0, e1, _q, _k, _v, _y, m = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        tau: float = ctx.saved_tau

        B, H, T = _y.size()
        dy = grad_y.view(B, T, H, -1).permute(0, 2, 1, 3).to(dtype=torch.float32, device="cpu")

        dq, dk, dv = torch.ops.torch_rosa.rosa_qkv_bwd(dy, q0, k0, e0, e1, _q, _k, _v, _y, 0, tau)

        dq = dq.type_as(grad_y).permute(0, 2, 1, 3).reshape(B, T, -1)
        dk = dk.type_as(grad_y).permute(0, 2, 1, 3).reshape(B, T, -1)
        dv = dv.type_as(grad_y).permute(0, 2, 1, 3).reshape(B, T, -1)

        demb0 = torch.where(m, 0, grad_y).sum(dim=(0, 1))
        demb1 = torch.where(m, grad_y, 0).sum(dim=(0, 1))

        return dq, dk, dv, demb0, demb1, None, None, None


class RosaQKVLayer(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_bits: int, bias: bool = False, tau: float = 1e-1) -> None:
        super().__init__()

        assert dims % num_heads == 0

        self.num_heads = num_heads
        self.num_x_bits = num_bits
        self.num_v_bits = dims // num_heads

        self.wq = nn.Linear(dims, self.num_heads * self.num_x_bits, bias=bias)
        self.wk = nn.Linear(dims, self.num_heads * self.num_x_bits, bias=bias)
        self.wv = nn.Linear(dims, self.num_heads * self.num_v_bits, bias=bias)

        self.emb0 = nn.Parameter(torch.full((dims,), -1e-5))
        self.emb1 = nn.Parameter(torch.full((dims,), +1e-5))

        self.tau = tau

    def forward(self, x: Tensor, tau: float | None = None) -> Tensor:
        tau = self.tau if tau is None else tau

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        x = RosaQKVFunction.apply(xq, xk, xv, self.emb0, self.emb1, self.num_x_bits, self.num_v_bits, tau)
        return x



if __name__ == "__main__":
    B, T, C, V = 8, 100, 256, 80
    emb = RosaEmbedding(C, V).cuda()
    x = torch.randint(0, V, (B, T)).cuda()

    x = emb(x)
    print(x)

    seq = RosaSeqLayer(C, 8, 8).cuda()
    x = seq(x)
    print(x)

    qkv = RosaQKVLayer(C, 8, 8).cuda()
    x = qkv(x)
    print(x)

    x.sum().backward()

    for name, p in emb.named_parameters():
        print(name, p.isnan().any())

    for name, p in seq.named_parameters():
        print(name, p.isnan().any())

    for name, p in qkv.named_parameters():
        print(name, p.isnan().any())
        