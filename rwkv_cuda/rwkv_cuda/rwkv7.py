import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from rwkv_cuda.ops import *

__all__ = [
    "RWKV7_CLAMPW_CUDA",
    "RWKV7_STATE_CLAMPW_CUDA",
    "RWKV7_STATE_PASSING_CLAMPW_CUDA",
    "RWKV7_ALBATROSS_W0_FP16_DITHER",
]

@torch.no_grad()
def RWKV7_RNN_OP(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    B, T, C = r.size()
    HEAD_SIZE = s0.size(-1)
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    DTYPE = v.dtype

    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

    if s0 is None:
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    else:
        state = s0.float()

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)
    
    return out.view(B, T, C).to(dtype=DTYPE), state.type_as(s0)

######################################################################################################

class _RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor):
        B, T, H, N = r.size()

        CHUNK_LEN = 16
        assert T % CHUNK_LEN == 0
        assert all(i.is_contiguous() for i in [r, w, k, v, a, b])

        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=r.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)

        rwkv7_clampw_forward(r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        return y
    
    @staticmethod
    def backward(ctx, dy: Tensor):
        assert all(i.is_contiguous() for i in [dy])
        
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
        
        rwkv7_clampw_backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
        return dr, dw, dk, dv, da, db
    
def RWKV7_CLAMPW_CUDA(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, HEAD_SIZE: int = 64) -> Tensor:
    B, T, C = r.size()
    r, w, k, v, a, b = [i.view(B, T, C // HEAD_SIZE, HEAD_SIZE) for i in [r, w, k, v, a, b]]
    return _RWKV7_CLAMPW_CUDA_OP.apply(r, w, k, v, a, b).view(B, T, C)

######################################################################################################

class _RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor):
        B, T, H, N = r.size()

        CHUNK_LEN = 16
        assert T % CHUNK_LEN == 0
        assert all(i.is_contiguous() for i in [s0, r, w, k, v, a, b])
        assert s0.dtype==torch.float32

        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=r.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)

        rwkv7_state_clampw_forward(s0, r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        return y
    
    @staticmethod
    def backward(ctx, dy: Tensor):
        assert all(i.is_contiguous() for i in [dy])

        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        B, T, H, N = r.size()
        
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
        ds0 = torch.empty(B, H, N, N, dtype=torch.float32, device=r.device)
        
        rwkv7_state_clampw_backward(r, w, k, v, a, b, dy, s, sa, ds0, dr, dw, dk, dv, da, db)
        return ds0, dr, dw, dk, dv, da, db
    
def RWKV7_STATE_CLAMPW_CUDA(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> Tensor:
    B, T, C = r.size()
    HEAD_SIZE = s0.size(-1)
    r, w, k, v, a, b = [i.view(B, T, C // HEAD_SIZE, HEAD_SIZE) for i in [r, w, k, v, a, b]]
    return _RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0, r, w, k, v, a, b).view(B, T, C)

######################################################################################################

class _RWKV7_STATE_PASSING_CLAMPW_CUDA_OP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor):
        B, T, H, N = r.size()

        CHUNK_LEN = 16
        assert T % CHUNK_LEN == 0
        assert all(i.is_contiguous() for i in [s0, r, w, k, v, a, b])
        assert s0.dtype==torch.float32

        y = torch.empty_like(v)
        sT = torch.empty_like(s0)
        s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=r.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)
        
        rwkv7_statepassing_clampw_forward(s0, r, w, k, v, a, b, y, sT, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        return y, sT
    
    @staticmethod
    def backward(ctx, dy: Tensor, dsT: Tensor):
        assert all(i.is_contiguous() for i in [dy, dsT])
        assert dsT.dtype==torch.float32

        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        ds0, dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [dsT, r, w, k, v, a, b]]

        rwkv7_statepassing_clampw_backward(r, w, k, v, a, b, dy, dsT, s, sa, ds0, dr, dw, dk, dv, da, db)
        return ds0, dr, dw, dk, dv, da, db
    
def RWKV7_STATE_PASSING_CLAMPW_CUDA(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    B, T, C = r.size()
    HEAD_SIZE = s0.size(-1)
    r, w, k, v, a, b = [i.view(B, T, C // HEAD_SIZE, HEAD_SIZE) for i in [r, w, k, v, a, b]]
    y, sT = _RWKV7_STATE_PASSING_CLAMPW_CUDA_OP.apply(s0, r, w, k, v, a, b)
    return y.view(B, T, C), sT

######################################################################################################

@torch.no_grad()
def RWKV7_ALBATROSS_W0_FP16_DITHER(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, elapsed_t: Tensor, inplace: bool = False) -> Tuple[Tensor, Tensor]:
    B, T, C = r.size()
    HEAD_SIZE = s0.size(-1)
    r, w, k, v, a, b = [i.view(B, T, C // HEAD_SIZE, HEAD_SIZE) for i in [r, w, k, v, a, b]]

    assert all(i.is_contiguous() for i in [s0, r, w, k, v, a, b])

    sT = s0 if inplace else s0.clone()
    y = torch.empty_like(v)

    rwkv7_albatross_forward_w0_fp16_dither(sT, r, w, k, v, a, b, y, elapsed_t)
    return y.view(B, T, C), sT

