import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from rwkv_cuda.ops import *

__all__ = [
    "RWKV7_CLAMPW_CUDA",
    "RWKV7_STATE_CLAMPW_CUDA",
    "RWKV7_STATE_PASSING_CLAMPW_CUDA",
    "RWKV7_ALBATROSS_W0_FP16_DITHER_SEQ",
    "RWKV7_ALBATROSS_W0_FP16_DITHER_ONE",
]


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

def RWKV7_ALBATROSS_W0_FP16_DITHER_SEQ(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, elapsed_t: Tensor) -> Tuple[Tensor, Tensor]:
    B, T, C = r.size()
    HEAD_SIZE = s0.size(-1)
    r, w, k, v, a, b = [i.view(B, T, C // HEAD_SIZE, HEAD_SIZE) for i in [r, w, k, v, a, b]]

    assert all(i.is_contiguous() for i in [s0, r, w, k, v, a, b])

    sT = s0.clone()
    y = torch.empty_like(v)

    rwkv7_albatross_forward_w0_fp16_dither_seq(sT, r, w, k, v, a, b, y, elapsed_t)
    return y.view(B, T, C), sT


def RWKV7_ALBATROSS_W0_FP16_DITHER_ONE(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, elapsed_t: Tensor) -> Tuple[Tensor, Tensor]:
    B, C = r.size()
    HEAD_SIZE = s0.size(-1)
    r, w, k, v, a, b = [i.view(B, C // HEAD_SIZE, HEAD_SIZE) for i in [r, w, k, v, a, b]]

    assert all(i.is_contiguous() for i in [s0, r, w, k, v, a, b])

    sT = s0.clone()
    y = torch.empty_like(v)

    rwkv7_albatross_forward_w0_fp16_dither_one(sT, r, w, k, v, a, b, y, elapsed_t)
    return y.view(B, C), sT
