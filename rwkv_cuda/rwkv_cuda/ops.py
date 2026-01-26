import torch
from torch import Tensor
from typing import Tuple

__all__ = [
    "rwkv7_clampw_forward",
    "rwkv7_clampw_backward",
    "rwkv7_state_clampw_forward",
    "rwkv7_state_clampw_backward",
    "rwkv7_statepassing_clampw_forward",
    "rwkv7_statepassing_clampw_backward",
    "rwkv7_albatross_forward_w0_fp16_dither",
]


def rwkv7_clampw_forward(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, y: Tensor, s: Tensor, sa: Tensor) -> None:
    return torch.ops.rwkv_cuda.rwkv7_clampw_forward(r, w, k, v, a, b, y, s, sa)

def rwkv7_clampw_backward(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, dy: Tensor, s: Tensor, sa: Tensor, dr: Tensor, dw: Tensor, dk: Tensor, dv: Tensor, da: Tensor, db: Tensor) -> None:
    return torch.ops.rwkv_cuda.rwkv7_clampw_backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)

def rwkv7_state_clampw_forward(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, y: Tensor, s: Tensor, sa: Tensor) -> None:
    return torch.ops.rwkv_cuda.rwkv7_state_clampw_forward(s0, r, w, k, v, a, b, y, s, sa)

def rwkv7_state_clampw_backward(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, dy: Tensor, s: Tensor, sa: Tensor, ds0: Tensor, dr: Tensor, dw: Tensor, dk: Tensor, dv: Tensor, da: Tensor, db: Tensor) -> None:
    return torch.ops.rwkv_cuda.rwkv7_state_clampw_backward(r, w, k, v, a, b, dy, s, sa, ds0, dr, dw, dk, dv, da, db)

def rwkv7_statepassing_clampw_forward(s0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, y: Tensor, sT: Tensor, s: Tensor, sa: Tensor) -> None:
    return torch.ops.rwkv_cuda.rwkv7_statepassing_clampw_forward(s0, r, w, k, v, a, b, y, sT, s, sa)

def rwkv7_statepassing_clampw_backward(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, dy: Tensor, dsT: Tensor, s: Tensor, sa: Tensor, ds0: Tensor, dr: Tensor, dw: Tensor, dk: Tensor, dv: Tensor, da: Tensor, db: Tensor) -> None:
    return torch.ops.rwkv_cuda.rwkv7_statepassing_clampw_backward(r, w, k, v, a, b, dy, dsT, s, sa, ds0, dr, dw, dk, dv, da, db)

def rwkv7_albatross_forward_w0_fp16_dither(state: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, y: Tensor, elapsed_t: Tensor) -> None:
    return torch.ops.rwkv_cuda.rwkv7_albatross_forward_w0_fp16_dither(state, r, w, k, v, a, b, y, elapsed_t)

