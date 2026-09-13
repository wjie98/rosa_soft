"""Dense QKV 4-bit adapter; keep the surrounding model's projections and FFN."""

import torch
from torch import nn

from rosa_soft import rosa_soft


class Rosa4Bit(nn.Module):
    def __init__(self, width, op=rosa_soft):
        super().__init__()
        if type(width) is not int or width <= 0 or width % 4:
            raise ValueError("width must be a positive multiple of four")
        self.op = op
        self.emb = nn.Parameter(torch.ones(1, 1, width))

    def forward(self, q, k, v):
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("Q/K/V must have matching [B,T,C] shapes")
        if q.size(-1) != self.emb.size(-1):
            raise ValueError("input width must match emb")
        shape = (*q.shape[:2], q.size(-1) // 4, 4)
        y = self.op(*(x.reshape(shape) for x in (q, k, v)))
        # Keep the amplitude outside sign quantization and preserve AMP output dtype.
        return (y.flatten(-2) * self.emb).to(y.dtype)
