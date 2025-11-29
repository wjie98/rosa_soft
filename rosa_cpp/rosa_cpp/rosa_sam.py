import torch

from torch import Tensor
from typing import *

from functools import lru_cache
from contextlib import contextmanager

from .ops import *


__all__ = [
    "RosaContext",
]


class RosaContext:
    def __init__(self):
        self.ctx = None
        self.device = None
        self.output_dtype = None

    def update(self, query: Tensor, key: Tensor, value: Tensor, mismatch: int = 0) -> Tensor:
        xq, xk, xv = self.dispatch(query=query, key=key, value=value)
        output = self.combine(xq=xq, xk=xk, xv=xv, mismatch=mismatch)
        return output
    
    def _init_ctx(self, query: Tensor, key: Tensor, value: Tensor):
        bsz, num_q_heads, seq_len, num_q_bits = query.size()
        bsz, num_k_heads, seq_len, num_k_bits = key.size()
        bsz, num_v_heads, seq_len, num_v_bits = value.size()

        assert num_k_heads == num_v_heads, f"Key and value must have the same number of heads, got {num_k_heads} and {num_v_heads}."
        assert num_q_bits == num_k_bits, f"Query and key must have the same bit width, got {num_q_bits} and {num_k_bits}."
        assert num_k_bits <= 64, f"Unsupported bit width for key: {num_k_bits}."
        assert num_v_bits <= 64, f"Unsupported bit width for value: {num_v_bits}."

        if self.ctx is None:
            self.n_ctx = bsz * num_q_heads
            self.n_rep = num_q_heads // num_k_heads

            self.ctx = rosa_sam_init(torch.zeros(self.n_ctx, dtype=torch.long, device="cpu"))

            self.num_q_bits = num_q_bits
            self.num_k_bits = num_k_bits
            self.num_v_bits = num_v_bits

            self.device = query.device
            self.output_dtype = value.dtype
        else:
            assert self.num_q_bits == num_q_bits
            assert self.num_k_bits == num_k_bits
            assert self.num_v_bits == num_v_bits
    
    def __del__(self):
        if self.ctx is not None:
            try:
                rosa_sam_free(self.ctx)
            except (AttributeError, TypeError):
                pass
            finally:
                self.ctx = None
    
    @torch.no_grad()
    def quantize(self, x: Tensor) -> Tensor:
        num_bits = x.size(-1)
        
        r = torch.arange(num_bits, device=x.device)
        x = ((x > 0).long() << r).sum(dim=-1)

        if num_bits <= 8:
            dtype = torch.uint8
        elif num_bits <= 16:
            dtype = torch.int16
        elif num_bits <= 32:
            dtype = torch.int32
        else:
            dtype = torch.int64

        x = x.to(dtype)
        return x
    
    def dispatch(self, query: Tensor, key: Tensor, value: Tensor):
        self._init_ctx(query=query, key=key, value=value)

        xq = self.quantize(query)
        xk = self.quantize(key)
        xv = self.quantize(value)

        with self.stream(post_wait=False):
            xq = xq.to("cpu", non_blocking=True)
            xk = xk.to("cpu", non_blocking=True)
            xv = xv.to("cpu", non_blocking=True)
        
        return xq, xk, xv

    def combine(self, xq: Tensor, xk: Tensor, xv: Tensor, mismatch: int = 0):
        bsz, num_q_heads, seq_len = xq.size()
        bsz, num_k_heads, seq_len = xk.size()
        bsz, num_v_heads, seq_len = xv.size()
        
        if self.n_rep > 1:
            xk = xk.view(bsz, num_k_heads, 1, seq_len).repeat(1, 1, self.n_rep, 1).view(bsz, num_q_heads, seq_len)
            xv = xv.view(bsz, num_v_heads, 1, seq_len).repeat(1, 1, self.n_rep, 1).view(bsz, num_q_heads, seq_len)

        xq = xq.view(-1, seq_len)
        xk = xk.view(-1, seq_len)
        xv = xv.view(-1, seq_len)

        xo = rosa_sam_update(
            self.ctx,
            xq.long(),
            xk.long(),
            xv.long(),
            mismatch,
        ).to(xv.dtype)

        with self.stream(prev_wait=False):
            xo = xo.to(self.device, non_blocking=True)
        
        r = torch.arange(self.num_v_bits, device=self.device)
        xo = (xo.view(bsz, num_q_heads, seq_len, 1) >> r) & 1
        return xo.to(self.output_dtype)

    @property
    def stream(self):
        return self._stream()
    
    @staticmethod
    @lru_cache(maxsize=None)
    def _stream():
        return RosaStream()


class RosaStream:
    def __init__(self):
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    @contextmanager
    def __call__(
            self,
            prev_wait: bool = True,
            post_wait: bool = True,
    ):
        if self.stream is None:
            yield
        else:
            if prev_wait:
                self.stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self.stream):
                yield self.stream
        
            if post_wait:
                torch.cuda.current_stream().wait_stream(self.stream)
            
