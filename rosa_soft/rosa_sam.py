import torch

from torch import Tensor
from typing import *

from functools import lru_cache
from contextlib import contextmanager

from .ops import *
from .utils import quantize, dequantize


__all__ = [
    "RosaWork",
    "RosaContext",
]


class RosaWork:
    def __init__(self):
        self._ctx: RosaContext
        self._qkv: Tuple[Tensor, ...]
    
    def wait(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self._ctx is None:
            raise RuntimeError("wait() called twice")
        ctx = self._ctx
        qkv = self._qkv

        self._ctx = None
        self._qkv = None

        return ctx._combine(*qkv)


class RosaContext:
    def __init__(self):
        self._sam: RosaSAM | None = None
    
    def update(self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        schmitt_trigger: float = 0.0,
        async_op: bool = False,
    ) -> Union[RosaWork, Tuple[Tensor, Dict[str, Tensor]]]:
        work = RosaWork()
        work._ctx = self
        work._qkv = self._dispatch(query=query, key=key, value=value, schmitt_trigger=schmitt_trigger)

        if async_op:
            return work
        return work.wait()
    
    def _init_sam(self, query: Tensor, key: Tensor, value: Tensor):
        bsz, num_q_heads, seq_len, num_q_bits = query.size()
        bsz, num_k_heads, seq_len, num_k_bits = key.size()
        bsz, num_v_heads, seq_len, num_v_bits = value.size()

        assert num_k_heads == num_v_heads, f"Key and value must have the same number of heads, got {num_k_heads} and {num_v_heads}."
        assert num_q_bits == num_k_bits, f"Query and key must have the same bit width, got {num_q_bits} and {num_k_bits}."
        assert num_k_bits <= 64, f"Unsupported bit width for key: {num_k_bits}."
        assert num_v_bits <= 64, f"Unsupported bit width for value: {num_v_bits}."

        if self._sam is None:
            self._sam = RosaSAM(bsz * num_q_heads)

            self.num_q_bits = num_q_bits
            self.num_k_bits = num_k_bits
            self.num_v_bits = num_v_bits

            self.device = query.device
            self.dtype = value.dtype
        else:
            assert self.num_q_bits == num_q_bits
            assert self.num_k_bits == num_k_bits
            assert self.num_v_bits == num_v_bits
    
    def _dispatch(self, query: Tensor, key: Tensor, value: Tensor, schmitt_trigger: float):
        self._init_sam(query=query, key=key, value=value)

        xq, mq = quantize(query, schmitt_trigger=schmitt_trigger)
        xk, mk = quantize(key, schmitt_trigger=schmitt_trigger)
        xv, _  = quantize(value)

        with self.stream(post_wait=False):
            xq = xq.to("cpu", non_blocking=True)
            xk = xk.to("cpu", non_blocking=True)
            xv = xv.to("cpu", non_blocking=True)
            mq = mq.to("cpu", non_blocking=True)
            mk = mk.to("cpu", non_blocking=True)

        return xq, xk, xv, mq, mk
    
    def _combine(self, xq: Tensor, xk: Tensor, xv: Tensor, mq: Tensor, mk: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        self.stream.synchronize()

        bsz, num_q_heads, seq_len = xq.size()
        bsz, num_k_heads, seq_len = xk.size()
        bsz, num_v_heads, seq_len = xv.size()
        
        n_rep = num_q_heads // num_k_heads
        if n_rep > 1:
            xk = xk.view(bsz, num_k_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
            xv = xv.view(bsz, num_v_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
            mk = mk.view(bsz, num_k_heads, 1, seq_len).repeat(1, 1, n_rep, 1)

        xq = xq.reshape(-1, seq_len)
        xk = xk.reshape(-1, seq_len)
        xv = xv.reshape(-1, seq_len)
        mq = mq.reshape(-1, seq_len)
        mk = mk.reshape(-1, seq_len)

        xo, mo, info = self._sam.update(xq, xk, xv, mq, mk)

        with self.stream(prev_wait=False):
            xo = xo.to(self.device, non_blocking=True)
            mo = mo.to(self.device, non_blocking=True)
            info = {key: val.to(self.device, non_blocking=True) for key, val in info.items()}
        
        xo = dequantize(xo, mo, self.num_v_bits).to(self.dtype)
        xo = xo.view(bsz, num_q_heads, seq_len, self.num_v_bits)
        info = {key: val.view(bsz, num_q_heads, seq_len) for key, val in info.items()}
        return xo, info

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
    
    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()
    
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
            

class RosaSAM:
    def __init__(self, n_ctx: int = 1):
        assert n_ctx >= 0
        self._objs = torch.zeros(n_ctx, dtype=torch.int64, device="cpu")
        self._objs = rosa_sam_init(self._objs)
    
    def __del__(self):
        try:
            rosa_sam_free(self._objs)
        except (AttributeError, TypeError):
            pass
    
    def __len__(self):
        return self._objs.numel()
    
    def update(self, xq: Tensor, xk: Tensor, xv: Tensor, mq: Optional[Tensor] = None, mk: Optional[Tensor] = None, u: int = 0) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        xq = xq.to(torch.int64)
        xk = xk.to(torch.int64)
        xv = xv.to(torch.int64)

        mq = torch.full_like(xq, -1) if mq is None else mq.to(torch.int64)
        mk = torch.full_like(xk, -1) if mk is None else mk.to(torch.int64)
        
        xo, endpos, length = rosa_sam_update(self._objs, xq, xk, xv, mq, mk, u)
        
        xo = xo.to(xv.dtype)
        mo = torch.zeros_like(xo)
        mo[length > 0] = -1 # mark no-match with "0" for dequantization

        info = {
            "endpos": endpos,
            "length": length,
        }
        return xo, mo, info
