import torch

from torch import Tensor
from typing import *

from functools import lru_cache
from contextlib import contextmanager

from .ops import *


__all__ = [
    "RosaContext",
    "RosaContextWork",
    "RosaCache",
    "RosaCacheWork",
]



class RosaContextWork:
    def __init__(self) -> None:
        self.context: RosaContext
        self._cache_work: Optional[RosaCacheWork]
        self._value_bits: int
        self._value_type: torch.dtype
    
    def wait(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self._cache_work is None:
            raise RuntimeError("wait() called twice on the same work")
        
        work = self._cache_work
        self._cache_work = None

        output, endpos = self.context._combine(work, self._value_bits, self._value_type)
        return output, endpos


class RosaContext:
    def __init__(self, batch_size: int, num_heads: int) -> None:
        self.cache = RosaCache(batch_size, num_heads)
    
    def destroy(self):
        self.cache.destroy()
    
    def update(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            varlen: Optional[Tensor] = None,
            schmitt_trigger: float = 0.0,
            async_op: bool = False,
    ):
        assert query.size(-1) <= 64, "query must have at most 64 bits"
        assert value.size(-1) <= 64, "value must have at most 64 bits"
        assert query.size(-1) == key.size(-1), "query and key must have the same head dimension"

        work = RosaContextWork()
        work.context = self
        work._value_bits = value.size(-1)
        work._value_type = value.dtype
        work._cache_work = self._dispatch(
            query=query,
            key=key,
            value=value,
            varlen=varlen,
            schmitt_trigger=schmitt_trigger,
        )

        if async_op:
            return work
        return work.wait()
    
    @torch.no_grad()
    def _dispatch(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        varlen: Optional[Tensor],
        schmitt_trigger: float,
    ):
        query, query_trigger = self._quantize(query, schmitt_trigger=schmitt_trigger)
        key,   key_trigger   = self._quantize(key, schmitt_trigger=schmitt_trigger)
        value, value_trigger = self._quantize(value)

        return self.cache._dispatch(
            query=query,
            key=key,
            value=value,
            query_trigger=query_trigger,
            key_trigger=key_trigger,
            varlen=varlen,
        )

    @torch.no_grad()
    def _combine(self, work, num_v_bits: int, dtype: torch.dtype):
        output, endpos = cast(RosaCacheWork, work).wait()

        output_trigger = torch.zeros_like(output)
        output_trigger[endpos >= 0] = -1
        output = self._dequantize(output, output_trigger, num_bits=num_v_bits)
        
        output = output.to(dtype)
        return output, endpos

    @staticmethod
    def _quantize(x: Tensor, schmitt_trigger: float = 0.0) -> Tuple[Tensor, Optional[Tensor]]:
        assert x.is_floating_point()

        num_bits = x.size(-1)
        r = torch.arange(num_bits, device=x.device)

        if num_bits <= 8:
            dtype = torch.uint8
        elif num_bits <= 16:
            dtype = torch.int16
        elif num_bits <= 32:
            dtype = torch.int32
        elif num_bits <= 64:
            dtype = torch.int64
        else:
            raise ValueError(f"num_bits={num_bits} is not supported")

        if schmitt_trigger > 0.0:
            m = (torch.abs(x) > schmitt_trigger).to(dtype)
            m = torch.sum(m << r, dim=-1)
        else:
            m = None

        x = (x > 0).to(dtype)
        x = torch.sum(x << r, dim=-1)

        return x, m
    
    @staticmethod
    def _dequantize(x: Tensor, m: Optional[Tensor], num_bits: int) -> Tensor:
        assert not x.is_floating_point()

        r = torch.arange(num_bits, device=x.device)
        x = (x.unsqueeze(-1) >> r) & 1
        x = torch.where(x != 0, 1.0, -1.0)

        if isinstance(m, Tensor):
            assert not m.is_floating_point()
            m = (m.unsqueeze(-1) >> r) & 1
            x = torch.where(m != 0, x, 0.0)

        return x
    

class RosaCacheWork:
    def __init__(self):
        self.cache: RosaCache
        self._args: Optional[Tuple[Tensor, ...]]
    
    def wait(self) -> Tuple[Tensor, ...]:
        if self._args is None:
            raise RuntimeError("wait() called twice on the same work")
        
        args = self._args
        self._args = None
        
        output, endpos = self.cache._combine(*args)

        return output, endpos

class RosaCache:
    def __init__(self, batch_size: int, num_heads: int):
        self.cache = torch.zeros(batch_size, dtype=torch.int64, device="cpu")
        rosa_cache_create_(self.cache, num_heads)
    
    def __del__(self):
        self.destroy()
    
    def destroy(self):
        try:
            rosa_cache_delete_(self.cache)
        except Exception:
            pass
    
    def update(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            query_trigger: Optional[Tensor] = None,
            key_trigger: Optional[Tensor] = None,
            varlen: Optional[Tensor] = None,
            async_op: bool = False,
    ):
        work = self._dispatch(
            query=query,
            key=key,
            value=value,
            query_trigger=query_trigger,
            key_trigger=key_trigger,
            varlen=varlen,
        )

        if async_op:
            return work
        return work.wait()
    
    def _dispatch(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            query_trigger: Optional[Tensor],
            key_trigger: Optional[Tensor],
            varlen: Optional[Tensor],
    ):
        device = value.device
        vshape = value.shape

        if isinstance(varlen, Tensor):
            ntk, nqk = query.size()
            ntk, nvv = value.size()

            batch = varlen[1:] - varlen[:-1] # [B]

            query = query.permute(1, 0).reshape(nqk, ntk).contiguous()
            key   = key.permute(1, 0).reshape(nqk, ntk).contiguous()
            value = value.permute(1, 0).reshape(nvv, ntk).contiguous()

            if isinstance(query_trigger, Tensor):
                query_trigger = query_trigger.permute(1, 0).reshape(nqk, ntk).contiguous()

            if isinstance(key_trigger, Tensor):
                key_trigger = key_trigger.permute(1, 0).reshape(nqk, ntk).contiguous()
        else:
            B, N, nqk = query.size()
            _, _, nvv = value.size()
            
            batch = torch.empty(B, dtype=torch.int64, device="cpu").fill_(N)

            query = query.permute(2, 0, 1).reshape(nqk, B * N).contiguous()
            key   = key.permute(2, 0, 1).reshape(nqk, B * N).contiguous()
            value = value.permute(2, 0, 1).reshape(nvv, B * N).contiguous()

            if isinstance(query_trigger, Tensor):
                query_trigger = query_trigger.permute(2, 0, 1).reshape(nqk, B * N).contiguous()

            if isinstance(key_trigger, Tensor):
                key_trigger = key_trigger.permute(2, 0, 1).reshape(nqk, B * N).contiguous()
        
        with self.stream(post_wait=False):
            batch = batch.to(device="cpu", non_blocking=True)
            query = query.to(device="cpu", non_blocking=True)
            key   = key.to(device="cpu", non_blocking=True)
            value = value.to(device="cpu", non_blocking=True)
            
            if isinstance(query_trigger, Tensor):
                query_trigger = query_trigger.to(device="cpu", non_blocking=True)

            if isinstance(key_trigger, Tensor):
                key_trigger = key_trigger.to(device="cpu", non_blocking=True)
        
        work = RosaCacheWork()
        work.cache = self
        work._args = (batch, query, key, value, query_trigger, key_trigger, device, vshape)
        
        return work

    def _combine(
        self,
        batch: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_trigger: Optional[Tensor],
        key_trigger: Optional[Tensor],
        device: torch.device,
        vshape: Tuple[int, ...],
    ):
        self.stream.synchronize()

        output, endpos = rosa_cache_update_(self.cache, batch, query, key, value, query_trigger, key_trigger)

        with self.stream(prev_wait=False):
            output = output.to(device=device, non_blocking=True)
            endpos = endpos.to(device=device, non_blocking=True)
        
        if len(vshape) == 2:
            ntk, _ = vshape
            output = output.permute(1, 0).reshape(ntk, -1)
            endpos = endpos.permute(1, 0).reshape(ntk, -1)
        elif len(vshape) == 3:
            B, N, _ = vshape
            output = output.permute(1, 0).reshape(B, N, -1)
            endpos = endpos.permute(1, 0).reshape(B, N, -1)
        else:
            raise ValueError(f"vshape {vshape} is not supported")
        
        return output, endpos

    @property
    def stream(self):
        return self._stream()

    @staticmethod
    @lru_cache(maxsize=None)
    def _stream():
        return _RosaStream()


class _RosaStream:
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
