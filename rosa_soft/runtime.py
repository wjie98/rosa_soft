from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor

from . import _C  # noqa: F401 - import registers torch.classes.rosa_soft.RosaRuntime

__all__ = ["RosaRuntime", "RosaRuntimeWork"]


@lru_cache(maxsize=1)
def _runtime_executor() -> ThreadPoolExecutor:
    workers = max(1, min(4, os.cpu_count() or 1))
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rosa-runtime")


def _make_cpu_cu_seqlens(batch: int, tokens: int) -> Tensor:
    return torch.arange(0, (batch + 1) * tokens, tokens, dtype=torch.int64, device="cpu")


def _normalize_cu_seqlens(cu_seqlens: Tensor) -> Tensor:
    if cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be a 1D tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype int32 or int64")
    if cu_seqlens.device.type != "cpu":
        cu_seqlens = cu_seqlens.to(device="cpu", non_blocking=False)
    return cu_seqlens.contiguous()


def _copy_to_cpu(x: Tensor, *, pin_memory: bool = False, non_blocking: bool = False) -> Tensor:
    if x.device.type == "cpu":
        return x.contiguous()
    if pin_memory:
        try:
            out = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
            out.copy_(x.contiguous(), non_blocking=non_blocking)
            return out
        except RuntimeError:
            pass
    return x.contiguous().to(device="cpu", non_blocking=non_blocking)


def _pin_cpu(x: Tensor) -> Tensor:
    if x.device.type != "cpu":
        raise ValueError("_pin_cpu expects a CPU tensor")
    try:
        out = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
        out.copy_(x)
        return out
    except RuntimeError:
        return x


def _pack_bits(x: Tensor) -> Tensor:
    if not x.is_floating_point():
        raise TypeError("unpacked RosaRuntime inputs must be floating-point logits")
    bits = x.size(-1)
    if bits <= 0 or bits > 8:
        raise ValueError(f"RosaRuntime supports 1..8 bits, got {bits}")
    weights = (1 << torch.arange(bits, device=x.device, dtype=torch.int16)).view(
        *([1] * (x.dim() - 1)), bits
    )
    return ((x > 0).to(torch.int16) * weights).sum(dim=-1).to(torch.uint8).contiguous()


def _unpack_bits(x: Tensor, bits: int, dtype: torch.dtype) -> Tensor:
    shifts = torch.arange(bits, device=x.device, dtype=torch.int16).view(
        *([1] * x.dim()), bits
    )
    out = ((x.to(torch.int16).unsqueeze(-1) >> shifts) & 1).to(dtype)
    return torch.where(out != 0, torch.ones((), dtype=dtype, device=x.device), -torch.ones((), dtype=dtype, device=x.device))


def _flatten_packed(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[int, ...], bool]:
    if query.dtype != torch.uint8 or key.dtype != torch.uint8 or value.dtype != torch.uint8:
        raise TypeError("packed RosaRuntime inputs must have dtype torch.uint8")
    if query.shape != key.shape:
        raise ValueError(f"query/key shape mismatch: {query.shape} vs {key.shape}")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")

    dense = cu_seqlens is None
    if dense:
        if query.dim() != 3 or value.dim() != 3:
            raise ValueError("dense packed inputs must be shaped [B, T, H] and [B, T, H_v]")
        if query.size(0) != value.size(0) or query.size(1) != value.size(1):
            raise ValueError("dense query/value batch and token dimensions must match")
        B, T, _ = query.shape
        cu = _make_cpu_cu_seqlens(B, T)
        return (
            query.reshape(B * T, query.size(2)).contiguous(),
            key.reshape(B * T, key.size(2)).contiguous(),
            value.reshape(B * T, value.size(2)).contiguous(),
            cu,
            tuple(value.shape),
            True,
        )

    if query.dim() != 2 or value.dim() != 2:
        raise ValueError("varlen packed inputs must be shaped [total, H] and [total, H_v]")
    if query.size(0) != value.size(0):
        raise ValueError("varlen query/value token dimensions must match")
    return (
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        _normalize_cu_seqlens(cu_seqlens),
        tuple(value.shape),
        False,
    )


class RosaRuntimeWork:
    def __init__(
        self,
        runtime: "RosaRuntime",
        future: Future,
        device: torch.device,
        output_shape: Tuple[int, ...],
        dense: bool,
        return_packed: bool,
        value_bits: int,
        output_dtype: torch.dtype,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._runtime = runtime
        self._future: Optional[Future] = future
        self._device = device
        self._output_shape = output_shape
        self._dense = dense
        self._return_packed = return_packed
        self._value_bits = value_bits
        self._output_dtype = output_dtype
        self._stream = stream

    def wait(self):
        if self._future is None:
            raise RuntimeError("wait() called twice on the same RosaRuntimeWork")
        future = self._future
        self._future = None
        output_cpu, endpos_cpu = future.result()
        return self._runtime._finish_output(
            output_cpu,
            endpos_cpu,
            self._device,
            self._output_shape,
            self._dense,
            self._return_packed,
            self._value_bits,
            self._output_dtype,
            self._stream,
        )


class RosaRuntime:
    """Stateful CPU suffix-automaton runtime with optional CUDA staging.

    The runtime stores state in a PyTorch custom class and must be closed
    explicitly when deterministic release is needed. It also supports context
    manager usage.
    """

    def __init__(
        self,
        num_heads: int,
        num_value_heads: Optional[int] = None,
        qk_bits: int = 8,
        value_bits: int = 8,
        backend: str = "compact",
    ) -> None:
        if num_value_heads is None:
            num_value_heads = num_heads
        self.num_heads = int(num_heads)
        self.num_value_heads = int(num_value_heads)
        self.qk_bits = int(qk_bits)
        self.value_bits = int(value_bits)
        self.backend = backend
        self._runtime = torch.classes.rosa_soft.RosaRuntime(
            self.num_heads,
            self.num_value_heads,
            self.qk_bits,
            self.value_bits,
            backend,
        )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._runtime.close()
            self._closed = True

    def __enter__(self) -> "RosaRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def stats(self) -> Tuple[int, int, int]:
        return tuple(int(x) for x in self._runtime.stats())

    def update(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        async_op: bool = False,
        return_packed: bool = False,
    ):
        if query.shape != key.shape:
            raise ValueError(f"query/key shape mismatch: {query.shape} vs {key.shape}")
        if query.size(-1) != self.qk_bits or key.size(-1) != self.qk_bits:
            raise ValueError("query/key last dimension must match qk_bits")
        if value.size(-1) != self.value_bits:
            raise ValueError("value last dimension must match value_bits")

        expected_dim = 4 if cu_seqlens is None else 3
        if query.dim() != expected_dim or value.dim() != expected_dim:
            layout = "dense" if cu_seqlens is None else "varlen"
            raise ValueError(f"{layout} inputs have wrong rank for RosaRuntime")

        return self.update_packed(
            _pack_bits(query),
            _pack_bits(key),
            _pack_bits(value),
            cu_seqlens=cu_seqlens,
            stream=stream,
            async_op=async_op,
            return_packed=return_packed,
            output_dtype=value.dtype,
        )

    def update_packed(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        async_op: bool = False,
        return_packed: bool = True,
        output_dtype: torch.dtype = torch.float32,
    ):
        if self._closed:
            raise RuntimeError("RosaRuntime is closed")
        query, key, value, cu, output_shape, dense = _flatten_packed(query, key, value, cu_seqlens)
        if query.size(1) != self.num_heads or key.size(1) != self.num_heads:
            raise ValueError("packed query/key head dimension must match num_heads")
        if value.size(1) != self.num_value_heads:
            raise ValueError("packed value head dimension must match num_value_heads")

        device = value.device
        if stream is not None and device.type != "cuda":
            raise ValueError("stream can only be used with CUDA inputs")
        if stream is not None and not async_op:
            async_op = True

        if stream is None:
            q_cpu = _copy_to_cpu(query)
            k_cpu = _copy_to_cpu(key)
            v_cpu = _copy_to_cpu(value)
            output_cpu, endpos_cpu = self._runtime.update_packed(cu, q_cpu, k_cpu, v_cpu)
            return self._finish_output(
                output_cpu,
                endpos_cpu,
                device,
                output_shape,
                dense,
                return_packed,
                self.value_bits,
                output_dtype,
                stream=None,
            )

        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            q_cpu = _copy_to_cpu(query, pin_memory=True, non_blocking=True)
            k_cpu = _copy_to_cpu(key, pin_memory=True, non_blocking=True)
            v_cpu = _copy_to_cpu(value, pin_memory=True, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)

        def run_cpu():
            event.synchronize()
            output_cpu, endpos_cpu = self._runtime.update_packed(cu, q_cpu, k_cpu, v_cpu)
            return _pin_cpu(output_cpu), _pin_cpu(endpos_cpu)

        work = RosaRuntimeWork(
            self,
            _runtime_executor().submit(run_cpu),
            device,
            output_shape,
            dense,
            return_packed,
            self.value_bits,
            output_dtype,
            stream,
        )
        if async_op:
            return work
        return work.wait()

    def _finish_output(
        self,
        output_cpu: Tensor,
        endpos_cpu: Tensor,
        device: torch.device,
        output_shape: Tuple[int, ...],
        dense: bool,
        return_packed: bool,
        value_bits: int,
        output_dtype: torch.dtype,
        stream: Optional[torch.cuda.Stream],
    ):
        if stream is None or device.type == "cpu":
            output = output_cpu.to(device=device, non_blocking=False)
            endpos = endpos_cpu.to(device=device, non_blocking=False)
        else:
            current = torch.cuda.current_stream(device)
            with torch.cuda.stream(stream):
                output = output_cpu.to(device=device, non_blocking=True)
                endpos = endpos_cpu.to(device=device, non_blocking=True)
            current.wait_stream(stream)

        if dense:
            output = output.reshape(output_shape[0], output_shape[1], self.num_heads)
            endpos = endpos.reshape(output_shape[0], output_shape[1], self.num_heads)
        if not return_packed:
            output = _unpack_bits(output, value_bits, output_dtype)
        return output, endpos
