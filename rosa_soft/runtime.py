from __future__ import annotations

import operator
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Tuple

import torch
from torch import Tensor

from . import _C  # noqa: F401 - registers torch.classes.rosa_soft.RosaRuntime

__all__ = ["RosaRuntime", "RosaRuntimeWork"]


def _dense_cu_seqlens(batch: int, tokens: int) -> Tensor:
    if tokens == 0:
        return torch.zeros(batch + 1, dtype=torch.int64)
    return torch.arange(
        0,
        (batch + 1) * tokens,
        tokens,
        dtype=torch.int64,
        device="cpu",
    )


def _integer_parameter(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _normalize_cu_seqlens(cu_seqlens: Tensor) -> Tensor:
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a 1D tensor")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must describe at least one sequence")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype int32 or int64")
    return cu_seqlens.to(device="cpu", non_blocking=False).contiguous()


def _copy_to_cpu(
    tensor: Tensor,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> Tensor:
    if tensor.device.type == "cpu":
        return tensor.contiguous()
    if pin_memory:
        try:
            output = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            output.copy_(tensor.contiguous(), non_blocking=non_blocking)
            return output
        except RuntimeError:
            pass
    return tensor.contiguous().to(
        device="cpu",
        non_blocking=non_blocking,
    )


def _pin_cpu(tensor: Tensor) -> Tensor:
    try:
        output = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        output.copy_(tensor)
        return output
    except RuntimeError:
        return tensor


def _pack_sign_bits(logits: Tensor) -> Tensor:
    if not logits.is_floating_point():
        raise TypeError("unpacked RosaRuntime inputs must be floating-point")
    bits = logits.size(-1)
    if not 1 <= bits <= 8:
        raise ValueError(f"RosaRuntime supports 1..8 bits, got {bits}")
    weights = (
        1
        << torch.arange(
            bits,
            device=logits.device,
            dtype=torch.int16,
        )
    ).view(*([1] * (logits.ndim - 1)), bits)
    return (
        (logits > 0).to(torch.int16) * weights
    ).sum(dim=-1).to(torch.uint8).contiguous()


def _unpack_sign_bits(
    packed: Tensor,
    bits: int,
    dtype: torch.dtype,
) -> Tensor:
    shifts = torch.arange(
        bits,
        device=packed.device,
        dtype=torch.int16,
    ).view(*([1] * packed.ndim), bits)
    binary = (
        (packed.to(torch.int16).unsqueeze(-1) >> shifts) & 1
    ).to(dtype)
    return binary.mul(2).sub(1)


def _flatten_packed(
    query_symbols: Tensor,
    key_symbols: Tensor,
    payload_symbols: Tensor,
    cu_seqlens: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[int, ...], bool]:
    if any(
        tensor.dtype != torch.uint8
        for tensor in (query_symbols, key_symbols, payload_symbols)
    ):
        raise TypeError("packed RosaRuntime inputs must have dtype torch.uint8")
    if query_symbols.shape != key_symbols.shape:
        raise ValueError(
            "packed query/key shapes differ: "
            f"{query_symbols.shape} vs {key_symbols.shape}"
        )
    if not (
        query_symbols.device
        == key_symbols.device
        == payload_symbols.device
    ):
        raise ValueError("all packed inputs must be on the same device")

    dense = cu_seqlens is None
    if dense:
        if query_symbols.ndim != 3 or payload_symbols.ndim != 3:
            raise ValueError(
                "dense packed inputs must have shapes [B, T, H] and "
                "[B, T, H_v]"
            )
        if (
            query_symbols.shape[:2]
            != payload_symbols.shape[:2]
        ):
            raise ValueError(
                "dense query and payload dimensions B,T must match"
            )
        batch, tokens, _ = query_symbols.shape
        if batch < 1:
            raise ValueError("dense batch size must be >= 1")
        return (
            query_symbols.reshape(
                batch * tokens,
                query_symbols.size(2),
            ).contiguous(),
            key_symbols.reshape(
                batch * tokens,
                key_symbols.size(2),
            ).contiguous(),
            payload_symbols.reshape(
                batch * tokens,
                payload_symbols.size(2),
            ).contiguous(),
            _dense_cu_seqlens(batch, tokens),
            tuple(payload_symbols.shape),
            True,
        )

    if query_symbols.ndim != 2 or payload_symbols.ndim != 2:
        raise ValueError(
            "varlen packed inputs must have shapes [total, H] and "
            "[total, H_v]"
        )
    if query_symbols.size(0) != payload_symbols.size(0):
        raise ValueError(
            "varlen query and payload token counts must match"
        )
    return (
        query_symbols.contiguous(),
        key_symbols.contiguous(),
        payload_symbols.contiguous(),
        _normalize_cu_seqlens(cu_seqlens),
        tuple(payload_symbols.shape),
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
        output_dtype: torch.dtype,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._runtime = runtime
        self._future: Optional[Future] = future
        self._device = device
        self._output_shape = output_shape
        self._dense = dense
        self._return_packed = return_packed
        self._output_dtype = output_dtype
        self._stream = stream

    def wait(self) -> Tuple[Tensor, Tensor]:
        if self._future is None:
            raise RuntimeError("RosaRuntimeWork.wait() may be called once")
        future = self._future
        self._future = None
        packed_output, end_positions = future.result()
        return self._runtime._finish_output(
            packed_output,
            end_positions,
            self._device,
            self._output_shape,
            self._dense,
            self._return_packed,
            self._output_dtype,
            self._stream,
        )


class RosaRuntime:
    """Stateful bounded-suffix runtime for exact RosaSoft hard routing."""

    def __init__(
        self,
        num_heads: int,
        num_value_heads: Optional[int] = None,
        qk_bits: int = 8,
        value_bits: int = 8,
        max_suffix_length: int = 32,
    ) -> None:
        if num_value_heads is None:
            num_value_heads = num_heads
        self.num_heads = _integer_parameter("num_heads", num_heads)
        self.num_value_heads = _integer_parameter(
            "num_value_heads",
            num_value_heads,
        )
        self.qk_bits = _integer_parameter("qk_bits", qk_bits)
        self.value_bits = _integer_parameter("value_bits", value_bits)
        self.max_suffix_length = _integer_parameter(
            "max_suffix_length",
            max_suffix_length,
        )
        self._native = torch.classes.rosa_soft.RosaRuntime(
            self.num_heads,
            self.num_value_heads,
            self.qk_bits,
            self.value_bits,
            self.max_suffix_length,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rosa-runtime",
        )
        self._lifecycle_lock = threading.Lock()
        self._close_future: Optional[Future] = None
        self._closed = False

    def close(self) -> None:
        with self._lifecycle_lock:
            owns_shutdown = not self._closed
            if owns_shutdown:
                self._closed = True
                self._close_future = self._executor.submit(
                    self._native.close
                )
            future = self._close_future
        if future is None:
            return
        future.result()
        if owns_shutdown:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "RosaRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _submit(self, function, *args) -> Future:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("RosaRuntime is closed")
            return self._executor.submit(function, *args)

    def stats(self) -> Tuple[int, int, int]:
        result = self._submit(self._native.stats).result()
        return tuple(int(value) for value in result)

    def update(
        self,
        query_logits: Tensor,
        key_logits: Tensor,
        payload_logits: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        async_op: bool = False,
        return_packed: bool = False,
    ):
        if query_logits.shape != key_logits.shape:
            raise ValueError(
                "query/key shapes differ: "
                f"{query_logits.shape} vs {key_logits.shape}"
            )
        if query_logits.size(-1) != self.qk_bits:
            raise ValueError(
                "query/key last dimension must equal qk_bits"
            )
        if payload_logits.size(-1) != self.value_bits:
            raise ValueError(
                "payload last dimension must equal value_bits"
            )

        expected_rank = 4 if cu_seqlens is None else 3
        if (
            query_logits.ndim != expected_rank
            or payload_logits.ndim != expected_rank
        ):
            layout = "dense" if cu_seqlens is None else "varlen"
            raise ValueError(
                f"{layout} inputs have the wrong rank for RosaRuntime"
            )

        return self.update_packed(
            _pack_sign_bits(query_logits),
            _pack_sign_bits(key_logits),
            _pack_sign_bits(payload_logits),
            cu_seqlens=cu_seqlens,
            stream=stream,
            async_op=async_op,
            return_packed=return_packed,
            output_dtype=payload_logits.dtype,
        )

    def update_packed(
        self,
        query_symbols: Tensor,
        key_symbols: Tensor,
        payload_symbols: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        async_op: bool = False,
        return_packed: bool = True,
        output_dtype: torch.dtype = torch.float32,
    ):
        (
            query_symbols,
            key_symbols,
            payload_symbols,
            offsets,
            output_shape,
            dense,
        ) = _flatten_packed(
            query_symbols,
            key_symbols,
            payload_symbols,
            cu_seqlens,
        )
        if query_symbols.size(1) != self.num_heads:
            raise ValueError(
                "packed query/key head count must equal num_heads"
            )
        if payload_symbols.size(1) != self.num_value_heads:
            raise ValueError(
                "packed payload head count must equal num_value_heads"
            )

        device = payload_symbols.device
        if stream is not None and device.type != "cuda":
            raise ValueError("stream requires CUDA inputs")
        if stream is not None and stream.device != device:
            raise ValueError("stream and inputs must use the same CUDA device")
        if (
            not return_packed
            and not torch.empty((), dtype=output_dtype).is_floating_point()
        ):
            raise TypeError(
                "output_dtype must be floating-point when unpacking"
            )

        if stream is None:
            query_cpu = _copy_to_cpu(query_symbols)
            key_cpu = _copy_to_cpu(key_symbols)
            payload_cpu = _copy_to_cpu(payload_symbols)
            future = self._submit(
                self._native.update_packed,
                offsets,
                query_cpu,
                key_cpu,
                payload_cpu,
            )
        else:
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                query_cpu = _copy_to_cpu(
                    query_symbols,
                    pin_memory=True,
                    non_blocking=True,
                )
                key_cpu = _copy_to_cpu(
                    key_symbols,
                    pin_memory=True,
                    non_blocking=True,
                )
                payload_cpu = _copy_to_cpu(
                    payload_symbols,
                    pin_memory=True,
                    non_blocking=True,
                )
                query_symbols.record_stream(stream)
                key_symbols.record_stream(stream)
                payload_symbols.record_stream(stream)
                ready = torch.cuda.Event()
                ready.record(stream)

            def update_after_transfer():
                ready.synchronize()
                packed_output, end_positions = (
                    self._native.update_packed(
                        offsets,
                        query_cpu,
                        key_cpu,
                        payload_cpu,
                    )
                )
                return (
                    _pin_cpu(packed_output),
                    _pin_cpu(end_positions),
                )

            future = self._submit(update_after_transfer)

        work = RosaRuntimeWork(
            self,
            future,
            device,
            output_shape,
            dense,
            return_packed,
            output_dtype,
            stream,
        )
        return work if async_op else work.wait()

    def _finish_output(
        self,
        packed_output_cpu: Tensor,
        end_positions_cpu: Tensor,
        device: torch.device,
        output_shape: Tuple[int, ...],
        dense: bool,
        return_packed: bool,
        output_dtype: torch.dtype,
        stream: Optional[torch.cuda.Stream],
    ) -> Tuple[Tensor, Tensor]:
        if stream is None or device.type == "cpu":
            packed_output = packed_output_cpu.to(
                device=device,
                non_blocking=False,
            )
            end_positions = end_positions_cpu.to(
                device=device,
                non_blocking=False,
            )
        else:
            current_stream = torch.cuda.current_stream(device)
            with torch.cuda.stream(stream):
                packed_output = packed_output_cpu.to(
                    device=device,
                    non_blocking=True,
                )
                end_positions = end_positions_cpu.to(
                    device=device,
                    non_blocking=True,
                )
            current_stream.wait_stream(stream)

        if dense:
            packed_output = packed_output.reshape(
                output_shape[0],
                output_shape[1],
                self.num_heads,
            )
            end_positions = end_positions.reshape(
                output_shape[0],
                output_shape[1],
                self.num_heads,
            )
        if return_packed:
            return packed_output, end_positions

        unpacked = _unpack_sign_bits(
            packed_output,
            self.value_bits,
            output_dtype,
        )
        unpacked = torch.where(
            end_positions.unsqueeze(-1) >= 0,
            unpacked,
            torch.zeros(
                (),
                dtype=output_dtype,
                device=device,
            ),
        )
        return unpacked, end_positions
