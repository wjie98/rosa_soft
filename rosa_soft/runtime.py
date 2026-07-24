from __future__ import annotations

import operator
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, Optional, Tuple

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


def _normalize_sequence_ids(
    sequence_ids: Optional[Sequence[int] | Tensor],
    batch: int,
) -> Optional[Tuple[int, ...]]:
    if sequence_ids is None:
        return None
    if isinstance(sequence_ids, Tensor):
        if sequence_ids.ndim != 1:
            raise ValueError("sequence_ids must be a 1D tensor")
        if sequence_ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("sequence_ids must contain integers")
        raw_ids = sequence_ids.detach().to(device="cpu").tolist()
    else:
        if isinstance(sequence_ids, (str, bytes)) or not isinstance(
            sequence_ids,
            Sequence,
        ):
            raise TypeError("sequence_ids must be an integer sequence")
        raw_ids = list(sequence_ids)

    normalized = tuple(
        _integer_parameter("sequence_ids", sequence_id)
        for sequence_id in raw_ids
    )
    if len(normalized) != batch:
        raise ValueError(
            f"sequence_ids must contain one ID per slot ({batch})"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("sequence_ids must be unique")
    return normalized


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


def _snapshot_to_cpu(tensor: Tensor) -> Tensor:
    cpu = _copy_to_cpu(tensor)
    return cpu.clone() if tensor.device.type == "cpu" else cpu


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
        self._wait_lock = threading.Lock()
        self._result: Optional[Tuple[Tensor, Tensor]] = None
        self._exception: Optional[BaseException] = None

    def wait(self) -> Tuple[Tensor, Tensor]:
        with self._wait_lock:
            if self._exception is not None:
                raise self._exception
            if self._result is not None:
                return self._result
            future = self._future
            if future is None:
                raise RuntimeError("RosaRuntimeWork has no pending result")
            try:
                packed_output, end_positions = future.result()
                self._result = self._runtime._finish_output(
                    packed_output,
                    end_positions,
                    self._device,
                    self._output_shape,
                    self._dense,
                    self._return_packed,
                    self._output_dtype,
                    self._stream,
                )
            except BaseException as error:
                self._exception = error
                raise
            finally:
                self._future = None
            return self._result


class _RuntimeState(Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class RosaRuntime:
    """Finite-horizon runtime for exact RosaSoft hard routing.

    Runtime slots are fixed by the first update. Supplying ``sequence_ids``
    on that update enables reorder checks.
    """

    def __init__(
        self,
        num_heads: int,
        num_payload_heads: Optional[int] = None,
        qk_bits: int = 8,
        payload_bits: Optional[int] = None,
        max_suffix_length: int = 32,
        *,
        num_value_heads: Optional[int] = None,
        value_bits: Optional[int] = None,
    ) -> None:
        if (
            num_payload_heads is not None
            and num_value_heads is not None
        ):
            raise TypeError(
                "specify only one of num_payload_heads and "
                "num_value_heads"
            )
        if payload_bits is not None and value_bits is not None:
            raise TypeError(
                "specify only one of payload_bits and value_bits"
            )
        if num_payload_heads is None:
            num_payload_heads = (
                num_heads
                if num_value_heads is None
                else num_value_heads
            )
        if payload_bits is None:
            payload_bits = 8 if value_bits is None else value_bits
        self.num_heads = _integer_parameter("num_heads", num_heads)
        self.num_payload_heads = _integer_parameter(
            "num_payload_heads",
            num_payload_heads,
        )
        self.qk_bits = _integer_parameter("qk_bits", qk_bits)
        self.payload_bits = _integer_parameter(
            "payload_bits",
            payload_bits,
        )
        self.max_suffix_length = _integer_parameter(
            "max_suffix_length",
            max_suffix_length,
        )
        self._native = torch.classes.rosa_soft.RosaRuntime(
            self.num_heads,
            self.num_payload_heads,
            self.qk_bits,
            self.payload_bits,
            self.max_suffix_length,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rosa-runtime",
        )
        self._lifecycle = threading.Condition()
        self._state = _RuntimeState.OPEN
        self._failure: Optional[BaseException] = None
        self._close_error: Optional[BaseException] = None
        self._executor_shutdown = False
        self._slots_initialized = False
        self._sequence_ids: Optional[Tuple[int, ...]] = None

    @property
    def num_value_heads(self) -> int:
        """Compatibility alias for ``num_payload_heads``."""
        return self.num_payload_heads

    @property
    def value_bits(self) -> int:
        """Compatibility alias for ``payload_bits``."""
        return self.payload_bits

    @property
    def state(self) -> str:
        with self._lifecycle:
            return self._state.value

    def close(self) -> None:
        with self._lifecycle:
            while self._state is _RuntimeState.CLOSING:
                self._lifecycle.wait()
            if self._state is _RuntimeState.CLOSED:
                return
            if self._executor_shutdown:
                if self._close_error is not None:
                    raise self._close_error
                return
            self._state = _RuntimeState.CLOSING

        error: Optional[BaseException] = None
        try:
            future = self._executor.submit(self._native.close)
            future.result()
        except BaseException as close_error:
            error = close_error
        finally:
            self._executor.shutdown(wait=True)
            with self._lifecycle:
                self._executor_shutdown = True
                self._close_error = error
                if error is None:
                    self._state = _RuntimeState.CLOSED
                    self._failure = None
                else:
                    self._state = _RuntimeState.FAILED
                    self._failure = error
                self._lifecycle.notify_all()
        if error is not None:
            raise error

    def __enter__(self) -> "RosaRuntime":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _state_error_locked(self) -> RuntimeError:
        if self._state is _RuntimeState.FAILED:
            detail = (
                f": {self._failure}"
                if self._failure is not None
                else ""
            )
            return RuntimeError(f"RosaRuntime is failed{detail}")
        return RuntimeError(
            f"RosaRuntime is {self._state.value.lower()}"
        )

    def _require_open(self) -> None:
        with self._lifecycle:
            if self._state is not _RuntimeState.OPEN:
                raise self._state_error_locked()

    def _submit(self, function, *args) -> Future:
        with self._lifecycle:
            if self._state is not _RuntimeState.OPEN:
                raise self._state_error_locked()
            return self._executor.submit(function, *args)

    def _mark_failed(self, error: BaseException) -> None:
        with self._lifecycle:
            self._failure = error
            if self._state is _RuntimeState.OPEN:
                self._state = _RuntimeState.FAILED

    def _check_sequence_ids(
        self,
        sequence_ids: Optional[Tuple[int, ...]],
    ) -> None:
        if not self._slots_initialized:
            self._slots_initialized = True
            self._sequence_ids = sequence_ids
            return
        if self._sequence_ids is None:
            if sequence_ids is not None:
                raise RuntimeError(
                    "sequence_ids checking must be enabled on the "
                    "first update after construction or reset"
                )
            return
        if sequence_ids is None:
            raise RuntimeError(
                "sequence_ids are required because slot checking is enabled"
            )
        if sequence_ids != self._sequence_ids:
            raise RuntimeError(
                "sequence_ids changed; RosaRuntime slots are fixed"
            )

    def _submit_update(
        self,
        function,
        sequence_ids: Optional[Tuple[int, ...]],
    ) -> Future:
        def run_update():
            with self._lifecycle:
                if self._failure is not None:
                    raise self._state_error_locked()
            self._check_sequence_ids(sequence_ids)
            try:
                return function()
            except BaseException as error:
                self._mark_failed(error)
                raise

        return self._submit(run_update)

    def stats(self) -> Tuple[int, int, int]:
        """Return exact state, edge, and deduplicated payload-symbol counts."""
        result = self._submit(self._native.stats).result()
        return tuple(int(value) for value in result)

    def memory_stats(self) -> Dict[str, int]:
        """Return exact logical state counts and bytes.

        ``logical_bytes`` counts stored SAM states, edges, finite-horizon key
        symbols, and payload symbols. It excludes allocator capacity and
        container overhead, so it is stable across standard-library builds.
        """
        result = self._submit(self._native.detailed_stats).result()
        names = (
            "states",
            "edges",
            "payload_symbols",
            "automata",
            "sequences",
            "logical_bytes",
        )
        return {
            name: int(value)
            for name, value in zip(names, result)
        }

    def reset(self) -> None:
        def reset_native() -> None:
            with self._lifecycle:
                if self._failure is not None:
                    raise self._state_error_locked()
            try:
                self._native.reset()
            except BaseException as error:
                self._mark_failed(error)
                raise
            self._slots_initialized = False
            self._sequence_ids = None

        self._submit(reset_native).result()

    def state_dict(self) -> Dict[str, Any]:
        self._require_open()
        raise NotImplementedError(
            "RosaRuntime checkpointing is unsupported: the native "
            "finite-horizon automata have no stable serialization schema"
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        del state_dict
        self._require_open()
        raise NotImplementedError(
            "RosaRuntime checkpoint restore is unsupported: the native "
            "finite-horizon automata have no stable serialization schema"
        )

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
        sequence_ids: Optional[Sequence[int] | Tensor] = None,
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
        if payload_logits.size(-1) != self.payload_bits:
            raise ValueError(
                "payload last dimension must equal payload_bits"
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
            sequence_ids=sequence_ids,
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
        sequence_ids: Optional[Sequence[int] | Tensor] = None,
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
        if payload_symbols.size(1) != self.num_payload_heads:
            raise ValueError(
                "packed payload head count must equal num_payload_heads"
            )
        batch = offsets.numel() - 1
        normalized_sequence_ids = _normalize_sequence_ids(
            sequence_ids,
            batch,
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

        offsets = offsets.clone()
        if stream is None:
            query_cpu = _snapshot_to_cpu(query_symbols)
            key_cpu = _snapshot_to_cpu(key_symbols)
            payload_cpu = _snapshot_to_cpu(payload_symbols)

            def update_cpu():
                return self._native.update_packed(
                    offsets,
                    query_cpu,
                    key_cpu,
                    payload_cpu,
                )

            future = self._submit_update(
                update_cpu,
                normalized_sequence_ids,
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

            future = self._submit_update(
                update_after_transfer,
                normalized_sequence_ids,
            )

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
            packed_output.record_stream(current_stream)
            end_positions.record_stream(current_stream)

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
            self.payload_bits,
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
