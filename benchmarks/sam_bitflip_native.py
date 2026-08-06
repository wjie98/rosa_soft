"""ctypes bridge for the standalone C++ SAM bitflip reference core."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "csrc" / "sam_bitflip_cpu.cpp"
DEFAULT_LIBRARY = ROOT / "build" / "sam_bitflip" / "librosa_sam_bitflip.so"

__all__ = [
    "DEFAULT_LIBRARY",
    "NativeFactorizedResult",
    "NativeSamBitflip",
    "NativeSamBitflipResult",
    "build_native_sam_bitflip",
]

_STAT_NAMES = (
    "forward_states",
    "forward_edges",
    "reverse_states",
    "reverse_edges",
    "predecessor_queries",
    "arithmetic_hits",
    "wavelet_fallbacks",
    "query_replay_steps",
    "query_branches_merged",
    "replacement_length_probes",
    "replacement_rows",
    "virtual_runs",
    "virtual_active_rows",
    "forward_sam_build_ns",
    "reverse_sam_build_ns",
    "endpos_build_ns",
    "trace_build_ns",
    "output_initialize_ns",
    "query_solve_ns",
    "key_solve_ns",
    "total_solve_ns",
    "forward_sam_bytes",
    "reverse_sam_bytes",
    "endpos_bytes",
    "solver_working_bytes",
    "materialized_output_bytes",
    "query_change_descriptors",
    "key_delete_descriptors",
    "key_override_descriptors",
    "factorized_result_bytes",
    "replacement_left_queries",
)


@dataclass(frozen=True)
class NativeSamBitflipResult:
    flipped_routes: Tensor
    flipped_lengths: Tensor
    profile: dict[str, int]


@dataclass(frozen=True)
class NativeFactorizedResult:
    """Exact base winners and affine counterfactual descriptor families."""

    sequence_length: int
    bit_width: int
    base_routes: Tensor
    base_lengths: Tensor
    query_offsets: Tensor
    query_changes: Tensor
    key_delete_offsets: Tensor
    key_delete_changes: Tensor
    key_override_offsets: Tensor
    key_overrides: Tensor
    profile: dict[str, int]

    @property
    def key_length(self) -> int:
        return max(self.sequence_length - 1, 0)

    @property
    def query_flip_count(self) -> int:
        return self.key_length * self.bit_width

    @property
    def descriptor_bytes(self) -> int:
        tensors = (
            self.base_routes,
            self.base_lengths,
            self.query_offsets,
            self.query_changes,
            self.key_delete_offsets,
            self.key_delete_changes,
            self.key_override_offsets,
            self.key_overrides,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @staticmethod
    def _apply_changes(routes, lengths, changes) -> None:
        for change in changes.tolist():
            start, stop, length_start, length_step, route_start, route_step = (
                change
            )
            rows = torch.arange(start, stop, dtype=torch.int64)
            offset = rows - start
            lengths[rows] = length_start + offset * length_step
            routes[rows] = route_start + offset * route_step

    @staticmethod
    def _apply_overrides(routes, lengths, changes) -> None:
        for change in changes.tolist():
            start, stop = change[:2]
            to_length_start, to_length_step = change[6:8]
            to_route_start, to_route_step = change[8:10]
            rows = torch.arange(start, stop, dtype=torch.int64)
            offset = rows - start
            lengths[rows] = to_length_start + offset * to_length_step
            routes[rows] = to_route_start + offset * to_route_step

    def materialize(self) -> tuple[Tensor, Tensor]:
        flip_count = 2 * self.query_flip_count
        routes = self.base_routes.to(torch.int64).repeat(flip_count, 1)
        lengths = self.base_lengths.to(torch.int64).repeat(flip_count, 1)
        for flip in range(self.query_flip_count):
            start = int(self.query_offsets[flip])
            stop = int(self.query_offsets[flip + 1])
            self._apply_changes(
                routes[flip],
                lengths[flip],
                self.query_changes[start:stop],
            )
        for local_flip in range(self.query_flip_count):
            flip = self.query_flip_count + local_flip
            key_position = local_flip // self.bit_width
            start = int(self.key_delete_offsets[key_position])
            stop = int(self.key_delete_offsets[key_position + 1])
            self._apply_changes(
                routes[flip],
                lengths[flip],
                self.key_delete_changes[start:stop],
            )
            start = int(self.key_override_offsets[local_flip])
            stop = int(self.key_override_offsets[local_flip + 1])
            self._apply_overrides(
                routes[flip],
                lengths[flip],
                self.key_overrides[start:stop],
            )
        return routes, lengths


def build_native_sam_bitflip(
    output: Path = DEFAULT_LIBRARY,
    *,
    compiler: str = "c++",
) -> Path:
    """Compile the independent C++17 research core."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-shared",
            str(SOURCE),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


class NativeSamBitflip:
    """Exact factorized and validation APIs for the C++ SAM core."""

    def __init__(self, library_path: Path | str | None = None) -> None:
        if library_path is None:
            configured = os.environ.get("ROSA_SAM_BITFLIP_PATH")
            library_path = Path(configured) if configured else DEFAULT_LIBRARY
        self.library_path = Path(library_path).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                f"native SAM bitflip core not built: {self.library_path}; run "
                "python benchmarks/sam_bitflip_native.py --build"
            )
        library = ctypes.CDLL(str(self.library_path))
        library.rosa_sam_bitflip_routes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int64,
        ]
        library.rosa_sam_bitflip_routes.restype = ctypes.c_int32
        library.rosa_sam_bitflip_factorized_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32),
        ]
        library.rosa_sam_bitflip_factorized_create.restype = ctypes.c_void_p
        library.rosa_sam_bitflip_factorized_sizes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
        ]
        library.rosa_sam_bitflip_factorized_sizes.restype = ctypes.c_int32
        library.rosa_sam_bitflip_factorized_copy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.rosa_sam_bitflip_factorized_copy.restype = ctypes.c_int32
        library.rosa_sam_bitflip_factorized_destroy.argtypes = [ctypes.c_void_p]
        library.rosa_sam_bitflip_factorized_destroy.restype = None
        self.library = library

    @classmethod
    def available(cls, library_path: Path | str | None = None) -> bool:
        try:
            cls(library_path)
        except (FileNotFoundError, OSError):
            return False
        return True

    @staticmethod
    def _validate(query_codes: Tensor, key_codes: Tensor, bit_width: int) -> None:
        if query_codes.ndim != 1 or key_codes.ndim != 1:
            raise ValueError("query_codes and key_codes must be vectors")
        if query_codes.shape != key_codes.shape:
            raise ValueError("query_codes and key_codes must have one shape")
        if query_codes.dtype != torch.uint8 or key_codes.dtype != torch.uint8:
            raise ValueError("packed codes must use torch.uint8")
        if query_codes.device.type != "cpu" or key_codes.device.type != "cpu":
            raise ValueError("native SAM bitflip is CPU-only")
        if not 1 <= bit_width <= 8:
            raise ValueError("bit_width must be in 1..8")
        if query_codes.numel() and (
            int(query_codes.max()) >= 1 << bit_width
            or int(key_codes.max()) >= 1 << bit_width
        ):
            raise ValueError("packed code uses bits outside bit_width")

    def solve(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        bit_width: int,
    ) -> NativeSamBitflipResult:
        self._validate(query_codes, key_codes, bit_width)
        query_codes = query_codes.contiguous()
        key_codes = key_codes.contiguous()
        sequence_length = query_codes.numel()
        flip_count = 2 * max(sequence_length - 1, 0) * bit_width
        routes = torch.empty(
            (flip_count, sequence_length),
            dtype=torch.int64,
        )
        lengths = torch.empty_like(routes)
        stats = (ctypes.c_uint64 * len(_STAT_NAMES))()

        result = self.library.rosa_sam_bitflip_routes(
            query_codes.data_ptr() if sequence_length else None,
            key_codes.data_ptr() if sequence_length else None,
            sequence_length,
            bit_width,
            routes.data_ptr() if routes.numel() else None,
            lengths.data_ptr() if lengths.numel() else None,
            stats,
            len(_STAT_NAMES),
        )
        if result != 0:
            raise RuntimeError(f"native SAM bitflip failed with code {result}")
        return NativeSamBitflipResult(
            flipped_routes=routes,
            flipped_lengths=lengths,
            profile={
                name: int(stats[index])
                for index, name in enumerate(_STAT_NAMES)
            },
        )

    def solve_factorized(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        bit_width: int,
    ) -> NativeFactorizedResult:
        self._validate(query_codes, key_codes, bit_width)
        query_codes = query_codes.contiguous()
        key_codes = key_codes.contiguous()
        sequence_length = query_codes.numel()
        stats = (ctypes.c_uint64 * len(_STAT_NAMES))()
        error = ctypes.c_int32()
        handle = self.library.rosa_sam_bitflip_factorized_create(
            query_codes.data_ptr() if sequence_length else None,
            key_codes.data_ptr() if sequence_length else None,
            sequence_length,
            bit_width,
            stats,
            len(_STAT_NAMES),
            ctypes.byref(error),
        )
        if not handle:
            raise RuntimeError(
                f"native factorized SAM bitflip failed with code {error.value}"
            )
        try:
            sizes = (ctypes.c_int64 * 7)()
            result = self.library.rosa_sam_bitflip_factorized_sizes(
                handle,
                sizes,
                len(sizes),
            )
            if result != 0:
                raise RuntimeError(
                    f"native factorized size query failed with code {result}"
                )
            (
                result_length,
                result_bit_width,
                query_flip_count,
                key_length,
                query_change_count,
                delete_change_count,
                override_count,
            ) = map(int, sizes)
            base_routes = torch.empty(result_length, dtype=torch.int32)
            base_lengths = torch.empty_like(base_routes)
            query_offsets = torch.empty(query_flip_count + 1, dtype=torch.int64)
            query_changes = torch.empty(
                (query_change_count, 6), dtype=torch.int32
            )
            delete_offsets = torch.empty(key_length + 1, dtype=torch.int64)
            delete_changes = torch.empty(
                (delete_change_count, 6), dtype=torch.int32
            )
            override_offsets = torch.empty(
                query_flip_count + 1, dtype=torch.int64
            )
            overrides = torch.empty((override_count, 10), dtype=torch.int32)

            result = self.library.rosa_sam_bitflip_factorized_copy(
                handle,
                base_routes.data_ptr() if base_routes.numel() else None,
                base_lengths.data_ptr() if base_lengths.numel() else None,
                query_offsets.data_ptr(),
                query_changes.data_ptr() if query_changes.numel() else None,
                delete_offsets.data_ptr(),
                delete_changes.data_ptr() if delete_changes.numel() else None,
                override_offsets.data_ptr(),
                overrides.data_ptr() if overrides.numel() else None,
            )
            if result != 0:
                raise RuntimeError(
                    f"native factorized copy failed with code {result}"
                )
        finally:
            self.library.rosa_sam_bitflip_factorized_destroy(handle)

        return NativeFactorizedResult(
            sequence_length=result_length,
            bit_width=result_bit_width,
            base_routes=base_routes,
            base_lengths=base_lengths,
            query_offsets=query_offsets,
            query_changes=query_changes,
            key_delete_offsets=delete_offsets,
            key_delete_changes=delete_changes,
            key_override_offsets=override_offsets,
            key_overrides=overrides,
            profile={
                name: int(stats[index])
                for index, name in enumerate(_STAT_NAMES)
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    if not args.build:
        parser.error("pass --build")
    print(build_native_sam_bitflip(args.output, compiler=args.compiler))


if __name__ == "__main__":
    main()
