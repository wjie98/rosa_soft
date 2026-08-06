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
)


@dataclass(frozen=True)
class NativeSamBitflipResult:
    flipped_routes: Tensor
    flipped_lengths: Tensor
    profile: dict[str, int]


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
    """Materialized exact counterfactual routes from the C++ SAM core."""

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
