"""Exact block, diagonal, and warp suffix-score CUDA experiments."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "block_suffix_scan"
METHODS = (
    "thread",
    "diagonal64",
    "warp_suffix",
    "diagonal32",
    "diagonal128",
)
TAIL_METHODS = (
    "physical_block64",
    "tail_thread",
    "tail_warp_suffix",
    "tail_diagonal_thread",
    "tail_diagonal_warp",
)
HYBRID_METHODS = (
    "full_diagonal_thread",
    "hybrid_tail_thread",
    "hybrid_tail_warp_suffix",
    "hybrid_tail_diagonal_warp",
)

_MODULE: ModuleType | None = None


def _matching_cuda_home() -> Path | None:
    import torch

    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_block_suffix_scan(*, verbose: bool = False) -> ModuleType:
    """Build the isolated suffix ownership study outside the package graph."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    cuda_home = _matching_cuda_home()
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = str(cuda_home)
    try:
        import ninja

        os.environ["PATH"] = os.pathsep.join(
            (ninja.BIN_DIR, os.environ.get("PATH", ""))
        )
    except ImportError:
        pass

    from torch.utils import cpp_extension

    if cuda_home is not None:
        cpp_extension.CUDA_HOME = str(cuda_home)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    _MODULE = cpp_extension.load(
        name="rosa_soft_block_suffix_scan_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_block_suffix.cpp"),
            str(SOURCE_ROOT / "rosa_soft_block_suffix_kernels.cu"),
        ],
        build_directory=str(BUILD_ROOT),
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-Xptxas",
            "-O3",
        ],
        with_cuda=True,
        verbose=verbose,
    )
    return _MODULE


def block_suffix_scores(
    packed_query: Tensor,
    packed_key: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    mismatch_scale: float = 3.0,
    method: str = "diagonal64",
    module: ModuleType | None = None,
) -> Tensor:
    """Return exact dense causal RosaSoft raw suffix scores."""

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if module is None:
        module = load_block_suffix_scan()
    return module.block_suffix_scores(
        packed_query.contiguous(),
        packed_key.contiguous(),
        int(symbol_dim),
        int(max_suffix_length),
        float(mismatch_scale),
        METHODS.index(method),
    )


def block_suffix_tail_scores(
    packed_query: Tensor,
    packed_key: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    route_start: int,
    active_queries: int,
    mismatch_scale: float = 3.0,
    method: str = "physical_block64",
    module: ModuleType | None = None,
) -> Tensor:
    """Return one exact causal-tail score tile for an ownership study."""

    if method not in TAIL_METHODS:
        raise ValueError(f"method must be one of {TAIL_METHODS}")
    if module is None:
        module = load_block_suffix_scan()
    return module.block_suffix_tail_scores(
        packed_query.contiguous(),
        packed_key.contiguous(),
        int(symbol_dim),
        int(max_suffix_length),
        float(mismatch_scale),
        int(route_start),
        int(active_queries),
        TAIL_METHODS.index(method),
    )


def block_suffix_hybrid_scores(
    packed_query: Tensor,
    packed_key: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    tile_start: int,
    tail_queries: int,
    mismatch_scale: float = 3.0,
    method: str = "full_diagonal_thread",
    module: ModuleType | None = None,
) -> Tensor:
    """Return a full 64x64 causal tile with an optional specialized tail."""

    if method not in HYBRID_METHODS:
        raise ValueError(f"method must be one of {HYBRID_METHODS}")
    if module is None:
        module = load_block_suffix_scan()
    return module.block_suffix_hybrid_scores(
        packed_query.contiguous(),
        packed_key.contiguous(),
        int(symbol_dim),
        int(max_suffix_length),
        float(mismatch_scale),
        int(tile_start),
        int(tail_queries),
        HYBRID_METHODS.index(method),
    )


__all__ = [
    "HYBRID_METHODS",
    "METHODS",
    "TAIL_METHODS",
    "block_suffix_hybrid_scores",
    "block_suffix_scores",
    "block_suffix_tail_scores",
    "load_block_suffix_scan",
]
