"""Isolated exact hard-forward indexing experiments."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "indexed_hard_forward"
INDEXED_METHODS = ("occurrence", "certificate", "hybrid")

_MODULE: ModuleType | None = None


def _matching_cuda_home() -> Path | None:
    import torch

    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_indexed_hard_forward(*, verbose: bool = False) -> ModuleType:
    """Build the hard-index study outside the production extension graph."""

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
        name="rosa_soft_indexed_hard_forward_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_indexed_hard.cpp"),
            str(SOURCE_ROOT / "rosa_soft_indexed_hard_kernels.cu"),
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


def pack_sign_bits(
    query: Tensor,
    key: Tensor,
    *,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor]:
    if module is None:
        module = load_indexed_hard_forward()
    return module.pack_sign_bits(query.contiguous(), key.contiguous())


def build_occurrence_index(
    packed_key: Tensor,
    *,
    symbol_dim: int,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor]:
    if module is None:
        module = load_indexed_hard_forward()
    return module.build_occurrence_index(
        packed_key.contiguous(),
        int(symbol_dim),
    )


def indexed_hard_forward_from_packed(
    packed_query: Tensor,
    packed_key: Tensor,
    value: Tensor,
    offsets: Tensor,
    occurrences: Tensor,
    *,
    max_suffix_length: int,
    method: str = "hybrid",
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if method not in INDEXED_METHODS:
        raise ValueError(f"method must be one of {INDEXED_METHODS}")
    if module is None:
        module = load_indexed_hard_forward()
    return module.indexed_hard_forward(
        packed_query.contiguous(),
        packed_key.contiguous(),
        value.contiguous(),
        offsets.contiguous(),
        occurrences.contiguous(),
        int(max_suffix_length),
        INDEXED_METHODS.index(method),
    )


def diagonal_hard_forward_from_packed(
    packed_query: Tensor,
    packed_key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if module is None:
        module = load_indexed_hard_forward()
    return module.diagonal_hard_forward(
        packed_query.contiguous(),
        packed_key.contiguous(),
        value.contiguous(),
        int(max_suffix_length),
    )


__all__ = [
    "INDEXED_METHODS",
    "build_occurrence_index",
    "diagonal_hard_forward_from_packed",
    "indexed_hard_forward_from_packed",
    "load_indexed_hard_forward",
    "pack_sign_bits",
]
