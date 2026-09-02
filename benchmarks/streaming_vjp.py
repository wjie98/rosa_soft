"""Direct benchmark wrapper for the production tiled-streaming VJP."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
PRODUCTION_CUDA_ROOT = ROOT / "rosa_soft" / "csrc" / "cuda"
BUILD_ROOT = ROOT / "build" / "streaming_vjp"

_MODULE: ModuleType | None = None


def _matching_cuda_home() -> Path | None:
    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_streaming_vjp(*, verbose: bool = False) -> ModuleType:
    """Build and cache the direct production-kernel wrapper."""

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
        name="rosa_soft_streaming_vjp_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_streaming_vjp.cpp"),
            str(PRODUCTION_CUDA_ROOT / "rosa_soft_streaming_kernels.cu"),
        ],
        build_directory=str(BUILD_ROOT),
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-Xptxas",
            "-O3",
        ],
        with_cuda=True,
        verbose=verbose,
    )
    return _MODULE


def streaming_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    gradient_mask: int = 7,
    query_tile_size: int = 32,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate every causal route without materializing quadratic state."""

    if module is None:
        module = load_streaming_vjp()
    return module.streaming_vjp(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        int(max_suffix_length),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        int(gradient_mask),
        int(query_tile_size),
    )


__all__ = ["load_streaming_vjp", "streaming_vjp"]
