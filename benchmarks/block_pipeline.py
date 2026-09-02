"""Warp-specialized double-buffer study for a block route sweep."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "block_pipeline"
METHODS = ("sequential", "warp_specialized")

_MODULE: ModuleType | None = None


def _matching_cuda_home() -> Path | None:
    import torch

    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_block_pipeline(*, verbose: bool = False) -> ModuleType:
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
        name="rosa_soft_block_pipeline_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_block_pipeline.cpp"),
            str(SOURCE_ROOT / "rosa_soft_block_pipeline_kernels.cu"),
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


def block_pipeline(
    packed_query: Tensor,
    packed_key: Tensor,
    grad_output: Tensor,
    value: Tensor,
    *,
    symbol_dim: int,
    mismatch_scale: float = 3.0,
    method: str = "sequential",
    module: ModuleType | None = None,
) -> Tensor:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if module is None:
        module = load_block_pipeline()
    return module.block_pipeline(
        packed_query.contiguous(),
        packed_key.contiguous(),
        grad_output.contiguous(),
        value.contiguous(),
        int(symbol_dim),
        float(mismatch_scale),
        METHODS.index(method),
    )


__all__ = ["METHODS", "block_pipeline", "load_block_pipeline"]
