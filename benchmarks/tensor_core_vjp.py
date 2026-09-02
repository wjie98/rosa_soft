"""Tensor-Core execution experiments for the frozen RosaSoft estimator.

The extension in this module is intentionally separate from the package build.
It exposes small, independently testable CUDA primitives before any component
is considered for a fused research VJP.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "tensor_core_vjp"

GATE_METHODS = ("scalar", "fp16", "int8", "b1")
MATMUL_METHODS = (
    "scalar",
    "fp16",
    "bfloat16",
    "tf32",
    "fp16_scaled",
    "fp16_hilo",
)
SUFFIX_METHODS = ("direct", "warp")

_MODULE: ModuleType | None = None


def _matching_cuda_home() -> Path | None:
    import torch

    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_tensor_core_vjp(*, verbose: bool = False) -> ModuleType:
    """Build and cache the isolated CUDA research extension."""

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
        name="rosa_soft_tensor_core_vjp_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_tensor_core.cpp"),
            str(SOURCE_ROOT / "rosa_soft_tensor_core_kernels.cu"),
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


def gate_matrix(
    query_codes: Tensor,
    key_codes: Tensor,
    *,
    symbol_bits: int,
    mismatch_scale: float = 3.0,
    method: str = "scalar",
    module: ModuleType | None = None,
) -> Tensor:
    """Compute the exponential-Hamming gate matrix by one CUDA method."""

    if method not in GATE_METHODS:
        raise ValueError(f"method must be one of {GATE_METHODS}")
    if module is None:
        module = load_tensor_core_vjp()
    return module.gate_matrix(
        query_codes.contiguous(),
        key_codes.contiguous(),
        int(symbol_bits),
        float(mismatch_scale),
        GATE_METHODS.index(method),
    )


def batched_matmul(
    left: Tensor,
    right: Tensor,
    *,
    method: str,
    module: ModuleType | None = None,
) -> Tensor:
    """Multiply FP32 matrices with an explicitly selected arithmetic path."""

    if method not in MATMUL_METHODS:
        raise ValueError(f"method must be one of {MATMUL_METHODS}")
    if module is None:
        module = load_tensor_core_vjp()
    return module.batched_matmul(
        left.contiguous(),
        right.contiguous(),
        MATMUL_METHODS.index(method),
    )


def suffix_scores(
    gates: Tensor,
    *,
    max_suffix_length: int,
    method: str,
    module: ModuleType | None = None,
) -> Tensor:
    """Evaluate independent diagonal suffix sequences."""

    if method not in SUFFIX_METHODS:
        raise ValueError(f"method must be one of {SUFFIX_METHODS}")
    if module is None:
        module = load_tensor_core_vjp()
    return module.suffix_scores(
        gates.contiguous(),
        int(max_suffix_length),
        SUFFIX_METHODS.index(method),
    )


__all__ = [
    "GATE_METHODS",
    "MATMUL_METHODS",
    "SUFFIX_METHODS",
    "batched_matmul",
    "gate_matrix",
    "load_tensor_core_vjp",
    "suffix_scores",
]
