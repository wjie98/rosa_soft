"""CUDA VJP over the exact native factorized SAM bitflip result."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

from benchmarks.sam_bitflip_native import NativeFactorizedResult


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "sam_bitflip_vjp"

_MODULE: ModuleType | None = None


def _matching_cuda_home() -> Path | None:
    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_descriptor_vjp(*, verbose: bool = False) -> ModuleType:
    """Build and cache the benchmark-only CUDA extension."""

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
        name="rosa_sam_bitflip_vjp_cuda",
        sources=[
            str(SOURCE_ROOT / "sam_bitflip_vjp.cpp"),
            str(SOURCE_ROOT / "sam_bitflip_vjp_cuda.cu"),
        ],
        build_directory=str(BUILD_ROOT),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )
    return _MODULE


def descriptor_vjp(
    factorized: NativeFactorizedResult,
    query_codes: Tensor,
    key_codes: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    module: ModuleType | None = None,
) -> Tensor:
    """Evaluate exact bitflip VJP without materializing route matrices."""

    if module is None:
        module = load_descriptor_vjp()
    if not value.is_cuda or not grad_output.is_cuda:
        raise ValueError("value and grad_output must be CUDA tensors")
    if value.ndim != 2 or value.shape != grad_output.shape:
        raise ValueError("value and grad_output must share shape [T, Dv]")
    if value.size(0) != factorized.sequence_length:
        raise ValueError("value length must match the factorized result")
    if value.dtype != grad_output.dtype:
        raise ValueError("value and grad_output must share dtype")
    if query_codes.ndim != 1 or key_codes.ndim != 1:
        raise ValueError("query_codes and key_codes must be vectors")
    if query_codes.dtype != torch.uint8 or key_codes.dtype != torch.uint8:
        raise ValueError("query_codes and key_codes must use torch.uint8")
    if query_codes.shape != key_codes.shape or query_codes.numel() != value.size(0):
        raise ValueError("packed codes must share the factorized sequence length")

    device = value.device
    arguments = (
        factorized.base_routes.to(device=device, non_blocking=True),
        factorized.query_offsets.to(device=device, non_blocking=True),
        factorized.query_changes.to(device=device, non_blocking=True),
        factorized.key_delete_offsets.to(device=device, non_blocking=True),
        factorized.key_delete_changes.to(device=device, non_blocking=True),
        factorized.key_override_offsets.to(device=device, non_blocking=True),
        factorized.key_overrides.to(device=device, non_blocking=True),
        query_codes.to(device=device, non_blocking=True),
        key_codes.to(device=device, non_blocking=True),
        value.contiguous(),
        grad_output.contiguous(),
        factorized.bit_width,
    )
    return module.descriptor_vjp(*arguments)


__all__ = ["descriptor_vjp", "load_descriptor_vjp"]
