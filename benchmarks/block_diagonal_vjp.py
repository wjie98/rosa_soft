"""Block-diagonal exact RosaSoft VJP research extension."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    make_dropout_seed,
    validate_rosa_soft_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "block_diagonal_vjp"
PLANS = (
    "baseline",
    "block_diagonal",
    "block_tf32",
    "block_tf32_pipeline",
)

_MODULE: ModuleType | None = None


def _matching_cuda_home() -> Path | None:
    import torch

    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_block_diagonal_vjp(*, verbose: bool = False) -> ModuleType:
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
        name="rosa_soft_block_diagonal_vjp_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_block_diagonal_vjp.cpp"),
            str(SOURCE_ROOT / "rosa_soft_block_diagonal_vjp_kernels.cu"),
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


def block_diagonal_vjp(
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
    gradient_mask: int,
    plan: str,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if plan not in PLANS:
        raise ValueError(f"plan must be one of {PLANS}")
    if module is None:
        module = load_block_diagonal_vjp()
    return module.block_diagonal_vjp(
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
        PLANS.index(plan),
    )


class _HardForwardBlockDiagonalVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_seed: Tensor,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        plan: str,
    ) -> Tensor:
        output, packed_query_symbols, packed_key_symbols = (
            torch.ops.rosa_soft.hard_forward(
                query,
                key,
                value,
            )
        )
        ctx.max_suffix_length = max_suffix_length
        ctx.scale = scale
        ctx.dropout_p = dropout_p
        ctx.mismatch_scale = mismatch_scale
        ctx.plan = plan
        ctx.save_for_backward(
            query,
            key,
            value,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        (
            query,
            key,
            value,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
        ) = ctx.saved_tensors
        requested = ctx.needs_input_grad[:3]
        gradient_mask = sum(
            bit for bit, needed in zip((1, 2, 4), requested) if needed
        )
        gradients = block_diagonal_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            max_suffix_length=ctx.max_suffix_length,
            scale=ctx.scale,
            dropout_p=ctx.dropout_p,
            mismatch_scale=ctx.mismatch_scale,
            gradient_mask=gradient_mask,
            plan=ctx.plan,
        )
        returned = tuple(
            gradient.to(input_tensor.dtype) if needed else None
            for gradient, input_tensor, needed in zip(
                gradients,
                (query, key, value),
                requested,
            )
        )
        return (*returned, None, None, None, None, None, None)


def block_diagonal_rosa_soft(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    plan: str = "block_tf32",
) -> Tensor:
    """Run hard ROSA forward with the block-diagonal research VJP."""

    if plan not in PLANS:
        raise ValueError(f"plan must be one of {PLANS}")
    max_suffix_length = validate_rosa_soft_inputs(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    if not query.is_cuda:
        raise ValueError("block_diagonal_rosa_soft requires CUDA tensors")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "block_diagonal_rosa_soft supports float32, float16, and bfloat16"
        )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward:
        return torch.ops.rosa_soft.hard_forward(
            query,
            key,
            value,
        )[0]
    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    return _HardForwardBlockDiagonalVjp.apply(
        query,
        key,
        value,
        dropout_seed,
        max_suffix_length,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        plan,
    )


__all__ = [
    "PLANS",
    "block_diagonal_rosa_soft",
    "block_diagonal_vjp",
    "load_block_diagonal_vjp",
]
