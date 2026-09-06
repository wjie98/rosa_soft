"""Exact unlimited macro-tile wavefront research operator."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

import rosa_soft  # noqa: F401 - register the exact hard-forward operators
from rosa_soft.soft_contract import make_dropout_seed


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "macro_wavefront_vjp"
EXECUTION_PLANS = (
    "multilaunch",
    "persistent",
    "persistent_barrier",
    "folded",
    "persistent_rows",
)
SCORE_EXECUTION_PLANS = EXECUTION_PLANS
TILE_SIZES = (0, 32, 64, 96, 128)

_MODULE: ModuleType | None = None


class _MacroWavefrontRosaSoftFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_seed: Tensor,
        tile_size: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        plan: str,
        module: ModuleType,
    ) -> Tensor:
        output, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
            query, key, value
        )
        ctx.tile_size = int(tile_size)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.plan = plan
        ctx.module = module
        ctx.save_for_backward(
            query,
            key,
            value,
            packed_query,
            packed_key,
            dropout_seed,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, packed_query, packed_key, seed = ctx.saved_tensors
        requested = ctx.needs_input_grad[:3]
        gradient_mask = sum(
            bit
            for bit, needed in zip((1, 2, 4), requested)
            if needed
        )
        gradients = macro_wavefront_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            tile_size=ctx.tile_size,
            scale=ctx.scale,
            dropout_p=ctx.dropout_p,
            mismatch_scale=ctx.mismatch_scale,
            gradient_mask=gradient_mask,
            plan=ctx.plan,
            module=ctx.module,
        )
        returned = tuple(
            gradient.to(input_tensor.dtype) if needed else None
            for gradient, input_tensor, needed in zip(
                gradients, (query, key, value), requested
            )
        )
        return (*returned, None, None, None, None, None, None, None)


def _matching_cuda_home() -> Path | None:
    import torch

    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_macro_wavefront_vjp(*, verbose: bool = False) -> ModuleType:
    """Build and return the isolated research extension."""

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
        name="rosa_soft_macro_wavefront_vjp_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_macro_wavefront.cpp"),
            str(SOURCE_ROOT / "rosa_soft_macro_wavefront_kernels.cu"),
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


def macro_wavefront_scores(
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    *,
    symbol_dim: int,
    tile_size: int = 0,
    mismatch_scale: float = 3.0,
    plan: str = "folded",
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact debug scores, compact tile edges, and launch metadata.

    Scores use the same route-coordinate layout as the PyTorch oracle: route
    zero is null/zero and key end ``k`` is stored at route ``k + 1``.
    """

    if plan not in SCORE_EXECUTION_PLANS:
        raise ValueError(f"plan must be one of {SCORE_EXECUTION_PLANS}")
    if tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {TILE_SIZES}")
    if module is None:
        module = load_macro_wavefront_vjp()
    return module.macro_wavefront_scores(
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        int(symbol_dim),
        int(tile_size),
        float(mismatch_scale),
        SCORE_EXECUTION_PLANS.index(plan),
    )


def macro_wavefront_stats(
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    symbol_dim: int,
    tile_size: int = 0,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    plan: str = "folded",
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return online ``(max, normalizer, expected utility)`` and edges."""

    if plan not in EXECUTION_PLANS:
        raise ValueError(f"plan must be one of {EXECUTION_PLANS}")
    if tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {TILE_SIZES}")
    if module is None:
        module = load_macro_wavefront_vjp()
    return module.macro_wavefront_stats(
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        int(symbol_dim),
        int(tile_size),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        EXECUTION_PLANS.index(plan),
    )


def macro_wavefront_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    tile_size: int = 0,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    gradient_mask: int = 7,
    plan: str = "folded",
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the exact unlimited macro-tile replay VJP."""

    if plan not in EXECUTION_PLANS:
        raise ValueError(f"plan must be one of {EXECUTION_PLANS}")
    if tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {TILE_SIZES}")
    if not 1 <= gradient_mask <= 7:
        raise ValueError("gradient_mask must be in [1, 7]")
    if module is None:
        module = load_macro_wavefront_vjp()
    return module.macro_wavefront_vjp(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        int(tile_size),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        int(gradient_mask),
        EXECUTION_PLANS.index(plan),
    )


def macro_wavefront_rosa_soft(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    tile_size: int = 0,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    plan: str = "folded",
    module: ModuleType | None = None,
) -> Tensor:
    """Use exact hard ROSA forward with the unlimited macro-tile VJP."""

    if plan not in EXECUTION_PLANS:
        raise ValueError(f"plan must be one of {EXECUTION_PLANS}")
    if tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {TILE_SIZES}")
    if module is None:
        module = load_macro_wavefront_vjp()
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward:
        return torch.ops.rosa_soft.hard_forward(query, key, value)[0]
    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    return _MacroWavefrontRosaSoftFunction.apply(
        query,
        key,
        value,
        dropout_seed,
        int(tile_size),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        plan,
        module,
    )
