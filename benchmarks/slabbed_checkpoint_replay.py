"""Exact unlimited RosaSoft VJP with fixed-width diagonal slab replay."""

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
BUILD_ROOT = ROOT / "build" / "slabbed_checkpoint_replay"

# This is an implementation constant, not an estimator or public tuning knob.
SLAB_DIAGONAL_CAPACITY = 512
EXECUTION_PLAN_CODES = {
    "baseline": 0,
    "fused_stats": 1,
    "fused_reverse": 2,
    "fused": 3,
    "tensor_value": 4,
    "fused_stats_value": 5,
    "fused_reverse_value": 6,
    "fused_all": 7,
    "tiled_symbol": 8,
    "tensor_value_tiled_symbol": 12,
    "fully_fused": 15,
}
EXECUTION_PLANS = tuple(EXECUTION_PLAN_CODES)

_MODULE: ModuleType | None = None


def slabbed_live_state_elements(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    *,
    needs_symbol_gradients: bool = True,
    slab_size: int = SLAB_DIAGONAL_CAPACITY,
) -> int:
    """Return the exact number of persistent FP32 scratch elements.

    CUDA pads rows to 32 and diagonal slots to four. The count includes the
    online row triple and one score slab, plus one utility slab when Q or K
    gradients are requested. Output gradients are intentionally excluded.
    """

    if batch_size < 1 or num_heads < 1 or seq_len < 1:
        raise ValueError("batch_size, num_heads, and seq_len must be positive")
    if slab_size < 32 or slab_size % 32:
        raise ValueError("slab_size must be a positive multiple of 32")
    slots = max(1, min(seq_len - 1, slab_size))
    padded_rows = ((seq_len + 31) // 32) * 32
    padded_slots = ((slots + 3) // 4) * 4
    slab_count = 2 if needs_symbol_gradients else 1
    series = batch_size * num_heads
    return series * (3 * seq_len + slab_count * padded_rows * padded_slots)


def _matching_cuda_home() -> Path | None:
    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_slabbed_checkpoint_replay(*, verbose: bool = False) -> ModuleType:
    """Build and return the isolated slabbed-replay research extension."""

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
        name="rosa_soft_slabbed_checkpoint_replay_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_slabbed_replay.cpp"),
            str(SOURCE_ROOT / "rosa_soft_slabbed_replay_kernels.cu"),
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


def slabbed_replay_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    gradient_mask: int = 7,
    plan: str = "baseline",
    value_tile_size: int = 16,
    diagonal_tile_size: int = 32,
    slab_size: int = SLAB_DIAGONAL_CAPACITY,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate every causal route using fixed-capacity replay slabs."""

    if not 1 <= gradient_mask <= 7:
        raise ValueError("gradient_mask must be in [1, 7]")
    if plan not in EXECUTION_PLANS:
        raise ValueError(f"plan must be one of {EXECUTION_PLANS}")
    if value_tile_size not in (16, 32):
        raise ValueError("value_tile_size must be 16 or 32")
    if diagonal_tile_size not in (16, 32):
        raise ValueError("diagonal_tile_size must be 16 or 32")
    if slab_size < 32 or slab_size % 32:
        raise ValueError("slab_size must be a positive multiple of 32")
    if module is None:
        module = load_slabbed_checkpoint_replay()
    return module.slabbed_replay_vjp(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        int(gradient_mask),
        EXECUTION_PLAN_CODES[plan],
        int(value_tile_size),
        int(diagonal_tile_size),
        int(slab_size),
    )


class _SlabbedRosaSoftFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_seed: Tensor,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        module: ModuleType,
    ) -> Tensor:
        output, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
            query, key, value
        )
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
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
            bit for bit, needed in zip((1, 2, 4), requested) if needed
        )
        gradients = slabbed_replay_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            scale=ctx.scale,
            dropout_p=ctx.dropout_p,
            mismatch_scale=ctx.mismatch_scale,
            gradient_mask=gradient_mask,
            module=ctx.module,
        )
        returned = tuple(
            gradient.to(input_tensor.dtype) if needed else None
            for gradient, input_tensor, needed in zip(
                gradients, (query, key, value), requested
            )
        )
        return (*returned, None, None, None, None, None)


def slabbed_checkpoint_rosa_soft(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    module: ModuleType | None = None,
) -> Tensor:
    """Run exact hard ROSA with the exact unlimited slabbed surrogate VJP."""

    if module is None:
        module = load_slabbed_checkpoint_replay()
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward:
        return torch.ops.rosa_soft.hard_forward(query, key, value)[0]
    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    return _SlabbedRosaSoftFunction.apply(
        query,
        key,
        value,
        dropout_seed,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        module,
    )
