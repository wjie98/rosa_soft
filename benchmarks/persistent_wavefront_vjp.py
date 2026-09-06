"""Persistent block-wavefront RosaSoft VJP research controls."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

import rosa_soft  # noqa: F401 - registers the hard-forward operators
from rosa_soft.soft_contract import make_dropout_seed


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
PACKAGE_CUDA_ROOT = ROOT / "rosa_soft" / "csrc" / "cuda"
BUILD_ROOT = ROOT / "build" / "persistent_wavefront_vjp"
SCORE_PLANS = ("sequential", "affine")
VJP_PLANS = ("multilaunch", "persistent")
UTILITY_PLANS = ("tiled", "scalar")
GROUPED_STATS_WORKSPACE_BYTES = 64 << 20
GROUPED_STATS_MAX_SIZE = 4096

_MODULE: ModuleType | None = None


class _UnboundedReplayRosaSoftFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_seed: Tensor,
        group_size: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        module: ModuleType,
    ) -> Tensor:
        output, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
            query, key, value
        )
        ctx.group_size = int(group_size)
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
            bit
            for bit, needed in zip((1, 2, 4), requested)
            if needed
        )
        gradients = unbounded_replay_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            group_size=ctx.group_size,
            scale=ctx.scale,
            dropout_p=ctx.dropout_p,
            mismatch_scale=ctx.mismatch_scale,
            gradient_mask=gradient_mask,
            module=ctx.module,
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


class _WavefrontRosaSoftFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        plan: str,
        module: ModuleType,
    ) -> Tensor:
        output, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
            query,
            key,
            value,
        )
        seed = (
            torch.tensor(0x5A17C9E3, dtype=torch.int64, device=query.device)
            if dropout_p
            else torch.empty(0, dtype=torch.int64, device=query.device)
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.plan = plan
        ctx.module = module
        ctx.save_for_backward(query, key, value, packed_query, packed_key, seed)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value, packed_query, packed_key, seed = ctx.saved_tensors
        gradient_mask = sum(
            bit
            for bit, needed in zip((1, 2, 4), ctx.needs_input_grad[:3])
            if needed
        )
        gradients = wavefront_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            max_suffix_length=ctx.max_suffix_length,
            scale=ctx.scale,
            dropout_p=ctx.dropout_p,
            mismatch_scale=ctx.mismatch_scale,
            gradient_mask=gradient_mask,
            plan=ctx.plan,
            module=ctx.module,
        )
        return (*gradients, None, None, None, None, None, None)


def _matching_cuda_home() -> Path | None:
    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_persistent_wavefront_vjp(*, verbose: bool = False) -> ModuleType:
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
        name="rosa_soft_persistent_wavefront_vjp_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_wavefront.cpp"),
            str(SOURCE_ROOT / "rosa_soft_wavefront_kernels.cu"),
            str(PACKAGE_CUDA_ROOT / "rosa_soft_grouped_checkpoint_kernels.cu"),
            str(PACKAGE_CUDA_ROOT / "rosa_soft_unbounded_kernels.cu"),
        ],
        build_directory=str(BUILD_ROOT),
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-DROSA_SOFT_BENCHMARK_COMPAT=1",
            "-Xptxas",
            "-O3",
        ],
        with_cuda=True,
        verbose=verbose,
    )
    return _MODULE


def unbounded_group_scores(
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    *,
    symbol_dim: int,
    group_start: int,
    group_width: int,
    mismatch_scale: float,
    module: ModuleType | None = None,
) -> Tensor:
    """Return one reusable ``[B,H,T,G]`` unbounded diagonal workspace."""

    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.unbounded_group_scores(
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        int(symbol_dim),
        int(group_start),
        int(group_width),
        float(mismatch_scale),
    )


def unbounded_replay_stats(
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    symbol_dim: int,
    group_size: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    module: ModuleType | None = None,
) -> Tensor:
    """Return ``(maximum, inverse_normalizer, expected_utility)`` per row."""

    if module is None:
        module = load_persistent_wavefront_vjp()
    group_size = min(int(group_size), packed_query_symbols.size(2))
    return module.unbounded_replay_stats(
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        int(symbol_dim),
        group_size,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
    )


def macro_checkpoint_stats(
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    symbol_dim: int,
    macro_diagonals: int,
    slab_size: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    module: ModuleType | None = None,
) -> Tensor:
    """Compute exact row statistics with grouped diagonal ownership."""

    if macro_diagonals not in (32, 64, 96):
        raise ValueError("macro_diagonals must be 32, 64, or 96")
    if module is None:
        module = load_persistent_wavefront_vjp()
    slab_size = min(int(slab_size), packed_query_symbols.size(2))
    return module.macro_checkpoint_stats(
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        int(symbol_dim),
        int(macro_diagonals),
        slab_size,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
    )


def unbounded_replay_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    group_size: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    gradient_mask: int,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run exact unbounded soft-DP VJP with fixed-group linear workspace."""

    if module is None:
        module = load_persistent_wavefront_vjp()
    group_size = min(int(group_size), query.size(1))
    return module.unbounded_replay_vjp(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        group_size,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        int(gradient_mask),
    )


def grouped_checkpoint_live_state_elements(
    seq_len: int,
    resident_ctas: int,
) -> int:
    """Return FP32 checkpoint elements for the reverse persistent grid."""

    if seq_len < 1 or resident_ctas < 1:
        raise ValueError("seq_len and resident_ctas must be positive")
    return resident_ctas * ((seq_len + 31) // 32) * 32


def grouped_checkpoint_stats_group_size(
    query: Tensor,
    *,
    workspace_bytes: int = GROUPED_STATS_WORKSPACE_BYTES,
) -> int:
    """Choose the largest fused-stats slab within a fixed memory budget."""

    if query.ndim != 4:
        raise ValueError("query must have shape [B, T, H, D]")
    batch_size, seq_len, num_heads, _ = query.shape
    if seq_len < 1:
        raise ValueError("sequence length must be positive")
    bytes_per_tile = batch_size * num_heads * seq_len * 3 * 4
    tile_count = max(1, workspace_bytes // max(1, bytes_per_tile))
    group_size = min(GROUPED_STATS_MAX_SIZE, tile_count * 32)
    return min(seq_len, max(1, group_size))


def grouped_checkpoint_reverse(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    row_stats: Tensor,
    *,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    gradient_mask: int = 7,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the checkpoint/replay reverse pass from completed row stats."""

    if not 1 <= gradient_mask <= 7:
        raise ValueError("gradient_mask must be in [1, 7]")
    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.grouped_checkpoint_reverse(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        row_stats.contiguous(),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        int(gradient_mask),
    )


def grouped_checkpoint_reverse_tensor_symbols(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    row_stats: Tensor,
    *,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    gradient_mask: int = 7,
    tensor_symbol_mask: int = 3,
    specialized_replay: bool = False,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the grouped reverse with WMMA Q/K credit contractions."""

    if not 1 <= gradient_mask <= 7:
        raise ValueError("gradient_mask must be in [1, 7]")
    if tensor_symbol_mask not in (0, 1, 2, 3):
        raise ValueError("tensor_symbol_mask must be 0, 1, 2, or 3")
    if specialized_replay and tensor_symbol_mask not in (0, 2, 3):
        raise ValueError(
            "unsupported tensor-symbol mask for specialized replay"
        )
    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.grouped_checkpoint_reverse_tensor_symbols(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        row_stats.contiguous(),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        int(gradient_mask),
        int(tensor_symbol_mask),
        bool(specialized_replay),
    )


def grouped_checkpoint_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    stats_group_size: int | None = None,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    gradient_mask: int = 7,
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run exact stats followed by linear-state grouped checkpoint reverse."""

    if module is None:
        module = load_persistent_wavefront_vjp()
    if stats_group_size is None:
        stats_group_size = grouped_checkpoint_stats_group_size(query)
    row_stats = unbounded_replay_stats(
        value,
        grad_output,
        packed_query_symbols,
        packed_key_symbols,
        dropout_seed,
        symbol_dim=query.size(-1),
        group_size=stats_group_size,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=mismatch_scale,
        module=module,
    )
    return grouped_checkpoint_reverse(
        query,
        key,
        value,
        grad_output,
        packed_query_symbols,
        packed_key_symbols,
        dropout_seed,
        row_stats,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=mismatch_scale,
        gradient_mask=gradient_mask,
        module=module,
    )


def unbounded_replay_rosa_soft(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    group_size: int = 256,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    module: ModuleType | None = None,
) -> Tensor:
    """Run exact hard ROSA with the unbounded diagonal-replay VJP."""

    if module is None:
        module = load_persistent_wavefront_vjp()
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward:
        return torch.ops.rosa_soft.hard_forward(query, key, value)[0]
    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    return _UnboundedReplayRosaSoftFunction.apply(
        query,
        key,
        value,
        dropout_seed,
        int(group_size),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        module,
    )


def wavefront_scores(
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    mismatch_scale: float,
    plan: str = "affine",
    module: ModuleType | None = None,
) -> Tensor:
    """Return debug scores for validating the wavefront state transition."""

    if plan not in SCORE_PLANS:
        raise ValueError(f"plan must be one of {SCORE_PLANS}")
    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.wavefront_scores(
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        int(symbol_dim),
        int(max_suffix_length),
        float(mismatch_scale),
        SCORE_PLANS.index(plan),
    )


def wavefront_stats(
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    utility_plan: str = "tiled",
    module: ModuleType | None = None,
) -> Tensor:
    """Return `(maximum, normalizer, expected_utility)` per query row."""

    if utility_plan not in UTILITY_PLANS:
        raise ValueError(f"utility_plan must be one of {UTILITY_PLANS}")
    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.wavefront_stats(
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        int(symbol_dim),
        int(max_suffix_length),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        UTILITY_PLANS.index(utility_plan),
    )


def persistent_wavefront_stats(
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    utility_plan: str = "tiled",
    module: ModuleType | None = None,
) -> Tensor:
    """Run the cooperative persistent online-statistics wavefront."""

    if utility_plan not in UTILITY_PLANS:
        raise ValueError(f"utility_plan must be one of {UTILITY_PLANS}")
    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.persistent_wavefront_stats(
        value.contiguous(),
        grad_output.contiguous(),
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        dropout_seed.contiguous(),
        int(symbol_dim),
        int(max_suffix_length),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        UTILITY_PLANS.index(utility_plan),
    )


def wavefront_log_gate_vjp(
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    raw_scores: Tensor,
    raw_score_vjp: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    mismatch_scale: float,
    module: ModuleType | None = None,
) -> Tensor:
    """Return the debug reverse-wavefront VJP with respect to log gates."""

    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.wavefront_log_gate_vjp(
        packed_query_symbols.contiguous(),
        packed_key_symbols.contiguous(),
        raw_scores.float().contiguous(),
        raw_score_vjp.float().contiguous(),
        int(symbol_dim),
        int(max_suffix_length),
        float(mismatch_scale),
    )


def wavefront_vjp(
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
    plan: str = "multilaunch",
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run an atomics-free multi-launch or persistent wavefront VJP."""

    if plan not in VJP_PLANS:
        raise ValueError(f"plan must be one of {VJP_PLANS}")
    if module is None:
        module = load_persistent_wavefront_vjp()
    return module.wavefront_vjp(
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
        VJP_PLANS.index(plan),
    )


def wavefront_rosa_soft(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = 2.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
    plan: str = "multilaunch",
    module: ModuleType | None = None,
) -> Tensor:
    """Use exact hard ROSA forward with the wavefront research VJP."""

    if plan not in VJP_PLANS:
        raise ValueError(f"plan must be one of {VJP_PLANS}")
    if module is None:
        module = load_persistent_wavefront_vjp()
    return _WavefrontRosaSoftFunction.apply(
        query,
        key,
        value,
        int(max_suffix_length),
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        plan,
        module,
    )
