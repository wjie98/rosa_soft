"""Configurable physical-tile builds for the block-diagonal VJP study."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "block_diagonal_tile_vjp"
THREAD_VARIANTS = (128, 192, 256)
PLANS = ("baseline", "block_diagonal", "block_tf32")

_MODULES: dict[tuple[int, int | None, int, int], ModuleType] = {}


def _matching_cuda_home() -> Path | None:
    import torch

    if torch.version.cuda is None:
        return None
    candidate = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    return candidate if (candidate / "bin" / "nvcc").is_file() else None


def load_block_diagonal_tile32(
    threads: int,
    *,
    max_registers: int | None = None,
    value_dim: int = 32,
    tile_size: int = 32,
    verbose: bool = False,
) -> ModuleType:
    """Build an isolated 32x32 TF32 VJP with a fixed CTA size."""

    if threads not in THREAD_VARIANTS:
        raise ValueError(f"threads must be one of {THREAD_VARIANTS}")
    if max_registers not in (None, 128):
        raise ValueError("max_registers must be None or 128")
    if value_dim not in (32, 64):
        raise ValueError("value_dim must be 32 or 64")
    if tile_size not in (32, 64):
        raise ValueError("tile_size must be 32 or 64")
    if tile_size == 64 and threads != 256:
        raise ValueError("the 64x64 control requires 256 threads")
    key = (threads, max_registers, value_dim, tile_size)
    if key in _MODULES:
        return _MODULES[key]
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
    suffix = f"threads_{threads}"
    if tile_size != 32:
        suffix = f"tile_{tile_size}_{suffix}"
    if value_dim != 32:
        suffix += f"_dv_{value_dim}"
    if max_registers is not None:
        suffix += f"_regs_{max_registers}"
    build_directory = BUILD_ROOT / suffix
    build_directory.mkdir(parents=True, exist_ok=True)
    cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-Xptxas",
        "-O3",
        f"-DROSA_BLOCK_ROWS={tile_size}",
        f"-DROSA_BLOCK_ROUTES={tile_size}",
        f"-DROSA_BLOCK_THREADS={threads}",
        f"-DROSA_BLOCK_VALUE_DIM={value_dim}",
    ]
    if max_registers is not None:
        cuda_flags.append(f"--maxrregcount={max_registers}")
        cuda_flags.append("-DROSA_BLOCK_MIN_BLOCKS=2")
    module_name = f"rosa_soft_block_diagonal_tile32_t{threads}_cuda"
    if value_dim != 32:
        module_name = (
            f"rosa_soft_block_diagonal_tile32_t{threads}v{value_dim}_cuda"
        )
    if max_registers is not None:
        module_name = (
            f"rosa_soft_block_diagonal_tile32_t{threads}v{value_dim}"
            f"r{max_registers}_cuda"
        )
    if tile_size != 32:
        module_name = (
            f"rosa_soft_block_diagonal_tile{tile_size}_t{threads}"
            f"v{value_dim}r{max_registers or 0}"
            "_cuda"
        )
    module = cpp_extension.load(
        name=module_name,
        sources=[
            str(SOURCE_ROOT / "rosa_soft_block_diagonal_vjp.cpp"),
            str(SOURCE_ROOT / "rosa_soft_block_diagonal_vjp_kernels.cu"),
        ],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        with_cuda=True,
        verbose=verbose,
    )
    _MODULES[key] = module
    return module


def block_diagonal_tile32_vjp(
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
    threads: int,
    max_registers: int | None = None,
    compiled_value_dim: int = 32,
    compiled_tile_size: int = 32,
    plan: str = "block_tf32",
    module: ModuleType | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if plan not in PLANS:
        raise ValueError(f"plan must be one of {PLANS}")
    if module is None:
        module = load_block_diagonal_tile32(
            threads,
            max_registers=max_registers,
            value_dim=compiled_value_dim,
            tile_size=compiled_tile_size,
        )
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


__all__ = [
    "THREAD_VARIANTS",
    "PLANS",
    "block_diagonal_tile32_vjp",
    "load_block_diagonal_tile32",
]
