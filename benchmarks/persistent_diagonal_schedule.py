"""Isolate full-diagonal load balancing from the RosaSoft VJP."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "benchmarks" / "csrc"
BUILD_ROOT = ROOT / "build" / "persistent_diagonal_schedule"
SCHEDULES = (
    "hardware_queue",
    "paired_launch",
    "fixed_workers",
    "persistent_queue",
    "fixed_paired",
)
_MODULE: ModuleType | None = None


def load_diagonal_schedule(*, verbose: bool = False) -> ModuleType:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    import ninja
    import torch
    from torch.utils import cpp_extension

    cuda_home = Path.home() / ".local" / f"cuda-{torch.version.cuda}"
    if (cuda_home / "bin" / "nvcc").is_file():
        os.environ["CUDA_HOME"] = str(cuda_home)
        cpp_extension.CUDA_HOME = str(cuda_home)
    os.environ["PATH"] = os.pathsep.join(
        (ninja.BIN_DIR, os.environ.get("PATH", ""))
    )
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    _MODULE = cpp_extension.load(
        name="rosa_soft_diagonal_schedule_cuda",
        sources=[
            str(SOURCE_ROOT / "rosa_soft_diagonal_schedule.cpp"),
            str(SOURCE_ROOT / "rosa_soft_diagonal_schedule_kernels.cu"),
        ],
        build_directory=str(BUILD_ROOT),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        with_cuda=True,
        verbose=verbose,
    )
    return _MODULE


def diagonal_schedule(
    packed_query: Tensor,
    packed_key: Tensor,
    *,
    symbol_dim: int,
    mismatch_scale: float = 3.0,
    schedule: str = "hardware_queue",
    worker_blocks: int = 0,
    module: ModuleType | None = None,
) -> Tensor:
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if module is None:
        module = load_diagonal_schedule()
    return module.diagonal_schedule(
        packed_query.contiguous(),
        packed_key.contiguous(),
        int(symbol_dim),
        float(mismatch_scale),
        SCHEDULES.index(schedule),
        int(worker_blocks),
    )


__all__ = ["SCHEDULES", "diagonal_schedule", "load_diagonal_schedule"]
