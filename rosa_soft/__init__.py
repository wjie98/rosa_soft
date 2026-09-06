"""ROSA hard routing and soft-gradient training."""

import importlib as _importlib

import torch as _torch

__version__ = "0.2.0"

try:
    _C = _importlib.import_module(f"{__name__}._C")
except ImportError as error:
    raise ImportError(
        "rosa_soft requires its native extension; reinstall with "
        "--no-build-isolation"
    ) from error

from .sam import RosaSam, rosa_hard


def _has_cuda() -> bool:
    try:
        return _torch._C._dispatch_has_kernel_for_dispatch_key(
            "rosa_soft::forward", "CUDA"
        ) and _torch._C._dispatch_has_kernel_for_dispatch_key(
            "rosa_soft::backward", "CUDA"
        )
    except RuntimeError:
        return False


if _has_cuda():
    from .soft import rosa_soft
else:

    def rosa_soft(
        q,
        k,
        v,
        cu_seqlens=None,
        *,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
    ):
        del q, k, v, cu_seqlens, scale, dropout_p, mismatch_scale
        raise RuntimeError("rosa_soft requires a CUDA build")


__all__ = ["__version__", "RosaSam", "rosa_hard", "rosa_soft"]
