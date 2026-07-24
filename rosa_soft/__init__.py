import torch

from .soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
)
from .soft_reference import (
    rosa_soft_reference,
)

try:
    from . import _C
except (ImportError, OSError):
    _C = None


def _has_custom_class(namespace: str, name: str) -> bool:
    try:
        getattr(getattr(torch.classes, namespace), name)
    except (AttributeError, RuntimeError):
        return False
    return True


def _has_dispatch_kernel(operator: str, dispatch_key: str) -> bool:
    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                operator,
                dispatch_key,
            )
        )
    except RuntimeError:
        return False


HAS_COMPILED_EXTENSION = _C is not None
HAS_ROSA_RUNTIME = HAS_COMPILED_EXTENSION and _has_custom_class(
    "rosa_soft",
    "RosaRuntime",
)
HAS_ROSA_SOFT_CUDA = (
    HAS_COMPILED_EXTENSION
    and _has_dispatch_kernel("rosa_soft::soft_forward", "CUDA")
    and _has_dispatch_kernel("rosa_soft::soft_backward", "CUDA")
)

__all__ = [
    "HAS_COMPILED_EXTENSION",
    "HAS_ROSA_RUNTIME",
    "HAS_ROSA_SOFT_CUDA",
    "ROSA_SOFT_DEFAULT_MISMATCH_PENALTY",
    "ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE",
    "rosa_soft_reference",
]

if HAS_ROSA_SOFT_CUDA:
    from .soft import rosa_soft

    __all__.append("rosa_soft")

if HAS_ROSA_RUNTIME:
    from .runtime import RosaRuntime, RosaRuntimeWork

    __all__ += [
        "RosaRuntime",
        "RosaRuntimeWork",
    ]
