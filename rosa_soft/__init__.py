import importlib as _importlib
from dataclasses import dataclass as _dataclass

import torch as _torch

from .soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P as _DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE as _DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE as _DEFAULT_SCALE,
)
from .soft_reference import (
    rosa_soft_reference,
    rosa_soft_varlen_reference,
)


__version__ = "0.1.0"


def _load_compiled_extension():
    module_name = f"{__name__}._C"
    try:
        return _importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return None


_C = _load_compiled_extension()


def _has_custom_class(namespace: str, name: str) -> bool:
    try:
        getattr(getattr(_torch.classes, namespace), name)
    except (AttributeError, RuntimeError):
        return False
    return True


def _has_dispatch_kernel(operator: str, dispatch_key: str) -> bool:
    try:
        return bool(
            _torch._C._dispatch_has_kernel_for_dispatch_key(
                operator,
                dispatch_key,
            )
        )
    except RuntimeError:
        return False


def _has_cuda_kernels(*operators: str) -> bool:
    return all(
        _has_dispatch_kernel(f"rosa_soft::{operator}", "CUDA")
        for operator in operators
    )


def _require_complete_cuda_registration(
    *operator_flags: bool,
) -> bool:
    if any(operator_flags) and not all(operator_flags):
        raise RuntimeError(
            "rosa_soft._C loaded with an incomplete CUDA operator "
            "registration; the extension is stale or incompatible with "
            "this package"
        )
    return all(operator_flags)


_has_compiled_extension = _C is not None
_has_rosa_runtime = _has_compiled_extension and _has_custom_class(
    "rosa_soft",
    "RosaRuntime",
)
_required_cuda_operators = (
    "hard_forward",
    "hard_forward_varlen",
    "surrogate_vjp_masked",
    "surrogate_vjp_varlen_masked",
)
_cuda_operator_flags = tuple(
    _has_compiled_extension and _has_cuda_kernels(operator)
    for operator in _required_cuda_operators
)
_has_rosa_soft_cuda = _require_complete_cuda_registration(
    *_cuda_operator_flags,
)

if _has_compiled_extension and not _has_rosa_runtime:
    raise RuntimeError(
        "rosa_soft._C loaded without the required RosaRuntime registration; "
        "the extension is stale or incompatible with this package"
    )

if _has_rosa_soft_cuda:
    _build_variant = "cuda"
elif _has_rosa_runtime:
    _build_variant = "cpu-runtime"
else:
    _build_variant = "reference"


@_dataclass(frozen=True, slots=True)
class BuildCapabilities:
    variant: str
    compiled_extension: bool
    rosa_runtime: bool
    rosa_soft_cuda: bool


BUILD_CAPABILITIES = BuildCapabilities(
    variant=_build_variant,
    compiled_extension=_has_compiled_extension,
    rosa_runtime=_has_rosa_runtime,
    rosa_soft_cuda=_has_rosa_soft_cuda,
)


def _raise_unavailable(feature: str, build_hint: str) -> None:
    raise RuntimeError(
        f"{feature} is unavailable in the "
        f"{BUILD_CAPABILITIES.variant!r} rosa_soft build. {build_hint}"
    )


if _has_rosa_soft_cuda:
    from .soft import rosa_soft, rosa_soft_varlen
else:
    def rosa_soft(
        query,
        key,
        value,
        *,
        max_suffix_length=32,
        scale=_DEFAULT_SCALE,
        dropout_p=_DEFAULT_DROPOUT_P,
        mismatch_scale=_DEFAULT_MISMATCH_SCALE,
    ):
        del (
            query,
            key,
            value,
            max_suffix_length,
            scale,
            dropout_p,
            mismatch_scale,
        )
        _raise_unavailable(
            "rosa_soft CUDA training operator",
            "Build with USE_CUDA=1 (or auto with CUDA_HOME available).",
        )

    def rosa_soft_varlen(
        query,
        key,
        value,
        cu_seqlens,
        *,
        max_suffix_length=32,
        scale=_DEFAULT_SCALE,
        dropout_p=_DEFAULT_DROPOUT_P,
        mismatch_scale=_DEFAULT_MISMATCH_SCALE,
    ):
        del (
            query,
            key,
            value,
            cu_seqlens,
            max_suffix_length,
            scale,
            dropout_p,
            mismatch_scale,
        )
        _raise_unavailable(
            "rosa_soft_varlen CUDA training operator",
            "Build with USE_CUDA=1 (or auto with CUDA_HOME available).",
        )


if _has_rosa_runtime:
    from .runtime import RosaRuntime
else:
    class RosaRuntime:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            _raise_unavailable(
                "RosaRuntime",
                "Build with ROSA_BUILD_EXTENSION=1.",
            )


__all__ = [
    "__version__",
    "BUILD_CAPABILITIES",
    "RosaRuntime",
    "rosa_soft",
    "rosa_soft_reference",
    "rosa_soft_varlen",
    "rosa_soft_varlen_reference",
]
