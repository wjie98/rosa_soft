import importlib
from typing import Optional

import torch

from .soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
)
from .soft_reference import rosa_soft_reference


__version__ = "0.1.0"

EXTENSION_IMPORT_ERROR: Optional[BaseException] = None
try:
    _C = importlib.import_module(f"{__name__}._C")
except (ImportError, OSError, RuntimeError) as error:
    _C = None
    EXTENSION_IMPORT_ERROR = error

EXTENSION_IMPORT_ERROR_MESSAGE = (
    None
    if EXTENSION_IMPORT_ERROR is None
    else (
        f"{type(EXTENSION_IMPORT_ERROR).__name__}: "
        f"{EXTENSION_IMPORT_ERROR}"
    )
)


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


def _has_cuda_kernels(*operators: str) -> bool:
    return all(
        _has_dispatch_kernel(f"rosa_soft::{operator}", "CUDA")
        for operator in operators
    )


HAS_COMPILED_EXTENSION = _C is not None
HAS_ROSA_RUNTIME = HAS_COMPILED_EXTENSION and _has_custom_class(
    "rosa_soft",
    "RosaRuntime",
)
HAS_ROSA_SOFT_CUDA = HAS_COMPILED_EXTENSION and _has_cuda_kernels(
    "soft_forward",
    "soft_backward",
)
HAS_RWKV7_CLAMPW_CUDA = HAS_COMPILED_EXTENSION and _has_cuda_kernels(
    "rwkv7_clampw_forward",
    "rwkv7_clampw_backward",
)
HAS_RWKV7_STATE_CLAMPW_CUDA = (
    HAS_COMPILED_EXTENSION
    and _has_cuda_kernels(
        "rwkv7_state_clampw_forward",
        "rwkv7_state_clampw_backward",
    )
)
HAS_RWKV7_STATE_PASSING_CLAMPW_CUDA = (
    HAS_COMPILED_EXTENSION
    and _has_cuda_kernels(
        "rwkv7_statepassing_clampw_forward",
        "rwkv7_statepassing_clampw_backward",
    )
)
HAS_RWKV7_ALBATROSS_CUDA = (
    HAS_COMPILED_EXTENSION
    and _has_cuda_kernels(
        "rwkv7_albatross_forward_w0_fp16_dither",
    )
)
HAS_RWKV7_CUDA = all(
    (
        HAS_RWKV7_CLAMPW_CUDA,
        HAS_RWKV7_STATE_CLAMPW_CUDA,
        HAS_RWKV7_STATE_PASSING_CLAMPW_CUDA,
        HAS_RWKV7_ALBATROSS_CUDA,
    )
)

if not HAS_COMPILED_EXTENSION:
    _build_variant = "reference"
elif HAS_RWKV7_CUDA:
    _build_variant = "cuda-rwkv7"
elif HAS_ROSA_SOFT_CUDA:
    _build_variant = "cuda"
elif HAS_ROSA_RUNTIME:
    _build_variant = "cpu-runtime"
else:
    _build_variant = "extension"

BUILD_CAPABILITIES = {
    "version": __version__,
    "variant": _build_variant,
    "compiled_extension": HAS_COMPILED_EXTENSION,
    "rosa_runtime": HAS_ROSA_RUNTIME,
    "rosa_soft_cuda": HAS_ROSA_SOFT_CUDA,
    "rwkv7_cuda": HAS_RWKV7_CUDA,
    "rwkv7_clampw_cuda": HAS_RWKV7_CLAMPW_CUDA,
    "rwkv7_state_clampw_cuda": HAS_RWKV7_STATE_CLAMPW_CUDA,
    "rwkv7_state_passing_clampw_cuda": (
        HAS_RWKV7_STATE_PASSING_CLAMPW_CUDA
    ),
    "rwkv7_albatross_cuda": HAS_RWKV7_ALBATROSS_CUDA,
    "extension_import_error": EXTENSION_IMPORT_ERROR_MESSAGE,
}


def _raise_unavailable(feature: str, build_hint: str) -> None:
    message = (
        f"{feature} is unavailable in the "
        f"{BUILD_CAPABILITIES['variant']!r} rosa_soft build. {build_hint}"
    )
    if EXTENSION_IMPORT_ERROR is not None:
        message += (
            " The compiled extension failed to import: "
            f"{EXTENSION_IMPORT_ERROR_MESSAGE}"
        )
    raise RuntimeError(message) from EXTENSION_IMPORT_ERROR


if HAS_ROSA_SOFT_CUDA:
    from .soft import rosa_soft
else:
    def rosa_soft(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length=32,
        route_temperature=ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
        mismatch_penalty=ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ):
        del (
            query_logits,
            key_logits,
            payload_logits,
            max_suffix_length,
            route_temperature,
            mismatch_penalty,
        )
        _raise_unavailable(
            "rosa_soft CUDA training operator",
            "Build with USE_CUDA=1 (or auto with CUDA_HOME available).",
        )


if HAS_ROSA_RUNTIME:
    from .runtime import RosaRuntime, RosaRuntimeWork
else:
    class RosaRuntime:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            _raise_unavailable(
                "RosaRuntime",
                "Build with ROSA_BUILD_EXTENSION=1.",
            )


    class RosaRuntimeWork:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            _raise_unavailable(
                "RosaRuntimeWork",
                "Build with ROSA_BUILD_EXTENSION=1.",
            )


__all__ = [
    "__version__",
    "BUILD_CAPABILITIES",
    "EXTENSION_IMPORT_ERROR",
    "EXTENSION_IMPORT_ERROR_MESSAGE",
    "HAS_COMPILED_EXTENSION",
    "HAS_ROSA_RUNTIME",
    "HAS_ROSA_SOFT_CUDA",
    "HAS_RWKV7_CUDA",
    "HAS_RWKV7_CLAMPW_CUDA",
    "HAS_RWKV7_STATE_CLAMPW_CUDA",
    "HAS_RWKV7_STATE_PASSING_CLAMPW_CUDA",
    "HAS_RWKV7_ALBATROSS_CUDA",
    "ROSA_SOFT_DEFAULT_MISMATCH_PENALTY",
    "ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE",
    "RosaRuntime",
    "RosaRuntimeWork",
    "rosa_soft",
    "rosa_soft_reference",
]
