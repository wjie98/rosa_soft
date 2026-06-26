import torch
from . import _C, ops

from .runtime import RosaRuntime, RosaRuntimeWork
from .anchor import (
    AttentionTelemetry,
    ROSA_ANCHOR_DEFAULT_LOGIT_EPSILON,
    ROSA_ANCHOR_DEFAULT_QK_DAMPER_STRENGTH,
    estimate_rosa_anchor_scale,
    resolve_rosa_anchor_scale,
    rosa_anchor_lambda,
    rosa_anchor_ops,
)
from .training import RosaAnchorScaleConfig, RosaAnchorScaleController

__all__ = [
    "AttentionTelemetry",
    "ROSA_ANCHOR_DEFAULT_LOGIT_EPSILON",
    "ROSA_ANCHOR_DEFAULT_QK_DAMPER_STRENGTH",
    "RosaAnchorScaleConfig",
    "RosaAnchorScaleController",
    "RosaRuntime",
    "RosaRuntimeWork",
    "estimate_rosa_anchor_scale",
    "ops",
    "resolve_rosa_anchor_scale",
    "rosa_anchor_lambda",
    "rosa_anchor_ops",
]
