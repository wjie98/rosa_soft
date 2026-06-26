import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor


__all__ = [
    "AttentionTelemetry",
    "estimate_rosa_anchor_scale",
    "resolve_rosa_anchor_scale",
    "rosa_anchor_lambda",
    "rosa_anchor_ops",
]


AUTO_SCALE_TARGET_TOP_PROB = 0.50
ROSA_ANCHOR_LAMBDA_POWER = 1.5

_ROSA_ANCHOR_AUTO_SCALE_INTERCEPT = -0.293063
_ROSA_ANCHOR_AUTO_SCALE_SEQ_EXP = 0.176767
_ROSA_ANCHOR_AUTO_SCALE_BITS_EXP = 0.762136
_ROSA_ANCHOR_AUTO_SCALE_WINDOW_EXP = -0.002577
_ROSA_ANCHOR_AUTO_SCALE_TARGET_LOGIT_EXP = 0.640061


@dataclass(frozen=True)
class AttentionTelemetry:
    top_prob: Tensor
    prob_gap: Tensor
    candidate_top_prob: Tensor
    candidate_prob_gap: Tensor
    candidate_mass: Tensor
    null_prob: Tensor
    entropy_norm: Tensor
    candidate_entropy_norm: Tensor
    rows: int
    scale: Tensor
    effective_window_mean: Optional[Tensor] = None
    effective_window_ratio: Optional[Tensor] = None
    truncated_fraction: Optional[Tensor] = None

    def as_float_dict(self) -> Dict[str, float]:
        values = {
            "top_prob": float(self.top_prob.detach().float().cpu()),
            "prob_gap": float(self.prob_gap.detach().float().cpu()),
            "candidate_top_prob": float(self.candidate_top_prob.detach().float().cpu()),
            "candidate_prob_gap": float(self.candidate_prob_gap.detach().float().cpu()),
            "candidate_mass": float(self.candidate_mass.detach().float().cpu()),
            "null_prob": float(self.null_prob.detach().float().cpu()),
            "entropy_norm": float(self.entropy_norm.detach().float().cpu()),
            "candidate_entropy_norm": float(self.candidate_entropy_norm.detach().float().cpu()),
            "rows": float(self.rows),
            "scale": float(self.scale.detach().float().cpu()),
        }
        if self.effective_window_mean is not None:
            values["effective_window_mean"] = float(self.effective_window_mean.detach().float().cpu())
        if self.effective_window_ratio is not None:
            values["effective_window_ratio"] = float(self.effective_window_ratio.detach().float().cpu())
        if self.truncated_fraction is not None:
            values["truncated_fraction"] = float(self.truncated_fraction.detach().float().cpu())
        return values


def _target_logit(target_top_prob: float) -> float:
    target = min(max(float(target_top_prob), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(target / (1.0 - target))


def estimate_rosa_anchor_scale(
    seq_len: int,
    qk_bits: int,
    window_size: int,
    target_top_prob: float = AUTO_SCALE_TARGET_TOP_PROB,
) -> float:
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if qk_bits < 1:
        raise ValueError("qk_bits must be >= 1")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    return math.exp(
        _ROSA_ANCHOR_AUTO_SCALE_INTERCEPT
        + _ROSA_ANCHOR_AUTO_SCALE_SEQ_EXP * math.log(float(seq_len))
        + _ROSA_ANCHOR_AUTO_SCALE_BITS_EXP * math.log(float(qk_bits))
        + _ROSA_ANCHOR_AUTO_SCALE_WINDOW_EXP * math.log(float(window_size))
        + _ROSA_ANCHOR_AUTO_SCALE_TARGET_LOGIT_EXP * _target_logit(target_top_prob)
    )


def resolve_rosa_anchor_scale(
    scale: Optional[float],
    window_size: int,
    seq_len: Optional[int] = None,
    qk_bits: Optional[int] = None,
) -> float:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if scale is None:
        if seq_len is not None and qk_bits is not None:
            return estimate_rosa_anchor_scale(
                seq_len=int(seq_len),
                qk_bits=int(qk_bits),
                window_size=int(window_size),
            )
        return 1.0 / math.sqrt(float(window_size))
    return float(scale)


def rosa_anchor_lambda(
    scale: float,
    window_size: int,
    power: float = ROSA_ANCHOR_LAMBDA_POWER,
) -> float:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    return math.log(float(window_size)) + float(power) * math.log1p(abs(float(scale)))


def _suffix_stats_to_metrics(stats: Tensor, like: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    stats_f = stats.detach().to(device=like.device, dtype=torch.float32)
    effective_offsets = stats_f[0]
    full_offsets = stats_f[1].clamp_min(1.0)
    truncated_candidates = stats_f[2]
    candidates = stats_f[3].clamp_min(1.0)
    return (
        effective_offsets / candidates,
        effective_offsets / full_offsets,
        truncated_candidates / candidates,
    )


def _attention_telemetry_from_cuda_stats(stats: Tensor, scale: float | Tensor) -> AttentionTelemetry:
    if stats.numel() < 14:
        raise ValueError("RosaAnchor CUDA telemetry stats must have 14 elements")
    stats_f = stats.detach().float()
    rows = stats_f[9].clamp_min(1.0)
    candidate_rows = stats_f[4].clamp_min(1.0)
    scale_t = torch.as_tensor(scale, device=stats.device, dtype=torch.float32).detach()
    effective_window_mean, effective_window_ratio, truncated_fraction = _suffix_stats_to_metrics(
        stats_f[10:14],
        stats,
    )
    return AttentionTelemetry(
        top_prob=stats_f[0] / rows,
        prob_gap=stats_f[1] / rows,
        candidate_top_prob=stats_f[2] / candidate_rows,
        candidate_prob_gap=stats_f[3] / candidate_rows,
        candidate_mass=stats_f[5] / rows,
        null_prob=stats_f[6] / rows,
        entropy_norm=stats_f[7] / rows,
        candidate_entropy_norm=stats_f[8] / candidate_rows,
        rows=int(float(rows.detach().cpu())),
        scale=scale_t,
        effective_window_mean=effective_window_mean,
        effective_window_ratio=effective_window_ratio,
        truncated_fraction=truncated_fraction,
    )


def _validate_inputs(query: Tensor, key: Tensor, value: Tensor, window_size: int, logit_epsilon: float) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (B, T, H, D)")
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value must share batch and sequence dimensions")
    if query.size(2) != key.size(2):
        raise ValueError("query and key must have the same number of heads")
    if query.size(2) % value.size(2) != 0:
        raise ValueError("query heads must be divisible by value heads")
    if query.size(3) != key.size(3):
        raise ValueError("query and key bit dimensions must match")
    if query.size(3) < 1 or query.size(3) > 32:
        raise ValueError("query/key bit dimension must be in [1, 32]")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size > 512:
        raise ValueError("RosaAnchor CUDA backward currently supports window_size <= 512")
    if logit_epsilon < 0.0:
        raise ValueError("logit_epsilon must be >= 0")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("RosaAnchor CUDA requires CUDA tensors")


def _op_args(scale: float, window_size: int, logit_epsilon: float) -> tuple:
    return (
        float(scale),
        int(window_size),
        float(logit_epsilon),
    )


class _RosaAnchorFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        window_size: int,
        logit_epsilon: float,
        qk_damper_strength: float,
    ) -> Tensor:
        ctx.scale = float(scale)
        ctx.window_size = int(window_size)
        ctx.logit_epsilon = float(logit_epsilon)
        ctx.qk_damper_strength = float(qk_damper_strength)
        query_c = query.contiguous()
        key_c = key.contiguous()
        value_c = value.contiguous()
        args = _op_args(ctx.scale, ctx.window_size, ctx.logit_epsilon)
        needs_backward = query.requires_grad or key.requires_grad or value.requires_grad
        if needs_backward:
            result = torch.ops.rosa_soft.rosa_anchor_forward_with_bits(query_c, key_c, value_c, *args)
            if len(result) >= 5:
                output, q_bits, k_bits, row_max, row_denom = result[:5]
                ctx.save_for_backward(query_c, key_c, value_c, q_bits, k_bits, row_max, row_denom)
                ctx.has_saved_row_stats = True
            else:
                output, q_bits, k_bits = result
                ctx.save_for_backward(query_c, key_c, value_c, q_bits, k_bits)
                ctx.has_saved_row_stats = False
            return output
        ctx.save_for_backward(query_c, key_c, value_c)
        ctx.has_saved_row_stats = False
        return torch.ops.rosa_soft.rosa_anchor_forward(query_c, key_c, value_c, *args)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        query, key, value = saved[:3]
        args = _op_args(ctx.scale, ctx.window_size, ctx.logit_epsilon)
        backward_tail = (float(ctx.qk_damper_strength),)
        if len(saved) >= 5:
            q_bits, k_bits = saved[3], saved[4]
            if getattr(ctx, "has_saved_row_stats", False):
                row_max, row_denom = saved[5], saved[6]
                grad_query, grad_key, grad_value = torch.ops.rosa_soft.rosa_anchor_backward_with_bits_and_stats(
                    query,
                    key,
                    value,
                    grad_output.contiguous(),
                    q_bits,
                    k_bits,
                    row_max,
                    row_denom,
                    *args,
                    *backward_tail,
                )
            else:
                grad_query, grad_key, grad_value = torch.ops.rosa_soft.rosa_anchor_backward_with_bits(
                    query,
                    key,
                    value,
                    grad_output.contiguous(),
                    q_bits,
                    k_bits,
                    *args,
                    *backward_tail,
                )
        else:
            grad_query, grad_key, grad_value = torch.ops.rosa_soft.rosa_anchor_backward(
                query,
                key,
                value,
                grad_output.contiguous(),
                *args,
                *backward_tail,
            )
        return (
            grad_query.to(dtype=query.dtype),
            grad_key.to(dtype=key.dtype),
            grad_value.to(dtype=value.dtype),
            None,
            None,
            None,
            None,
        )


class _RosaAnchorWithTelemetryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        window_size: int,
        logit_epsilon: float,
        qk_damper_strength: float,
    ) -> tuple[Tensor, Tensor]:
        ctx.scale = float(scale)
        ctx.window_size = int(window_size)
        ctx.logit_epsilon = float(logit_epsilon)
        ctx.qk_damper_strength = float(qk_damper_strength)
        query_c = query.contiguous()
        key_c = key.contiguous()
        value_c = value.contiguous()
        output, stats, q_bits, k_bits, row_max, row_denom = torch.ops.rosa_soft.rosa_anchor_forward_with_stats(
            query_c,
            key_c,
            value_c,
            *_op_args(ctx.scale, ctx.window_size, ctx.logit_epsilon),
        )
        ctx.save_for_backward(query_c, key_c, value_c, q_bits, k_bits, row_max, row_denom)
        ctx.mark_non_differentiable(stats)
        return output, stats

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_stats: Tensor):
        del grad_stats
        query, key, value, q_bits, k_bits, row_max, row_denom = ctx.saved_tensors
        grad_query, grad_key, grad_value = torch.ops.rosa_soft.rosa_anchor_backward_with_bits_and_stats(
            query,
            key,
            value,
            grad_output.contiguous(),
            q_bits,
            k_bits,
            row_max,
            row_denom,
            *_op_args(ctx.scale, ctx.window_size, ctx.logit_epsilon),
            float(ctx.qk_damper_strength),
        )
        return (
            grad_query.to(dtype=query.dtype),
            grad_key.to(dtype=key.dtype),
            grad_value.to(dtype=value.dtype),
            None,
            None,
            None,
            None,
        )


def rosa_anchor_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    window_size: int = 32,
    scale: Optional[float] = None,
    return_telemetry: bool = False,
    logit_epsilon: float = 0.0,
    qk_damper_strength: float = 0.0,
) -> Union[Tensor, Tuple[Tensor, AttentionTelemetry]]:
    """Production RosaAnchor CUDA proxy.

    Inputs use `(B, T, H, D)` layout. Q/K bits must be in `[1, 32]`, and value
    heads may be grouped-query style as long as `H_q % H_v == 0`.
    """

    _validate_inputs(query, key, value, int(window_size), float(logit_epsilon))
    resolved_scale = resolve_rosa_anchor_scale(
        scale,
        int(window_size),
        seq_len=query.size(1),
        qk_bits=query.size(3),
    )
    if return_telemetry:
        output, stats = _RosaAnchorWithTelemetryFunction.apply(
            query,
            key,
            value,
            float(resolved_scale),
            int(window_size),
            float(logit_epsilon),
            float(qk_damper_strength),
        )
        return output, _attention_telemetry_from_cuda_stats(stats, resolved_scale)
    return _RosaAnchorFunction.apply(
        query,
        key,
        value,
        float(resolved_scale),
        int(window_size),
        float(logit_epsilon),
        float(qk_damper_strength),
    )
