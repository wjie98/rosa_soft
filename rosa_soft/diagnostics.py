"""Read-only summaries for RosaSoft routes, symbols, and gradients."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from numbers import Real
from typing import Dict

import torch
from torch import Tensor

from .testing import RosaSoftInspection


__all__ = [
    "RosaSoftDiagnostics",
    "RosaSoftGradientDiagnostics",
    "TensorGradientDiagnostics",
    "summarize_rosa_soft",
    "summarize_rosa_soft_gradients",
]


@dataclass(frozen=True)
class RosaSoftDiagnostics:
    scale: float
    dropout_p: float
    mismatch_scale: float
    top_k: int
    quantile: float
    route_rows: int
    competitive_route_count: int
    hard_selected_route_probability_mean: Tensor
    soft_route_entropy_mean: Tensor
    effective_route_count_mean: Tensor
    null_route_probability_mean: Tensor
    soft_hard_route_agreement: Tensor
    hard_nonnull_route_fraction: Tensor
    hard_route_lag_mean: Tensor
    max_exact_suffix_length: Tensor
    occupied_exact_suffix_length_count_mean: Tensor
    selected_suffix_length_candidate_count_mean: Tensor
    selected_suffix_length_probability_mass_mean: Tensor
    numerically_nonzero_route_fraction: Tensor
    minimum_nonzero_route_probability: Tensor
    soft_hard_score_gap_mean: Tensor
    soft_hard_score_gap_quantile: Tensor
    query_positive_fraction: Tensor
    key_positive_fraction: Tensor
    query_bit_imbalance: Tensor
    key_bit_imbalance: Tensor
    query_sign_margin_mean: Tensor
    key_sign_margin_mean: Tensor
    query_sign_margin_low_quantile: Tensor
    key_sign_margin_low_quantile: Tensor
    query_softsign_derivative_mean: Tensor
    key_softsign_derivative_mean: Tensor

    def as_float_dict(self) -> Dict[str, float]:
        values = {
            "scale": self.scale,
            "dropout_p": self.dropout_p,
            "mismatch_scale": self.mismatch_scale,
            "top_k": float(self.top_k),
            "quantile": self.quantile,
            "route_rows": float(self.route_rows),
            "competitive_route_count": float(
                self.competitive_route_count
            ),
        }
        for name in self.__dataclass_fields__:
            if name in values:
                continue
            value = getattr(self, name)
            values[name] = float(value.detach().float().cpu())
        return values


@dataclass(frozen=True)
class TensorGradientDiagnostics:
    l2_norm: Tensor
    rms: Tensor
    max_abs: Tensor
    finite_fraction: Tensor
    nonzero_fraction: Tensor
    feature_rms_cv: Tensor


@dataclass(frozen=True)
class RosaSoftGradientDiagnostics:
    query: TensorGradientDiagnostics
    key: TensorGradientDiagnostics
    value: TensorGradientDiagnostics
    qk_to_value_norm_ratio: Tensor

    def as_float_dict(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for role in ("query", "key", "value"):
            summary = getattr(self, role)
            for name in summary.__dataclass_fields__:
                value = getattr(summary, name)
                values[f"{role}_{name}"] = float(
                    value.detach().float().cpu()
                )
        values["qk_to_value_norm_ratio"] = float(
            self.qk_to_value_norm_ratio.detach().float().cpu()
        )
        return values


def _valid_route_masks(
    inspection: RosaSoftInspection,
) -> tuple[Tensor, Tensor, Tensor]:
    seq_len = inspection.soft_suffix_scores.size(-1)
    route_index = torch.arange(
        seq_len,
        device=inspection.soft_suffix_scores.device,
    ).view(1, 1, 1, seq_len)
    valid = inspection.causal_route_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    ).expand_as(inspection.soft_suffix_scores)
    return valid, valid & (route_index > 0), route_index


def _competitive_route_mask(
    inspection: RosaSoftInspection,
    top_k: int,
) -> Tensor:
    _, nonnull, _ = _valid_route_masks(inspection)
    selected = torch.zeros_like(nonnull)
    count = min(int(top_k), max(nonnull.size(-1) - 1, 1))
    for scores in (
        inspection.exact_suffix_lengths,
        inspection.soft_suffix_scores,
    ):
        indices = scores.masked_fill(
            ~nonnull,
            -torch.inf,
        ).topk(count, dim=-1).indices
        selected.scatter_(-1, indices, True)
    return selected & nonnull


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    try:
        normalized_top_k = operator.index(top_k)
    except TypeError as error:
        raise TypeError("top_k must be an integer") from error
    if normalized_top_k < 1:
        raise ValueError("top_k must be >= 1")
    return normalized_top_k


def _validate_quantile(quantile: float) -> float:
    if isinstance(quantile, bool) or not isinstance(quantile, Real):
        raise TypeError("quantile must be a real number")
    normalized_quantile = float(quantile)
    if (
        not math.isfinite(normalized_quantile)
        or not 0.0 <= normalized_quantile <= 1.0
    ):
        raise ValueError("quantile must be in [0, 1]")
    return normalized_quantile


def _nan_scalar(reference: Tensor) -> Tensor:
    return torch.full(
        (),
        float("nan"),
        device=reference.device,
        dtype=torch.float32,
    )


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return _nan_scalar(values)
    return selected.float().mean()


def _symbol_health(
    logits: Tensor,
    low_quantile: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    logits = logits.float()
    positive_fraction = (logits > 0).float().mean()
    per_bit_positive = (logits > 0).float().mean(dim=(0, 1, 2))
    bit_imbalance = (
        2.0 * (per_bit_positive - 0.5).abs()
    ).mean()
    margins = logits.abs()
    return (
        positive_fraction,
        bit_imbalance,
        margins.mean(),
        torch.quantile(margins, low_quantile),
        (1.0 + margins).square().reciprocal().mean(),
    )


@torch.no_grad()
def summarize_rosa_soft(
    inspection: RosaSoftInspection,
    top_k: int = 4,
    quantile: float = 0.95,
) -> RosaSoftDiagnostics:
    top_k = _validate_top_k(top_k)
    quantile = _validate_quantile(quantile)
    valid, nonnull, route_index = _valid_route_masks(inspection)
    competitive = _competitive_route_mask(inspection, top_k)

    residuals = (
        inspection.soft_suffix_scores - inspection.exact_suffix_lengths
    ).abs().masked_select(competitive)
    if residuals.numel() == 0:
        residual_mean = _nan_scalar(inspection.soft_suffix_scores)
        residual_quantile = residual_mean
    else:
        residuals = residuals.float()
        residual_mean = residuals.mean()
        residual_quantile = torch.quantile(residuals, quantile)

    probabilities = inspection.route_probabilities.float()
    selected_routes = inspection.selected_route_indices
    selected_probability = torch.gather(
        probabilities,
        -1,
        selected_routes.unsqueeze(-1),
    ).mean()
    entropy_terms = torch.where(
        probabilities > 0,
        probabilities * probabilities.log(),
        torch.zeros_like(probabilities),
    )
    entropy = -entropy_terms.sum(dim=-1).mean()
    effective_route_count = (
        probabilities.square().sum(dim=-1).reciprocal().mean()
    )

    best_probability = probabilities.amax(dim=-1, keepdim=True)
    soft_winner = torch.where(
        probabilities == best_probability,
        route_index,
        0,
    ).amax(dim=-1)
    hard_nonnull = selected_routes > 0
    row_index = torch.arange(
        probabilities.size(-2),
        device=probabilities.device,
    ).view(1, 1, -1)
    hard_route_lag = _masked_mean(
        row_index - selected_routes,
        hard_nonnull,
    )

    occupied_suffix_length_count = torch.zeros_like(
        selected_routes,
        dtype=torch.float32,
    )
    for suffix_length in range(
        inspection.effective_max_suffix_length + 1
    ):
        occupied_suffix_length_count += (
            nonnull
            & (inspection.exact_suffix_lengths == suffix_length)
        ).any(dim=-1)

    selected_suffix_length = torch.gather(
        inspection.exact_suffix_lengths,
        -1,
        selected_routes.unsqueeze(-1),
    )
    selected_suffix_length_routes = nonnull & (
        inspection.exact_suffix_lengths == selected_suffix_length
    )
    selected_suffix_length_candidate_count = _masked_mean(
        selected_suffix_length_routes.sum(dim=-1),
        hard_nonnull,
    )
    selected_suffix_length_probability_mass = _masked_mean(
        (probabilities * selected_suffix_length_routes).sum(dim=-1),
        hard_nonnull,
    )

    nonzero_probabilities = probabilities.masked_select(
        valid & (probabilities > 0)
    )
    if nonzero_probabilities.numel() == 0:
        minimum_nonzero_probability = _nan_scalar(probabilities)
    else:
        minimum_nonzero_probability = nonzero_probabilities.min()
    numerically_nonzero_route_fraction = (
        valid & (probabilities > 0)
    ).sum().float() / valid.sum().clamp_min(1)

    query_health = _symbol_health(
        inspection.query,
        1.0 - quantile,
    )
    key_health = _symbol_health(
        inspection.key,
        1.0 - quantile,
    )
    return RosaSoftDiagnostics(
        scale=inspection.scale,
        dropout_p=inspection.dropout_p,
        mismatch_scale=inspection.mismatch_scale,
        top_k=top_k,
        quantile=quantile,
        route_rows=selected_routes.numel(),
        competitive_route_count=int(competitive.sum().item()),
        hard_selected_route_probability_mean=selected_probability,
        soft_route_entropy_mean=entropy,
        effective_route_count_mean=effective_route_count,
        null_route_probability_mean=probabilities[..., 0].mean(),
        soft_hard_route_agreement=(
            soft_winner == selected_routes
        ).float().mean(),
        hard_nonnull_route_fraction=hard_nonnull.float().mean(),
        hard_route_lag_mean=hard_route_lag,
        max_exact_suffix_length=(
            inspection.exact_suffix_lengths.max()
        ),
        occupied_exact_suffix_length_count_mean=(
            occupied_suffix_length_count.mean()
        ),
        selected_suffix_length_candidate_count_mean=(
            selected_suffix_length_candidate_count
        ),
        selected_suffix_length_probability_mass_mean=(
            selected_suffix_length_probability_mass
        ),
        numerically_nonzero_route_fraction=(
            numerically_nonzero_route_fraction
        ),
        minimum_nonzero_route_probability=(
            minimum_nonzero_probability
        ),
        soft_hard_score_gap_mean=residual_mean,
        soft_hard_score_gap_quantile=residual_quantile,
        query_positive_fraction=query_health[0],
        key_positive_fraction=key_health[0],
        query_bit_imbalance=query_health[1],
        key_bit_imbalance=key_health[1],
        query_sign_margin_mean=query_health[2],
        key_sign_margin_mean=key_health[2],
        query_sign_margin_low_quantile=query_health[3],
        key_sign_margin_low_quantile=key_health[3],
        query_softsign_derivative_mean=query_health[4],
        key_softsign_derivative_mean=key_health[4],
    )


def _summarize_gradient(gradient: Tensor) -> TensorGradientDiagnostics:
    values = gradient.detach().float()
    finite = torch.isfinite(values)
    finite_values = values.masked_select(finite)
    if finite_values.numel() == 0:
        nan = _nan_scalar(values)
        l2_norm = rms = max_abs = feature_rms_cv = nan
    else:
        l2_norm = finite_values.norm()
        rms = finite_values.square().mean().sqrt()
        max_abs = finite_values.abs().max()
        feature_rms = values.square().mean(
            dim=tuple(range(values.ndim - 1))
        ).sqrt()
        feature_mean = feature_rms.mean()
        feature_rms_cv = feature_rms.std(
            unbiased=False
        ) / feature_mean.clamp_min(torch.finfo(torch.float32).tiny)
    return TensorGradientDiagnostics(
        l2_norm=l2_norm,
        rms=rms,
        max_abs=max_abs,
        finite_fraction=finite.float().mean(),
        nonzero_fraction=(finite & (values != 0)).float().mean(),
        feature_rms_cv=feature_rms_cv,
    )


@torch.no_grad()
def summarize_rosa_soft_gradients(
    grad_query: Tensor,
    grad_key: Tensor,
    grad_value: Tensor,
) -> RosaSoftGradientDiagnostics:
    query = _summarize_gradient(grad_query)
    key = _summarize_gradient(grad_key)
    value = _summarize_gradient(grad_value)
    qk_norm = (
        query.l2_norm.square() + key.l2_norm.square()
    ).sqrt()
    if float(value.l2_norm) == 0.0:
        qk_to_value_norm_ratio = torch.where(
            qk_norm == 0,
            _nan_scalar(qk_norm),
            torch.full_like(qk_norm, float("inf")),
        )
    else:
        qk_to_value_norm_ratio = qk_norm / value.l2_norm
    return RosaSoftGradientDiagnostics(
        query=query,
        key=key,
        value=value,
        qk_to_value_norm_ratio=qk_to_value_norm_ratio,
    )
