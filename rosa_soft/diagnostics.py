"""Read-only summaries for a detached RosaSoft inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor

from .testing import RosaSoftInspection


__all__ = ["RosaSoftDiagnostics", "summarize_rosa_soft"]


@dataclass(frozen=True)
class RosaSoftDiagnostics:
    route_temperature: float
    mismatch_penalty: float
    route_rows: int
    competitive_route_count: int
    selected_route_probability: Tensor
    effective_route_count: Tensor
    proxy_exact_length_error_mean: Tensor
    proxy_exact_length_error_quantile: Tensor
    proxy_hard_route_agreement: Tensor
    hard_nonnull_route_fraction: Tensor
    max_exact_suffix_length: Tensor

    def as_float_dict(self) -> Dict[str, float]:
        values = {
            "route_temperature": self.route_temperature,
            "mismatch_penalty": self.mismatch_penalty,
            "route_rows": float(self.route_rows),
            "competitive_route_count": float(
                self.competitive_route_count
            ),
        }
        for name in (
            "selected_route_probability",
            "effective_route_count",
            "proxy_exact_length_error_mean",
            "proxy_exact_length_error_quantile",
            "proxy_hard_route_agreement",
            "hard_nonnull_route_fraction",
            "max_exact_suffix_length",
        ):
            value = getattr(self, name)
            values[name] = float(value.detach().float().cpu())
        return values


def _competitive_route_mask(
    inspection: RosaSoftInspection,
    top_k: int,
) -> Tensor:
    seq_len = inspection.proxy_scores.size(-1)
    route_index = torch.arange(
        seq_len,
        device=inspection.proxy_scores.device,
    ).view(
        1,
        1,
        1,
        seq_len,
    )
    valid = inspection.valid_routes.view(1, 1, seq_len, seq_len) & (
        route_index > 0
    )
    valid = valid.expand_as(inspection.proxy_scores)
    selected = torch.zeros_like(valid)
    count = min(int(top_k), max(seq_len - 1, 1))
    for scores in (
        inspection.exact_suffix_lengths,
        inspection.proxy_scores,
    ):
        indices = scores.masked_fill(~valid, -torch.inf).topk(count, dim=-1).indices
        selected.scatter_(-1, indices, True)
    return selected & valid


@torch.no_grad()
def summarize_rosa_soft(
    inspection: RosaSoftInspection,
    top_k: int = 4,
    quantile: float = 0.95,
) -> RosaSoftDiagnostics:
    if int(top_k) < 1:
        raise ValueError("top_k must be >= 1")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    competitive = _competitive_route_mask(inspection, int(top_k))
    errors = (
        inspection.proxy_scores - inspection.exact_suffix_lengths
    ).abs().masked_select(competitive)
    if errors.numel() == 0:
        error_mean = torch.full(
            (),
            float("nan"),
            device=inspection.proxy_scores.device,
            dtype=torch.float32,
        )
        error_quantile = error_mean
    else:
        errors = errors.float()
        error_mean = errors.mean()
        error_quantile = torch.quantile(errors, float(quantile))

    selected_route_probability = torch.gather(
        inspection.route_probabilities,
        -1,
        inspection.selected_routes.unsqueeze(-1),
    ).mean()
    effective_route_count = (
        inspection.route_probabilities.square()
        .sum(dim=-1)
        .reciprocal()
        .mean()
    )
    route_scores = inspection.route_scores
    best_proxy_score = route_scores.amax(
        dim=-1,
        keepdim=True,
    )
    route_index = torch.arange(
        route_scores.size(-1),
        device=route_scores.device,
    )
    proxy_winner = torch.where(
        route_scores == best_proxy_score,
        route_index,
        0,
    ).amax(dim=-1)
    return RosaSoftDiagnostics(
        route_temperature=inspection.route_temperature,
        mismatch_penalty=inspection.mismatch_penalty,
        route_rows=inspection.selected_routes.numel(),
        competitive_route_count=int(competitive.sum().item()),
        selected_route_probability=selected_route_probability,
        effective_route_count=effective_route_count,
        proxy_exact_length_error_mean=error_mean,
        proxy_exact_length_error_quantile=error_quantile,
        proxy_hard_route_agreement=(
            proxy_winner == inspection.selected_routes
        ).float().mean(),
        hard_nonnull_route_fraction=(
            inspection.selected_routes != 0
        ).float().mean(),
        max_exact_suffix_length=inspection.exact_suffix_lengths.max(),
    )
