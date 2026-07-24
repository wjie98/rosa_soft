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
    rows: int
    competitive_candidates: int
    selected_route_probability: Tensor
    effective_route_count: Tensor
    proxy_hard_error_mean: Tensor
    proxy_hard_error_quantile: Tensor
    proxy_winner_agreement: Tensor
    nonnull_fraction: Tensor
    longest_hard_suffix: Tensor

    def as_float_dict(self) -> Dict[str, float]:
        values = {
            "route_temperature": self.route_temperature,
            "mismatch_penalty": self.mismatch_penalty,
            "rows": float(self.rows),
            "competitive_candidates": float(self.competitive_candidates),
        }
        for name in (
            "selected_route_probability",
            "effective_route_count",
            "proxy_hard_error_mean",
            "proxy_hard_error_quantile",
            "proxy_winner_agreement",
            "nonnull_fraction",
            "longest_hard_suffix",
        ):
            value = getattr(self, name)
            values[name] = float(value.detach().float().cpu())
        return values


def _competitive_mask(
    inspection: RosaSoftInspection,
    top_k: int,
) -> Tensor:
    seq_len = inspection.proxy_scores.size(-1)
    action = torch.arange(
        seq_len,
        device=inspection.proxy_scores.device,
    ).view(
        1,
        1,
        1,
        seq_len,
    )
    valid = inspection.valid_actions.view(1, 1, seq_len, seq_len) & (
        action > 0
    )
    valid = valid.expand_as(inspection.proxy_scores)
    selected = torch.zeros_like(valid)
    count = min(int(top_k), max(seq_len - 1, 1))
    for scores in (inspection.hard_lengths, inspection.proxy_scores):
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

    competitive = _competitive_mask(inspection, int(top_k))
    errors = (
        inspection.proxy_scores - inspection.hard_lengths
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
        inspection.selected_actions.unsqueeze(-1),
    ).mean()
    effective_route_count = (
        inspection.route_probabilities.square()
        .sum(dim=-1)
        .reciprocal()
        .mean()
    )
    best_proxy_score = inspection.route_scores.amax(
        dim=-1,
        keepdim=True,
    )
    action = torch.arange(
        inspection.route_scores.size(-1),
        device=inspection.route_scores.device,
    )
    proxy_winner = torch.where(
        inspection.route_scores == best_proxy_score,
        action,
        0,
    ).amax(dim=-1)
    return RosaSoftDiagnostics(
        route_temperature=inspection.route_temperature,
        mismatch_penalty=inspection.mismatch_penalty,
        rows=inspection.selected_actions.numel(),
        competitive_candidates=int(competitive.sum().item()),
        selected_route_probability=selected_route_probability,
        effective_route_count=effective_route_count,
        proxy_hard_error_mean=error_mean,
        proxy_hard_error_quantile=error_quantile,
        proxy_winner_agreement=(
            proxy_winner == inspection.selected_actions
        ).float().mean(),
        nonnull_fraction=(inspection.selected_actions != 0).float().mean(),
        longest_hard_suffix=inspection.hard_lengths.max(),
    )
