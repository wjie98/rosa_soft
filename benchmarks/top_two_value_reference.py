"""Unbounded RosaSoft reference with explicit route/value gradient roles.

This research-only module keeps the exact hard ROSA forward.  Its default
backward is algebraically identical to the full-horizon discovery estimator,
but writes the Q/K and value paths separately so value-credit ablations do not
silently change route credit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    validate_rosa_soft_inputs,
)
from rosa_soft.soft_reference import (
    _causal_route_mask,
    _expand_value_heads,
    _gather_routed_values,
    _hard_route_forward,
    _hard_sign,
    _hard_sign_with_softsign_vjp,
    _masked_route_scores,
    _pairwise_soft_match_gates,
    _reference_compute_dtype,
    _suffix_score_utility,
)


QK_GRADIENTS = (
    "dense",
    "top2",
    "winner_rest",
    "hard_winner_rest",
    "no_prior",
    "hard_null_gate",
)
VALUE_GRADIENTS = ("dense", "selected")

_QK_IDS = {name: index for index, name in enumerate(QK_GRADIENTS)}
_QK_BY_ID = {index: name for name, index in _QK_IDS.items()}
_VALUE_IDS = {name: index for index, name in enumerate(VALUE_GRADIENTS)}
_VALUE_BY_ID = {index: name for name, index in _VALUE_IDS.items()}


@dataclass(frozen=True)
class CompetitionState:
    """Detached diagnostics for the winner-versus-rest decomposition."""

    logits: Tensor
    probabilities: Tensor
    hard_winner: Tensor
    soft_winner: Tensor
    soft_runner_up: Tensor
    top_two_mass: Tensor
    nonnull_top_two_mass: Tensor
    runner_up_share_of_rest: Tensor
    tail_mass: Tensor
    hard_soft_winner_agreement: Tensor
    soft_winner_is_null: Tensor
    top_two_value_collision: Tensor
    top_two_nonnull_value_collision: Tensor
    has_two_nonnull_routes: Tensor


def _needs_backward(*tensors: Tensor) -> bool:
    return torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in tensors
    )


def _validate_modes(qk_gradient: str, value_gradient: str) -> None:
    if qk_gradient not in QK_GRADIENTS:
        raise ValueError(f"qk_gradient must be one of {QK_GRADIENTS}")
    if value_gradient not in VALUE_GRADIENTS:
        raise ValueError(f"value_gradient must be one of {VALUE_GRADIENTS}")


def unbounded_suffix_scores(local_match: Tensor) -> Tensor:
    """Evaluate S[t,a] = g[t,a] * (1 + S[t-1,a-1]) in O(T^2)."""

    if local_match.ndim < 2 or local_match.size(-2) != local_match.size(-1):
        raise ValueError("local_match must end in one square route matrix")
    seq_len = local_match.size(-1)
    previous = local_match.new_zeros(*local_match.shape[:-2], seq_len)
    rows = []
    for row_index in range(seq_len):
        if row_index == 0:
            current = previous
        else:
            active = local_match[..., row_index, 1 : row_index + 1] * (
                1.0 + previous[..., :row_index]
            )
            current = F.pad(active, (1, seq_len - row_index - 1))
        rows.append(current)
        previous = current
    return torch.stack(rows, dim=-2)


def _route_logits(
    route_scores: Tensor,
    causal_route_mask: Tensor,
    scale: float,
) -> Tensor:
    seq_len = route_scores.size(-1)
    route_index = torch.arange(
        seq_len, device=route_scores.device
    ).view(1, 1, 1, seq_len)
    nonnull = causal_route_mask.view(1, 1, seq_len, seq_len) & (
        route_index > 0
    )
    nonnull_count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)
    logits = route_scores * float(scale)
    logits = logits - torch.where(
        nonnull,
        nonnull_count.to(route_scores.dtype).log(),
        torch.zeros((), dtype=route_scores.dtype, device=route_scores.device),
    )
    return logits.masked_fill(
        ~causal_route_mask.view(1, 1, seq_len, seq_len),
        -torch.inf,
    )


def _soft_route_state(
    query: Tensor,
    key: Tensor,
    *,
    scale: float,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    causal_route_mask = _causal_route_mask(query.size(1), query.device)
    local_match = _pairwise_soft_match_gates(
        query,
        key,
        causal_route_mask,
        mismatch_scale,
    )
    suffix_scores = unbounded_suffix_scores(local_match)
    route_scores = _masked_route_scores(
        _suffix_score_utility(suffix_scores),
        causal_route_mask,
    )
    logits = _route_logits(route_scores, causal_route_mask, scale)
    centered = logits - logits.amax(dim=-1, keepdim=True)
    probabilities = torch.softmax(centered, dim=-1)
    return logits, probabilities, causal_route_mask


def top2_route_probabilities(logits: Tensor) -> Tensor:
    """Renormalize only the two largest finite logits for Q/K ablation."""

    if logits.size(-1) == 1:
        return torch.ones_like(logits)
    top_values, top_indices = torch.topk(logits, k=2, dim=-1)
    top_probabilities = torch.softmax(top_values, dim=-1)
    return torch.zeros_like(logits).scatter(
        -1, top_indices, top_probabilities
    )


def no_prior_route_probabilities(
    logits: Tensor,
    causal_route_mask: Tensor,
) -> Tensor:
    """Undo the non-null ``-log(N)`` shift relative to the null route."""

    seq_len = logits.size(-1)
    route_index = torch.arange(
        seq_len, device=logits.device
    ).view(1, 1, 1, seq_len)
    nonnull = causal_route_mask.view(1, 1, seq_len, seq_len) & (
        route_index > 0
    )
    nonnull_count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)
    adjusted = logits + torch.where(
        nonnull,
        nonnull_count.to(logits.dtype).log(),
        torch.zeros((), dtype=logits.dtype, device=logits.device),
    )
    return torch.softmax(adjusted - adjusted.amax(dim=-1, keepdim=True), dim=-1)


def hard_null_gate_probabilities(
    logits: Tensor,
    hard_winner: Tensor,
) -> Tensor:
    """Remove null from backward competition after a hard non-null match."""

    if hard_winner.shape != logits.shape[:-1]:
        raise ValueError("hard_winner must match logits without route axis")
    null_logits = torch.where(
        (hard_winner > 0).unsqueeze(-1),
        torch.full_like(logits[..., :1], -torch.inf),
        logits[..., :1],
    )
    adjusted = torch.cat((null_logits, logits[..., 1:]), dim=-1)
    return torch.softmax(adjusted - adjusted.amax(dim=-1, keepdim=True), dim=-1)


def _signed_route_values(value: Tensor, query_heads: int) -> Tensor:
    route_values = _expand_value_heads(
        _hard_sign_with_softsign_vjp(value),
        query_heads,
    )
    nonnull = torch.arange(
        value.size(1), device=value.device
    ).view(1, 1, -1, 1) != 0
    return torch.where(nonnull, route_values, torch.zeros_like(route_values))


def _route_carrier(probabilities: Tensor, route_values: Tensor) -> Tensor:
    return torch.einsum(
        "bhta,bhad->bhtd", probabilities, route_values
    ).permute(0, 2, 1, 3)


def hierarchical_route_carrier(
    probabilities: Tensor,
    route_values: Tensor,
) -> Tensor:
    """Reconstruct ordinary attention as top-one versus conditional rest."""

    winner = probabilities.argmax(dim=-1, keepdim=True)
    winner_probability = probabilities.gather(-1, winner)
    winner_mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter(
        -1, winner, True
    )
    rest_probability = probabilities.masked_fill(winner_mask, 0.0)
    rest_mass = rest_probability.sum(dim=-1, keepdim=True)
    conditional_rest = torch.where(
        rest_mass > 0.0,
        rest_probability / rest_mass.clamp_min(torch.finfo(probabilities.dtype).tiny),
        torch.zeros_like(rest_probability),
    )
    winner_value = torch.gather(
        route_values,
        2,
        winner.expand(*winner.shape[:-1], route_values.size(-1)),
    )
    rest_value = torch.einsum(
        "bhta,bhad->bhtd", conditional_rest, route_values
    )
    carrier = winner_probability * winner_value + rest_mass * rest_value
    return carrier.permute(0, 2, 1, 3)


def winner_rest_route_carrier(
    probabilities: Tensor,
    route_values: Tensor,
    *,
    winner: Tensor | None = None,
) -> Tensor:
    """Keep only winner-versus-rest Q/K credit without dropping candidates.

    Numerically this is ordinary attention.  The conditional distribution
    inside the rest set is detached, so Q/K only see the binary margin between
    ``winner`` and ``logsumexp(rest)``.  Every finite rest candidate still gets
    credit through the derivative of that log-sum-exp.
    """

    if winner is None:
        winner = probabilities.detach().argmax(dim=-1)
    if winner.shape == probabilities.shape[:-1]:
        winner = winner.unsqueeze(-1)
    if winner.shape != probabilities.shape[:-1] + (1,):
        raise ValueError("winner must match probabilities without route axis")

    winner_mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter(
        -1, winner, True
    )
    winner_probability = probabilities.gather(-1, winner)
    rest_probability = probabilities.masked_fill(winner_mask, 0.0)
    rest_mass = rest_probability.sum(dim=-1, keepdim=True)
    conditional_rest = torch.where(
        rest_mass > 0.0,
        rest_probability / rest_mass.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ),
        torch.zeros_like(rest_probability),
    ).detach()
    winner_value = torch.gather(
        route_values,
        2,
        winner.expand(*winner.shape[:-1], route_values.size(-1)),
    )
    rest_value = torch.einsum(
        "bhta,bhad->bhtd", conditional_rest, route_values
    )
    carrier = (
        winner_probability * winner_value
        + rest_mass * rest_value
    )
    return carrier.permute(0, 2, 1, 3)


class _HardForwardRoleSeparatedVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        mismatch_scale: float,
        qk_gradient: int,
        value_gradient: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, selected_routes, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
        )
        ctx.scale = float(scale)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.qk_gradient = _QK_BY_ID[int(qk_gradient)]
        ctx.value_gradient = _VALUE_BY_ID[int(value_gradient)]
        ctx.save_for_backward(query, key, value, selected_routes)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, selected_routes = ctx.saved_tensors
        needs_query, needs_key, needs_value = ctx.needs_input_grad[:3]
        gradients: list[Tensor | None] = [None, None, None]
        compute_dtype = _reference_compute_dtype(query.dtype)

        with torch.enable_grad():
            query_leaf = query.detach().requires_grad_(needs_query)
            key_leaf = key.detach().requires_grad_(needs_key)
            value_leaf = value.detach().requires_grad_(needs_value)
            logits, dense_probabilities, causal_route_mask = _soft_route_state(
                query_leaf.to(compute_dtype),
                key_leaf.to(compute_dtype),
                scale=ctx.scale,
                mismatch_scale=ctx.mismatch_scale,
            )

            qk_indices = [
                index
                for index, needed in enumerate((needs_query, needs_key))
                if needed
            ]
            if qk_indices:
                qk_values = _signed_route_values(
                    value.detach().to(compute_dtype),
                    query.size(2),
                )
                if ctx.qk_gradient == "dense":
                    qk_carrier = _route_carrier(
                        dense_probabilities,
                        qk_values,
                    )
                elif ctx.qk_gradient == "top2":
                    qk_carrier = _route_carrier(
                        top2_route_probabilities(logits),
                        qk_values,
                    )
                elif ctx.qk_gradient == "no_prior":
                    qk_carrier = _route_carrier(
                        no_prior_route_probabilities(
                            logits, causal_route_mask
                        ),
                        qk_values,
                    )
                elif ctx.qk_gradient == "hard_null_gate":
                    qk_carrier = _route_carrier(
                        hard_null_gate_probabilities(
                            logits, selected_routes
                        ),
                        qk_values,
                    )
                else:
                    winner = (
                        selected_routes
                        if ctx.qk_gradient == "hard_winner_rest"
                        else None
                    )
                    qk_carrier = winner_rest_route_carrier(
                        dense_probabilities,
                        qk_values,
                        winner=winner,
                    )
                qk_carrier = qk_carrier.to(query.dtype)
                qk_leaves = (query_leaf, key_leaf)
                qk_gradients = torch.autograd.grad(
                    qk_carrier,
                    tuple(qk_leaves[index] for index in qk_indices),
                    grad_output,
                    create_graph=False,
                    retain_graph=needs_value,
                )
                for index, gradient in zip(qk_indices, qk_gradients):
                    gradients[index] = gradient

            if needs_value:
                route_values = _signed_route_values(
                    value_leaf.to(compute_dtype),
                    query.size(2),
                )
                if ctx.value_gradient == "dense":
                    value_carrier = _route_carrier(
                        dense_probabilities.detach(),
                        route_values,
                    )
                else:
                    value_carrier = _gather_routed_values(
                        route_values,
                        selected_routes,
                    ).permute(0, 2, 1, 3)
                gradients[2] = torch.autograd.grad(
                    value_carrier.to(query.dtype),
                    value_leaf,
                    grad_output,
                    create_graph=False,
                )[0]

        return (
            gradients[0],
            gradients[1],
            gradients[2],
            None,
            None,
            None,
            None,
        )


def rosa_top_two_value_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    qk_gradient: str = "dense",
    value_gradient: str = "dense",
) -> Tensor:
    """Run exact hard ROSA with an unlimited, role-separated surrogate VJP."""

    _validate_modes(qk_gradient, value_gradient)
    validate_rosa_soft_inputs(
        query,
        key,
        value,
        query.size(1),
        scale,
        0.0,
        mismatch_scale,
    )
    _reference_compute_dtype(query.dtype)
    if not _needs_backward(query, key, value):
        return _hard_route_forward(query, key, value)[0].to(query.dtype)
    return _HardForwardRoleSeparatedVjp.apply(
        query,
        key,
        value,
        float(scale),
        float(mismatch_scale),
        _QK_IDS[qk_gradient],
        _VALUE_IDS[value_gradient],
    )


@torch.no_grad()
def competition_state(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> CompetitionState:
    """Inspect whether ordinary attention has reduced to top-two competition."""

    validate_rosa_soft_inputs(
        query,
        key,
        value,
        query.size(1),
        scale,
        0.0,
        mismatch_scale,
    )
    compute_dtype = _reference_compute_dtype(query.dtype)
    query_work = query.to(compute_dtype)
    key_work = key.to(compute_dtype)
    value_work = value.to(compute_dtype)
    logits, probabilities, causal_route_mask = _soft_route_state(
        query_work,
        key_work,
        scale=scale,
        mismatch_scale=mismatch_scale,
    )
    _, _, hard_winner, _ = _hard_route_forward(
        query_work, key_work, value_work
    )
    top_count = min(2, probabilities.size(-1))
    top_probabilities, top_indices = torch.topk(
        probabilities, k=top_count, dim=-1
    )
    soft_winner = top_indices[..., 0]
    soft_runner_up = (
        top_indices[..., 1] if top_count == 2 else soft_winner
    )
    top_two_mass = top_probabilities.sum(dim=-1).clamp(0.0, 1.0)
    winner_probability = top_probabilities[..., 0]
    runner_probability = (
        top_probabilities[..., 1]
        if top_count == 2
        else torch.zeros_like(winner_probability)
    )
    rest_mass = (1.0 - winner_probability).clamp_min(0.0)
    runner_share = torch.where(
        rest_mass > 0.0,
        runner_probability / rest_mass,
        torch.ones_like(rest_mass),
    ).clamp(0.0, 1.0)

    route_values = _expand_value_heads(
        _hard_sign(value_work), query.size(2)
    )
    route_values[..., 0, :] = 0.0
    winner_value = torch.gather(
        route_values,
        2,
        soft_winner.unsqueeze(-1).expand(
            *soft_winner.shape, route_values.size(-1)
        ),
    )
    runner_value = torch.gather(
        route_values,
        2,
        soft_runner_up.unsqueeze(-1).expand(
            *soft_runner_up.shape, route_values.size(-1)
        ),
    )
    collision = (winner_value == runner_value).all(dim=-1)

    nonnull_probabilities = probabilities[..., 1:]
    nonnull_top_count = min(2, nonnull_probabilities.size(-1))
    if nonnull_top_count:
        nonnull_top_probabilities, nonnull_top_indices = torch.topk(
            nonnull_probabilities, k=nonnull_top_count, dim=-1
        )
        nonnull_top_two_mass = nonnull_top_probabilities.sum(dim=-1)
        nonnull_winner = nonnull_top_indices[..., 0] + 1
        nonnull_runner = (
            nonnull_top_indices[..., 1] + 1
            if nonnull_top_count == 2
            else nonnull_winner
        )
        nonnull_winner_value = torch.gather(
            route_values,
            2,
            nonnull_winner.unsqueeze(-1).expand(
                *nonnull_winner.shape, route_values.size(-1)
            ),
        )
        nonnull_runner_value = torch.gather(
            route_values,
            2,
            nonnull_runner.unsqueeze(-1).expand(
                *nonnull_runner.shape, route_values.size(-1)
            ),
        )
        nonnull_collision = (
            nonnull_winner_value == nonnull_runner_value
        ).all(dim=-1)
    else:
        nonnull_top_two_mass = probabilities.new_zeros(
            probabilities.shape[:-1]
        )
        nonnull_collision = torch.zeros_like(
            soft_winner, dtype=torch.bool
        )
    nonnull_count = causal_route_mask[:, 1:].sum(dim=-1).view(1, 1, -1)
    return CompetitionState(
        logits=logits.detach(),
        probabilities=probabilities.detach(),
        hard_winner=hard_winner.detach(),
        soft_winner=soft_winner.detach(),
        soft_runner_up=soft_runner_up.detach(),
        top_two_mass=top_two_mass.detach(),
        nonnull_top_two_mass=nonnull_top_two_mass.detach(),
        runner_up_share_of_rest=runner_share.detach(),
        tail_mass=(1.0 - top_two_mass).clamp_min(0.0).detach(),
        hard_soft_winner_agreement=(hard_winner == soft_winner).detach(),
        soft_winner_is_null=soft_winner.eq(0).detach(),
        top_two_value_collision=collision.detach(),
        top_two_nonnull_value_collision=nonnull_collision.detach(),
        has_two_nonnull_routes=nonnull_count.ge(2).expand_as(
            soft_winner
        ).detach(),
    )
