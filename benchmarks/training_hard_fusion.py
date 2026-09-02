"""Training-only hard/soft fusion feasibility study.

A finite-W dense surrogate scan already observes whether each candidate
matches all W suffix symbols. Such candidates are the only ones whose exact
hard length may still grow. This module measures that ambiguity set and the
extra exact comparisons required to preserve unlimited hard semantics.

It also implements the two-accumulator softmax identity needed to consume
saved row normalizers in one later route pass without knowing expected utility
in advance. This is a research model, not a production operator path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class AmbiguityStats:
    tokens: int
    window: int
    causal_candidates: int
    fallback_rows: int
    continuation_candidates: int
    continuation_comparisons: int
    direct_hard_comparisons: int
    capped_route_errors: int
    max_continuation_candidates: int

    @property
    def fallback_row_fraction(self) -> float:
        return self.fallback_rows / max(self.tokens - 1, 1)

    @property
    def continuation_candidate_fraction(self) -> float:
        return self.continuation_candidates / max(self.causal_candidates, 1)

    @property
    def continuation_vs_direct_work(self) -> float:
        return self.continuation_comparisons / max(
            self.direct_hard_comparisons,
            1,
        )


def suffix_length_matrix(query: Tensor, key: Tensor) -> Tensor:
    """Return exact causal suffix lengths indexed by query/key end."""

    if query.ndim != 1 or key.shape != query.shape:
        raise ValueError("query and key must be equal-length vectors")
    tokens = query.numel()
    dtype = torch.int16 if tokens <= 32767 else torch.int32
    lengths = torch.zeros(tokens, tokens, dtype=dtype, device=query.device)
    previous = torch.zeros(tokens, dtype=dtype, device=query.device)
    for query_end in range(1, tokens):
        current = torch.zeros_like(previous)
        predecessors = torch.cat((previous.new_zeros(1), previous[: query_end - 1]))
        current[:query_end] = torch.where(
            query[query_end] == key[:query_end],
            predecessors + 1,
            0,
        )
        lengths[query_end] = current
        previous = current
    return lengths


def _latest_routes(lengths: Tensor) -> tuple[Tensor, Tensor]:
    best_lengths = lengths.amax(dim=1)
    key_ends = torch.arange(lengths.size(1), device=lengths.device)
    routes = torch.where(
        lengths == best_lengths[:, None],
        key_ends,
        -1,
    ).amax(dim=1)
    routes = torch.where(best_lengths > 0, routes, -1)
    return routes, best_lengths


def _ambiguity_stats_from_lengths(
    query: Tensor,
    key: Tensor,
    lengths: Tensor,
    window: int,
) -> AmbiguityStats:
    if window < 1:
        raise ValueError("window must be positive")
    lengths = lengths.to(torch.int32)
    tokens = query.numel()
    query_ends = torch.arange(tokens, device=query.device)[:, None]
    key_ends = torch.arange(tokens, device=query.device)[None, :]
    causal = key_ends < query_ends
    maximum_lengths = torch.minimum(query_ends + 1, key_ends + 1)

    exact_routes, _ = _latest_routes(lengths)
    capped_lengths = torch.minimum(lengths, lengths.new_tensor(window))
    capped_routes, _ = _latest_routes(capped_lengths)

    continuation = causal & (maximum_lengths > window) & (lengths >= window)
    continuation_counts = continuation.sum(dim=1)
    fallback_rows = continuation_counts > 0

    matching_beyond_window = torch.clamp(lengths - window, min=0)
    ends_in_mismatch = continuation & (lengths < maximum_lengths)
    continuation_work = torch.where(
        continuation,
        matching_beyond_window + ends_in_mismatch.to(lengths.dtype),
        0,
    )
    direct_work = torch.where(
        causal,
        lengths + (lengths < maximum_lengths).to(lengths.dtype),
        0,
    )

    return AmbiguityStats(
        tokens=tokens,
        window=window,
        causal_candidates=int(causal.sum()),
        fallback_rows=int(fallback_rows.sum()),
        continuation_candidates=int(continuation.sum()),
        continuation_comparisons=int(continuation_work.sum()),
        direct_hard_comparisons=int(direct_work.sum()),
        capped_route_errors=int((capped_routes != exact_routes).sum()),
        max_continuation_candidates=int(continuation_counts.max()),
    )


def ambiguity_stats(query: Tensor, key: Tensor, window: int) -> AmbiguityStats:
    return _ambiguity_stats_from_lengths(
        query,
        key,
        suffix_length_matrix(query, key),
        window,
    )


def dual_accumulator_softmax_vjp(
    logits: Tensor,
    utility: Tensor,
    score_jacobian: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute ``d E[utility] / dx`` before materializing expected utility.

    ``score_jacobian[j, p]`` is ``d logits[j] / d x[p]``. The two route-pass
    accumulators are returned so their extra state is explicit.
    """

    if logits.ndim != 1 or utility.shape != logits.shape:
        raise ValueError("logits and utility must be equal-length vectors")
    if score_jacobian.ndim != 2 or score_jacobian.size(0) != logits.numel():
        raise ValueError("score_jacobian must have shape [routes, parameters]")
    probability = logits.softmax(dim=0)
    expected_utility = (probability * utility).sum()
    utility_accumulator = torch.einsum(
        "r,r,rp->p",
        probability,
        utility,
        score_jacobian,
    )
    normalizer_accumulator = torch.einsum(
        "r,rp->p",
        probability,
        score_jacobian,
    )
    gradient = utility_accumulator - expected_utility * normalizer_accumulator
    return (
        gradient,
        utility_accumulator,
        normalizer_accumulator,
        expected_utility,
    )


def _make_codes(tokens: int, bits: int, pattern: str, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed + tokens + bits)
    maximum = 1 << bits
    if pattern == "random":
        return (
            torch.randint(maximum, (tokens,), generator=generator),
            torch.randint(maximum, (tokens,), generator=generator),
        )
    if pattern == "skewed":
        query = torch.randint(maximum, (tokens,), generator=generator)
        key = torch.randint(maximum, (tokens,), generator=generator)
        query[torch.rand(tokens, generator=generator) < 0.9] = 0
        key[torch.rand(tokens, generator=generator) < 0.9] = 0
        return query, key
    if pattern == "all_match":
        codes = torch.zeros(tokens, dtype=torch.int64)
        return codes, codes.clone()
    if pattern.startswith("periodic"):
        period = int(pattern.removeprefix("periodic"))
        motif = torch.randint(maximum, (period,), generator=generator)
        codes = motif[torch.arange(tokens) % period]
        return codes, codes.clone()
    raise ValueError(f"unknown pattern: {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["random", "skewed", "periodic64", "all_match"],
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for tokens in args.tokens:
        for bits in args.bits:
            for pattern in args.patterns:
                query, key = _make_codes(tokens, bits, pattern, args.seed)
                query = query.to(args.device)
                key = key.to(args.device)
                lengths = suffix_length_matrix(query, key)
                for window in args.windows:
                    stats = _ambiguity_stats_from_lengths(
                        query,
                        key,
                        lengths,
                        window,
                    )
                    row = {
                        "bits": bits,
                        "pattern": pattern,
                        **asdict(stats),
                        "fallback_row_fraction": stats.fallback_row_fraction,
                        "continuation_candidate_fraction": (
                            stats.continuation_candidate_fraction
                        ),
                        "continuation_vs_direct_work": (
                            stats.continuation_vs_direct_work
                        ),
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "settings": vars(args) | {"output": str(args.output)},
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
