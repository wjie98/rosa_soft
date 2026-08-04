"""Q/K codebook diagnostics for pretraining checkpoints.

The core evaluator consumes matched query/key trajectories rather than a
particular model class.  A training job can therefore collect residual-stream
occurrences, run its Q/K symbol heads, and pass ``[N, L, D]`` logits plus the
continuation labels to :func:`evaluate_codebook`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HorizonMetrics:
    horizon: int
    query_key_trajectory_alignment: float
    route_accuracy: float
    strict_route_accuracy: float
    null_fraction: float
    target_suffix_length_mean: float
    distractor_suffix_length_mean: float
    positive_suffix_margin_mean: float
    key_states: int
    conflicting_states: int
    conflicting_sample_fraction: float
    max_state_occupancy: int
    continuation_conditional_entropy_bits: float


@dataclass(frozen=True)
class CodebookMetrics:
    samples: int
    trajectory_length: int
    bits: int
    bit_agreement: float
    symbol_agreement: float
    mean_bit_entropy: float
    dead_bit_fraction: float
    horizons: tuple[HorizonMetrics, ...]


def _hard_bits(logits: Tensor) -> Tensor:
    return logits > 0


def _binary_entropy(probability: Tensor) -> Tensor:
    probability = probability.double().clamp(1e-12, 1.0 - 1e-12)
    return -(
        probability * probability.log2()
        + (1.0 - probability) * (1.0 - probability).log2()
    )


def _validate_inputs(
    query_logits: Tensor,
    key_logits: Tensor,
    continuation_labels: Tensor,
) -> None:
    if query_logits.shape != key_logits.shape or query_logits.ndim != 3:
        raise ValueError("query/key logits must share shape [N, L, D]")
    if query_logits.size(0) < 1:
        raise ValueError("at least one trajectory is required")
    if query_logits.size(1) < 1 or query_logits.size(2) < 1:
        raise ValueError("trajectory length and bit width must be positive")
    if continuation_labels.shape != (query_logits.size(0),):
        raise ValueError("continuation_labels must have shape [N]")


def _suffix_lengths(query_bits: Tensor, key_bits: Tensor, horizon: int) -> Tensor:
    local_matches = (
        query_bits[:, None, -horizon:]
        == key_bits[None, :, -horizon:]
    ).all(dim=-1)
    reverse_matches = local_matches.flip(-1).to(torch.int64)
    return reverse_matches.cumprod(dim=-1).sum(dim=-1)


def _conflict_metrics(
    key_bits: Tensor,
    labels: Tensor,
    horizon: int,
) -> tuple[int, int, float, int, float]:
    fingerprints = key_bits[:, -horizon:].reshape(key_bits.size(0), -1).cpu()
    labels = labels.to(torch.int64).cpu()
    groups: dict[tuple[bool, ...], list[int]] = {}
    for sample, fingerprint in enumerate(fingerprints.tolist()):
        groups.setdefault(tuple(fingerprint), []).append(sample)

    conflicting_states = 0
    conflicting_samples = 0
    max_occupancy = 0
    conditional_entropy = 0.0
    for indices in groups.values():
        max_occupancy = max(max_occupancy, len(indices))
        _, counts = torch.unique(labels[indices], return_counts=True)
        if counts.numel() > 1:
            conflicting_states += 1
            conflicting_samples += len(indices)
        probabilities = counts.float() / len(indices)
        entropy = -(probabilities * probabilities.log2()).sum()
        conditional_entropy += (
            len(indices) / labels.numel() * float(entropy)
        )
    return (
        len(groups),
        conflicting_states,
        conflicting_samples / labels.numel(),
        max_occupancy,
        conditional_entropy,
    )


def evaluate_codebook(
    query_logits: Tensor,
    key_logits: Tensor,
    continuation_labels: Tensor,
    *,
    horizons: tuple[int, ...] | None = None,
) -> CodebookMetrics:
    """Measure alignment, exact routing, and incompatible key collisions."""

    _validate_inputs(query_logits, key_logits, continuation_labels)
    sample_count, trajectory_length, bits = query_logits.shape
    if horizons is None:
        horizons = tuple(range(1, trajectory_length + 1))
    if not horizons or any(
        horizon < 1 or horizon > trajectory_length for horizon in horizons
    ):
        raise ValueError("horizons must lie in [1, trajectory_length]")

    query_bits = _hard_bits(query_logits).cpu()
    key_bits = _hard_bits(key_logits).cpu()
    labels = continuation_labels.to(torch.int64).cpu()
    bit_probability = key_bits.float().mean(dim=0)
    bit_entropy = _binary_entropy(bit_probability)
    dead_bits = (bit_probability == 0.0) | (bit_probability == 1.0)
    horizon_records = []
    target_indices = torch.arange(sample_count)
    for horizon in horizons:
        suffix_lengths = _suffix_lengths(query_bits, key_bits, horizon)
        target_lengths = suffix_lengths[target_indices, target_indices]
        if sample_count == 1:
            distractor_lengths = torch.zeros_like(target_lengths)
        else:
            without_target = suffix_lengths.clone()
            without_target[target_indices, target_indices] = -1
            distractor_lengths = without_target.max(dim=-1).values
        route_positions = torch.arange(1, sample_count + 1)
        encoded = suffix_lengths * (sample_count + 1) + route_positions
        selected = encoded.argmax(dim=-1)
        selected_lengths = suffix_lengths[target_indices, selected]
        selected = torch.where(
            selected_lengths > 0,
            selected,
            torch.full_like(selected, -1),
        )
        selected_labels = torch.full_like(labels, -1)
        nonnull = selected >= 0
        selected_labels[nonnull] = labels[selected[nonnull]]
        (
            key_states,
            conflicting_states,
            conflicting_sample_fraction,
            max_state_occupancy,
            conditional_entropy,
        ) = _conflict_metrics(key_bits, labels, horizon)
        horizon_records.append(
            HorizonMetrics(
                horizon=horizon,
                query_key_trajectory_alignment=float(
                    (query_bits[:, -horizon:] == key_bits[:, -horizon:])
                    .all(dim=(-1, -2))
                    .float()
                    .mean()
                ),
                route_accuracy=float(
                    (selected_labels == labels).float().mean()
                ),
                strict_route_accuracy=float(
                    (selected == target_indices).float().mean()
                ),
                null_fraction=float((~nonnull).float().mean()),
                target_suffix_length_mean=float(target_lengths.float().mean()),
                distractor_suffix_length_mean=float(
                    distractor_lengths.float().mean()
                ),
                positive_suffix_margin_mean=float(
                    (target_lengths - distractor_lengths)
                    .clamp_min(0)
                    .float()
                    .mean()
                ),
                key_states=key_states,
                conflicting_states=conflicting_states,
                conflicting_sample_fraction=conflicting_sample_fraction,
                max_state_occupancy=max_state_occupancy,
                continuation_conditional_entropy_bits=conditional_entropy,
            )
        )
    return CodebookMetrics(
        samples=sample_count,
        trajectory_length=trajectory_length,
        bits=bits,
        bit_agreement=float((query_bits == key_bits).float().mean()),
        symbol_agreement=float(
            (query_bits == key_bits).all(dim=-1).float().mean()
        ),
        mean_bit_entropy=float(bit_entropy.mean()),
        dead_bit_fraction=float(dead_bits.float().mean()),
        horizons=tuple(horizon_records),
    )


def _unique_trajectories(
    concepts: int,
    trajectory_length: int,
    bits: int,
    generator: torch.Generator,
) -> Tensor:
    capacity_bits = trajectory_length * bits
    if capacity_bits < math.ceil(math.log2(concepts)):
        raise ValueError("trajectory capacity cannot encode every concept")
    trajectories = torch.randint(
        0,
        2,
        (concepts, capacity_bits),
        generator=generator,
        dtype=torch.bool,
    )
    seen: set[tuple[bool, ...]] = set()
    for concept in range(concepts):
        fingerprint = tuple(trajectories[concept].tolist())
        while fingerprint in seen:
            trajectories[concept] = torch.randint(
                0,
                2,
                (capacity_bits,),
                generator=generator,
                dtype=torch.bool,
            )
            fingerprint = tuple(trajectories[concept].tolist())
        seen.add(fingerprint)
    return trajectories.view(concepts, trajectory_length, bits)


def synthetic_snapshot(
    *,
    seed: int,
    concepts: int,
    trajectory_length: int,
    bits: int,
    mode: str,
    corruption_p: float,
    active_bits: int,
    labels: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create controlled aligned, corrupted, drifted, or collapsed logits."""

    if mode not in {"aligned", "corrupted", "role_drift", "collapsed"}:
        raise ValueError("unknown synthetic mode")
    if not 0.0 <= corruption_p <= 1.0:
        raise ValueError("corruption_p must be in [0, 1]")
    if not 1 <= active_bits <= bits:
        raise ValueError("active_bits must be in [1, bits]")
    if labels < 1:
        raise ValueError("labels must be positive")
    generator = torch.Generator().manual_seed(seed)
    target = _unique_trajectories(
        concepts,
        trajectory_length,
        bits,
        generator,
    )
    key_bits = target.clone()
    query_bits = target.clone()
    if mode == "corrupted":
        flips = torch.rand(
            query_bits.shape,
            generator=generator,
        ) < corruption_p
        query_bits ^= flips
    elif mode == "role_drift":
        role_flips = torch.rand(
            trajectory_length,
            bits,
            generator=generator,
        ) < corruption_p
        query_bits ^= role_flips.unsqueeze(0)
    elif mode == "collapsed":
        key_bits[..., active_bits:] = False
        query_bits[..., active_bits:] = False
    query_logits = torch.where(query_bits, 1.0, -1.0)
    key_logits = torch.where(key_bits, 1.0, -1.0)
    continuation_labels = torch.arange(concepts) % labels
    return query_logits, key_logits, continuation_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--concepts", type=int, default=64)
    parser.add_argument("--trajectory-length", type=int, default=8)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["aligned", "corrupted", "role_drift", "collapsed"],
        default=["aligned", "corrupted", "role_drift", "collapsed"],
    )
    parser.add_argument("--corruption-p", type=float, default=0.1)
    parser.add_argument("--active-bits", type=int, default=1)
    parser.add_argument("--labels", type=int, default=64)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    records = []
    for mode in args.modes:
        for seed in args.seeds:
            query, key, labels = synthetic_snapshot(
                seed=seed,
                concepts=args.concepts,
                trajectory_length=args.trajectory_length,
                bits=args.bits,
                mode=mode,
                corruption_p=args.corruption_p,
                active_bits=args.active_bits,
                labels=args.labels,
            )
            metrics = evaluate_codebook(query, key, labels)
            records.append(
                {
                    "mode": mode,
                    "seed": seed,
                    **asdict(metrics),
                }
            )
    result = {"config": vars(args), "records": records}
    payload = json.dumps(result, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as output:
            output.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
