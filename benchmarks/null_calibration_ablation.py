"""Probe RosaSoft null calibration as the candidate set grows.

The benchmark is score-level on purpose.  It separates three questions that
are easy to conflate in an end-to-end fit:

1. Does random background mass force the null route to disappear as N grows?
2. Can one planted exact suffix overcome the N-way candidate prior?
3. If recall is enabled, does the planted candidate receive useful mass?

All reported probabilities use the expected random-background partition.  A
Monte Carlo estimate is needed only for the current square-root route score.
The collision likelihood ratio has unit expectation under independent,
uniform hard bits by construction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor


NULL_ROUTE_SCORE = 0.5
SQRT_SUFFIX_SCALE = math.sqrt(2.0) + 1.0


def normalized_sqrt_utility(raw_suffix_score: Tensor) -> Tensor:
    """Production suffix utility, calibrated by U(0)=0 and U(1)=1."""

    return SQRT_SUFFIX_SCALE * (torch.sqrt(1.0 + raw_suffix_score) - 1.0)


def null_local_gate_mean(qk_bits: int, mismatch_scale: float) -> float:
    """Return E[exp(-lambda H / D)] for H ~ Binomial(D, 1/2)."""

    if qk_bits < 1:
        raise ValueError("qk_bits must be positive")
    if mismatch_scale <= 0:
        raise ValueError("mismatch_scale must be positive")
    one_bit_mean = 0.5 * (
        1.0 + math.exp(-float(mismatch_scale) / qk_bits)
    )
    return one_bit_mean**qk_bits


def planted_route_state(
    *,
    qk_bits: int,
    max_suffix_length: int,
    planted_suffix_length: int,
    mismatch_scale: float,
    scale: float,
) -> Dict[str, float]:
    """Return deterministic scores for an exact prefix followed by mismatches."""

    if not 1 <= planted_suffix_length <= max_suffix_length:
        raise ValueError(
            "planted_suffix_length must be in [1, max_suffix_length]"
        )
    exact = torch.zeros(planted_suffix_length, dtype=torch.float64)
    tail = torch.full(
        (max_suffix_length - planted_suffix_length,),
        float(qk_bits),
        dtype=torch.float64,
    )
    mismatch_count = torch.cat((exact, tail))
    log_gate = -float(mismatch_scale) * mismatch_count / qk_bits
    prefix_log_gate = log_gate.cumsum(dim=0)
    raw_suffix_score = prefix_log_gate.exp().sum()
    route_score = normalized_sqrt_utility(raw_suffix_score)

    gate_mean = null_local_gate_mean(qk_bits, mismatch_scale)
    prefix_log_bf = prefix_log_gate - torch.arange(
        1,
        max_suffix_length + 1,
        dtype=torch.float64,
    ) * math.log(gate_mean)
    route_log_bf = torch.logsumexp(prefix_log_bf, dim=0) - math.log(
        max_suffix_length
    )
    return {
        "raw_suffix_score": float(raw_suffix_score),
        "route_score": float(route_score),
        "route_log_weight": float(scale * route_score),
        "route_log_likelihood_ratio": float(route_log_bf),
    }


def estimate_null_route_partition(
    *,
    qk_bits: int,
    max_suffix_length: int,
    mismatch_scale: float,
    scale: float,
    sample_count: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """Estimate E[exp(scale U(S))] for one random non-null route."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    generator = torch.Generator(device=device).manual_seed(seed)
    weight_sum = 0.0
    weight_square_sum = 0.0
    raw_sum = 0.0
    completed = 0
    while completed < sample_count:
        count = min(chunk_size, sample_count - completed)
        mismatch_count = torch.zeros(
            count,
            max_suffix_length,
            dtype=torch.float32,
            device=device,
        )
        for _ in range(qk_bits):
            mismatch_count.add_(
                torch.randint(
                    0,
                    2,
                    mismatch_count.shape,
                    generator=generator,
                    device=device,
                    dtype=torch.int64,
                )
            )
        gates = torch.exp(
            -float(mismatch_scale) * mismatch_count / qk_bits
        )
        raw_score = gates.cumprod(dim=-1).sum(dim=-1)
        weight = torch.exp(float(scale) * normalized_sqrt_utility(raw_score))
        weight64 = weight.to(torch.float64)
        weight_sum += float(weight64.sum())
        weight_square_sum += float(weight64.square().sum())
        raw_sum += float(raw_score.to(torch.float64).sum())
        completed += count

    mean_weight = weight_sum / sample_count
    variance = max(
        0.0,
        weight_square_sum / sample_count - mean_weight * mean_weight,
    )
    gate_mean = null_local_gate_mean(qk_bits, mismatch_scale)
    expected_raw = sum(
        gate_mean**length for length in range(1, max_suffix_length + 1)
    )
    return {
        "mean_route_weight": mean_weight,
        "route_weight_standard_error": math.sqrt(variance / sample_count),
        "mean_raw_suffix_score": raw_sum / sample_count,
        "expected_raw_suffix_score": expected_raw,
        "local_gate_mean": gate_mean,
    }


def _sigmoid(log_odds: float) -> float:
    if log_odds >= 0:
        inverse = math.exp(-log_odds)
        return 1.0 / (1.0 + inverse)
    odds = math.exp(log_odds)
    return odds / (1.0 + odds)


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(
        math.exp(left - maximum) + math.exp(right - maximum)
    )


def calibration_record(
    *,
    qk_bits: int,
    max_suffix_length: int,
    planted_suffix_length: int,
    candidate_count: int,
    mismatch_scale: float,
    scale: float,
    null_partition: Dict[str, float],
) -> Dict[str, object]:
    """Compare null and planted-route behavior for one analytical cell."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    planted = planted_route_state(
        qk_bits=qk_bits,
        max_suffix_length=max_suffix_length,
        planted_suffix_length=planted_suffix_length,
        mismatch_scale=mismatch_scale,
        scale=scale,
    )
    log_n = math.log(candidate_count)
    log_background_count = (
        math.log(candidate_count - 1) if candidate_count > 1 else -math.inf
    )
    mean_route_weight = float(null_partition["mean_route_weight"])
    log_mean_route_weight = math.log(mean_route_weight)
    target_log_weight = float(planted["route_log_weight"])
    log_joint_weight = _logaddexp(
        log_background_count + log_mean_route_weight,
        target_log_weight,
    )
    log_mean_joint_weight = log_joint_weight - log_n
    null_logit = float(scale) * NULL_ROUTE_SCORE

    current_background_recall = _sigmoid(
        log_mean_route_weight - null_logit
    )
    current_planted_recall = _sigmoid(log_mean_joint_weight - null_logit)
    current_target_share = math.exp(target_log_weight - log_joint_weight)

    uncorrected_background_recall = _sigmoid(
        log_n + log_mean_route_weight - null_logit
    )
    uncorrected_planted_recall = _sigmoid(log_joint_weight - null_logit)

    calibrated_background_recall = 0.5
    calibrated_planted_recall = _sigmoid(
        log_mean_joint_weight - log_mean_route_weight
    )

    target_log_bf = float(planted["route_log_likelihood_ratio"])
    log_joint_bf = _logaddexp(log_background_count, target_log_bf)
    log_mean_joint_bf = log_joint_bf - log_n
    collision_background_recall = 0.5
    collision_planted_recall = _sigmoid(log_mean_joint_bf)
    collision_target_share = math.exp(target_log_bf - log_joint_bf)

    longer_collision_probability = 2.0 ** (
        -qk_bits * (planted_suffix_length + 1)
    )
    hard_not_beaten_probability = math.exp(
        (candidate_count - 1)
        * math.log1p(-longer_collision_probability)
    )
    return {
        "qk_bits": qk_bits,
        "max_suffix_length": max_suffix_length,
        "planted_suffix_length": planted_suffix_length,
        "candidate_count": candidate_count,
        "planted": planted,
        "hard_target_not_beaten_probability": hard_not_beaten_probability,
        "current": {
            "background_recall_probability": current_background_recall,
            "planted_recall_probability": current_planted_recall,
            "planted_target_probability": (
                current_planted_recall * current_target_share
            ),
            "effective_candidate_capacity": math.exp(
                target_log_weight - log_mean_route_weight
            ),
        },
        "no_candidate_correction": {
            "background_recall_probability": (
                uncorrected_background_recall
            ),
            "planted_recall_probability": uncorrected_planted_recall,
            "planted_target_probability": (
                uncorrected_planted_recall * current_target_share
            ),
        },
        "moment_calibrated_sqrt": {
            "background_recall_probability": (
                calibrated_background_recall
            ),
            "planted_recall_probability": calibrated_planted_recall,
            "planted_target_probability": (
                calibrated_planted_recall * current_target_share
            ),
        },
        "collision_lr_joint": {
            "background_recall_probability": collision_background_recall,
            "planted_recall_probability": collision_planted_recall,
            "planted_target_probability": (
                collision_planted_recall * collision_target_share
            ),
            "effective_candidate_capacity": math.exp(target_log_bf),
        },
        "collision_lr_null_sqrt_route": {
            "background_recall_probability": collision_background_recall,
            "planted_recall_probability": collision_planted_recall,
            "planted_target_probability": (
                collision_planted_recall * current_target_share
            ),
        },
    }


def run_ablation(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    partitions: Dict[tuple[int, int], Dict[str, float]] = {}
    for qk_bits in args.qk_bits:
        for window in args.windows:
            partitions[(qk_bits, window)] = estimate_null_route_partition(
                qk_bits=qk_bits,
                max_suffix_length=window,
                mismatch_scale=args.mismatch_scale,
                scale=args.scale,
                sample_count=args.sample_count,
                chunk_size=args.chunk_size,
                seed=args.seed + 1009 * qk_bits + window,
                device=device,
            )

    records = []
    for qk_bits in args.qk_bits:
        for window in args.windows:
            for planted_length in args.planted_lengths:
                if planted_length > window:
                    continue
                for candidate_count in args.candidate_counts:
                    records.append(
                        calibration_record(
                            qk_bits=qk_bits,
                            max_suffix_length=window,
                            planted_suffix_length=planted_length,
                            candidate_count=candidate_count,
                            mismatch_scale=args.mismatch_scale,
                            scale=args.scale,
                            null_partition=partitions[(qk_bits, window)],
                        )
                    )
    return {
        "device": args.device,
        "torch_version": torch.__version__,
        "assumptions": {
            "background_bits": "independent_uniform_hard_bits",
            "partition": "expected_random_background_partition",
            "planted_tail": "all_bits_mismatch_after_exact_suffix",
            "null_prior_odds": 1.0,
        },
        "scale": args.scale,
        "mismatch_scale": args.mismatch_scale,
        "sample_count": args.sample_count,
        "partitions": {
            f"D{bits}:W{window}": state
            for (bits, window), state in partitions.items()
        },
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qk-bits", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--windows", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument(
        "--planted-lengths",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--candidate-counts",
        nargs="+",
        type=int,
        default=[1_000, 1_000_000, 100_000_000],
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--sample-count", type=int, default=131_072)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report = run_ablation(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
