"""Compare the RosaSoft surrogate VJP with exhaustive hard bit flips.

The probe is intentionally small and CPU-only.  For every query and key
logit, it compares the surrogate directional derivative toward the opposite
hard sign with the exact hard-loss change produced by that bit flip.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosa_soft.soft_reference import rosa_soft_reference


DEFAULT_SEEDS = (7, 19, 31)
DEFAULT_SEQUENCE_LENGTHS = (4, 6)
DEFAULT_SYMBOL_DIMS = (2, 4)
DEFAULT_MAX_SUFFIX_LENGTHS = (1, 3)
DEFAULT_SCALES = (0.5, 1.0, 2.0)
DEFAULT_MISMATCH_SCALES = (3.0, 9.0)


@dataclass(frozen=True)
class BitFlipOracle:
    """Exact hard-loss changes for every query and key logit."""

    base_loss: Tensor
    query_loss_deltas: Tensor
    key_loss_deltas: Tensor


def _hard_sign(values: Tensor) -> Tensor:
    return torch.where(values > 0, 1.0, -1.0)


def _validate_loss_weights(
    query_logits: Tensor,
    value_logits: Tensor,
    loss_weights: Tensor,
) -> None:
    expected_shape = (
        query_logits.size(0),
        query_logits.size(1),
        query_logits.size(2),
        value_logits.size(3),
    )
    if tuple(loss_weights.shape) != expected_shape:
        raise ValueError(
            f"loss_weights must have shape {expected_shape}, "
            f"got {tuple(loss_weights.shape)}"
        )
    if loss_weights.device != query_logits.device:
        raise ValueError("loss_weights must be on the input device")
    if loss_weights.dtype != query_logits.dtype:
        raise ValueError("loss_weights must have the input dtype")


def hard_linear_loss(
    query_logits: Tensor,
    key_logits: Tensor,
    value_logits: Tensor,
    loss_weights: Tensor,
    *,
    max_suffix_length: int,
) -> Tensor:
    """Evaluate ``sum(hard_output * loss_weights)`` without sampling."""

    _validate_loss_weights(query_logits, value_logits, loss_weights)
    with torch.no_grad():
        hard_output = rosa_soft_reference(
            query_logits.detach(),
            key_logits.detach(),
            value_logits.detach(),
            max_suffix_length=max_suffix_length,
        )
        return (hard_output * loss_weights.detach()).sum()


def _copy_with_flipped_logit(logits: Tensor, flat_index: int) -> Tensor:
    flipped = logits.detach().clone(memory_format=torch.contiguous_format)
    flat = flipped.view(-1)
    value = flat[flat_index]
    magnitude = value.abs()
    replacement = torch.where(
        value > 0,
        -magnitude,
        torch.where(
            magnitude > 0,
            magnitude,
            torch.ones_like(magnitude),
        ),
    )
    flat[flat_index] = replacement
    return flipped


def discrete_bit_flip_oracle(
    query_logits: Tensor,
    key_logits: Tensor,
    value_logits: Tensor,
    loss_weights: Tensor,
    *,
    max_suffix_length: int,
) -> BitFlipOracle:
    """Exhaustively evaluate the hard-loss delta of every Q/K sign flip."""

    base_loss = hard_linear_loss(
        query_logits,
        key_logits,
        value_logits,
        loss_weights,
        max_suffix_length=max_suffix_length,
    )
    query_deltas = torch.empty(
        query_logits.shape,
        dtype=base_loss.dtype,
        device=base_loss.device,
    )
    key_deltas = torch.empty(
        key_logits.shape,
        dtype=base_loss.dtype,
        device=base_loss.device,
    )

    for flat_index in range(query_logits.numel()):
        flipped_query = _copy_with_flipped_logit(
            query_logits,
            flat_index,
        )
        flipped_loss = hard_linear_loss(
            flipped_query,
            key_logits,
            value_logits,
            loss_weights,
            max_suffix_length=max_suffix_length,
        )
        query_deltas.view(-1)[flat_index] = flipped_loss - base_loss

    for flat_index in range(key_logits.numel()):
        flipped_key = _copy_with_flipped_logit(
            key_logits,
            flat_index,
        )
        flipped_loss = hard_linear_loss(
            query_logits,
            flipped_key,
            value_logits,
            loss_weights,
            max_suffix_length=max_suffix_length,
        )
        key_deltas.view(-1)[flat_index] = flipped_loss - base_loss

    return BitFlipOracle(
        base_loss=base_loss,
        query_loss_deltas=query_deltas,
        key_loss_deltas=key_deltas,
    )


def surrogate_flip_directions(
    query_logits: Tensor,
    key_logits: Tensor,
    value_logits: Tensor,
    loss_weights: Tensor,
    *,
    max_suffix_length: int,
    scale: float,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return hard loss and VJP derivatives toward each opposite hard sign."""

    _validate_loss_weights(query_logits, value_logits, loss_weights)
    query_leaf = query_logits.detach().clone().requires_grad_()
    key_leaf = key_logits.detach().clone().requires_grad_()
    output = rosa_soft_reference(
        query_leaf,
        key_leaf,
        value_logits.detach(),
        max_suffix_length=max_suffix_length,
        scale=scale,
        mismatch_scale=mismatch_scale,
    )
    loss = (output * loss_weights.detach()).sum()
    grad_query, grad_key = torch.autograd.grad(
        loss,
        (query_leaf, key_leaf),
    )

    query_direction = -_hard_sign(query_logits) * grad_query
    key_direction = -_hard_sign(key_logits) * grad_key
    return (
        loss.detach(),
        query_direction.detach(),
        key_direction.detach(),
    )


def _fraction(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_alignment(
    surrogate_direction: Tensor,
    oracle_loss_delta: Tensor,
    *,
    top_k: int,
) -> Dict[str, object]:
    """Summarize directional agreement without dropping any coordinates."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    surrogate = surrogate_direction.detach().double().flatten()
    oracle = oracle_loss_delta.detach().double().flatten()
    if surrogate.shape != oracle.shape:
        raise ValueError(
            "surrogate_direction and oracle_loss_delta must match"
        )
    if surrogate.numel() == 0:
        raise ValueError("alignment tensors must be nonempty")
    if not bool(torch.isfinite(surrogate).all()):
        raise ValueError("surrogate_direction must be finite")
    if not bool(torch.isfinite(oracle).all()):
        raise ValueError("oracle_loss_delta must be finite")

    surrogate_norm = surrogate.norm()
    oracle_norm = oracle.norm()
    if float(surrogate_norm) == 0.0 or float(oracle_norm) == 0.0:
        cosine_similarity = None
    else:
        cosine_similarity = float(
            (
                torch.dot(surrogate, oracle)
                / (surrogate_norm * oracle_norm)
            )
            .clamp(-1.0, 1.0)
            .item()
        )

    surrogate_nonzero = surrogate != 0
    oracle_nonzero = oracle != 0
    oracle_useful = oracle < 0
    predicted_useful = surrogate < 0
    sign_comparable = surrogate_nonzero & oracle_nonzero
    sign_agree = sign_comparable & (
        torch.sign(surrogate) == torch.sign(oracle)
    )

    recommended_indices = predicted_useful.nonzero().flatten()
    if recommended_indices.numel() > 0:
        order = torch.argsort(surrogate[recommended_indices])
        top_indices = recommended_indices[order[:top_k]]
    else:
        top_indices = recommended_indices
    top_deltas = oracle[top_indices]
    top_count = top_indices.numel()
    top_oracle_useful_count = int((top_deltas < 0).sum().item())

    surrogate_zero = ~surrogate_nonzero
    missed_oracle_nonzero = surrogate_zero & oracle_nonzero
    missed_oracle_useful = surrogate_zero & oracle_useful

    candidate_count = surrogate.numel()
    surrogate_nonzero_count = int(surrogate_nonzero.sum().item())
    oracle_nonzero_count = int(oracle_nonzero.sum().item())
    oracle_useful_count = int(oracle_useful.sum().item())
    predicted_useful_count = int(predicted_useful.sum().item())
    sign_comparable_count = int(sign_comparable.sum().item())
    sign_agreement_count = int(sign_agree.sum().item())
    surrogate_zero_count = int(surrogate_zero.sum().item())
    missed_oracle_nonzero_count = int(missed_oracle_nonzero.sum().item())
    missed_oracle_useful_count = int(missed_oracle_useful.sum().item())

    return {
        "candidate_count": candidate_count,
        "surrogate_direction_l2_norm": float(surrogate_norm.item()),
        "oracle_delta_l2_norm": float(oracle_norm.item()),
        "cosine_similarity": cosine_similarity,
        "surrogate_nonzero_count": surrogate_nonzero_count,
        "oracle_nonzero_count": oracle_nonzero_count,
        "oracle_useful_count": oracle_useful_count,
        "predicted_useful_count": predicted_useful_count,
        "sign_comparable_count": sign_comparable_count,
        "sign_agreement_count": sign_agreement_count,
        "sign_agreement": _fraction(
            sign_agreement_count,
            sign_comparable_count,
        ),
        "top_k": {
            "requested": top_k,
            "evaluated_recommendations": top_count,
            "oracle_useful_count": top_oracle_useful_count,
            "oracle_useful_fraction": _fraction(
                top_oracle_useful_count,
                top_count,
            ),
            "mean_oracle_loss_delta": (
                float(top_deltas.mean().item()) if top_count else None
            ),
        },
        "zero_support": {
            "surrogate_zero_count": surrogate_zero_count,
            "surrogate_zero_fraction": _fraction(
                surrogate_zero_count,
                candidate_count,
            ),
            "missed_oracle_nonzero_count": missed_oracle_nonzero_count,
            "missed_oracle_nonzero_fraction": _fraction(
                missed_oracle_nonzero_count,
                oracle_nonzero_count,
            ),
            "missed_oracle_useful_count": missed_oracle_useful_count,
            "missed_oracle_useful_fraction": _fraction(
                missed_oracle_useful_count,
                oracle_useful_count,
            ),
        },
    }


def _nonzero_randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> Tensor:
    values = torch.randn(
        shape,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )
    signs = torch.where(values >= 0, 1.0, -1.0)
    return signs * (values.abs() + 0.2)


def measure_configuration(
    *,
    seed: int,
    seq_len: int,
    symbol_dim: int,
    max_suffix_length: int,
    scale: float,
    mismatch_scale: float,
    value_dim: int = 4,
    top_k: int = 8,
) -> Dict[str, object]:
    """Run one reproducible exhaustive CPU alignment configuration."""

    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if symbol_dim < 1 or symbol_dim > 32:
        raise ValueError("symbol_dim must be in [1, 32]")
    if value_dim < 1:
        raise ValueError("value_dim must be >= 1")

    data_generator = torch.Generator(device="cpu").manual_seed(seed)
    query = _nonzero_randn(
        (1, seq_len, 1, symbol_dim),
        generator=data_generator,
    )
    key = _nonzero_randn(
        (1, seq_len, 1, symbol_dim),
        generator=data_generator,
    )
    value = _nonzero_randn(
        (1, seq_len, 1, value_dim),
        generator=data_generator,
    )
    loss_weights = _nonzero_randn(
        (1, seq_len, 1, value_dim),
        generator=data_generator,
    )
    oracle = discrete_bit_flip_oracle(
        query,
        key,
        value,
        loss_weights,
        max_suffix_length=max_suffix_length,
    )
    (
        surrogate_forward_loss,
        query_direction,
        key_direction,
    ) = surrogate_flip_directions(
        query,
        key,
        value,
        loss_weights,
        max_suffix_length=max_suffix_length,
        scale=scale,
        mismatch_scale=mismatch_scale,
    )
    combined_direction = torch.cat(
        (query_direction.flatten(), key_direction.flatten())
    )
    combined_oracle_delta = torch.cat(
        (
            oracle.query_loss_deltas.flatten(),
            oracle.key_loss_deltas.flatten(),
        )
    )

    return {
        "seed": seed,
        "sequence_length": seq_len,
        "symbol_dim": symbol_dim,
        "max_suffix_length": min(max_suffix_length, seq_len),
        "scale": float(scale),
        "mismatch_scale": float(mismatch_scale),
        "base_hard_loss": float(oracle.base_loss.item()),
        "surrogate_forward_loss": float(surrogate_forward_loss.item()),
        "hard_forward_loss_difference": float(
            (surrogate_forward_loss - oracle.base_loss).item()
        ),
        "query": summarize_alignment(
            query_direction,
            oracle.query_loss_deltas,
            top_k=top_k,
        ),
        "key": summarize_alignment(
            key_direction,
            oracle.key_loss_deltas,
            top_k=top_k,
        ),
        "combined": summarize_alignment(
            combined_direction,
            combined_oracle_delta,
            top_k=top_k,
        ),
    }


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _aggregate_results(
    results: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    combined = [result["combined"] for result in results]
    top_k = [summary["top_k"] for summary in combined]
    zero_support = [summary["zero_support"] for summary in combined]

    candidate_count = sum(summary["candidate_count"] for summary in combined)
    oracle_useful_count = sum(
        summary["oracle_useful_count"] for summary in combined
    )
    missed_oracle_useful_count = sum(
        summary["missed_oracle_useful_count"]
        for summary in zero_support
    )
    top_count = sum(
        summary["evaluated_recommendations"] for summary in top_k
    )
    top_oracle_useful_count = sum(
        summary["oracle_useful_count"] for summary in top_k
    )

    return {
        "configuration_count": len(results),
        "candidate_count": candidate_count,
        "mean_combined_cosine_similarity": _mean_optional(
            [summary["cosine_similarity"] for summary in combined]
        ),
        "mean_combined_sign_agreement": _mean_optional(
            [summary["sign_agreement"] for summary in combined]
        ),
        "oracle_useful_count": oracle_useful_count,
        "missed_oracle_useful_count": missed_oracle_useful_count,
        "missed_oracle_useful_fraction": _fraction(
            missed_oracle_useful_count,
            oracle_useful_count,
        ),
        "top_k_evaluated_recommendations": top_count,
        "top_k_oracle_useful_count": top_oracle_useful_count,
        "top_k_oracle_useful_fraction": _fraction(
            top_oracle_useful_count,
            top_count,
        ),
    }


def _require_nonempty(name: str, values: Sequence[object]) -> None:
    if not values:
        raise ValueError(f"{name} must be nonempty")


def run_experiment(
    *,
    seeds: Sequence[int],
    sequence_lengths: Sequence[int],
    symbol_dims: Sequence[int],
    max_suffix_lengths: Sequence[int],
    scales: Sequence[float],
    mismatch_scales: Sequence[float],
    value_dim: int,
    top_k: int,
) -> Dict[str, object]:
    """Evaluate the full Cartesian matrix, retaining every Q/K coordinate."""

    matrix_values = {
        "seeds": seeds,
        "sequence_lengths": sequence_lengths,
        "symbol_dims": symbol_dims,
        "max_suffix_lengths": max_suffix_lengths,
        "scales": scales,
        "mismatch_scales": mismatch_scales,
    }
    for name, values in matrix_values.items():
        _require_nonempty(name, values)

    results: List[Dict[str, object]] = []
    for (
        seed,
        seq_len,
        symbol_dim,
        scale,
        mismatch_scale,
    ) in itertools.product(
        seeds,
        sequence_lengths,
        symbol_dims,
        scales,
        mismatch_scales,
    ):
        effective_lengths = sorted(
            {
                min(int(max_suffix_length), int(seq_len))
                for max_suffix_length in max_suffix_lengths
            }
        )
        for max_suffix_length in effective_lengths:
            results.append(
                measure_configuration(
                    seed=int(seed),
                    seq_len=int(seq_len),
                    symbol_dim=int(symbol_dim),
                    max_suffix_length=max_suffix_length,
                    scale=float(scale),
                    mismatch_scale=float(mismatch_scale),
                    value_dim=value_dim,
                    top_k=top_k,
                )
            )

    report = {
        "schema_version": 2,
        "device": "cpu",
        "dtype": "float32",
        "methodology": {
            "hard_loss": "sum(hard_output * fixed_loss_weights)",
            "surrogate_direction": (
                "-hard_sign(logit) * dloss/dlogit; negative recommends "
                "moving toward the opposite hard sign"
            ),
            "oracle_delta": "loss(bit_flipped) - loss(base)",
            "sign_agreement_support": (
                "coordinates where surrogate direction and oracle delta "
                "are both nonzero"
            ),
            "top_k": (
                "most negative surrogate directions among recommended flips"
            ),
            "candidate_semantics": (
                "dense exhaustive enumeration of every query and key logit"
            ),
        },
        "matrix": {
            name: list(values) for name, values in matrix_values.items()
        }
        | {
            "value_dim": value_dim,
            "top_k": top_k,
        },
        "aggregate": _aggregate_results(results),
        "results": results,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively compare RosaSoft CPU surrogate directions with "
            "hard Q/K bit flips."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
    )
    parser.add_argument(
        "--symbol-dims",
        type=int,
        nargs="+",
        default=DEFAULT_SYMBOL_DIMS,
    )
    parser.add_argument(
        "--max-suffix-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_MAX_SUFFIX_LENGTHS,
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
    )
    parser.add_argument(
        "--mismatch-scales",
        type=float,
        nargs="+",
        default=DEFAULT_MISMATCH_SCALES,
    )
    parser.add_argument("--value-dim", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_experiment(
        seeds=args.seeds,
        sequence_lengths=args.sequence_lengths,
        symbol_dims=args.symbol_dims,
        max_suffix_lengths=args.max_suffix_lengths,
        scales=args.scales,
        mismatch_scales=args.mismatch_scales,
        value_dim=args.value_dim,
        top_k=args.top_k,
    )
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
