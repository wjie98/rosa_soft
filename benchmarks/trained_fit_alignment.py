"""Analyze production and bitflip VJPs at trained fitting checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft
from benchmarks.discrete_gradient_alignment import summarize_alignment
from benchmarks.estimator_fit_ablation import (
    _bitflip_vjp_for_input,
    rosa_soft_exact_bitflip,
)
from examples.fit_soft_reference import (
    TARGET_MODES,
    TinyRosaFitLM,
    build_target_mask,
    hard_feature_collision_stats,
    loss_and_accuracy,
    make_copy_tokens,
)
from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.soft_reference import (
    _hard_route_forward,
    _hard_sign,
    rosa_soft_reference,
)
from rosa_soft.testing import inspect_rosa_soft


def _flatten_parameter_gradients(
    model: TinyRosaFitLM,
    names: Sequence[str],
) -> Tensor:
    parameters = dict(model.named_parameters())
    gradients = []
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            raise RuntimeError(f"missing gradient for {name}")
        gradients.append(gradient.detach().flatten())
    return torch.cat(gradients)


def _parameter_vjp(
    model: TinyRosaFitLM,
    tokens: Tensor,
    target_mask: Tensor,
    operator,
) -> tuple[Tensor, float]:
    model.zero_grad(set_to_none=True)
    residual, query, key, value = model.project_symbols(tokens)
    routed = operator(
        query,
        key,
        value,
        max_suffix_length=model.max_suffix_length,
        scale=model.scale,
        mismatch_scale=model.mismatch_scale,
    )
    logits = model.head(
        model.output_norm(
            residual + model.output(routed.flatten(2))
        )
    )
    loss, _ = loss_and_accuracy(logits, tokens, target_mask)
    loss.backward()
    return (
        _flatten_parameter_gradients(
            model,
            ("query.weight", "key.weight"),
        ),
        float(loss.detach()),
    )


def _cosine(first: Tensor, second: Tensor) -> Optional[float]:
    first = first.double()
    second = second.double()
    denominator = first.norm() * second.norm()
    if float(denominator) == 0.0:
        return None
    return float(
        (torch.dot(first, second) / denominator)
        .clamp(-1.0, 1.0)
        .item()
    )


def _optional_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return statistics.mean(finite_values)


def analyze_checkpoint(
    model: TinyRosaFitLM,
    tokens: Tensor,
    target_mask: Tensor,
    *,
    top_k: int,
) -> Dict[str, object]:
    model.eval()
    with torch.no_grad():
        residual, query, key, value = model.project_symbols(tokens)
        routed, inspection = inspect_rosa_soft(
            query,
            key,
            value,
            max_suffix_length=model.max_suffix_length,
            scale=model.scale,
            dropout_p=0.0,
            mismatch_scale=model.mismatch_scale,
        )
        collision_stats = hard_feature_collision_stats(
            tokens,
            target_mask,
            routed,
            inspection.selected_route_indices,
        )

    routed_leaf = routed.detach().requires_grad_()
    logits = model.head(
        model.output_norm(
            residual.detach()
            + model.output(routed_leaf.flatten(2))
        )
    )
    loss, accuracy = loss_and_accuracy(logits, tokens, target_mask)
    (grad_output,) = torch.autograd.grad(loss, routed_leaf)

    query_leaf = query.detach().requires_grad_()
    key_leaf = key.detach().requires_grad_()
    value_leaf = value.detach().requires_grad_()
    surrogate_output = rosa_soft_reference(
        query_leaf,
        key_leaf,
        value_leaf,
        max_suffix_length=model.max_suffix_length,
        scale=model.scale,
        dropout_p=0.0,
        mismatch_scale=model.mismatch_scale,
    )
    surrogate_query, surrogate_key = torch.autograd.grad(
        (surrogate_output * grad_output).sum(),
        (query_leaf, key_leaf),
    )
    surrogate_directions = (
        -_hard_sign(query) * surrogate_query,
        -_hard_sign(key) * surrogate_key,
    )

    hard_output, _, _, _ = _hard_route_forward(
        query.detach(),
        key.detach(),
        value.detach(),
        model.max_suffix_length,
    )
    bitflip_started = time.perf_counter()
    bitflip_query_gradient = _bitflip_vjp_for_input(
        0,
        query.detach(),
        key.detach(),
        value.detach(),
        hard_output,
        grad_output,
        model.max_suffix_length,
    )
    bitflip_key_gradient = _bitflip_vjp_for_input(
        1,
        query.detach(),
        key.detach(),
        value.detach(),
        hard_output,
        grad_output,
        model.max_suffix_length,
    )
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    bitflip_seconds = time.perf_counter() - bitflip_started
    oracle_deltas = (
        -_hard_sign(query) * bitflip_query_gradient,
        -_hard_sign(key) * bitflip_key_gradient,
    )

    production_parameter_gradient, production_loss = _parameter_vjp(
        model,
        tokens,
        target_mask,
        rosa_soft_reference,
    )
    bitflip_parameter_gradient, bitflip_loss = _parameter_vjp(
        model,
        tokens,
        target_mask,
        rosa_soft_exact_bitflip,
    )
    if not math.isclose(
        production_loss,
        bitflip_loss,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("production and bitflip hard forwards differ")

    combined_surrogate = torch.cat(
        [direction.flatten() for direction in surrogate_directions]
    )
    combined_oracle = torch.cat(
        [delta.flatten() for delta in oracle_deltas]
    )
    qk_margins = torch.cat(
        [query.detach().abs().flatten(), key.detach().abs().flatten()]
    ).float()
    final_loss = float(loss.detach())
    return {
        "final_loss": final_loss,
        "final_accuracy": accuracy,
        **collision_stats,
        "fit_loss_above_hard_feature_entropy": max(
            0.0,
            final_loss
            - float(collision_stats["hard_feature_conditional_entropy"]),
        ),
        "query_alignment": summarize_alignment(
            surrogate_directions[0],
            oracle_deltas[0],
            top_k=top_k,
        ),
        "key_alignment": summarize_alignment(
            surrogate_directions[1],
            oracle_deltas[1],
            top_k=top_k,
        ),
        "combined_alignment": summarize_alignment(
            combined_surrogate,
            combined_oracle,
            top_k=top_k,
        ),
        "parameter_alignment": {
            "production_l2_norm": float(
                production_parameter_gradient.norm()
            ),
            "bitflip_l2_norm": float(
                bitflip_parameter_gradient.norm()
            ),
            "cosine_similarity": _cosine(
                production_parameter_gradient,
                bitflip_parameter_gradient,
            ),
        },
        "qk_margin": {
            "minimum": float(qk_margins.min()),
            "mean": float(qk_margins.mean()),
            "maximum": float(qk_margins.max()),
            "softsign_derivative_mean": float(
                (1.0 + qk_margins).square().reciprocal().mean()
            ),
        },
        "bitflip_probe_seconds": bitflip_seconds,
        "diagnostic_dropout_p": 0.0,
    }


def run_seed(
    args: argparse.Namespace,
    model_seed: int,
    dropout_seed: int,
) -> Dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(300_000 + model_seed)
    tokens = make_copy_tokens(
        seq_len=args.sequence_length + 1,
        vocab_size=args.vocab_size,
        motif_min=args.motif_min,
        motif_max=args.motif_max,
        seed=100_000 + model_seed,
    ).to(device)
    target_mask = build_target_mask(
        tokens,
        args.max_suffix_length,
        args.target_mode,
    )
    if not bool(target_mask.any()):
        raise ValueError("generated sequence has no selected targets")
    model = TinyRosaFitLM(
        vocab_size=args.vocab_size,
        num_heads=args.heads,
        qk_bits=args.qk_bits,
        value_heads=args.value_heads,
        value_bits=args.value_bits,
        max_suffix_length=args.max_suffix_length,
        scale=args.scale,
        dropout_p=args.dropout_p,
        mismatch_scale=args.mismatch_scale,
        operator=args.operator,
    ).to(device)
    torch.manual_seed(400_000 + dropout_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(400_000 + dropout_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    for _ in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_and_accuracy(
            model(tokens),
            tokens,
            target_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()

    result = analyze_checkpoint(
        model,
        tokens,
        target_mask,
        top_k=args.top_k,
    )
    return {
        "model_seed": model_seed,
        "dropout_seed": dropout_seed,
        "fit_target_count": int(target_mask.sum()),
        **result,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    if args.operator == "cuda":
        if torch.device(args.device).type != "cuda":
            raise ValueError("--operator cuda requires a CUDA device")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")
    dropout_seeds = (
        args.model_seeds
        if args.dropout_seeds is None
        else args.dropout_seeds
    )
    if len(dropout_seeds) != len(args.model_seeds):
        raise ValueError(
            "dropout seeds must match model seed count for paired runs"
        )
    runs = [
        run_seed(args, model_seed, dropout_seed)
        for model_seed, dropout_seed in zip(
            args.model_seeds,
            dropout_seeds,
        )
    ]
    return {
        "schema_version": 1,
        "operator": args.operator,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "model_seeds": args.model_seeds,
        "dropout_seeds": dropout_seeds,
        "steps": args.steps,
        "target_mode": args.target_mode,
        "dropout_p": args.dropout_p,
        "scale": args.scale,
        "mismatch_scale": args.mismatch_scale,
        "top_k": args.top_k,
        "runs": runs,
        "summary": {
            "mean_final_loss": statistics.mean(
                float(run["final_loss"]) for run in runs
            ),
            "collision_free_runs": sum(
                float(run["hard_feature_conditional_entropy"]) == 0.0
                for run in runs
            ),
            "mean_combined_cosine": _optional_mean(
                run["combined_alignment"]["cosine_similarity"]
                for run in runs
            ),
            "mean_parameter_cosine": _optional_mean(
                run["parameter_alignment"]["cosine_similarity"]
                for run in runs
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        choices=("reference", "cuda"),
        default="reference",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[1, 7],
    )
    parser.add_argument("--dropout-seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default="any-candidate",
    )
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--motif-min", type=int, default=4)
    parser.add_argument("--motif-max", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--max-suffix-length", type=int, default=8)
    parser.add_argument(
        "--scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_SCALE,
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=ROSA_SOFT_DEFAULT_DROPOUT_P,
    )
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_experiment(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
