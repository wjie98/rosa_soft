"""Single-sample hard-forward fitting benchmark for RosaSoft."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft
from rosa_soft import rosa_soft_reference
from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.testing import inspect_rosa_soft


TARGET_MODES = (
    "any-candidate",
    "strict-longest-latest",
)


def make_copy_tokens(
    seq_len: int,
    vocab_size: int,
    motif_min: int,
    motif_max: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    motif_len = int(
        torch.randint(motif_min, motif_max + 1, (1,), generator=generator).item()
    )
    motif = torch.randint(0, vocab_size, (motif_len,), generator=generator)
    tokens = motif.repeat(math.ceil(seq_len / motif_len))[:seq_len].clone()

    if seq_len >= motif_len * 4:
        block_count = max(1, seq_len // max(1, motif_len * 2))
        for _ in range(block_count):
            max_span = min(seq_len, motif_len * 4)
            span = int(
                torch.randint(
                    motif_len,
                    max_span + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            src = int(
                torch.randint(
                    0,
                    max(1, seq_len - span + 1),
                    (1,),
                    generator=generator,
                ).item()
            )
            dst = int(
                torch.randint(
                    0,
                    max(1, seq_len - span + 1),
                    (1,),
                    generator=generator,
                ).item()
            )
            tokens[dst : dst + span] = tokens[src : src + span].clone()
    return tokens.unsqueeze(0)


def historical_target_mask(
    tokens: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Select next-token targets with at least one correct historical route."""

    if tokens.ndim != 2 or tokens.size(1) < 2:
        raise ValueError("tokens must have shape (B, T) with T >= 2")
    if max_suffix_length < 1:
        raise ValueError("max_suffix_length must be >= 1")

    working_tokens = tokens.detach().cpu()
    batch, sequence_length = working_tokens.shape
    mask = torch.zeros(
        batch,
        sequence_length - 1,
        dtype=torch.bool,
    )
    for batch_index in range(batch):
        for query_position in range(sequence_length - 1):
            target = working_tokens[batch_index, query_position + 1]
            found = False
            for route_value_position in range(1, query_position + 1):
                if (
                    working_tokens[batch_index, route_value_position]
                    != target
                ):
                    continue
                suffix_steps = min(
                    max_suffix_length,
                    query_position + 1,
                    route_value_position,
                )
                for suffix_length in range(1, suffix_steps + 1):
                    query_start = query_position + 1 - suffix_length
                    key_start = route_value_position - suffix_length
                    if torch.equal(
                        working_tokens[
                            batch_index,
                            query_start : query_position + 1,
                        ],
                        working_tokens[
                            batch_index,
                            key_start:route_value_position,
                        ],
                    ):
                        mask[batch_index, query_position] = True
                        found = True
                        break
                if found:
                    break
    return mask.to(tokens.device)


def strict_longest_latest_target_mask(
    tokens: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Select targets returned by exact token longest/latest routing."""

    if tokens.ndim != 2 or tokens.size(1) < 2:
        raise ValueError("tokens must have shape (B, T) with T >= 2")
    if max_suffix_length < 1:
        raise ValueError("max_suffix_length must be >= 1")

    working_tokens = tokens.detach().cpu()
    batch, sequence_length = working_tokens.shape
    mask = torch.zeros(
        batch,
        sequence_length - 1,
        dtype=torch.bool,
    )
    for batch_index in range(batch):
        for query_position in range(sequence_length - 1):
            selected_route = 0
            selected_length = 0
            for route_value_position in range(1, query_position + 1):
                suffix_steps = min(
                    max_suffix_length,
                    query_position + 1,
                    route_value_position,
                )
                suffix_length = 0
                for candidate_length in range(1, suffix_steps + 1):
                    query_start = (
                        query_position + 1 - candidate_length
                    )
                    key_start = (
                        route_value_position - candidate_length
                    )
                    if not torch.equal(
                        working_tokens[
                            batch_index,
                            query_start : query_position + 1,
                        ],
                        working_tokens[
                            batch_index,
                            key_start:route_value_position,
                        ],
                    ):
                        break
                    suffix_length = candidate_length
                if suffix_length >= selected_length and suffix_length > 0:
                    selected_route = route_value_position
                    selected_length = suffix_length
            if (
                selected_route > 0
                and working_tokens[batch_index, selected_route]
                == working_tokens[batch_index, query_position + 1]
            ):
                mask[batch_index, query_position] = True
    return mask.to(tokens.device)


def build_target_mask(
    tokens: Tensor,
    max_suffix_length: int,
    target_mode: str,
) -> Tensor:
    if target_mode == "any-candidate":
        return historical_target_mask(tokens, max_suffix_length)
    if target_mode == "strict-longest-latest":
        return strict_longest_latest_target_mask(
            tokens,
            max_suffix_length,
        )
    raise ValueError(
        f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}"
    )


def loss_and_accuracy(
    logits: Tensor,
    tokens: Tensor,
    target_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, float]:
    prediction = logits[:, :-1].float()
    targets = tokens[:, 1:]
    if target_mask is not None:
        if target_mask.shape != targets.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be bool with shape (B, T - 1)")
        prediction = prediction[target_mask]
        targets = targets[target_mask]
    loss = F.cross_entropy(
        prediction.reshape(-1, prediction.size(-1)),
        targets.reshape(-1),
    )
    accuracy = float((prediction.argmax(dim=-1) == targets).float().mean().item())
    return loss, accuracy


@torch.no_grad()
def hard_feature_collision_stats(
    tokens: Tensor,
    target_mask: Tensor,
    routed: Tensor,
    selected_route_indices: Tensor,
) -> Dict[str, float]:
    """Measure the irreducible target entropy of this tiny model's hard input."""

    if target_mask.shape != tokens[:, :-1].shape:
        raise ValueError("target_mask must have shape (B, T - 1)")
    if routed.shape[:2] != tokens.shape:
        raise ValueError("routed must have leading shape (B, T)")
    if selected_route_indices.shape[:1] != tokens.shape[:1] or (
        selected_route_indices.shape[2] != tokens.shape[1]
    ):
        raise ValueError(
            "selected_route_indices must have shape (B, H, T)"
        )

    tokens_cpu = tokens.detach().cpu()
    target_mask_cpu = target_mask.detach().cpu()
    routed_cpu = routed.detach().cpu()
    routes_cpu = selected_route_indices.detach().cpu()
    groups: Dict[Tuple[int, ...], list[Tuple[int, Tuple[int, ...]]]] = {}
    for batch_index, query_position in target_mask_cpu.nonzero().tolist():
        feature = (
            int(tokens_cpu[batch_index, query_position]),
            *(
                int(value)
                for value in routed_cpu[
                    batch_index,
                    query_position,
                ].reshape(-1).tolist()
            ),
        )
        route_tokens = tuple(
            -1
            if route_index == 0
            else int(tokens_cpu[batch_index, route_index])
            for route_index in routes_cpu[
                batch_index,
                :,
                query_position,
            ].tolist()
        )
        target = int(tokens_cpu[batch_index, query_position + 1])
        groups.setdefault(feature, []).append((target, route_tokens))

    target_count = sum(len(rows) for rows in groups.values())
    conditional_entropy = 0.0
    conflicting_groups = 0
    conflicting_targets = 0
    route_value_collision_groups = 0
    quantized_value_collision_groups = 0
    largest_conflicting_group = 0
    for rows in groups.values():
        target_counts: Dict[int, int] = {}
        route_token_signatures = set()
        for target, route_tokens in rows:
            target_counts[target] = target_counts.get(target, 0) + 1
            route_token_signatures.add(route_tokens)
        row_count = len(rows)
        for count in target_counts.values():
            probability = count / row_count
            conditional_entropy -= (
                row_count / target_count
            ) * probability * math.log(probability)
        if len(target_counts) <= 1:
            continue
        conflicting_groups += 1
        conflicting_targets += row_count
        largest_conflicting_group = max(
            largest_conflicting_group,
            row_count,
        )
        if len(route_token_signatures) == 1:
            route_value_collision_groups += 1
        else:
            quantized_value_collision_groups += 1

    return {
        "hard_feature_conditional_entropy": conditional_entropy,
        "hard_feature_unique_count": float(len(groups)),
        "hard_feature_conflicting_group_count": float(
            conflicting_groups
        ),
        "hard_feature_conflicting_target_count": float(
            conflicting_targets
        ),
        "hard_feature_largest_conflicting_group": float(
            largest_conflicting_group
        ),
        "hard_route_value_collision_group_count": float(
            route_value_collision_groups
        ),
        "hard_quantized_value_collision_group_count": float(
            quantized_value_collision_groups
        ),
    }


class TinyRosaFitLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        qk_bits: int,
        value_heads: int,
        value_bits: int,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        operator: str,
    ) -> None:
        super().__init__()
        if num_heads % value_heads != 0:
            raise ValueError("num_heads must be divisible by value_heads")
        self.num_heads = num_heads
        self.qk_bits = qk_bits
        self.value_heads = value_heads
        self.value_bits = value_bits
        self.max_suffix_length = max_suffix_length
        self.scale = scale
        self.dropout_p = dropout_p
        self.mismatch_scale = mismatch_scale
        self.operator = operator

        hidden_size = num_heads * qk_bits
        routed_width = num_heads * value_bits
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, value_heads * value_bits, bias=False)
        self.output = nn.Linear(routed_width, hidden_size, bias=False)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def project_symbols(
        self,
        tokens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        residual = self.embedding(tokens)
        hidden = self.input_norm(residual)
        query = self.query(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.num_heads,
            self.qk_bits,
        )
        key = self.key(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.num_heads,
            self.qk_bits,
        )
        value = self.value(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.value_heads,
            self.value_bits,
        )
        return residual, query, key, value

    def forward(self, tokens: Tensor) -> Tensor:
        residual, query, key, value = self.project_symbols(tokens)
        training_operator = (
            rosa_soft.rosa_soft
            if self.operator == "cuda"
            else rosa_soft_reference
        )
        routed = training_operator(
            query,
            key,
            value,
            max_suffix_length=self.max_suffix_length,
            scale=self.scale,
            dropout_p=self.dropout_p if self.training else 0.0,
            mismatch_scale=self.mismatch_scale,
        )
        hidden = residual + self.output(routed.flatten(2))
        return self.head(self.output_norm(hidden))

    @torch.no_grad()
    def route_stats(
        self,
        tokens: Tensor,
        target_mask: Tensor,
    ) -> Dict[str, float]:
        _, query, key, value = self.project_symbols(tokens)
        routed, inspection = inspect_rosa_soft(
            query,
            key,
            value,
            max_suffix_length=self.max_suffix_length,
            scale=self.scale,
            dropout_p=self.dropout_p,
            mismatch_scale=self.mismatch_scale,
        )
        probabilities = inspection.route_probabilities[:, :, :-1]
        selected_route_indices = inspection.selected_route_indices[:, :, :-1]
        selected = torch.gather(
            probabilities,
            -1,
            selected_route_indices.unsqueeze(-1),
        ).squeeze(-1)
        effective = probabilities.square().sum(dim=-1).reciprocal()
        scoped_mask = target_mask.unsqueeze(1).expand_as(selected)
        return {
            "hard_top_probability": float(selected[scoped_mask].mean().item()),
            "effective_routes": float(effective[scoped_mask].mean().item()),
            "hard_nonnull_route_fraction": float(
                (selected_route_indices[scoped_mask] != 0).float().mean().item()
            ),
            "observed_max_suffix_length": float(
                inspection.exact_suffix_lengths.max().item()
            ),
            **hard_feature_collision_stats(
                tokens,
                target_mask,
                routed,
                inspection.selected_route_indices,
            ),
        }


@torch.no_grad()
def evaluate(
    model: TinyRosaFitLM,
    tokens: Tensor,
    target_mask: Optional[Tensor] = None,
) -> Tuple[float, float]:
    loss, accuracy = loss_and_accuracy(model(tokens), tokens, target_mask)
    return float(loss.item()), accuracy


def _validate_success_loss_threshold(
    success_loss_threshold: Optional[float],
) -> None:
    if success_loss_threshold is None:
        return
    if (
        not math.isfinite(success_loss_threshold)
        or success_loss_threshold < 0.0
    ):
        raise ValueError("success_loss_threshold must be finite and >= 0")


def evaluate_success(
    final_loss: float,
    success_loss_threshold: Optional[float],
) -> Optional[bool]:
    _validate_success_loss_threshold(success_loss_threshold)
    if success_loss_threshold is None:
        return None
    return math.isfinite(final_loss) and final_loss <= success_loss_threshold


def fit_exit_code(result: Dict[str, object]) -> int:
    return 1 if result.get("success") is False else 0


def fit(args: argparse.Namespace) -> Dict[str, object]:
    success_loss_threshold = args.success_loss_threshold
    _validate_success_loss_threshold(success_loss_threshold)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.operator == "cuda":
        if device.type != "cuda":
            raise ValueError("--operator cuda requires --device cuda")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")
    if not math.isfinite(args.scale) or args.scale <= 0.0:
        raise ValueError("scale must be finite and > 0")
    if (
        not math.isfinite(args.dropout_p)
        or not 0.0 <= args.dropout_p <= 1.0 - 2.0**-24
    ):
        raise ValueError(
            "dropout_p must be finite and in [0, 1 - 2^-24]"
        )
    if not math.isfinite(args.mismatch_scale) or args.mismatch_scale <= 0.0:
        raise ValueError("mismatch_scale must be finite and > 0")

    torch.manual_seed(300_000 + args.seed)
    tokens = make_copy_tokens(
        seq_len=args.seq + 1,
        vocab_size=args.vocab_size,
        motif_min=args.motif_min,
        motif_max=args.motif_max,
        seed=100_000 + args.seed,
    )
    target_mask = build_target_mask(
        tokens,
        args.max_suffix_length,
        args.target_mode,
    )
    if not bool(target_mask.any()):
        raise ValueError(
            "generated sequence has no historically recallable targets"
        )
    tokens = tokens.to(device)
    target_mask = target_mask.to(device)
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

    dropout_seed = (
        args.seed if args.dropout_seed < 0 else args.dropout_seed
    )
    torch.manual_seed(400_000 + dropout_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(400_000 + dropout_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_loss, initial_accuracy = evaluate(model, tokens, target_mask)
    full_initial_loss, full_initial_accuracy = evaluate(model, tokens)
    best_loss = initial_loss
    best_accuracy = initial_accuracy
    best_step = 0
    first_below = {"0.1": -1, "0.01": -1, "0.001": -1}
    started = time.perf_counter()
    print(f"tokens={tokens[0].tolist()}", flush=True)
    print(
        f"operator={args.operator} step=0 fit_loss={initial_loss:.9f} "
        f"acc={initial_accuracy:.4f} scale={args.scale:.6f} "
        f"dropout_p={args.dropout_p:.6f} "
        f"mismatch_scale={args.mismatch_scale:.6f}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy = loss_and_accuracy(
            model(tokens),
            tokens,
            target_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_accuracy = accuracy
            best_step = step
        for threshold in first_below:
            if first_below[threshold] < 0 and loss_value < float(threshold):
                first_below[threshold] = step
        if step % args.log_every == 0 or step == args.steps:
            print(
                f"operator={args.operator} step={step} fit_loss={loss_value:.9f} "
                f"acc={accuracy:.4f} best={best_loss:.9f}@{best_step}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = evaluate(model, tokens, target_mask)
    full_final_loss, full_final_accuracy = evaluate(model, tokens)
    if final_loss < best_loss:
        best_loss = final_loss
        best_accuracy = final_accuracy
        best_step = args.steps
    stats = model.route_stats(tokens, target_mask)
    stats["fit_loss_above_hard_feature_entropy"] = max(
        0.0,
        final_loss - stats["hard_feature_conditional_entropy"],
    )
    success = evaluate_success(final_loss, success_loss_threshold)
    result: Dict[str, object] = {
        "operator": args.operator,
        "device": str(device),
        "seed": args.seed,
        "dropout_seed": dropout_seed,
        "target_mode": args.target_mode,
        "steps": args.steps,
        "tokens": tokens[0].tolist(),
        "max_suffix_length": args.max_suffix_length,
        "scale": args.scale,
        "dropout_p": args.dropout_p,
        "mismatch_scale": args.mismatch_scale,
        "fit_target_count": int(target_mask.sum().item()),
        "excluded_cold_start_target_count": int(
            target_mask.numel() - target_mask.sum().item()
        ),
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "full_initial_loss": full_initial_loss,
        "full_initial_accuracy": full_initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "full_final_loss": full_final_loss,
        "full_final_accuracy": full_final_accuracy,
        "best_loss": best_loss,
        "best_accuracy": best_accuracy,
        "best_step": best_step,
        "first_below": first_below,
        "success_loss_threshold": success_loss_threshold,
        "success": success,
        "elapsed_seconds": elapsed,
        "step_ms": elapsed * 1000.0 / args.steps,
        **stats,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=["reference", "cuda"], default="reference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout-seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--motif-min", type=int, default=4)
    parser.add_argument("--motif-max", type=int, default=8)
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default="any-candidate",
        help=(
            "any-candidate is the historical combinatorial stress mask; "
            "strict-longest-latest keeps only targets selected by exact "
            "token routing"
        ),
    )
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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--success-loss-threshold",
        type=float,
        default=None,
        help=(
            "exit nonzero unless historically recallable-target final_loss "
            "is at or below this value; disabled by default"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = fit(build_parser().parse_args(argv))
    exit_code = fit_exit_code(result)
    if exit_code:
        print(
            "success gate failed: final_loss "
            f"{result['final_loss']:.9f} did not meet <= "
            f"{result['success_loss_threshold']:.9f}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
