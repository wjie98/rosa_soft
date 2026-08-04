"""Offline conflict statistics and deterministic dormant-bit growth.

This is a structural research prototype, not a production training policy.
It preallocates a projection bank, exposes only the active prefix, and uses
hard key-code conflicts to initialize one dormant read/write bit at a time.
The split consumes continuation labels and is therefore reported as an
external supervised edit, not as a gradient estimator improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft  # noqa: E402
from rosa_soft.soft_reference import _hard_sign  # noqa: E402
from rosa_soft.testing import inspect_rosa_soft  # noqa: E402


@dataclass(frozen=True)
class ConflictStats:
    active_bits: int
    samples: int
    states: int
    conflicting_states: int
    conflicting_samples: int
    conflicting_pairs: int
    max_state_size: int
    conditional_entropy_bits: float


@dataclass(frozen=True)
class GrowthBatch:
    features: Tensor
    type_symbol: Tensor
    value: Tensor
    target: Tensor
    labels: Tensor
    cue_features: Tensor
    query_positions: Tensor
    expected_routes: Tensor

    def to(self, device: torch.device) -> "GrowthBatch":
        return GrowthBatch(
            features=self.features.to(device),
            type_symbol=self.type_symbol.to(device),
            value=self.value.to(device),
            target=self.target.to(device),
            labels=self.labels.to(device),
            cue_features=self.cue_features.to(device),
            query_positions=self.query_positions.to(device),
            expected_routes=self.expected_routes.to(device),
        )


def _binary_codes(code_ids: Tensor, width: int) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64, device=code_ids.device)
    bits = ((code_ids.unsqueeze(-1) >> shifts) & 1).bool()
    return torch.where(bits, 1.0, -1.0)


def _hard_code_ids(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.size(1) > 62:
        raise ValueError("logits must have shape [N, D] with D <= 62")
    bits = (_hard_sign(logits) > 0).to(torch.int64)
    shifts = torch.arange(logits.size(1), device=logits.device)
    return (bits << shifts).sum(dim=-1)


def collect_conflict_stats(logits: Tensor, labels: Tensor) -> ConflictStats:
    if labels.ndim != 1 or logits.size(0) != labels.numel():
        raise ValueError("labels must have shape [N]")
    if labels.numel() < 1:
        raise ValueError("at least one labeled state is required")
    code_ids = _hard_code_ids(logits).cpu()
    labels_cpu = labels.to(torch.int64).cpu()
    groups: dict[int, list[int]] = {}
    for index, code in enumerate(code_ids.tolist()):
        groups.setdefault(int(code), []).append(index)

    conflicting_states = 0
    conflicting_samples = 0
    conflicting_pairs = 0
    max_state_size = 0
    conditional_entropy = 0.0
    sample_count = labels.numel()
    for indices in groups.values():
        max_state_size = max(max_state_size, len(indices))
        group_labels = labels_cpu[indices]
        _, counts = torch.unique(group_labels, return_counts=True)
        if counts.numel() > 1:
            conflicting_states += 1
            conflicting_samples += len(indices)
            all_pairs = len(indices) * (len(indices) - 1) // 2
            same_pairs = int(((counts * (counts - 1)) // 2).sum())
            conflicting_pairs += all_pairs - same_pairs
        probabilities = counts.float() / len(indices)
        entropy = -(probabilities * probabilities.log2()).sum()
        conditional_entropy += len(indices) / sample_count * float(entropy)
    return ConflictStats(
        active_bits=logits.size(1),
        samples=sample_count,
        states=len(groups),
        conflicting_states=conflicting_states,
        conflicting_samples=conflicting_samples,
        conflicting_pairs=conflicting_pairs,
        max_state_size=max_state_size,
        conditional_entropy_bits=conditional_entropy,
    )


class DormantBitProjection(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        max_bits: int,
        active_bits: int,
        aligned_head_init: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= active_bits <= max_bits:
            raise ValueError("active_bits must be in [1, max_bits]")
        self.feature_size = int(feature_size)
        self.max_bits = int(max_bits)
        self.active_bits = int(active_bits)
        self.query_weight = nn.Parameter(torch.empty(max_bits, feature_size))
        self.key_weight = nn.Parameter(torch.empty(max_bits, feature_size))
        nn.init.normal_(self.query_weight, std=0.25)
        nn.init.normal_(self.key_weight, std=0.25)
        if aligned_head_init:
            with torch.no_grad():
                self.key_weight.copy_(self.query_weight)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        query = F.linear(features, self.query_weight[: self.active_bits])
        key = F.linear(features, self.key_weight[: self.active_bits])
        return query.unsqueeze(-2), key.unsqueeze(-2)

    @torch.no_grad()
    def activate_split(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        margin: float = 1.0,
    ) -> dict[str, object]:
        if self.active_bits >= self.max_bits:
            raise RuntimeError("all projection bits are already active")
        _, key = self(features)
        key_logits = key[:, 0]
        before = collect_conflict_stats(key_logits, labels)
        if before.conflicting_pairs == 0:
            raise RuntimeError("cannot split a conflict-free codebook")

        code_ids = _hard_code_ids(key_logits).cpu()
        labels_cpu = labels.to(torch.int64).cpu()
        desired = torch.empty(labels.numel(), dtype=features.dtype)
        groups: dict[int, list[int]] = {}
        for index, code in enumerate(code_ids.tolist()):
            groups.setdefault(int(code), []).append(index)
        for code, indices in sorted(groups.items()):
            unique_labels = sorted(set(labels_cpu[indices].tolist()))
            negative_count = (len(unique_labels) + 1) // 2
            side = {
                label: (-1.0 if rank < negative_count else 1.0)
                for rank, label in enumerate(unique_labels)
            }
            if len(unique_labels) == 1:
                side[unique_labels[0]] = -1.0 if code % 2 == 0 else 1.0
            for index in indices:
                desired[index] = side[int(labels_cpu[index])]

        desired = desired.to(features.device) * float(margin)
        solution = torch.linalg.lstsq(features.float(), desired.float()).solution
        solution = solution.to(self.query_weight.dtype)
        bit = self.active_bits
        self.query_weight[bit].copy_(solution)
        self.key_weight[bit].copy_(solution)
        self.active_bits += 1
        _, new_key = self(features)
        after = collect_conflict_stats(new_key[:, 0], labels)
        return {
            "activated_bit": bit,
            "before": asdict(before),
            "after": asdict(after),
            "split_fit_accuracy": float(
                ((_hard_sign(features @ solution) == _hard_sign(desired)))
                .float()
                .mean()
            ),
        }


def make_growth_batch(candidate_count: int) -> GrowthBatch:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    value_bits = max(2, math.ceil(math.log2(candidate_count)))
    memory_length = 2 * candidate_count
    query_positions = torch.arange(
        memory_length + 1,
        memory_length + 1 + candidate_count,
    )
    sequence_length = int(query_positions[-1]) + 1
    expected_routes = torch.arange(candidate_count) * 2 + 1
    cue_features = torch.eye(candidate_count)
    features = torch.zeros(1, sequence_length, candidate_count)
    type_symbol = -torch.ones(1, sequence_length, 1, 1)
    value = -torch.ones(1, sequence_length, 1, value_bits)
    labels = torch.arange(candidate_count, dtype=torch.int64)
    payload_codes = _binary_codes(labels, value_bits)
    for cue in range(candidate_count):
        features[0, 2 * cue] = cue_features[cue]
        features[0, query_positions[cue]] = cue_features[cue]
        type_symbol[0, 2 * cue, 0, 0] = 1.0
        type_symbol[0, query_positions[cue], 0, 0] = 1.0
        value[0, expected_routes[cue], 0] = payload_codes[cue]
    return GrowthBatch(
        features=features,
        type_symbol=type_symbol,
        value=value,
        target=payload_codes.unsqueeze(0),
        labels=labels,
        cue_features=cue_features,
        query_positions=query_positions,
        expected_routes=expected_routes,
    )


class GrowthRouter(nn.Module):
    def __init__(
        self,
        *,
        candidate_count: int,
        max_bits: int,
        active_bits: int,
        aligned_head_init: bool,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        operator: str,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.dropout_p = float(dropout_p)
        self.mismatch_scale = float(mismatch_scale)
        self.operator = operator
        self.projection = DormantBitProjection(
            feature_size=candidate_count,
            max_bits=max_bits,
            active_bits=active_bits,
            aligned_head_init=aligned_head_init,
        )

    def project(self, batch: GrowthBatch) -> tuple[Tensor, Tensor]:
        query, key = self.projection(batch.features)
        return (
            torch.cat((query, batch.type_symbol), dim=-1),
            torch.cat((key, batch.type_symbol), dim=-1),
        )

    def forward(self, batch: GrowthBatch) -> Tensor:
        query, key = self.project(batch)
        operator = (
            rosa_soft.rosa_soft
            if self.operator == "cuda"
            else rosa_soft.rosa_soft_reference
        )
        routed = operator(
            query,
            key,
            batch.value,
            max_suffix_length=1,
            scale=self.scale,
            dropout_p=self.dropout_p if self.training else 0.0,
            mismatch_scale=self.mismatch_scale,
        )
        return routed[:, batch.query_positions, 0]


def growth_loss(output: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(output.float(), target.float())


@torch.no_grad()
def evaluate(model: GrowthRouter, batch: GrowthBatch) -> dict[str, object]:
    model.eval()
    query, key = model.project(batch)
    routed, inspection = inspect_rosa_soft(
        query,
        key,
        batch.value,
        max_suffix_length=1,
        scale=model.scale,
        dropout_p=0.0,
        mismatch_scale=model.mismatch_scale,
    )
    output = routed[:, batch.query_positions, 0]
    routes = inspection.selected_route_indices[:, 0, batch.query_positions]
    cue_query, cue_key = model.projection(batch.cue_features)
    conflicts = collect_conflict_stats(cue_key[:, 0], batch.labels)
    return {
        "active_bits": model.projection.active_bits,
        "loss": float(growth_loss(output, batch.target)),
        "route_accuracy": float(
            (routes == batch.expected_routes).float().mean()
        ),
        "value_accuracy": float(
            (output == batch.target).all(dim=-1).float().mean()
        ),
        "query_key_alignment_fraction": float(
            (_hard_sign(cue_query) == _hard_sign(cue_key))
            .all(dim=(-1, -2))
            .float()
            .mean()
        ),
        "conflicts": asdict(conflicts),
    }


def _optimization_step(
    model: GrowthRouter,
    batch: GrowthBatch,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = growth_loss(output, batch.target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())


def run_strategy(
    args: argparse.Namespace,
    *,
    seed: int,
    strategy: str,
) -> dict[str, object]:
    device = torch.device(args.device)
    batch = make_growth_batch(args.candidate_count).to(device)
    torch.manual_seed(100_000 + seed)
    model = GrowthRouter(
        candidate_count=args.candidate_count,
        max_bits=args.max_bits,
        active_bits=(args.max_bits if strategy == "fixed" else args.initial_bits),
        aligned_head_init=args.head_init == "aligned",
        scale=args.scale,
        dropout_p=args.dropout_p,
        mismatch_scale=args.mismatch_scale,
        operator=args.operator,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, object]] = []
    split_events: list[dict[str, object]] = []
    initial = evaluate(model, batch)
    history.append({"step": 0, **initial})
    first_exact_step = 0 if initial["route_accuracy"] == 1.0 else -1
    started = time.perf_counter()
    for step in range(args.steps):
        if (
            strategy == "growth"
            and step % args.growth_interval == 0
            and model.projection.active_bits < args.max_bits
        ):
            current = history[-1] if history[-1]["step"] == step else evaluate(model, batch)
            if current["conflicts"]["conflicting_pairs"] > 0:
                event = model.projection.activate_split(
                    batch.cue_features,
                    batch.labels,
                    margin=args.split_margin,
                )
                split_events.append({"step": step, **event})
                split_metrics = evaluate(model, batch)
                history.append(
                    {"step": step, "after_split": True, **split_metrics}
                )
                if (
                    first_exact_step < 0
                    and split_metrics["route_accuracy"] == 1.0
                ):
                    first_exact_step = step
        _optimization_step(model, batch, optimizer, args.grad_clip)
        completed_step = step + 1
        if completed_step % args.growth_interval == 0 or completed_step == args.steps:
            metrics = evaluate(model, batch)
            history.append({"step": completed_step, **metrics})
            if first_exact_step < 0 and metrics["route_accuracy"] == 1.0:
                first_exact_step = completed_step
    if batch.features.is_cuda:
        torch.cuda.synchronize(batch.features.device)
    final = evaluate(model, batch)
    return {
        "seed": seed,
        "strategy": strategy,
        "initial": initial,
        "final": final,
        "first_exact_step": first_exact_step,
        "split_events": split_events,
        "external_split_count": len(split_events),
        "history": history,
        "step_ms": (time.perf_counter() - started) * 1000.0 / max(args.steps, 1),
    }


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if args.growth_interval < 1:
        raise ValueError("growth_interval must be positive")
    if args.operator == "cuda":
        if device.type != "cuda":
            raise ValueError("--operator cuda requires a CUDA device")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")
    runs = [
        run_strategy(args, seed=seed, strategy=strategy)
        for strategy in args.strategies
        for seed in args.seeds
    ]
    summary = {}
    for strategy in args.strategies:
        selected = [run for run in runs if run["strategy"] == strategy]
        summary[strategy] = {
            "runs": len(selected),
            "mean_final_route_accuracy": statistics.mean(
                float(run["final"]["route_accuracy"]) for run in selected
            ),
            "mean_final_conflicting_pairs": statistics.mean(
                float(run["final"]["conflicts"]["conflicting_pairs"])
                for run in selected
            ),
            "mean_external_split_count": statistics.mean(
                int(run["external_split_count"]) for run in selected
            ),
        }
    return {
        "schema_version": 1,
        "operator": args.operator,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "candidate_count": args.candidate_count,
        "initial_bits": args.initial_bits,
        "max_bits": args.max_bits,
        "runs": runs,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("reference", "cuda"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("fixed", "growth"),
        default=["fixed", "growth"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--initial-bits", type=int, default=2)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--head-init", choices=("aligned", "independent"), default="aligned")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--growth-interval", type=int, default=50)
    parser.add_argument("--split-margin", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gate(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
