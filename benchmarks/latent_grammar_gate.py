"""Finite-capacity latent-grammar gate for stateful ROSA symbols.

Each memory phrase exposes its cue only at the first position and then emits
identical blank inputs.  With one hard bit per position, N candidates require
at least ceil(log2(N)) suffix positions.  A stateless projector can vary only
the first bit; the stateful projector can unfold the cue into a trajectory.
Payload values are injected directly, so the only trainable path is routing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft  # noqa: E402
from benchmarks.internal_language import StatefulSymbolizer  # noqa: E402
from rosa_soft.soft_reference import _hard_route_forward, _hard_sign  # noqa: E402
from rosa_soft.testing import inspect_rosa_soft  # noqa: E402


@dataclass(frozen=True)
class GrammarBatch:
    inputs: Tensor
    reset_mask: Tensor
    content_mask: Tensor
    query_phase: Tensor
    key_phase: Tensor
    value: Tensor
    target: Tensor
    query_cues: Tensor
    query_position: int
    candidate_routes: Tensor
    phrase_length: int

    def to(self, device: torch.device) -> "GrammarBatch":
        return GrammarBatch(
            inputs=self.inputs.to(device),
            reset_mask=self.reset_mask.to(device),
            content_mask=self.content_mask.to(device),
            query_phase=self.query_phase.to(device),
            key_phase=self.key_phase.to(device),
            value=self.value.to(device),
            target=self.target.to(device),
            query_cues=self.query_cues.to(device),
            query_position=self.query_position,
            candidate_routes=self.candidate_routes.to(device),
            phrase_length=self.phrase_length,
        )


def _binary_codes(code_ids: Tensor, width: int) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64)
    bits = ((code_ids.unsqueeze(-1) >> shifts) & 1).bool()
    return torch.where(bits, 1.0, -1.0)


def required_candidate_count(phrase_length: int) -> int:
    if phrase_length not in (2, 4, 8):
        raise ValueError("phrase_length must be one of 2, 4, or 8")
    return (1 << (phrase_length // 2)) + 1


def active_content_positions(phrase_length: int) -> tuple[int, ...]:
    """Place one outline bit first and the remaining bits at phrase end."""

    required_candidate_count(phrase_length)
    tail_count = phrase_length // 2
    return (0, *range(phrase_length - tail_count, phrase_length))


def make_grammar_batch(
    *,
    seed: int,
    pairs: int,
    phrase_length: int,
    candidate_count: Optional[int] = None,
    value_bits: int = 6,
) -> GrammarBatch:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if phrase_length < 1:
        raise ValueError("phrase_length must be positive")
    if candidate_count is None:
        candidate_count = required_candidate_count(phrase_length)
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if value_bits < 2 or candidate_count > (1 << value_bits) - 2:
        raise ValueError("value_bits cannot encode the requested payloads")

    generator = torch.Generator().manual_seed(seed)
    batch_size = 2 * pairs
    input_size = candidate_count + 1
    blank = candidate_count
    chunk_length = phrase_length + 1
    memory_length = candidate_count * chunk_length
    query_start = memory_length + 1
    sequence_length = query_start + phrase_length
    query_position = sequence_length - 1
    candidate_routes = (
        torch.arange(candidate_count, dtype=torch.int64) * chunk_length
        + phrase_length
    )

    inputs = torch.zeros(batch_size, sequence_length, input_size)
    reset_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    content_mask = torch.zeros(
        batch_size,
        sequence_length,
        1,
        1,
        dtype=torch.bool,
    )
    query_phase = torch.ones(batch_size, sequence_length, 1, 1)
    key_phase = torch.ones(batch_size, sequence_length, 1, 1)
    values = -torch.ones(batch_size, sequence_length, 1, value_bits)
    targets = torch.empty(batch_size, value_bits)
    query_cues = torch.empty(batch_size, dtype=torch.int64)
    usable_codes = torch.arange(1, (1 << value_bits) - 1)
    complement_mask = (1 << value_bits) - 1

    for cue in range(candidate_count):
        start = cue * chunk_length
        inputs[:, start, cue] = 1.0
        inputs[:, start + 1 : start + phrase_length, blank] = 1.0
        inputs[:, start + phrase_length, blank] = 1.0
        reset_mask[:, start] = True
        for local_position in active_content_positions(phrase_length):
            content_mask[:, start + local_position, 0, 0] = True
        key_phase[:, start, 0, 0] = 1.0
        key_phase[:, start + 1 : start + chunk_length, 0, 0] = -1.0
    inputs[:, memory_length, blank] = 1.0
    reset_mask[:, memory_length] = True

    for pair in range(pairs):
        query_cue = pair % candidate_count
        permutation = usable_codes[
            torch.randperm(usable_codes.numel(), generator=generator)
        ][:candidate_count]
        for side in range(2):
            batch_index = 2 * pair + side
            codes = permutation if side == 0 else permutation ^ complement_mask
            values[batch_index, candidate_routes, 0] = _binary_codes(
                codes,
                value_bits,
            )
            targets[batch_index] = values[
                batch_index,
                candidate_routes[query_cue],
                0,
            ]
            query_cues[batch_index] = query_cue
            inputs[batch_index, query_start, query_cue] = 1.0
            inputs[
                batch_index,
                query_start + 1 : sequence_length,
                blank,
            ] = 1.0
            reset_mask[batch_index, query_start] = True
            for local_position in active_content_positions(phrase_length):
                content_mask[
                    batch_index,
                    query_start + local_position,
                    0,
                    0,
                ] = True
            query_phase[batch_index, query_start, 0, 0] = 1.0
            query_phase[
                batch_index,
                query_start + 1 : sequence_length,
                0,
                0,
            ] = -1.0

    return GrammarBatch(
        inputs=inputs,
        reset_mask=reset_mask,
        content_mask=content_mask,
        query_phase=query_phase,
        key_phase=key_phase,
        value=values,
        target=targets,
        query_cues=query_cues,
        query_position=query_position,
        candidate_routes=candidate_routes,
        phrase_length=phrase_length,
    )


class GrammarRouter(torch.nn.Module):
    def __init__(
        self,
        *,
        candidate_count: int,
        state_size: int,
        update_rate: float,
        stateful: bool,
        aligned_head_init: bool,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        operator: str,
    ) -> None:
        super().__init__()
        self.max_suffix_length = int(max_suffix_length)
        self.scale = float(scale)
        self.dropout_p = float(dropout_p)
        self.mismatch_scale = float(mismatch_scale)
        self.operator = operator
        self.symbolizer = StatefulSymbolizer(
            input_size=candidate_count + 1,
            state_size=state_size,
            heads=1,
            bits=1,
            update_rate=update_rate,
            stateful=stateful,
        )
        if aligned_head_init:
            with torch.no_grad():
                self.symbolizer.key_projection.weight.copy_(
                    self.symbolizer.query_projection.weight
                )

    def project(self, batch: GrammarBatch) -> tuple[Tensor, Tensor]:
        symbols = self.symbolizer(
            batch.inputs,
            reset_mask=batch.reset_mask,
        )
        query_content = torch.where(
            batch.content_mask,
            symbols.query,
            -torch.ones_like(symbols.query),
        )
        key_content = torch.where(
            batch.content_mask,
            symbols.key,
            -torch.ones_like(symbols.key),
        )
        return (
            torch.cat((query_content, batch.query_phase), dim=-1),
            torch.cat((key_content, batch.key_phase), dim=-1),
        )

    def forward(self, batch: GrammarBatch) -> Tensor:
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
            max_suffix_length=self.max_suffix_length,
            scale=self.scale,
            dropout_p=self.dropout_p if self.training else 0.0,
            mismatch_scale=self.mismatch_scale,
        )
        return routed[:, batch.query_position, 0]


def grammar_loss(output: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(output.float(), target.float())


def oracle_route_accuracy(batch: GrammarBatch, window: int) -> float:
    """Route with explicit binary trajectories to verify the gate itself."""

    batch_size, sequence_length, _ = batch.inputs.shape
    candidate_count = batch.candidate_routes.numel()
    content_positions = active_content_positions(batch.phrase_length)
    codes = _binary_codes(
        torch.arange(candidate_count),
        len(content_positions),
    ).to(batch.inputs.device)
    trajectories = torch.full(
        (candidate_count, batch.phrase_length),
        -1.0,
        device=batch.inputs.device,
    )
    trajectories[:, content_positions[0]] = codes[:, -1]
    for code_index, local_position in enumerate(content_positions[1:]):
        trajectories[:, local_position] = codes[:, code_index]
    query_content = torch.full(
        (batch_size, sequence_length, 1, 1),
        -1.0,
        device=batch.inputs.device,
    )
    key_content = query_content.clone()
    chunk_length = batch.phrase_length + 1
    query_start = batch.query_position - batch.phrase_length + 1
    for cue in range(candidate_count):
        start = cue * chunk_length
        key_content[:, start : start + batch.phrase_length, 0, 0] = (
            trajectories[cue]
        )
    for row, cue in enumerate(batch.query_cues.tolist()):
        query_content[
            row,
            query_start : batch.query_position + 1,
            0,
            0,
        ] = trajectories[cue]
    query = torch.cat((query_content, batch.query_phase), dim=-1)
    key = torch.cat((key_content, batch.key_phase), dim=-1)
    _, _, routes, _ = _hard_route_forward(query, key, batch.value, window)
    expected = batch.candidate_routes[batch.query_cues]
    return float((routes[:, 0, batch.query_position] == expected).float().mean())


@torch.no_grad()
def evaluate(model: GrammarRouter, batch: GrammarBatch) -> dict[str, float]:
    model.eval()
    query, key = model.project(batch)
    routed, inspection = inspect_rosa_soft(
        query,
        key,
        batch.value,
        max_suffix_length=model.max_suffix_length,
        scale=model.scale,
        dropout_p=0.0,
        mismatch_scale=model.mismatch_scale,
    )
    output = routed[:, batch.query_position, 0]
    selected = inspection.selected_route_indices[:, 0, batch.query_position]
    expected = batch.candidate_routes[batch.query_cues]
    exact_value = (output == batch.target).all(dim=-1)
    paired_values_differ = (
        output[0::2] != output[1::2]
    ).any(dim=-1)
    query_start = batch.query_position - batch.phrase_length + 1
    paired_query_symbols_equal = torch.equal(
        _hard_sign(query[0::2, query_start:]),
        _hard_sign(query[1::2, query_start:]),
    )
    return {
        "loss": float(grammar_loss(output, batch.target)),
        "route_accuracy": float((selected == expected).float().mean()),
        "value_accuracy": float(exact_value.float().mean()),
        "nonnull_fraction": float((selected != 0).float().mean()),
        "paired_value_difference_fraction": float(
            paired_values_differ.float().mean()
        ),
        "paired_query_symbols_equal": float(paired_query_symbols_equal),
    }


def _train(
    model: GrammarRouter,
    batch: GrammarBatch,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    first_exact = -1
    last_exact = -1
    exact_steps = 0
    best_loss = float("inf")
    started = time.perf_counter()
    for step in range(steps + 1):
        model.train()
        output = model(batch)
        loss = grammar_loss(output, batch.target)
        loss_value = float(loss.detach())
        best_loss = min(best_loss, loss_value)
        if loss_value == 0.0:
            exact_steps += 1
            last_exact = step
            if first_exact < 0:
                first_exact = step
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    if batch.inputs.is_cuda:
        torch.cuda.synchronize(batch.inputs.device)
    return {
        "final_training_loss": float(loss.detach()),
        "best_training_loss": best_loss,
        "first_exact_step": first_exact,
        "last_exact_step": last_exact,
        "exact_step_count": exact_steps,
        "step_ms": (time.perf_counter() - started) * 1000.0 / max(steps, 1),
    }


def run_condition(
    args: argparse.Namespace,
    *,
    seed: int,
    phrase_length: int,
    window: int,
    stateful: bool,
) -> dict[str, object]:
    candidate_count = required_candidate_count(phrase_length)
    pairs = max(args.pairs, candidate_count)
    device = torch.device(args.device)
    batch = make_grammar_batch(
        seed=100_000 + seed,
        pairs=pairs,
        phrase_length=phrase_length,
        candidate_count=candidate_count,
        value_bits=args.value_bits,
    ).to(device)
    torch.manual_seed(200_000 + seed)
    model = GrammarRouter(
        candidate_count=candidate_count,
        state_size=args.state_size,
        update_rate=args.update_rate,
        stateful=stateful,
        aligned_head_init=args.head_init == "aligned",
        max_suffix_length=window,
        scale=args.scale,
        dropout_p=args.dropout_p,
        mismatch_scale=args.mismatch_scale,
        operator=args.operator,
    ).to(device)
    initial = evaluate(model, batch)
    training = _train(
        model,
        batch,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    final = evaluate(model, batch)
    active_in_window = sum(
        position >= phrase_length - window
        for position in active_content_positions(phrase_length)
    )
    theoretical_capacity = (
        1 << active_in_window
        if stateful
        else (2 if window >= phrase_length else 1)
    )
    theoretical_route_ceiling = min(1.0, theoretical_capacity / candidate_count)
    passed = bool(
        final["value_accuracy"] >= 0.99
        and final["route_accuracy"] >= 0.99
        and final["paired_value_difference_fraction"] >= 0.99
        and final["paired_query_symbols_equal"] == 1.0
    )
    return {
        "seed": seed,
        "stateful": stateful,
        "phrase_length": phrase_length,
        "window": window,
        "candidate_count": candidate_count,
        "theoretical_capacity": theoretical_capacity,
        "theoretical_route_ceiling": theoretical_route_ceiling,
        "capacity_sufficient": theoretical_capacity >= candidate_count,
        "oracle_route_accuracy": oracle_route_accuracy(batch, window),
        "initial": initial,
        "training": training,
        "final": final,
        "passed": passed,
    }


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if args.operator == "cuda":
        if device.type != "cuda":
            raise ValueError("--operator cuda requires a CUDA device")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")
    modes = [True, False] if args.mode == "both" else [args.mode == "stateful"]
    runs = []
    for phrase_length in args.phrase_lengths:
        for window in args.windows:
            if window > phrase_length:
                continue
            for stateful in modes:
                for seed in args.seeds:
                    runs.append(
                        run_condition(
                            args,
                            seed=seed,
                            phrase_length=phrase_length,
                            window=window,
                            stateful=stateful,
                        )
                    )
    feasible = [run for run in runs if run["capacity_sufficient"]]
    infeasible = [run for run in runs if not run["capacity_sufficient"]]
    return {
        "schema_version": 1,
        "operator": args.operator,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "head_init": args.head_init,
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "feasible_run_count": len(feasible),
            "feasible_pass_count": sum(bool(run["passed"]) for run in feasible),
            "infeasible_false_pass_count": sum(
                bool(run["passed"]) for run in infeasible
            ),
            "mean_feasible_final_route_accuracy": (
                statistics.mean(
                    float(run["final"]["route_accuracy"]) for run in feasible
                )
                if feasible
                else 0.0
            ),
            "mean_infeasible_final_route_accuracy": (
                statistics.mean(
                    float(run["final"]["route_accuracy"]) for run in infeasible
                )
                if infeasible
                else 0.0
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("reference", "cuda"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("stateful", "stateless", "both"), default="both")
    parser.add_argument("--head-init", choices=("aligned", "independent"), default="aligned")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--phrase-lengths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--state-size", type=int, default=32)
    parser.add_argument("--value-bits", type=int, default=6)
    parser.add_argument("--update-rate", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
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
