"""Shortcut-free two-hop ROSA recall gate.

The first exact-hard route returns an intermediate address.  That hard value
is the only input used to form the second query, whose route returns the final
answer.  Source-to-intermediate maps and answer tables vary per sample, so a
direct source-to-answer shortcut cannot solve the batch.
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
from rosa_soft.testing import inspect_rosa_soft  # noqa: E402


@dataclass(frozen=True)
class MultihopBatch:
    inputs: Tensor
    value: Tensor
    target: Tensor
    expected_feedback: Tensor
    query_sources: Tensor
    first_query_position: int
    source_routes: Tensor
    intermediate_routes: Tensor
    expected_second_routes: Tensor

    def to(self, device: torch.device) -> "MultihopBatch":
        return MultihopBatch(
            inputs=self.inputs.to(device),
            value=self.value.to(device),
            target=self.target.to(device),
            expected_feedback=self.expected_feedback.to(device),
            query_sources=self.query_sources.to(device),
            first_query_position=self.first_query_position,
            source_routes=self.source_routes.to(device),
            intermediate_routes=self.intermediate_routes.to(device),
            expected_second_routes=self.expected_second_routes.to(device),
        )


def _binary_codes(code_ids: Tensor, width: int) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64)
    bits = ((code_ids.unsqueeze(-1) >> shifts) & 1).bool()
    return torch.where(bits, 1.0, -1.0)


def _draw_addresses(
    generator: torch.Generator,
    count: int,
    bits: int,
) -> Tensor:
    usable = torch.arange(1, (1 << bits) - 1)
    ids = usable[torch.randperm(usable.numel(), generator=generator)[:count]]
    return _binary_codes(ids, bits)


def make_multihop_batch(
    *,
    seed: int,
    pairs: int,
    address_count: int,
    address_bits: int,
) -> MultihopBatch:
    if pairs < 1 or address_count < 2:
        raise ValueError("pairs must be positive and address_count >= 2")
    if 2 * address_count > (1 << address_bits) - 2:
        raise ValueError("address_bits are insufficient for unique addresses")

    generator = torch.Generator().manual_seed(seed)
    source_addresses = _draw_addresses(generator, address_count, address_bits)
    intermediate_addresses = _draw_addresses(
        generator,
        address_count,
        address_bits,
    )
    while any(
        torch.equal(source, intermediate)
        for source in source_addresses
        for intermediate in intermediate_addresses
    ):
        intermediate_addresses = _draw_addresses(
            generator,
            address_count,
            address_bits,
        )

    batch_size = 2 * pairs
    memory_entries = 2 * address_count
    memory_length = 2 * memory_entries
    first_query_position = memory_length + 1
    sequence_length = first_query_position + 1
    source_routes = torch.arange(address_count) * 2 + 1
    intermediate_routes = (
        torch.arange(address_count) * 2 + 1 + 2 * address_count
    )
    inputs = torch.zeros(batch_size, sequence_length, address_bits)
    value = -torch.ones(batch_size, sequence_length, 1, address_bits)
    target = torch.empty(batch_size, address_bits)
    expected_feedback = torch.empty(batch_size, address_bits)
    query_sources = torch.empty(batch_size, dtype=torch.int64)
    expected_second_routes = torch.empty(batch_size, dtype=torch.int64)

    for source in range(address_count):
        inputs[:, 2 * source] = source_addresses[source]
    intermediate_offset = 2 * address_count
    for intermediate in range(address_count):
        inputs[:, intermediate_offset + 2 * intermediate] = (
            intermediate_addresses[intermediate]
        )

    usable_answers = torch.arange(1, (1 << address_bits) - 1)
    for pair in range(pairs):
        source = pair % address_count
        first_map = torch.randperm(address_count, generator=generator)
        second_map = torch.randperm(address_count, generator=generator)
        if second_map[source] == first_map[source]:
            swap = (source + 1) % address_count
            second_map[source], second_map[swap] = (
                second_map[swap].clone(),
                second_map[source].clone(),
            )
        first_answers = usable_answers[
            torch.randperm(usable_answers.numel(), generator=generator)
        ][:address_count]
        second_answers = usable_answers[
            torch.randperm(usable_answers.numel(), generator=generator)
        ][:address_count]
        first_target_code = first_answers[first_map[source]]
        second_answers[second_map[source]] = (
            ((1 << address_bits) - 1) ^ first_target_code
        )

        for side, (mapping, answers) in enumerate(
            ((first_map, first_answers), (second_map, second_answers))
        ):
            row = 2 * pair + side
            mapped_addresses = intermediate_addresses[mapping]
            value[row, source_routes, 0] = mapped_addresses
            value[row, intermediate_routes, 0] = _binary_codes(
                answers,
                address_bits,
            )
            mapped = int(mapping[source])
            expected_feedback[row] = intermediate_addresses[mapped]
            expected_second_routes[row] = intermediate_routes[mapped]
            target[row] = value[row, expected_second_routes[row], 0]
            query_sources[row] = source
            inputs[row, first_query_position] = source_addresses[source]

    return MultihopBatch(
        inputs=inputs,
        value=value,
        target=target,
        expected_feedback=expected_feedback,
        query_sources=query_sources,
        first_query_position=first_query_position,
        source_routes=source_routes,
        intermediate_routes=intermediate_routes,
        expected_second_routes=expected_second_routes,
    )


class MultihopRouter(torch.nn.Module):
    def __init__(
        self,
        *,
        address_bits: int,
        state_size: int,
        qk_bits: int,
        update_rate: float,
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
        self.symbolizer = StatefulSymbolizer(
            input_size=address_bits,
            state_size=state_size,
            heads=1,
            bits=qk_bits,
            update_rate=update_rate,
            stateful=True,
        )
        if aligned_head_init:
            with torch.no_grad():
                self.symbolizer.key_projection.weight.copy_(
                    self.symbolizer.query_projection.weight
                )

    def _operator(self):
        return (
            rosa_soft.rosa_soft
            if self.operator == "cuda"
            else rosa_soft.rosa_soft_reference
        )

    def _project_reset_positions(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        reset = torch.ones(
            inputs.shape[:2],
            dtype=torch.bool,
            device=inputs.device,
        )
        symbols = self.symbolizer(inputs, reset_mask=reset)
        return symbols.query, symbols.key

    def first_hop(self, batch: MultihopBatch) -> tuple[Tensor, Tensor, Tensor]:
        query, key = self._project_reset_positions(batch.inputs)
        routed = self._operator()(
            query,
            key,
            batch.value,
            max_suffix_length=1,
            scale=self.scale,
            dropout_p=self.dropout_p if self.training else 0.0,
            mismatch_scale=self.mismatch_scale,
        )
        return routed[:, batch.first_query_position, 0], query, key

    def forward(
        self,
        batch: MultihopBatch,
        *,
        feedback_mode: str = "routed",
    ) -> Tensor:
        feedback, query, key = self.first_hop(batch)
        if feedback_mode == "detached":
            feedback = feedback.detach()
        elif feedback_mode == "zero":
            feedback = torch.zeros_like(feedback)
        elif feedback_mode == "oracle":
            feedback = batch.expected_feedback
        elif feedback_mode == "shuffled":
            feedback = feedback.roll(1, dims=0)
        elif feedback_mode != "routed":
            raise ValueError(f"unknown feedback_mode: {feedback_mode}")

        second_query, second_key = self._project_reset_positions(
            feedback.unsqueeze(1)
        )
        query = torch.cat((query, second_query), dim=1)
        key = torch.cat((key, second_key), dim=1)
        value = torch.cat(
            (
                batch.value,
                -torch.ones_like(batch.value[:, :1]),
            ),
            dim=1,
        )
        routed = self._operator()(
            query,
            key,
            value,
            max_suffix_length=1,
            scale=self.scale,
            dropout_p=self.dropout_p if self.training else 0.0,
            mismatch_scale=self.mismatch_scale,
        )
        return routed[:, -1, 0]


def multihop_loss(output: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(output.float(), target.float())


@torch.no_grad()
def _exact_accuracy(output: Tensor, target: Tensor) -> float:
    return float((output == target).all(dim=-1).float().mean())


@torch.no_grad()
def evaluate(model: MultihopRouter, batch: MultihopBatch) -> dict[str, float]:
    model.eval()
    first_feedback, query, key = model.first_hop(batch)
    _, first_inspection = inspect_rosa_soft(
        query,
        key,
        batch.value,
        max_suffix_length=1,
        scale=model.scale,
        dropout_p=0.0,
        mismatch_scale=model.mismatch_scale,
    )
    first_routes = first_inspection.selected_route_indices[
        :, 0, batch.first_query_position
    ]
    expected_first_routes = batch.source_routes[batch.query_sources]

    outputs = {
        mode: model(batch, feedback_mode=mode)
        for mode in ("routed", "zero", "shuffled", "oracle")
    }
    second_query, second_key = model._project_reset_positions(
        first_feedback.unsqueeze(1)
    )
    second_query = torch.cat((query, second_query), dim=1)
    second_key = torch.cat((key, second_key), dim=1)
    second_value = torch.cat(
        (batch.value, -torch.ones_like(batch.value[:, :1])),
        dim=1,
    )
    _, second_inspection = inspect_rosa_soft(
        second_query,
        second_key,
        second_value,
        max_suffix_length=1,
        scale=model.scale,
        dropout_p=0.0,
        mismatch_scale=model.mismatch_scale,
    )
    second_routes = second_inspection.selected_route_indices[:, 0, -1]
    paired_feedback_differs = (
        first_feedback[0::2] != first_feedback[1::2]
    ).any(dim=-1)
    return {
        "loss": float(multihop_loss(outputs["routed"], batch.target)),
        "answer_accuracy": _exact_accuracy(outputs["routed"], batch.target),
        "first_route_accuracy": float(
            (first_routes == expected_first_routes).float().mean()
        ),
        "first_value_accuracy": _exact_accuracy(
            first_feedback,
            batch.expected_feedback,
        ),
        "second_route_accuracy": float(
            (second_routes == batch.expected_second_routes).float().mean()
        ),
        "zero_feedback_accuracy": _exact_accuracy(
            outputs["zero"],
            batch.target,
        ),
        "shuffled_feedback_accuracy": _exact_accuracy(
            outputs["shuffled"],
            batch.target,
        ),
        "oracle_feedback_accuracy": _exact_accuracy(
            outputs["oracle"],
            batch.target,
        ),
        "paired_feedback_difference_fraction": float(
            paired_feedback_differs.float().mean()
        ),
    }


def _train(
    model: MultihopRouter,
    batch: MultihopBatch,
    *,
    feedback_mode: str,
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
        output = model(batch, feedback_mode=feedback_mode)
        loss = multihop_loss(output, batch.target)
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


def run_seed(args: argparse.Namespace, seed: int, feedback_mode: str) -> dict[str, object]:
    device = torch.device(args.device)
    batch = make_multihop_batch(
        seed=100_000 + seed,
        pairs=args.pairs,
        address_count=args.address_count,
        address_bits=args.address_bits,
    ).to(device)
    torch.manual_seed(200_000 + seed)
    model = MultihopRouter(
        address_bits=args.address_bits,
        state_size=args.state_size,
        qk_bits=args.qk_bits,
        update_rate=args.update_rate,
        aligned_head_init=args.head_init == "aligned",
        scale=args.scale,
        dropout_p=args.dropout_p,
        mismatch_scale=args.mismatch_scale,
        operator=args.operator,
    ).to(device)
    initial = evaluate(model, batch)
    training = _train(
        model,
        batch,
        feedback_mode=feedback_mode,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    final = evaluate(model, batch)
    passed = bool(
        final["answer_accuracy"] >= 0.99
        and final["first_route_accuracy"] >= 0.99
        and final["first_value_accuracy"] >= 0.99
        and final["second_route_accuracy"] >= 0.99
        and final["zero_feedback_accuracy"] <= 0.25
        and final["shuffled_feedback_accuracy"] <= 0.25
        and final["paired_feedback_difference_fraction"] >= 0.99
    )
    return {
        "seed": seed,
        "training_feedback_mode": feedback_mode,
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
    runs = [
        run_seed(args, seed, feedback_mode)
        for feedback_mode in args.training_feedback_modes
        for seed in args.seeds
    ]
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
            "pass_count": sum(bool(run["passed"]) for run in runs),
            "mean_answer_accuracy": statistics.mean(
                float(run["final"]["answer_accuracy"]) for run in runs
            ),
            "mean_zero_feedback_accuracy": statistics.mean(
                float(run["final"]["zero_feedback_accuracy"]) for run in runs
            ),
            "mean_shuffled_feedback_accuracy": statistics.mean(
                float(run["final"]["shuffled_feedback_accuracy"]) for run in runs
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("reference", "cuda"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--training-feedback-modes",
        nargs="+",
        choices=("routed", "detached", "zero", "oracle"),
        default=["routed", "zero"],
    )
    parser.add_argument("--head-init", choices=("aligned", "independent"), default="aligned")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--address-count", type=int, default=6)
    parser.add_argument("--address-bits", type=int, default=8)
    parser.add_argument("--state-size", type=int, default=32)
    parser.add_argument("--qk-bits", type=int, default=8)
    parser.add_argument("--update-rate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
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
