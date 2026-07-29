"""Shortcut-free associative-route training gate for RosaSoft.

Paired rows have identical current inputs and complementary targets. The
target appears once in historical value, so neither a residual-only path
nor selecting the current value can pass the gate. Recall queries begin
with a wrong sign bit and must learn the discrete route through RosaSoft's
surrogate VJP.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft
from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)


@dataclass(frozen=True)
class RecallTask:
    base_query: Tensor
    key: Tensor
    value: Tensor
    targets: Tensor
    recall_positions: Tensor
    expected_route_indices: Tensor
    query_selector: Tensor
    target_query_logits: Tensor
    initial_query_logits: Tensor

    def query(self, query_logits: Tensor) -> Tensor:
        updates = torch.einsum(
            "ad,at->td",
            query_logits,
            self.query_selector,
        )
        return self.base_query + updates.view(
            1,
            self.base_query.size(1),
            1,
            self.base_query.size(3),
        )


def _codes(code_ids: Sequence[int], width: int) -> Tensor:
    ids = torch.tensor(code_ids, dtype=torch.int64)
    shifts = torch.arange(width, dtype=torch.int64)
    bits = ((ids.unsqueeze(-1) >> shifts) & 1).bool()
    return torch.where(bits, 1.0, -1.0)


def _sample_unique_ids(
    count: int,
    width: int,
    generator: torch.Generator,
    *,
    block_complements: bool = False,
) -> Tuple[List[int], set[int]]:
    upper_bound = 1 << width
    complement_mask = upper_bound - 1
    selected: List[int] = []
    blocked: set[int] = set()
    while len(selected) < count:
        value = int(
            torch.randint(
                0,
                upper_bound,
                (1,),
                dtype=torch.int64,
                generator=generator,
            ).item()
        )
        if value in blocked:
            continue
        selected.append(value)
        blocked.add(value)
        if block_complements:
            blocked.add(value ^ complement_mask)
    return selected, blocked


def _validate_task_shape(
    associations: int,
    qk_bits: int,
    value_bits: int,
    initial_mismatches: int,
) -> None:
    if associations < 1:
        raise ValueError("associations must be >= 1")
    if not 1 <= qk_bits <= 32:
        raise ValueError("qk_bits must be in [1, 32]")
    if not 1 <= value_bits <= 32:
        raise ValueError("value_bits must be in [1, 32]")
    if (1 << qk_bits) < associations + 1:
        raise ValueError("qk_bits cannot encode all cues plus a distractor")
    if (1 << (value_bits - 1)) <= associations:
        raise ValueError(
            "value_bits cannot encode complementary targets plus a decoy"
        )
    if not 1 <= initial_mismatches < qk_bits:
        raise ValueError("initial_mismatches must be in [1, qk_bits)")


def make_recall_task(
    *,
    seed: int,
    associations: int,
    qk_bits: int,
    value_bits: int,
    initial_mismatches: int,
) -> RecallTask:
    """Build paired episodes whose target route starts with wrong Q bits."""

    _validate_task_shape(
        associations,
        qk_bits,
        value_bits,
        initial_mismatches,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    cue_ids, _ = _sample_unique_ids(
        associations + 1,
        qk_bits,
        generator,
    )
    target_ids, blocked_value_ids = _sample_unique_ids(
        associations,
        value_bits,
        generator,
        block_complements=True,
    )
    while True:
        decoy_id = int(
            torch.randint(
                0,
                1 << value_bits,
                (1,),
                dtype=torch.int64,
                generator=generator,
            ).item()
        )
        if decoy_id not in blocked_value_ids:
            break

    cues = _codes(cue_ids[:associations], qk_bits)
    cue_distractor = _codes([cue_ids[-1]], qk_bits)[0]
    first_targets = _codes(target_ids, value_bits)
    targets = torch.stack((first_targets, -first_targets))
    value_decoy = _codes([decoy_id], value_bits)[0]

    seq_len = 2 * associations + 1
    base_query = cue_distractor.view(1, 1, 1, qk_bits).repeat(
        2,
        seq_len,
        1,
        1,
    )
    key = base_query.clone()
    value = value_decoy.view(1, 1, 1, value_bits).repeat(
        2,
        seq_len,
        1,
        1,
    )

    expected_route_indices = torch.arange(1, associations + 1)
    recall_positions = torch.arange(associations + 1, seq_len)
    query_selector = torch.nn.functional.one_hot(
        recall_positions,
        num_classes=seq_len,
    ).to(torch.float32)
    key[:, :associations, 0] = cues
    base_query[:, recall_positions, 0] = 0.0
    value[:, expected_route_indices, 0] = targets
    initial_query_signs = cues.clone()
    for association in range(associations):
        mismatch_bits = torch.randperm(
            qk_bits,
            generator=generator,
        )[:initial_mismatches]
        initial_query_signs[association, mismatch_bits] *= -1.0
    initial_query_logits = initial_query_signs * (
        0.05
        + 0.1
        * torch.rand(
            associations,
            qk_bits,
            generator=generator,
        )
    )

    return RecallTask(
        base_query=base_query,
        key=key,
        value=value,
        targets=targets,
        recall_positions=recall_positions,
        expected_route_indices=expected_route_indices,
        query_selector=query_selector,
        target_query_logits=cues,
        initial_query_logits=initial_query_logits,
    )


def _exact_accuracy(actual: Tensor, expected: Tensor) -> float:
    return float((actual == expected).all(dim=-1).float().mean().item())


def _bit_accuracy(actual: Tensor, expected: Tensor) -> float:
    return float((actual == expected).float().mean().item())


def _target_values_are_unique(task: RecallTask) -> bool:
    for batch_index in range(task.value.size(0)):
        for association_index, route_index in enumerate(
            task.expected_route_indices.tolist()
        ):
            target = task.targets[batch_index, association_index]
            matches = (task.value[batch_index, :, 0] == target).all(dim=-1)
            if matches.nonzero(as_tuple=False).flatten().tolist() != [route_index]:
                return False
    return True


def run_seed(
    operator_name: str,
    operator: Callable[..., Tensor],
    *,
    device: torch.device,
    seed: int,
    associations: int,
    qk_bits: int,
    value_bits: int,
    initial_mismatches: int,
    steps: int,
    learning_rate: float,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
) -> Dict[str, object]:
    task = make_recall_task(
        seed=seed,
        associations=associations,
        qk_bits=qk_bits,
        value_bits=value_bits,
        initial_mismatches=initial_mismatches,
    )
    base_query = task.base_query.to(device)
    key = task.key.to(device)
    value = task.value.to(device)
    targets = task.targets.to(device)
    query_selector = task.query_selector.to(device)
    query_logits = torch.nn.Parameter(
        task.initial_query_logits.to(device)
    )
    optimizer = torch.optim.SGD(
        [query_logits],
        lr=learning_rate,
    )
    recall_positions = task.recall_positions.to(device)

    def build_query() -> Tensor:
        updates = torch.einsum(
            "ad,at->td",
            query_logits,
            query_selector,
        )
        return base_query + updates.view(
            1,
            base_query.size(1),
            1,
            base_query.size(3),
        )

    def recall_accuracy(query: Tensor) -> Tuple[float, float]:
        with torch.no_grad():
            output = operator(
                query,
                key,
                value,
                max_suffix_length=1,
                scale=scale,
                dropout_p=dropout_p,
                mismatch_scale=mismatch_scale,
            )
        recalled = output[:, recall_positions, 0]
        return (
            _exact_accuracy(recalled, targets),
            _bit_accuracy(recalled, targets),
        )

    torch.manual_seed(700_000 + seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(700_000 + seed)
    initial_exact_accuracy, initial_bit_accuracy = recall_accuracy(
        build_query()
    )
    target_query = base_query + torch.einsum(
        "ad,at->td",
        task.target_query_logits.to(device),
        query_selector,
    ).view(
        1,
        base_query.size(1),
        1,
        base_query.size(3),
    )
    target_query_exact_accuracy, _ = recall_accuracy(target_query)
    active = torch.ones(
        associations,
        dtype=torch.bool,
        device=device,
    )
    success_steps = torch.full(
        (associations,),
        -1,
        dtype=torch.int64,
        device=device,
    )

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = operator(
            build_query(),
            key,
            value,
            max_suffix_length=1,
            scale=scale,
            dropout_p=dropout_p,
            mismatch_scale=mismatch_scale,
        )
        recalled = output[:, recall_positions, 0]
        per_association_loss = -(
            recalled * targets
        ).mean(dim=(0, 2))
        (
            per_association_loss
            * active.to(per_association_loss.dtype)
        ).sum().backward()
        optimizer.step()

        with torch.no_grad():
            hard_output = operator(
                build_query(),
                key,
                value,
                max_suffix_length=1,
                scale=scale,
                dropout_p=dropout_p,
                mismatch_scale=mismatch_scale,
            )
            association_correct = (
                hard_output[:, recall_positions, 0] == targets
            ).all(dim=(0, 2))
            newly_correct = active & association_correct
            success_steps[newly_correct] = step
            active &= ~association_correct
        if not active.any():
            break

    final_exact_accuracy, final_bit_accuracy = recall_accuracy(
        build_query()
    )
    current_value = value[:, recall_positions, 0]
    paired_current_inputs_identical = bool(
        torch.equal(
            build_query()[0, recall_positions],
            build_query()[1, recall_positions],
        )
        and torch.equal(current_value[0], current_value[1])
    )
    paired_targets_are_complements = bool(
        torch.equal(targets[0], -targets[1])
    )
    targets_are_unique_history = _target_values_are_unique(task)
    routes_are_strictly_historical = bool(
        torch.all(task.expected_route_indices < task.recall_positions).item()
    )
    current_value_exact_accuracy = _exact_accuracy(
        current_value,
        targets,
    )
    passed = bool(
        initial_exact_accuracy < 1.0
        and final_exact_accuracy == 1.0
        and target_query_exact_accuracy == 1.0
        and current_value_exact_accuracy == 0.0
        and paired_current_inputs_identical
        and paired_targets_are_complements
        and targets_are_unique_history
        and routes_are_strictly_historical
    )
    return {
        "operator": operator_name,
        "seed": seed,
        "steps": steps,
        "executed_steps": step,
        "success_step": int(success_steps.max().item()),
        "initial_hard_recall_exact_accuracy": initial_exact_accuracy,
        "initial_hard_recall_bit_accuracy": initial_bit_accuracy,
        "hard_recall_exact_accuracy": final_exact_accuracy,
        "hard_recall_bit_accuracy": final_bit_accuracy,
        "target_query_exact_accuracy": target_query_exact_accuracy,
        "current_value_exact_accuracy": current_value_exact_accuracy,
        "current_value_bit_accuracy": _bit_accuracy(
            current_value,
            targets,
        ),
        "residual_only_exact_accuracy_ceiling": 0.5,
        "paired_current_inputs_identical": paired_current_inputs_identical,
        "paired_targets_are_complements": paired_targets_are_complements,
        "targets_are_unique_history": targets_are_unique_history,
        "routes_are_strictly_historical": routes_are_strictly_historical,
        "passed": passed,
    }


def _operators(
    operator_name: str,
    device: torch.device,
) -> Dict[str, Callable[..., Tensor]]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if operator_name in {"cuda", "both"}:
        if device.type != "cuda":
            raise ValueError("CUDA operator requires --device cuda")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")

    operators: Dict[str, Callable[..., Tensor]] = {}
    if operator_name in {"reference", "both"}:
        operators["reference"] = rosa_soft.rosa_soft_reference
    if operator_name in {"cuda", "both"}:
        operators["cuda"] = rosa_soft.rosa_soft
    return operators


def run_gate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    operators = _operators(args.operator, device)
    runs = [
        run_seed(
            operator_name,
            operator,
            device=device,
            seed=seed,
            associations=args.associations,
            qk_bits=args.qk_bits,
            value_bits=args.value_bits,
            initial_mismatches=args.initial_mismatches,
            steps=args.steps,
            learning_rate=args.learning_rate,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
        )
        for operator_name, operator in operators.items()
        for seed in args.seeds
    ]
    return {
        "operator": args.operator,
        "device": str(device),
        "seeds": args.seeds,
        "associations": args.associations,
        "qk_bits": args.qk_bits,
        "value_bits": args.value_bits,
        "initial_mismatches": args.initial_mismatches,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "max_suffix_length": 1,
        "scale": args.scale,
        "dropout_p": args.dropout_p,
        "mismatch_scale": args.mismatch_scale,
        "runs": runs,
        "passed": all(bool(run["passed"]) for run in runs),
    }


def gate_exit_code(report: Dict[str, object]) -> int:
    return 0 if report.get("passed") is True else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        choices=["reference", "cuda", "both"],
        default="reference",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--associations", type=int, default=4)
    parser.add_argument("--qk-bits", type=int, default=8)
    parser.add_argument("--value-bits", type=int, default=8)
    parser.add_argument("--initial-mismatches", type=int, default=1)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.5)
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
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gate(args)
    encoded = json.dumps(report, indent=2)
    print(encoded, flush=True)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return gate_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
