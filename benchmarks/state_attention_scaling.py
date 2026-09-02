"""Scaling matrix for the parameter-free quadratic state-attention VJP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.contextual_estimator_recall import (  # noqa: E402
    build_parser as build_contextual_parser,
    run_benchmark,
)


def _matrix_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: dict[tuple[int, int, int], set[str]] = {}

    def add(associations: int, heads: int, depth: int, axis: str) -> None:
        configs.setdefault((associations, heads, depth), set()).add(axis)

    for associations in args.association_values:
        add(associations, args.base_heads, args.base_depth, "associations")
    for heads in args.head_values:
        add(args.base_associations, heads, args.base_depth, "heads")
    for depth in args.depth_values:
        add(args.base_associations, args.base_heads, depth, "depth")
    return [
        {
            "associations": associations,
            "heads": heads,
            "context_depth": depth,
            "axes": sorted(axes),
        }
        for (associations, heads, depth), axes in sorted(configs.items())
    ]


def _contextual_args(
    args: argparse.Namespace,
    config: dict[str, object],
) -> argparse.Namespace:
    contextual = build_contextual_parser().parse_args([])
    for name in (
        "operator",
        "device",
        "seeds",
        "train_pairs",
        "validation_pairs",
        "hidden_size",
        "qk_bits",
        "value_heads",
        "value_bits",
        "context_scale",
        "steps",
        "baseline_steps",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "scale",
        "mismatch_scale",
    ):
        setattr(contextual, name, getattr(args, name))
    contextual.estimators = list(args.estimators)
    contextual.dropout_p = 0.0
    contextual.associations = int(config["associations"])
    contextual.heads = int(config["heads"])
    contextual.context_depth = int(config["context_depth"])
    contextual.json_out = ""
    contextual.summary_only = False
    return contextual


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    configs = _matrix_configs(args)
    records = []
    for config in configs:
        contextual = _contextual_args(args, config)
        if contextual.heads % contextual.value_heads:
            raise ValueError("every head count must be divisible by value_heads")
        if contextual.associations > 1 << contextual.value_bits:
            raise ValueError("association count exceeds the value code count")
        report = run_benchmark(contextual)
        records.append({"config": config, "report": report})
    return {
        "schema_version": 1,
        "objective": "shortcut-free contextual recall scaling",
        "estimators": list(args.estimators),
        "seeds": list(args.seeds),
        "matrix": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["production", "state_quadratic_attention"],
    )
    parser.add_argument("--operator", choices=("reference", "cuda"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--association-values", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--head-values", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--depth-values", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--base-associations", type=int, default=4)
    parser.add_argument("--base-heads", type=int, default=2)
    parser.add_argument("--base-depth", type=int, default=1)
    parser.add_argument("--train-pairs", type=int, default=16)
    parser.add_argument("--validation-pairs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--qk-bits", type=int, default=8)
    parser.add_argument("--value-heads", type=int, default=1)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--context-scale", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--baseline-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _summary(report: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            **record["config"],
            "summary": record["report"]["summary"],
            "shortcut_checks_passed": record["report"]["shortcut_checks_passed"],
        }
        for record in report["matrix"]
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_matrix(args)
    output = _summary(report) if args.summary_only else report
    print(json.dumps(output, indent=2, allow_nan=False))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
