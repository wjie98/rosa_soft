"""Ablate controlled Q/K/V initialization on next-token ROSA training."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.state_attention_pretraining import (  # noqa: E402
    build_parser as build_pretraining_parser,
    run_benchmark,
)


EXPERIMENTS = {
    "E0_default_dv4": {
        "qk_init": "default",
        "value_bits": 4,
        "value_init": "default",
    },
    "E1_shared_dv4": {
        "qk_init": "partial_shared_orthogonal",
        "value_bits": 4,
        "value_init": "default",
    },
    "E2_default_dv8": {
        "qk_init": "default",
        "value_bits": 8,
        "value_init": "orthogonal",
    },
    "E3_shared_dv8": {
        "qk_init": "partial_shared_orthogonal",
        "value_bits": 8,
        "value_init": "orthogonal",
    },
    "E4_shared_dv16": {
        "qk_init": "partial_shared_orthogonal",
        "value_bits": 16,
        "value_init": "orthogonal",
    },
    "E5_shared_dv8_paired": {
        "qk_init": "partial_shared_orthogonal",
        "value_bits": 8,
        "value_init": "paired_output",
    },
    "E6_shared_dv4_orthogonal": {
        "qk_init": "partial_shared_orthogonal",
        "value_bits": 4,
        "value_init": "orthogonal",
    },
    "E7_independent_qk_dv4": {
        "qk_init": "partial_shared_orthogonal",
        "qk_correlation": 0.0,
        "value_bits": 4,
        "value_init": "default",
    },
}


def _initialization_summary(report: dict[str, object]) -> dict[str, float]:
    metrics = [run["initialization"] for run in report["runs"]]
    return {
        name: statistics.mean(float(metric[name]) for metric in metrics)
        for name in metrics[0]
    }


def run_ablation(args: argparse.Namespace) -> dict[str, object]:
    records = []
    for experiment in args.experiments:
        config = EXPERIMENTS[experiment]
        local_args = copy.deepcopy(args)
        local_args.json_out = ""
        local_args.controlled_module_reset = True
        for name, value in config.items():
            setattr(local_args, name, value)
        report = run_benchmark(local_args)
        records.append(
            {
                "experiment": experiment,
                "config": config,
                "initialization_summary": _initialization_summary(report),
                "report": report,
            }
        )
    return {
        "schema_version": 1,
        "objective": "controlled state-attention initialization ablation",
        "experiments": records,
    }


def _summary(report: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "experiment": record["experiment"],
            "config": record["config"],
            "initialization": record["initialization_summary"],
            "estimators": record["report"]["summary"],
        }
        for record in report["experiments"]
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = build_pretraining_parser()
    parser.description = __doc__
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_ablation(args)
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
