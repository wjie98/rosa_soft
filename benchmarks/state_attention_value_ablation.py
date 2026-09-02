"""Ablate binary, quantized, exponential, and continuous ROSA values."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.state_attention_pretraining import (  # noqa: E402
    NextTokenEstimatorLM,
    NextTokenRecallBatch,
    build_parser as build_pretraining_parser,
    run_benchmark,
)
from benchmarks.value_codec_proxy import (  # noqa: E402
    VALUE_CODECS,
    VALUE_GRADIENTS,
    encode_values,
    rosa_value_codec_state_proxy,
)
from rosa_soft.soft_reference import _expand_value_heads  # noqa: E402


def _experiment(
    codec: str,
    *,
    value_bits: int = 4,
    quant_bits: int = 4,
    exponent_min: float = -4.0,
    value_gradient: str = "proxy",
) -> dict[str, object]:
    return {
        "qk_init": "partial_shared_orthogonal",
        "value_init": "orthogonal",
        "value_bits": value_bits,
        "value_codec": codec,
        "value_quant_bits": quant_bits,
        "value_exponent_min": exponent_min,
        "value_gradient": value_gradient,
    }


EXPERIMENTS = {
    "binary_v4_proxy": _experiment("binary", quant_bits=1),
    "binary_v8_proxy": _experiment("binary", value_bits=8, quant_bits=1),
    "uniform2_v4_proxy": _experiment("uniform", quant_bits=2),
    "uniform4_v4_proxy": _experiment("uniform", quant_bits=4),
    "exp2_v4_proxy": _experiment("exponential", quant_bits=2),
    "exp3_v4_proxy": _experiment("exponential", quant_bits=3),
    "exp4_v4_proxy": _experiment("exponential", quant_bits=4),
    "exp4e8_v4_proxy": _experiment(
        "exponential",
        quant_bits=4,
        exponent_min=-8.0,
    ),
    "rms_v4_proxy": _experiment("rms"),
    "float_v4_proxy": _experiment("float"),
    "uniform2_v8_proxy": _experiment("uniform", value_bits=8, quant_bits=2),
    "uniform4_v8_proxy": _experiment("uniform", value_bits=8, quant_bits=4),
    "exp2_v8_proxy": _experiment("exponential", value_bits=8, quant_bits=2),
    "exp3_v8_proxy": _experiment("exponential", value_bits=8, quant_bits=3),
    "exp4_v8_proxy": _experiment("exponential", value_bits=8, quant_bits=4),
    "rms_v8_proxy": _experiment("rms", value_bits=8),
    "float_v8_proxy": _experiment("float", value_bits=8),
    "binary_v4_selected": _experiment(
        "binary",
        quant_bits=1,
        value_gradient="selected",
    ),
    "binary_v8_selected": _experiment(
        "binary",
        value_bits=8,
        quant_bits=1,
        value_gradient="selected",
    ),
    "uniform2_v4_selected": _experiment(
        "uniform",
        quant_bits=2,
        value_gradient="selected",
    ),
    "uniform4_v4_selected": _experiment(
        "uniform",
        quant_bits=4,
        value_gradient="selected",
    ),
    "exp2_v4_selected": _experiment(
        "exponential",
        quant_bits=2,
        value_gradient="selected",
    ),
    "exp3_v4_selected": _experiment(
        "exponential",
        quant_bits=3,
        value_gradient="selected",
    ),
    "exp4_v4_selected": _experiment(
        "exponential",
        quant_bits=4,
        value_gradient="selected",
    ),
    "exp4e8_v4_selected": _experiment(
        "exponential",
        quant_bits=4,
        exponent_min=-8.0,
        value_gradient="selected",
    ),
    "rms_v4_selected": _experiment("rms", value_gradient="selected"),
    "float_v4_selected": _experiment("float", value_gradient="selected"),
}


class ValueCodecNextTokenEstimatorLM(NextTokenEstimatorLM):
    def __init__(
        self,
        *,
        value_codec: str,
        value_quant_bits: int,
        value_exponent_min: float,
        value_gradient: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if value_codec not in VALUE_CODECS:
            raise ValueError(f"value_codec must be one of {VALUE_CODECS}")
        if value_gradient not in VALUE_GRADIENTS:
            raise ValueError(f"value_gradient must be one of {VALUE_GRADIENTS}")
        self.value_codec = value_codec
        self.value_quant_bits = int(value_quant_bits)
        self.value_exponent_min = float(value_exponent_min)
        self.value_gradient = value_gradient

    def represent_value(self, value: Tensor) -> Tensor:
        return encode_values(
            value,
            codec=self.value_codec,
            quant_bits=self.value_quant_bits,
            exponent_min=self.value_exponent_min,
        )

    def _routed_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        route_mode: str,
    ) -> Tensor:
        if route_mode == "current-value":
            return _expand_value_heads(
                self.represent_value(value),
                self.num_heads,
            ).permute(0, 2, 1, 3)
        if route_mode != "rosa":
            return super()._routed_values(query, key, value, route_mode)
        if self.estimator != "state_quadratic_attention":
            raise ValueError(
                "value codec experiments require state_quadratic_attention"
            )
        return rosa_value_codec_state_proxy(
            query,
            key,
            value,
            codec=self.value_codec,
            quant_bits=self.value_quant_bits,
            exponent_min=self.value_exponent_min,
            value_gradient=self.value_gradient,
            max_suffix_length=1,
            mismatch_scale=self.mismatch_scale,
        )

    @torch.no_grad()
    def value_evaluation_metrics(
        self,
        value: Tensor,
        batch: NextTokenRecallBatch,
    ) -> dict[str, float]:
        represented = self.represent_value(value).float()
        storage = represented[:, batch.storage_payload_positions]
        unique_fractions = []
        for batch_index in range(storage.size(0)):
            for value_head in range(storage.size(2)):
                codes = storage[batch_index, :, value_head]
                unique_fractions.append(
                    codes.unique(dim=0).size(0) / batch.associations
                )
        return {
            "value_representation_rms": float(represented.square().mean().sqrt()),
            "value_representation_max_abs": float(represented.abs().max()),
            "value_storage_unique_code_fraction": statistics.mean(
                unique_fractions
            ),
            "value_projection_weight_rms": float(
                self.value.weight.float().square().mean().sqrt()
            ),
            "output_projection_weight_rms": float(
                self.output.weight.float().square().mean().sqrt()
            ),
        }


def _model_factory(args: argparse.Namespace):
    def build(**kwargs) -> ValueCodecNextTokenEstimatorLM:
        return ValueCodecNextTokenEstimatorLM(
            value_codec=args.value_codec,
            value_quant_bits=args.value_quant_bits,
            value_exponent_min=args.value_exponent_min,
            value_gradient=args.value_gradient,
            **kwargs,
        )

    return build


def _initialization_summary(report: dict[str, object]) -> dict[str, float]:
    metrics = [run["initialization"] for run in report["runs"]]
    return {
        name: statistics.mean(float(metric[name]) for metric in metrics)
        for name in metrics[0]
    }


def run_ablation(args: argparse.Namespace) -> dict[str, object]:
    if list(args.estimators) != ["state_quadratic_attention"]:
        raise ValueError(
            "value codec ablation supports only state_quadratic_attention"
        )
    records = []
    for experiment in args.experiments:
        config = EXPERIMENTS[experiment]
        local_args = copy.deepcopy(args)
        local_args.json_out = ""
        local_args.controlled_module_reset = True
        for name, value in config.items():
            setattr(local_args, name, value)
        report = run_benchmark(
            local_args,
            model_factory=_model_factory(local_args),
        )
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
        "objective": "state-attention value representation ablation",
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
    parser.set_defaults(estimators=["state_quadratic_attention"])
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--value-codec", choices=VALUE_CODECS, default="binary")
    parser.add_argument("--value-quant-bits", type=int, default=4)
    parser.add_argument("--value-exponent-min", type=float, default=-4.0)
    parser.add_argument(
        "--value-gradient",
        choices=VALUE_GRADIENTS,
        default="proxy",
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
