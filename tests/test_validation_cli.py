import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_script = _load_script(
    "_rosa_soft_benchmark_cli",
    "benchmarks/rosa_soft.py",
)
fit_script = _load_script(
    "_rosa_soft_fit_cli",
    "examples/fit_soft_reference.py",
)
recall_script = _load_script(
    "_rosa_soft_recall_cli",
    "examples/associative_recall_gate.py",
)


def test_benchmark_random_inputs_ignore_global_rng_state():
    args = SimpleNamespace(
        device="cpu",
        dtype="float32",
        seed=17,
        batch=1,
        heads=2,
        bits=3,
        value_heads=1,
        value_dim=2,
        pattern="random",
        mode="forward",
        max_suffix_length=2,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=9.0,
        warmup=0,
        repeats=1,
    )
    captured = []

    def operator(query, key, value, **_kwargs):
        captured.append((query.clone(), key.clone(), value.clone()))
        return query

    torch.manual_seed(1)
    benchmark_script.benchmark(operator, seq_len=4, args=args)
    torch.manual_seed(999)
    torch.randn(32)
    benchmark_script.benchmark(operator, seq_len=4, args=args)

    for first, second in zip(captured[0], captured[1]):
        assert torch.equal(first, second)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or not benchmark_script.rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="two CUDA devices and RosaSoft CUDA are required",
)
def test_benchmark_records_events_on_noncurrent_cuda_device():
    current_device = torch.cuda.current_device()
    target_device = (current_device + 1) % torch.cuda.device_count()
    args = SimpleNamespace(
        device=f"cuda:{target_device}",
        dtype="float32",
        seed=19,
        batch=1,
        heads=1,
        bits=3,
        value_heads=1,
        value_dim=2,
        pattern="random",
        mode="forward",
        max_suffix_length=2,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=9.0,
        warmup=1,
        repeats=2,
    )

    result = benchmark_script.benchmark(
        benchmark_script.rosa_soft.rosa_soft,
        seq_len=4,
        args=args,
    )

    assert result["step_ms"] >= 0.0
    assert result["peak_operator_mib"] >= 0.0
    assert torch.cuda.current_device() == current_device


@pytest.mark.parametrize(
    ("cli_args", "expected_exit_code", "expected_success"),
    [
        ([], 0, None),
        (["--success-loss-threshold", "0.5"], 0, True),
        (["--success-loss-threshold", "0.49"], 1, False),
    ],
)
def test_fit_cli_success_threshold_is_optional_and_controls_exit(
    monkeypatch,
    cli_args,
    expected_exit_code,
    expected_success,
):
    observed = {}

    def fake_fit(args):
        success = fit_script.evaluate_success(
            0.5,
            args.success_loss_threshold,
        )
        observed["success"] = success
        return {
            "final_loss": 0.5,
            "success_loss_threshold": args.success_loss_threshold,
            "success": success,
        }

    monkeypatch.setattr(fit_script, "fit", fake_fit)

    assert fit_script.main(cli_args) == expected_exit_code
    assert observed["success"] is expected_success


def test_fit_success_threshold_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite and >= 0"):
        fit_script.evaluate_success(0.5, float("nan"))


def test_fit_mask_selects_only_targets_with_a_correct_historical_route():
    tokens = torch.tensor([[0, 1, 0, 1, 0]])

    mask = fit_script.historical_target_mask(tokens, max_suffix_length=4)

    assert torch.equal(
        mask,
        torch.tensor([[False, False, True, True]]),
    )


def test_fit_loss_excludes_cold_start_targets():
    tokens = torch.tensor([[0, 1, 0, 1, 0]])
    mask = fit_script.historical_target_mask(tokens, max_suffix_length=4)
    logits = torch.full((1, 5, 2), -8.0)
    logits[:, :-1].scatter_(
        -1,
        tokens[:, 1:].unsqueeze(-1),
        8.0,
    )
    logits[:, :2] *= -1

    fit_loss, fit_accuracy = fit_script.loss_and_accuracy(
        logits,
        tokens,
        mask,
    )
    _, full_accuracy = fit_script.loss_and_accuracy(logits, tokens)

    assert fit_loss < 1e-6
    assert fit_accuracy == 1.0
    assert full_accuracy == 0.5


def test_reference_fit_reaches_near_zero_on_historical_targets(capsys):
    args = fit_script.build_parser().parse_args(
        [
            "--operator",
            "reference",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--steps",
            "80",
            "--log-every",
            "80",
            "--success-loss-threshold",
            "0.01",
        ]
    )

    result = fit_script.fit(args)
    capsys.readouterr()

    assert result["success"] is True
    assert result["final_loss"] < 0.01
    assert result["final_accuracy"] == 1.0
    assert result["fit_target_count"] == 12
    assert result["excluded_cold_start_target_count"] == 4
    assert 0.0 < result["hard_nonnull_route_fraction"] <= 1.0


def test_recall_cli_runs_multiple_seeds_and_writes_json(tmp_path, capsys):
    output_path = tmp_path / "recall.json"

    exit_code = recall_script.main(
        [
            "--operator",
            "reference",
            "--device",
            "cpu",
            "--seeds",
            "3",
            "7",
            "--json-out",
            str(output_path),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["passed"] is True
    assert [run["seed"] for run in report["runs"]] == [3, 7]
    assert all(
        run["hard_recall_exact_accuracy"] == 1.0
        and run["current_value_exact_accuracy"] == 0.0
        and run["residual_only_exact_accuracy_ceiling"] == 0.5
        for run in report["runs"]
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("passed", "expected_exit_code"),
    [(True, 0), (False, 1)],
)
def test_recall_gate_status_controls_exit_code(passed, expected_exit_code):
    assert recall_script.gate_exit_code({"passed": passed}) == expected_exit_code


def test_recall_cli_returns_nonzero_for_failed_report(monkeypatch, capsys):
    monkeypatch.setattr(
        recall_script,
        "run_gate",
        lambda _args: {"passed": False},
    )

    assert recall_script.main([]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
