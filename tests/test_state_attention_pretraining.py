import json

import pytest
import torch

from benchmarks.state_attention_pretraining import (
    NextTokenEstimatorLM,
    _theoretical_limits,
    build_parser,
    make_next_token_recall_batch,
    run_benchmark,
)


def test_next_token_batch_places_recalled_payload_after_each_query():
    batch = make_next_token_recall_batch(
        seed=9,
        pairs=2,
        associations=3,
        payload_bits=3,
    )
    storage_payloads = batch.tokens[:, batch.storage_payload_positions]
    recalled_payloads = batch.tokens[:, batch.query_positions + 1]
    complement_mask = (1 << batch.payload_bits) - 1
    codes = recalled_payloads - batch.associations

    assert torch.equal(storage_payloads, recalled_payloads)
    assert torch.equal(codes[0::2] ^ complement_mask, codes[1::2])
    assert all(row.unique().numel() == 3 for row in codes)


def test_next_token_batch_seed_is_deterministic_and_changes_the_mapping():
    kwargs = {"pairs": 4, "associations": 4, "payload_bits": 4}
    first = make_next_token_recall_batch(seed=23, **kwargs)
    repeated = make_next_token_recall_batch(seed=23, **kwargs)
    different = make_next_token_recall_batch(seed=24, **kwargs)

    assert torch.equal(first.tokens, repeated.tokens)
    assert not torch.equal(first.tokens, different.tokens)


def test_next_token_theoretical_limits_include_random_storage_targets():
    limits = _theoretical_limits(associations=4, payload_bits=4)

    expected_floor = sum(torch.log(torch.tensor([16.0, 15.0, 14.0, 13.0]))) / 19
    assert limits["loss_floor"] == pytest.approx(float(expected_floor))
    assert limits["top1_accuracy_ceiling"] < 1.0


def test_query_residual_is_identical_across_complementary_assignments():
    batch = make_next_token_recall_batch(
        seed=11,
        pairs=2,
        associations=2,
        payload_bits=2,
    )
    model = NextTokenEstimatorLM(
        associations=2,
        hidden_size=8,
        payload_bits=2,
        num_heads=1,
        qk_bits=2,
        value_heads=1,
        value_bits=2,
        context_scale=0.25,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
        estimator="state_quadratic_attention",
        bit_temperature=0.5,
        antithetic_pairs=1,
        context_depth=2,
    )

    residual = model.encode_residual(batch.tokens)
    query = residual[:, batch.query_positions]

    torch.testing.assert_close(query[0::2], query[1::2], rtol=0, atol=0)


def test_next_token_benchmark_smoke_schema():
    args = build_parser().parse_args(
        [
            "--estimators",
            "production",
            "bitflip",
            "state_quadratic_attention",
            "--seeds",
            "0",
            "--train-pairs",
            "1",
            "--validation-pairs",
            "1",
            "--associations",
            "2",
            "--hidden-size",
            "8",
            "--context-depth",
            "2",
            "--heads",
            "1",
            "--qk-bits",
            "2",
            "--value-heads",
            "1",
            "--payload-bits",
            "2",
            "--value-bits",
            "2",
            "--steps",
            "1",
        ]
    )

    report = run_benchmark(args)

    assert report["schema_version"] == 2
    assert report["objective"] == "uniform full-sequence next-token cross entropy"
    assert report["payload_bits"] == 2
    assert report["value_dim"] == 2
    assert report["bitflip_gradient_scale"] == 1.0
    assert report["initialization"]["controlled_module_reset"] is False
    assert report["training_data"] == (
        "fresh complementary mappings each optimizer step"
    )
    assert report["shortcut_checks_passed"] is True
    assert set(report["summary"]) == set(args.estimators)
    json.dumps(report, allow_nan=False)
