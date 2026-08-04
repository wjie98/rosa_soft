import torch

from benchmarks.global_bit_fit import (
    _loss_and_upstream,
    build_parser,
    make_task,
    run_fit,
)
from benchmarks.global_bit_oracle import (
    exact_bitflip_vjp,
    exact_margin_edit_oracle,
    exact_shared_bit_oracle,
)
from rosa_soft.soft_reference import _hard_route_forward


def test_joint_suffix_task_requires_a_coordinated_edit():
    task = make_task("joint_suffix", model_seed=3)
    output, _, _, _ = _hard_route_forward(
        task.query,
        task.initial_key,
        task.value,
        2,
    )
    loss, grad_output = _loss_and_upstream(output, task.target_output)
    masks = {
        "query_stochastic_mask": task.query_stochastic_mask,
        "key_stochastic_mask": task.key_stochastic_mask,
    }
    _, bitflip_key, bitflip_bits, _ = exact_bitflip_vjp(
        task.query,
        task.initial_key,
        task.value,
        grad_output,
        max_suffix_length=2,
        **masks,
    )
    stochastic = exact_shared_bit_oracle(
        task.query,
        task.initial_key,
        task.value,
        grad_output,
        max_suffix_length=2,
        bit_temperature=0.5,
        **masks,
    )
    margin = exact_margin_edit_oracle(
        task.query,
        task.initial_key,
        task.value,
        grad_output,
        max_suffix_length=2,
        eta=1.0,
        **masks,
    )

    assert loss == 4.0
    assert task.improving_single_bit_edits == 0
    assert torch.count_nonzero(bitflip_bits) == 0
    assert torch.count_nonzero(bitflip_key) == 0
    assert torch.count_nonzero(stochastic.bit_gradient) == 3
    assert stochastic.bit_gradient[0] > 0
    assert stochastic.bit_gradient[1] < 0
    assert margin.flipped_bits == 2
    assert margin.target_routes[0, 0, -1].item() == task.target_route


def test_margin_fit_crosses_the_joint_suffix_plateau():
    args = build_parser().parse_args(
        [
            "--estimators",
            "margin_edit",
            "--tasks",
            "joint_suffix",
            "--model-seeds",
            "0",
            "--steps",
            "20",
        ]
    )
    result = run_fit("margin_edit", "joint_suffix", 0, 0, args)

    assert result["initial_loss"] == 4.0
    assert result["final_loss"] == 0.0
    assert result["final_route"] == result["target_route"] == 2
