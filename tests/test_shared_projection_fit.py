import torch

from benchmarks.global_bit_fit import _loss_and_upstream
from benchmarks.global_bit_oracle import (
    exact_bitflip_vjp,
    exact_shared_bit_oracle,
)
from benchmarks.shared_projection_fit import (
    _hard_loss,
    build_parser,
    make_shared_projection_task,
    run_fit,
)
from rosa_soft.soft_reference import _hard_route_forward


def test_shared_projection_couples_three_key_logits_through_two_parameters():
    task = make_shared_projection_task(model_seed=0)
    initial_key = task.project_key(task.initial_parameters)
    target_key = task.project_key(task.target_parameters)
    _, _, initial_routes, _ = _hard_route_forward(
        task.route_task.query,
        initial_key,
        task.route_task.value,
        2,
    )
    _, _, target_routes, _ = _hard_route_forward(
        task.route_task.query,
        target_key,
        task.route_task.value,
        2,
    )

    assert torch.linalg.matrix_rank(task.features) == 2
    assert torch.equal(
        torch.sign(initial_key[0, :3, 0, 0]),
        torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64),
    )
    assert torch.equal(
        torch.sign(target_key[0, :3, 0, 0]),
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64),
    )
    assert initial_routes[0, 0, -1].item() == 3
    assert target_routes[0, 0, -1].item() == 2


def test_shared_projection_preserves_joint_signal_after_aggregation():
    task = make_shared_projection_task(model_seed=2)
    key = task.project_key(task.initial_parameters)
    output, _, _, _ = _hard_route_forward(
        task.route_task.query,
        key,
        task.route_task.value,
        2,
    )
    _, grad_output = _loss_and_upstream(
        output,
        task.route_task.target_output,
    )
    masks = {
        "query_stochastic_mask": task.route_task.query_stochastic_mask,
        "key_stochastic_mask": task.route_task.key_stochastic_mask,
    }
    _, bitflip_key, _, _ = exact_bitflip_vjp(
        task.route_task.query,
        key,
        task.route_task.value,
        grad_output,
        max_suffix_length=2,
        **masks,
    )
    stochastic = exact_shared_bit_oracle(
        task.route_task.query,
        key,
        task.route_task.value,
        grad_output,
        max_suffix_length=2,
        bit_temperature=0.5,
        **masks,
    )
    bitflip_parameter = task.features.T @ bitflip_key[0, :3, 0, 0]
    stochastic_parameter = (
        task.features.T @ stochastic.key_gradient[0, :3, 0, 0]
    )

    assert torch.count_nonzero(bitflip_parameter) == 0
    assert torch.count_nonzero(stochastic_parameter) == 2


def test_margin_edit_fits_through_shared_projection():
    args = build_parser().parse_args(
        [
            "--estimators",
            "margin_edit",
            "--model-seeds",
            "0",
            "--steps",
            "20",
        ]
    )
    result = run_fit("margin_edit", 0, 0, args)
    task = make_shared_projection_task(model_seed=0)
    final_loss, final_route, _ = _hard_loss(
        task,
        torch.tensor(result["final_parameters"], dtype=torch.float64),
        2,
    )

    assert result["initial_loss"] == 4.0
    assert final_loss == 0.0
    assert final_route == 2
