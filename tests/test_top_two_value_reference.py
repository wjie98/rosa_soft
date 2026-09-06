import pytest
import torch

from benchmarks.top_two_value_reference import (
    competition_state,
    hierarchical_route_carrier,
    hard_null_gate_probabilities,
    no_prior_route_probabilities,
    rosa_top_two_value_reference,
    top2_route_probabilities,
    unbounded_suffix_scores,
    winner_rest_route_carrier,
)
from rosa_soft.soft_reference import (
    _expand_value_heads,
    _gather_routed_values,
    _hard_route_forward,
    _hard_sign_with_softsign_vjp,
    _suffix_prefix_product_scores,
    rosa_soft_reference,
)


def _inputs(*, requires_grad=True, seed=17):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(
        2,
        7,
        2,
        4,
        generator=generator,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        query.shape,
        generator=generator,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    value = torch.randn(
        2,
        7,
        1,
        5,
        generator=generator,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    grad_output = torch.randn(
        2, 7, 2, 5, generator=generator, dtype=torch.float64
    )
    return (query, key, value), grad_output


def _run(inputs, grad_output, **kwargs):
    leaves = tuple(
        tensor.detach().clone().requires_grad_() for tensor in inputs
    )
    output = rosa_top_two_value_reference(*leaves, **kwargs)
    gradients = torch.autograd.grad(output, leaves, grad_output)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def test_unbounded_recurrence_matches_full_prefix_product_sum():
    generator = torch.Generator().manual_seed(3)
    local_match = torch.rand(
        2, 3, 8, 8, generator=generator, dtype=torch.float64
    )
    route_index = torch.arange(8).view(1, 8)
    row_index = torch.arange(8).view(8, 1)
    causal = (route_index > 0) & (route_index <= row_index)
    local_match = local_match * causal

    expected = _suffix_prefix_product_scores(local_match, 8)
    actual = unbounded_suffix_scores(local_match)

    torch.testing.assert_close(actual, expected, rtol=1e-15, atol=1e-15)


def test_dense_role_separation_matches_full_horizon_reference():
    inputs, grad_output = _inputs()
    actual_output, actual_gradients = _run(inputs, grad_output)

    reference_leaves = tuple(
        tensor.detach().clone().requires_grad_() for tensor in inputs
    )
    reference_output = rosa_soft_reference(
        *reference_leaves,
        max_suffix_length=reference_leaves[0].size(1),
    )
    reference_gradients = torch.autograd.grad(
        reference_output, reference_leaves, grad_output
    )

    assert torch.equal(actual_output, reference_output)
    for actual, expected in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


def test_hierarchical_carrier_is_exact_ordinary_attention():
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(
        2, 3, 5, 5, generator=generator, dtype=torch.float64
    )
    probabilities = torch.softmax(logits, dim=-1)
    route_values = torch.randn(
        2, 3, 5, 7, generator=generator, dtype=torch.float64
    )
    direct = torch.einsum(
        "bhta,bhad->bhtd", probabilities, route_values
    ).permute(0, 2, 1, 3)

    hierarchical = hierarchical_route_carrier(
        probabilities, route_values
    )

    torch.testing.assert_close(hierarchical, direct, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("use_explicit_winner", [False, True])
def test_winner_rest_carrier_preserves_value_but_changes_logit_vjp(
    use_explicit_winner,
):
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(
        2, 3, 5, 5, generator=generator, dtype=torch.float64,
        requires_grad=True,
    )
    route_values = torch.randn(
        2, 3, 5, 7, generator=generator, dtype=torch.float64
    )
    grad_output = torch.randn(
        2, 5, 3, 7, generator=generator, dtype=torch.float64
    )
    probabilities = torch.softmax(logits, dim=-1)
    direct = torch.einsum(
        "bhta,bhad->bhtd", probabilities, route_values
    ).permute(0, 2, 1, 3)
    winner = (
        torch.roll(probabilities.detach().argmax(dim=-1), 1, dims=-1)
        if use_explicit_winner
        else None
    )
    reduced = winner_rest_route_carrier(
        probabilities, route_values, winner=winner
    )

    torch.testing.assert_close(reduced, direct, rtol=2e-15, atol=2e-15)
    direct_gradient = torch.autograd.grad(
        direct, logits, grad_output, retain_graph=True
    )[0]
    reduced_gradient = torch.autograd.grad(reduced, logits, grad_output)[0]
    assert not torch.equal(reduced_gradient, direct_gradient)
    assert torch.isfinite(reduced_gradient).all()
    torch.testing.assert_close(
        reduced_gradient.sum(dim=-1),
        torch.zeros_like(reduced_gradient[..., 0]),
        rtol=0,
        atol=2e-15,
    )


def test_selected_value_keeps_qk_vjp_and_changes_only_value_vjp():
    inputs, grad_output = _inputs(seed=23)
    dense_output, dense_gradients = _run(
        inputs,
        grad_output,
        qk_gradient="dense",
        value_gradient="dense",
    )
    selected_output, selected_gradients = _run(
        inputs,
        grad_output,
        qk_gradient="dense",
        value_gradient="selected",
    )

    assert torch.equal(selected_output, dense_output)
    for selected, dense in zip(selected_gradients[:2], dense_gradients[:2]):
        torch.testing.assert_close(selected, dense, rtol=0, atol=0)
    assert not torch.equal(selected_gradients[2], dense_gradients[2])

    value_leaf = inputs[2].detach().clone().requires_grad_()
    _, _, selected_routes, _ = _hard_route_forward(
        inputs[0].detach(), inputs[1].detach(), value_leaf
    )
    route_values = _expand_value_heads(
        _hard_sign_with_softsign_vjp(value_leaf), inputs[0].size(2)
    )
    route_values[..., 0, :] = 0.0
    selected_carrier = _gather_routed_values(
        route_values, selected_routes
    ).permute(0, 2, 1, 3)
    expected_value_gradient = torch.autograd.grad(
        selected_carrier, value_leaf, grad_output
    )[0]
    torch.testing.assert_close(
        selected_gradients[2], expected_value_gradient, rtol=0, atol=0
    )


def test_top2_qk_changes_only_qk_vjp_and_preserves_dense_value_credit():
    inputs, grad_output = _inputs(seed=29)
    dense_output, dense_gradients = _run(inputs, grad_output)
    top2_output, top2_gradients = _run(
        inputs, grad_output, qk_gradient="top2"
    )

    assert torch.equal(top2_output, dense_output)
    assert any(
        not torch.equal(top2, dense)
        for top2, dense in zip(top2_gradients[:2], dense_gradients[:2])
    )
    torch.testing.assert_close(
        top2_gradients[2], dense_gradients[2], rtol=0, atol=0
    )
    assert all(torch.isfinite(gradient).all() for gradient in top2_gradients)


def test_top2_distribution_handles_a_single_finite_route():
    logits = torch.tensor(
        [[[[0.5, -torch.inf, -torch.inf]]]], dtype=torch.float64
    )
    probabilities = top2_route_probabilities(logits)

    torch.testing.assert_close(
        probabilities,
        torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(probabilities).all()


def test_no_prior_only_changes_null_vs_nonnull_mass():
    inputs, _ = _inputs(requires_grad=False, seed=41)
    from benchmarks.top_two_value_reference import _soft_route_state

    logits, dense, causal = _soft_route_state(
        inputs[0], inputs[1], scale=1.0, mismatch_scale=3.0
    )
    no_prior = no_prior_route_probabilities(logits, causal)
    dense_nonnull = dense[..., 1:] / dense[..., 1:].sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(dense.dtype).tiny)
    no_prior_nonnull = no_prior[..., 1:] / no_prior[..., 1:].sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(no_prior.dtype).tiny)

    torch.testing.assert_close(
        dense_nonnull, no_prior_nonnull, rtol=2e-14, atol=2e-14
    )
    assert bool((no_prior[..., 0] <= dense[..., 0]).all())


def test_hard_null_gate_masks_only_rows_with_a_nonnull_hard_winner():
    logits = torch.tensor(
        [[[[0.0, -1.0, -2.0], [0.0, 1.0, -1.0]]]],
        dtype=torch.float64,
    )
    winner = torch.tensor([[[0, 1]]])
    probabilities = hard_null_gate_probabilities(logits, winner)

    torch.testing.assert_close(
        probabilities[..., 0, :],
        torch.softmax(logits[..., 0, :], dim=-1),
        rtol=0,
        atol=0,
    )
    assert probabilities[..., 1, 0].eq(0).all()
    torch.testing.assert_close(
        probabilities[..., 1, 1:].sum(dim=-1),
        torch.ones_like(probabilities[..., 1, 0]),
        rtol=0,
        atol=2e-15,
    )


@pytest.mark.parametrize(
    "qk_gradient",
    [
        "dense",
        "top2",
        "winner_rest",
        "hard_winner_rest",
        "no_prior",
        "hard_null_gate",
    ],
)
@pytest.mark.parametrize("value_gradient", ["dense", "selected"])
def test_every_ablation_preserves_hard_forward_and_partial_gradients(
    qk_gradient,
    value_gradient,
):
    inputs, grad_output = _inputs(seed=31)
    expected = rosa_soft_reference(
        *(tensor.detach() for tensor in inputs), max_suffix_length=7
    )
    for gradient_mask in range(1, 8):
        leaves = tuple(
            tensor.detach().clone().requires_grad_(bool(gradient_mask & bit))
            for tensor, bit in zip(inputs, (1, 2, 4))
        )
        output = rosa_top_two_value_reference(
            *leaves,
            qk_gradient=qk_gradient,
            value_gradient=value_gradient,
        )
        requested = tuple(
            leaf for leaf in leaves if leaf.requires_grad
        )
        gradients = torch.autograd.grad(output, requested, grad_output)

        assert torch.equal(output, expected)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_competition_diagnostics_are_bounded_and_consistent():
    inputs, _ = _inputs(requires_grad=False, seed=37)
    state = competition_state(*inputs)

    assert state.probabilities.shape == (2, 2, 7, 7)
    torch.testing.assert_close(
        state.probabilities.sum(dim=-1),
        torch.ones_like(state.top_two_mass),
        rtol=2e-15,
        atol=2e-15,
    )
    for metric in (
        state.top_two_mass,
        state.nonnull_top_two_mass,
        state.runner_up_share_of_rest,
        state.tail_mass,
    ):
        assert bool(((metric >= 0.0) & (metric <= 1.0)).all())
    torch.testing.assert_close(
        state.top_two_mass + state.tail_mass,
        torch.ones_like(state.top_two_mass),
        rtol=0,
        atol=0,
    )
    assert state.has_two_nonnull_routes.shape == state.soft_winner.shape
    assert state.top_two_nonnull_value_collision.shape == state.soft_winner.shape


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("qk_gradient", "unknown"), ("value_gradient", "unknown")],
)
def test_invalid_ablation_mode_is_rejected(keyword, value):
    inputs, _ = _inputs(requires_grad=False)
    with pytest.raises(ValueError, match=keyword):
        rosa_top_two_value_reference(*inputs, **{keyword: value})
