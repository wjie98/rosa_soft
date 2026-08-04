"""Paired fitting comparison for hard-forward RosaSoft VJPs.

This is a research benchmark, not a production API. Every estimator returns
the same exact hard ROSA value. They differ only in the custom backward:

1. deterministic raw Hamming surrogate;
2. independently perturbed mismatch scores;
3. exact single-bit hard counterfactuals;
4. production deterministic surrogate with post-softmax attention dropout;
5. research-only long-suffix residual dropout.

Model/data seeds and stochastic-estimator seeds can be crossed independently.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.fit_soft_reference import (
    TinyRosaFitLM,
    historical_target_mask,
    loss_and_accuracy,
    make_copy_tokens,
)
from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.soft_reference import (
    _build_vjp_carrier,
    _causal_route_mask,
    _hard_route_forward,
    _hard_sign,
    _hard_sign_with_softsign_vjp,
    _masked_route_scores,
    _pairwise_soft_match_gates,
    _reference_compute_dtype,
    _route_probabilities,
    _suffix_score_utility,
    _suffix_prefix_product_scores,
    _validate_reference_call,
    rosa_soft_reference,
)


Estimator = Callable[..., Tensor]
ESTIMATOR_NAMES = (
    "deterministic",
    "mismatch_random",
    "bitflip",
    "attention_dropout",
    "suffix_dropout",
)
_MISMATCH_RANDOM_MODE = 1


def _needs_backward(*tensors: Tensor) -> bool:
    return torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in tensors
    )


def _pairwise_random_match_gates(
    query: Tensor,
    key: Tensor,
    causal_route_mask: Tensor,
    mismatch_scale: float,
    uniforms: Tensor,
) -> Tensor:
    """Use random numerical mismatch weights and a hard-Hamming local VJP."""

    query_h = query.permute(0, 2, 1, 3)
    key_h = key.permute(0, 2, 1, 3)[..., :-1, :]
    query_hard = _hard_sign(query_h)
    key_hard = _hard_sign(key_h)
    query_surrogate = _hard_sign_with_softsign_vjp(query_h)
    key_surrogate = _hard_sign_with_softsign_vjp(key_h)

    hard_mismatch = 0.5 * (
        1.0
        - query_hard.unsqueeze(-2) * key_hard.unsqueeze(-3)
    )
    mismatch_weights = 1.0 - 0.5 * uniforms.pow(3)
    hard_mismatch_rate = hard_mismatch.mean(dim=-1)
    random_mismatch_rate = (
        hard_mismatch * mismatch_weights
    ).mean(dim=-1)
    surrogate_mismatch_rate = 0.5 * (
        1.0
        - query_surrogate.unsqueeze(-2)
        * key_surrogate.unsqueeze(-3)
    ).mean(dim=-1)

    mismatch_scale = float(mismatch_scale)
    hard_vjp_scale = torch.exp(
        -mismatch_scale
        * (hard_mismatch_rate - random_mismatch_rate)
    ).detach()
    proxy = surrogate_mismatch_rate * hard_vjp_scale
    energy = (
        random_mismatch_rate.detach() + proxy - proxy.detach()
    )
    local_gates = torch.exp(-mismatch_scale * energy)
    seq_len = query.size(1)
    return torch.nn.functional.pad(
        local_gates,
        (1, 0),
        value=0.0,
    ) * causal_route_mask.view(1, 1, seq_len, seq_len)


class _HardForwardResearchSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        stochastic_state: Tensor,
        max_suffix_length: int,
        scale: float,
        mismatch_scale: float,
        mode: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
            int(max_suffix_length),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.mode = int(mode)
        ctx.save_for_backward(query, key, value, stochastic_state)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, stochastic_state = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(need)
                for tensor, need in zip(
                    (query, key, value),
                    needs,
                )
            )
            query_leaf, key_leaf, value_leaf = leaves
            compute_dtype = _reference_compute_dtype(query.dtype)
            query_work = query_leaf.to(compute_dtype)
            key_work = key_leaf.to(compute_dtype)
            value_work = value_leaf.to(compute_dtype)
            causal_route_mask = _causal_route_mask(
                query.size(1),
                query.device,
            )
            if ctx.mode != _MISMATCH_RANDOM_MODE:
                raise RuntimeError(
                    f"unknown research surrogate mode: {ctx.mode}"
                )
            local_match_gates = _pairwise_random_match_gates(
                query_work,
                key_work,
                causal_route_mask,
                ctx.mismatch_scale,
                stochastic_state,
            )
            soft_suffix_scores = _suffix_prefix_product_scores(
                local_match_gates,
                ctx.max_suffix_length,
            )
            probabilities = _route_probabilities(
                _masked_route_scores(
                    _suffix_score_utility(soft_suffix_scores),
                    causal_route_mask,
                ),
                causal_route_mask,
                ctx.scale,
            )
            carrier = _build_vjp_carrier(
                value_work,
                probabilities,
                query.size(2),
            ).to(query.dtype)
            required_indices = [
                index for index, need in enumerate(needs) if need
            ]
            required_grads = torch.autograd.grad(
                carrier,
                tuple(leaves[index] for index in required_indices),
                grad_output,
                create_graph=False,
            )
            gradients = [None, None, None]
            for index, gradient in zip(
                required_indices,
                required_grads,
            ):
                gradients[index] = gradient
        return (
            gradients[0],
            gradients[1],
            gradients[2],
            None,
            None,
            None,
            None,
            None,
        )


def rosa_soft_mismatch_random(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        0.0,
        mismatch_scale,
    )
    if not _needs_backward(query, key, value):
        return rosa_soft_reference(
            query,
            key,
            value,
            max_suffix_length=max_suffix_length,
            scale=scale,
            mismatch_scale=mismatch_scale,
        )
    batch, seq_len, heads, bits = query.shape
    uniforms = torch.rand(
        batch,
        heads,
        seq_len,
        max(seq_len - 1, 0),
        bits,
        device=query.device,
        dtype=_reference_compute_dtype(query.dtype),
    )
    return _HardForwardResearchSurrogate.apply(
        query,
        key,
        value,
        uniforms,
        max_suffix_length,
        float(scale),
        float(mismatch_scale),
        _MISMATCH_RANDOM_MODE,
    )


def make_attention_dropout_estimator(
    dropout_p: float,
) -> Estimator:
    if (
        not math.isfinite(dropout_p)
        or not 0.0 <= dropout_p <= 1.0 - 2.0**-24
    ):
        raise ValueError(
            "attention dropout must be finite and in [0, 1 - 2^-24]"
        )

    def estimator(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        max_suffix_length: int = 32,
        scale: float = ROSA_SOFT_DEFAULT_SCALE,
        mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ) -> Tensor:
        return rosa_soft_reference(
            query,
            key,
            value,
            max_suffix_length=max_suffix_length,
            scale=scale,
            dropout_p=dropout_p,
            mismatch_scale=mismatch_scale,
        )

    return estimator


def _stratified_row_weights(
    batch: int,
    heads: int,
    seq_len: int,
    dropout_p: float,
    row_uniforms: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Select one query row per stratum and return unbiased row weights."""

    if (
        not math.isfinite(dropout_p)
        or not 0.0 <= dropout_p <= 1.0 - 2.0**-24
    ):
        raise ValueError(
            "suffix dropout must be finite and in [0, 1 - 2^-24]"
        )
    if float(dropout_p) == 0.0:
        return torch.ones(
            batch,
            heads,
            seq_len,
            device=row_uniforms.device,
            dtype=dtype,
        )

    keep_probability = 1.0 - float(dropout_p)
    sample_count = max(1, min(seq_len, round(keep_probability * seq_len)))
    stratum_index = torch.arange(
        sample_count,
        device=row_uniforms.device,
        dtype=torch.int64,
    )
    starts = torch.div(
        stratum_index * seq_len,
        sample_count,
        rounding_mode="floor",
    )
    ends = torch.div(
        (stratum_index + 1) * seq_len,
        sample_count,
        rounding_mode="floor",
    )
    widths = ends - starts
    offsets = torch.floor(
        row_uniforms[..., :sample_count]
        * widths.to(row_uniforms.dtype)
    ).to(torch.int64)
    selected_rows = starts + offsets
    weights = torch.zeros(
        batch,
        heads,
        seq_len,
        device=row_uniforms.device,
        dtype=dtype,
    )
    return weights.scatter(
        dim=-1,
        index=selected_rows,
        src=widths.to(dtype).view(1, 1, -1).expand(
            batch,
            heads,
            -1,
        ),
    )


class _HardForwardSuffixDropout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        row_uniforms: Tensor,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
            int(max_suffix_length),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.save_for_backward(query, key, value, row_uniforms)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, row_uniforms = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(need)
                for tensor, need in zip((query, key, value), needs)
            )
            query_leaf, key_leaf, value_leaf = leaves
            compute_dtype = _reference_compute_dtype(query.dtype)
            query_work = query_leaf.to(compute_dtype)
            key_work = key_leaf.to(compute_dtype)
            value_work = value_leaf.to(compute_dtype)
            causal_route_mask = _causal_route_mask(
                query.size(1),
                query.device,
            )
            local_match_gates = _pairwise_soft_match_gates(
                query_work,
                key_work,
                causal_route_mask,
                ctx.mismatch_scale,
            )
            full_suffix_scores = _suffix_prefix_product_scores(
                local_match_gates,
                ctx.max_suffix_length,
            )
            full_probabilities = _route_probabilities(
                _masked_route_scores(
                    _suffix_score_utility(full_suffix_scores),
                    causal_route_mask,
                ),
                causal_route_mask,
                ctx.scale,
            )
            full_carrier = _build_vjp_carrier(
                value_work,
                full_probabilities,
                query.size(2),
            )
            row_weights = _stratified_row_weights(
                query.size(0),
                query.size(2),
                query.size(1),
                ctx.dropout_p,
                row_uniforms,
                compute_dtype,
            ).permute(0, 2, 1).unsqueeze(-1)
            local_probabilities = _route_probabilities(
                _masked_route_scores(
                    _suffix_score_utility(local_match_gates),
                    causal_route_mask,
                ),
                causal_route_mask,
                ctx.scale,
            )
            local_carrier = _build_vjp_carrier(
                value_work,
                local_probabilities,
                query.size(2),
            )
            carrier = (
                local_carrier
                + row_weights * (full_carrier - local_carrier)
            )

            carrier = carrier.to(query.dtype)
            required_indices = [
                index for index, need in enumerate(needs) if need
            ]
            required_grads = torch.autograd.grad(
                carrier,
                tuple(leaves[index] for index in required_indices),
                grad_output,
                create_graph=False,
            )
            gradients = [None, None, None]
            for index, gradient in zip(
                required_indices,
                required_grads,
            ):
                gradients[index] = gradient

        return (
            gradients[0],
            gradients[1],
            gradients[2],
            None,
            None,
            None,
            None,
            None,
        )


def make_suffix_dropout_estimator(dropout_p: float) -> Estimator:
    if (
        not math.isfinite(dropout_p)
        or not 0.0 <= dropout_p <= 1.0 - 2.0**-24
    ):
        raise ValueError(
            "suffix dropout must be finite and in [0, 1 - 2^-24]"
        )

    def estimator(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        max_suffix_length: int = 32,
        scale: float = ROSA_SOFT_DEFAULT_SCALE,
        mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ) -> Tensor:
        max_suffix_length = _validate_reference_call(
            query,
            key,
            value,
            max_suffix_length,
            scale,
            0.0,
            mismatch_scale,
        )
        if float(dropout_p) == 0.0 or not _needs_backward(
            query,
            key,
            value,
        ):
            return rosa_soft_reference(
                query,
                key,
                value,
                max_suffix_length=max_suffix_length,
                scale=scale,
                mismatch_scale=mismatch_scale,
            )
        batch, seq_len, heads, _ = query.shape
        row_uniforms = torch.rand(
            batch,
            heads,
            seq_len,
            device=query.device,
            dtype=_reference_compute_dtype(query.dtype),
        )
        return _HardForwardSuffixDropout.apply(
            query,
            key,
            value,
            row_uniforms,
            max_suffix_length,
            float(scale),
            float(dropout_p),
            float(mismatch_scale),
        )

    return estimator


def _bitflip_vjp_for_input(
    role: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    base_output: Tensor,
    grad_output: Tensor,
    max_suffix_length: int,
) -> Tensor:
    inputs = (query, key, value)
    selected = inputs[role]
    if selected.size(0) != 1:
        raise ValueError("exact bitflip research VJP requires batch size 1")
    bit_count = selected.numel()
    variants = selected.expand(
        bit_count,
        *selected.shape[1:],
    ).clone()
    flat_variants = variants.reshape(bit_count, bit_count)
    flat_signs = _hard_sign(selected).reshape(-1)
    bit_indices = torch.arange(bit_count, device=selected.device)
    flat_variants[bit_indices, bit_indices] = -flat_signs

    expanded_inputs = [
        tensor.expand(bit_count, *tensor.shape[1:])
        for tensor in inputs
    ]
    expanded_inputs[role] = variants
    flipped_output, _, _, _ = _hard_route_forward(
        expanded_inputs[0],
        expanded_inputs[1],
        expanded_inputs[2],
        max_suffix_length,
    )
    loss_delta = (
        (flipped_output - base_output)
        * grad_output
    ).flatten(1).sum(dim=1)
    gradient = -flat_signs * loss_delta
    return gradient.reshape_as(selected)


class _HardForwardExactBitflip(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
            int(max_suffix_length),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.save_for_backward(query, key, value)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value = ctx.saved_tensors
        compute_dtype = _reference_compute_dtype(query.dtype)
        work_inputs = tuple(
            tensor.to(compute_dtype)
            for tensor in (query, key, value)
        )
        base_output, _, _, _ = _hard_route_forward(
            *work_inputs,
            ctx.max_suffix_length,
        )
        grad_output_work = grad_output.to(compute_dtype)
        gradients = []
        for role, need in enumerate(ctx.needs_input_grad[:3]):
            if not need:
                gradients.append(None)
                continue
            gradients.append(
                _bitflip_vjp_for_input(
                    role,
                    *work_inputs,
                    base_output,
                    grad_output_work,
                    ctx.max_suffix_length,
                ).to(query.dtype)
            )
        return gradients[0], gradients[1], gradients[2], None


def rosa_soft_exact_bitflip(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        0.0,
        mismatch_scale,
    )
    if query.size(0) != 1:
        raise ValueError("exact bitflip research VJP requires batch size 1")
    if not _needs_backward(query, key, value):
        return rosa_soft_reference(
            query,
            key,
            value,
            max_suffix_length=max_suffix_length,
        )
    return _HardForwardExactBitflip.apply(
        query,
        key,
        value,
        max_suffix_length,
    )


class _ResearchFitLM(TinyRosaFitLM):
    def __init__(self, *args, training_operator: Estimator, **kwargs):
        super().__init__(*args, operator="reference", **kwargs)
        self.training_operator = training_operator

    def forward(self, tokens: Tensor) -> Tensor:
        residual, query, key, value = self.project_symbols(tokens)
        routed = self.training_operator(
            query,
            key,
            value,
            max_suffix_length=self.max_suffix_length,
            scale=self.scale,
            mismatch_scale=self.mismatch_scale,
        )
        hidden = residual + self.output(routed.flatten(2))
        return self.head(self.output_norm(hidden))


@torch.no_grad()
def _evaluate(
    model: _ResearchFitLM,
    tokens: Tensor,
    target_mask: Tensor,
) -> tuple[float, float]:
    loss, accuracy = loss_and_accuracy(
        model(tokens),
        tokens,
        target_mask,
    )
    return float(loss.item()), accuracy


def _make_estimator(
    name: str,
    dropout_p: float,
) -> Estimator:
    if name == "deterministic":
        return rosa_soft_reference
    if name == "mismatch_random":
        return rosa_soft_mismatch_random
    if name == "bitflip":
        return rosa_soft_exact_bitflip
    if name == "attention_dropout":
        return make_attention_dropout_estimator(dropout_p)
    if name == "suffix_dropout":
        return make_suffix_dropout_estimator(dropout_p)
    raise ValueError(f"unknown estimator: {name}")


def run_fit(
    estimator_name: str,
    model_seed: int,
    noise_seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(300_000 + model_seed)
    tokens = make_copy_tokens(
        seq_len=args.sequence_length + 1,
        vocab_size=args.vocab_size,
        motif_min=args.motif_min,
        motif_max=args.motif_max,
        seed=100_000 + model_seed,
    )
    target_mask = historical_target_mask(
        tokens,
        args.max_suffix_length,
    )
    tokens = tokens.to(device)
    target_mask = target_mask.to(device)
    model = _ResearchFitLM(
        vocab_size=args.vocab_size,
        num_heads=args.heads,
        qk_bits=args.qk_bits,
        value_heads=args.value_heads,
        value_bits=args.value_bits,
        max_suffix_length=args.max_suffix_length,
        scale=args.scale,
        dropout_p=0.0,
        mismatch_scale=args.mismatch_scale,
        training_operator=_make_estimator(
            estimator_name,
            args.dropout_p,
        ),
    ).to(device)
    torch.manual_seed(400_000 + noise_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(400_000 + noise_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    initial_loss, initial_accuracy = _evaluate(
        model,
        tokens,
        target_mask,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    best_loss = initial_loss
    best_step = 0
    first_below = -1
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_and_accuracy(
            model(tokens),
            tokens,
            target_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
        if (
            first_below < 0
            and loss_value < args.success_threshold
        ):
            first_below = step
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = _evaluate(
        model,
        tokens,
        target_mask,
    )
    if final_loss < best_loss:
        best_loss = final_loss
        best_step = args.steps
    return {
        "estimator": estimator_name,
        "model_seed": model_seed,
        "noise_seed": noise_seed,
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "best_loss": best_loss,
        "best_step": best_step,
        "first_below_threshold": first_below,
        "ever_success": first_below >= 0 or best_loss < args.success_threshold,
        "final_success": (
            math.isfinite(final_loss)
            and final_loss < args.success_threshold
        ),
        "step_ms": elapsed * 1000.0 / args.steps,
        "fit_target_count": int(target_mask.sum().item()),
    }


def run_matrix(args: argparse.Namespace) -> Dict[str, object]:
    if args.noise_seeds is None:
        seed_pairs = [
            (model_seed, model_seed)
            for model_seed in args.model_seeds
        ]
        seed_pairing = "matched"
    else:
        seed_pairs = [
            (model_seed, noise_seed)
            for model_seed in args.model_seeds
            for noise_seed in args.noise_seeds
        ]
        seed_pairing = "cartesian"
    runs = [
        run_fit(estimator, model_seed, noise_seed, args)
        for estimator in args.estimators
        for model_seed, noise_seed in seed_pairs
    ]
    summaries = {}
    for estimator in args.estimators:
        selected = [
            run for run in runs if run["estimator"] == estimator
        ]
        ever_successes = [
            run for run in selected if run["ever_success"]
        ]
        final_successes = [
            run for run in selected if run["final_success"]
        ]
        success_steps = [
            int(run["first_below_threshold"])
            for run in ever_successes
            if int(run["first_below_threshold"]) >= 0
        ]
        summaries[estimator] = {
            "ever_successes": len(ever_successes),
            "final_successes": len(final_successes),
            "runs": len(selected),
            "ever_success_rate": len(ever_successes) / len(selected),
            "final_success_rate": len(final_successes) / len(selected),
            "median_final_loss": statistics.median(
                float(run["final_loss"]) for run in selected
            ),
            "median_best_loss": statistics.median(
                float(run["best_loss"]) for run in selected
            ),
            "median_success_step": (
                statistics.median(success_steps)
                if success_steps
                else None
            ),
            "mean_step_ms": statistics.mean(
                float(run["step_ms"]) for run in selected
            ),
        }
    return {
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "model_seeds": args.model_seeds,
        "noise_seeds": (
            args.model_seeds
            if args.noise_seeds is None
            else args.noise_seeds
        ),
        "seed_pairing": seed_pairing,
        "estimators": args.estimators,
        "steps": args.steps,
        "success_threshold": args.success_threshold,
        "dropout_p": args.dropout_p,
        "attention_dropout_location": "post_softmax_attention_weights",
        "suffix_dropout_sample_unit": "query_row",
        "suffix_dropout_base_length": 1,
        "suffix_dropout_sampling": "fixed_count_stratified",
        "sequence_length": args.sequence_length,
        "vocab_size": args.vocab_size,
        "motif_length_range": [args.motif_min, args.motif_max],
        "heads": args.heads,
        "qk_bits": args.qk_bits,
        "value_heads": args.value_heads,
        "value_bits": args.value_bits,
        "max_suffix_length": args.max_suffix_length,
        "scale": args.scale,
        "mismatch_scale": args.mismatch_scale,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "runs": runs,
        "summaries": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATOR_NAMES,
        default=list(ESTIMATOR_NAMES),
    )
    parser.add_argument(
        "--model-seeds",
        "--seeds",
        dest="model_seeds",
        type=int,
        nargs="+",
        default=list(range(8)),
    )
    parser.add_argument(
        "--noise-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "cross these stochastic-estimator seeds with every model seed; "
            "by default each model seed uses the matching noise seed"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--success-threshold", type=float, default=1e-3)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--motif-min", type=int, default=4)
    parser.add_argument("--motif-max", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--max-suffix-length", type=int, default=8)
    parser.add_argument(
        "--scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_SCALE,
    )
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_matrix(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
