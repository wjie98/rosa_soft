"""Matched ablation of RosaSoft mismatch and suffix-score geometry.

This module is a research harness, not a public operator API. Every proxy
executes the same exact hard ROSA forward. The production-shaped variants
differ only in two orthogonal backward choices:

* mismatch count normalized by ``D`` or ``sqrt(D)``;
* raw expected suffix length or a normalized concave power utility, including
  the logarithmic limit.

The additional ``collision_lr`` proxy replaces only the backward route
distribution with a random-collision likelihood ratio. It remains a research
control and is not a public RosaSoft estimator mode.

The fitting model keeps its residual width fixed while Q/K symbol width varies.
This avoids changing the whole model capacity when measuring the Q/K bit axis.
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
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import once_differentiable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.fit_soft_reference import (
    build_target_mask,
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
    _hard_sign_with_softsign_vjp,
    _masked_route_scores,
    _reference_compute_dtype,
    _route_probabilities,
    _suffix_score_utility as _production_suffix_score_utility,
    _suffix_prefix_product_scores,
    _validate_reference_call,
)


ProxyOperator = Callable[..., Tensor]
PROXY_CONFIGS = {
    "baseline": ("mean", "linear"),
    "power075_suffix": ("mean", "power075"),
    "sqrt_suffix": ("mean", "sqrt"),
    "power025_suffix": ("mean", "power025"),
    "log_suffix": ("mean", "log"),
    "collision_lr": ("mean", "collision_lr"),
    "sqrt_dim": ("sqrt", "linear"),
    "sqrt_dim_suffix": ("sqrt", "sqrt"),
}
_DIMENSION_MODES = {"mean": 0, "sqrt": 1}
_DIMENSION_MODES_BY_ID = {value: key for key, value in _DIMENSION_MODES.items()}
_SUFFIX_MODES = {
    "linear": 0,
    "power075": 1,
    "sqrt": 2,
    "power025": 3,
    "log": 4,
    "collision_lr": 5,
}
_SUFFIX_MODES_BY_ID = {value: key for key, value in _SUFFIX_MODES.items()}
_SUFFIX_POWER_ALPHA = {
    "power075": 0.75,
    "sqrt": 0.5,
    "power025": 0.25,
}
_SQRT_SUFFIX_SCALE = math.sqrt(2.0) + 1.0


def _mismatch_denominator(symbol_dim: int, mode: str) -> float:
    if mode == "mean":
        return float(symbol_dim)
    if mode == "sqrt":
        return math.sqrt(float(symbol_dim))
    raise ValueError(f"unknown mismatch normalization: {mode}")


def _suffix_score_utility(raw_score: Tensor, mode: str) -> Tensor:
    if mode == "linear":
        return raw_score
    if mode == "sqrt":
        return _production_suffix_score_utility(raw_score)
    if mode == "log":
        return torch.log1p(raw_score) / math.log(2.0)
    if mode in _SUFFIX_POWER_ALPHA:
        alpha = _SUFFIX_POWER_ALPHA[mode]
        numerator = torch.expm1(alpha * torch.log1p(raw_score))
        denominator = math.expm1(alpha * math.log(2.0))
        return numerator / denominator
    raise ValueError(f"unknown suffix utility: {mode}")


def _collision_log_likelihood_ratio_scores(
    local_gates: Tensor,
    *,
    symbol_dim: int,
    max_suffix_length: int,
    mismatch_scale: float,
) -> Tensor:
    """Return a uniform-length mixture likelihood ratio in log space.

    For uniform independent hard bits, each normalized local gate has mean
    one. Consequently every prefix product and their uniform mixture have
    mean one under the random-collision null.
    """

    null_gate_mean = (
        0.5
        * (
            1.0
            + math.exp(-float(mismatch_scale) / float(symbol_dim))
        )
    ) ** symbol_dim
    valid = local_gates > 0
    local_log_bf = torch.where(
        valid,
        local_gates.clamp_min(torch.finfo(local_gates.dtype).tiny).log()
        - math.log(null_gate_mean),
        torch.zeros_like(local_gates),
    )
    prefix_log_bf = local_log_bf
    route_log_bf = prefix_log_bf
    prefix_valid = valid
    prefix_count = valid.to(torch.int64)
    max_offsets = min(
        int(max_suffix_length),
        local_gates.size(-2),
        local_gates.size(-1),
    )
    for _ in range(1, max_offsets):
        previous_log_bf = torch.nn.functional.pad(
            prefix_log_bf[..., :-1, :-1],
            (1, 0, 1, 0),
            value=0.0,
        )
        previous_valid = torch.nn.functional.pad(
            prefix_valid[..., :-1, :-1],
            (1, 0, 1, 0),
            value=False,
        )
        prefix_valid = valid & previous_valid
        prefix_log_bf = local_log_bf + previous_log_bf
        route_log_bf = torch.where(
            prefix_valid,
            torch.logaddexp(route_log_bf, prefix_log_bf),
            route_log_bf,
        )
        prefix_count = prefix_count + prefix_valid
    return route_log_bf - prefix_count.clamp_min(1).to(
        route_log_bf.dtype
    ).log()


def _proxy_route_probabilities(
    route_scores: Tensor,
    causal_route_mask: Tensor,
    *,
    scale: float,
    null_score: float,
) -> Tensor:
    seq_len = route_scores.size(-1)
    route_index = torch.arange(
        seq_len,
        device=route_scores.device,
    ).view(1, 1, 1, seq_len)
    causal = causal_route_mask.view(1, 1, seq_len, seq_len)
    nonnull = causal & (route_index > 0)
    candidate_count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)
    scores = route_scores.clone()
    scores[..., 0] = float(null_score)
    logits = scores * float(scale)
    logits = logits - torch.where(
        nonnull,
        candidate_count.to(route_scores.dtype).log(),
        torch.zeros((), dtype=route_scores.dtype, device=route_scores.device),
    )
    logits = logits.masked_fill(~causal, -torch.inf)
    return torch.softmax(logits - logits.amax(dim=-1, keepdim=True), dim=-1)


def _pairwise_proxy_match_gates(
    query: Tensor,
    key: Tensor,
    causal_route_mask: Tensor,
    mismatch_scale: float,
    dimension_mode: str,
) -> Tensor:
    query_symbols = _hard_sign_with_softsign_vjp(
        query.permute(0, 2, 1, 3)
    )
    key_symbols = _hard_sign_with_softsign_vjp(
        key.permute(0, 2, 1, 3)[..., :-1, :]
    )
    mismatch_count = 0.5 * (
        1.0
        - query_symbols.unsqueeze(-2) * key_symbols.unsqueeze(-3)
    ).sum(dim=-1)
    distance = mismatch_count / _mismatch_denominator(
        query.size(-1),
        dimension_mode,
    )
    local_gates = torch.exp(-float(mismatch_scale) * distance)
    seq_len = query.size(1)
    return torch.nn.functional.pad(
        local_gates,
        (1, 0),
        value=0.0,
    ) * causal_route_mask.view(1, 1, seq_len, seq_len)


def _proxy_state(
    query: Tensor,
    key: Tensor,
    *,
    max_suffix_length: int,
    scale: float,
    mismatch_scale: float,
    dimension_mode: str,
    suffix_mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    causal_route_mask = _causal_route_mask(query.size(1), query.device)
    local_gates = _pairwise_proxy_match_gates(
        query,
        key,
        causal_route_mask,
        mismatch_scale,
        dimension_mode,
    )
    raw_suffix_scores = _suffix_prefix_product_scores(
        local_gates,
        max_suffix_length,
    )
    if suffix_mode == "collision_lr":
        route_scores = _collision_log_likelihood_ratio_scores(
            local_gates,
            symbol_dim=query.size(-1),
            max_suffix_length=max_suffix_length,
            mismatch_scale=mismatch_scale,
        )
        probabilities = _proxy_route_probabilities(
            route_scores,
            causal_route_mask,
            scale=scale,
            null_score=0.0,
        )
    else:
        route_scores = _suffix_score_utility(raw_suffix_scores, suffix_mode)
        probabilities = _route_probabilities(
            _masked_route_scores(route_scores, causal_route_mask),
            causal_route_mask,
            scale,
        )
    return local_gates, raw_suffix_scores, probabilities


def _surrogate_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int,
    scale: float,
    mismatch_scale: float,
    dimension_mode: str,
    suffix_mode: str,
) -> Tensor:
    _, _, probabilities = _proxy_state(
        query,
        key,
        max_suffix_length=max_suffix_length,
        scale=scale,
        mismatch_scale=mismatch_scale,
        dimension_mode=dimension_mode,
        suffix_mode=suffix_mode,
    )
    return _build_vjp_carrier(value, probabilities, query.size(2))


class _HardForwardSuffixProxy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
        scale: float,
        mismatch_scale: float,
        dimension_mode: int,
        suffix_mode: int,
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
        ctx.dimension_mode = _DIMENSION_MODES_BY_ID[int(dimension_mode)]
        ctx.suffix_mode = _SUFFIX_MODES_BY_ID[int(suffix_mode)]
        ctx.save_for_backward(query, key, value)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(need)
                for tensor, need in zip((query, key, value), needs)
            )
            compute_dtype = _reference_compute_dtype(query.dtype)
            carrier = _surrogate_carrier(
                leaves[0].to(compute_dtype),
                leaves[1].to(compute_dtype),
                leaves[2].to(compute_dtype),
                max_suffix_length=ctx.max_suffix_length,
                scale=ctx.scale,
                mismatch_scale=ctx.mismatch_scale,
                dimension_mode=ctx.dimension_mode,
                suffix_mode=ctx.suffix_mode,
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
            for index, gradient in zip(required_indices, required_grads):
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


def rosa_soft_suffix_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    proxy: str,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    if proxy not in PROXY_CONFIGS:
        raise ValueError(f"proxy must be one of {tuple(PROXY_CONFIGS)}")
    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        0.0,
        mismatch_scale,
    )
    dimension_mode, suffix_mode = PROXY_CONFIGS[proxy]
    return _HardForwardSuffixProxy.apply(
        query,
        key,
        value,
        max_suffix_length,
        float(scale),
        float(mismatch_scale),
        _DIMENSION_MODES[dimension_mode],
        _SUFFIX_MODES[suffix_mode],
    )


def make_proxy_operator(proxy: str) -> ProxyOperator:
    if proxy not in PROXY_CONFIGS:
        raise ValueError(f"proxy must be one of {tuple(PROXY_CONFIGS)}")

    def operator(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        max_suffix_length: int,
        scale: float,
        mismatch_scale: float,
    ) -> Tensor:
        return rosa_soft_suffix_proxy(
            query,
            key,
            value,
            proxy=proxy,
            max_suffix_length=max_suffix_length,
            scale=scale,
            mismatch_scale=mismatch_scale,
        )

    return operator


class FixedWidthRosaFitLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        qk_bits: int,
        value_heads: int,
        value_bits: int,
        max_suffix_length: int,
        scale: float,
        mismatch_scale: float,
        proxy: str,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if num_heads % value_heads != 0:
            raise ValueError("num_heads must be divisible by value_heads")
        self.num_heads = num_heads
        self.qk_bits = qk_bits
        self.value_heads = value_heads
        self.value_bits = value_bits
        self.max_suffix_length = max_suffix_length
        self.scale = scale
        self.mismatch_scale = mismatch_scale
        self.proxy = proxy

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, num_heads * qk_bits, bias=False)
        self.key = nn.Linear(hidden_size, num_heads * qk_bits, bias=False)
        self.value = nn.Linear(
            hidden_size,
            value_heads * value_bits,
            bias=False,
        )
        self.output = nn.Linear(
            num_heads * value_bits,
            hidden_size,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._reset_parameters(initialization_seed)

    @staticmethod
    def _generator(seed: int, offset: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed + offset)

    def _reset_parameters(self, seed: int) -> None:
        nn.init.normal_(
            self.embedding.weight,
            generator=self._generator(seed, 1),
        )
        for offset, layer in enumerate(
            (self.query, self.key, self.value, self.output, self.head),
            start=2,
        ):
            nn.init.kaiming_uniform_(
                layer.weight,
                a=math.sqrt(5.0),
                generator=self._generator(seed, offset),
            )
        for norm in (self.input_norm, self.output_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def project_symbols(
        self,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        residual = self.embedding(tokens)
        hidden = self.input_norm(residual)
        query = self.query(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.num_heads,
            self.qk_bits,
        )
        key = self.key(hidden).view_as(query)
        value = self.value(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.value_heads,
            self.value_bits,
        )
        return residual, query, key, value

    def forward(self, tokens: Tensor) -> Tensor:
        residual, query, key, value = self.project_symbols(tokens)
        routed = rosa_soft_suffix_proxy(
            query,
            key,
            value,
            proxy=self.proxy,
            max_suffix_length=self.max_suffix_length,
            scale=self.scale,
            mismatch_scale=self.mismatch_scale,
        )
        hidden = residual + self.output(routed.flatten(2))
        return self.head(self.output_norm(hidden))


@torch.no_grad()
def _evaluate(
    model: FixedWidthRosaFitLM,
    tokens: Tensor,
    target_mask: Tensor,
) -> tuple[float, float]:
    loss, accuracy = loss_and_accuracy(model(tokens), tokens, target_mask)
    return float(loss.item()), accuracy


def run_fit(
    *,
    proxy: str,
    qk_bits: int,
    max_suffix_length: int,
    mismatch_scale: float,
    model_seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = torch.device(args.device)
    tokens = make_copy_tokens(
        seq_len=args.sequence_length + 1,
        vocab_size=args.vocab_size,
        motif_min=args.motif_min,
        motif_max=args.motif_max,
        seed=100_000 + model_seed,
    )
    target_mask = build_target_mask(
        tokens,
        max_suffix_length,
        args.target_mode,
    )
    if not bool(target_mask.any()):
        raise ValueError("generated sequence has no fitting targets")
    tokens = tokens.to(device)
    target_mask = target_mask.to(device)
    model = FixedWidthRosaFitLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        qk_bits=qk_bits,
        value_heads=args.value_heads,
        value_bits=args.value_bits,
        max_suffix_length=max_suffix_length,
        scale=args.scale,
        mismatch_scale=mismatch_scale,
        proxy=proxy,
        initialization_seed=300_000 + model_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    initial_loss, initial_accuracy = _evaluate(model, tokens, target_mask)
    best_loss = initial_loss
    best_step = 0
    first_below = -1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_and_accuracy(model(tokens), tokens, target_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
        if first_below < 0 and loss_value < args.success_threshold:
            first_below = step
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = _evaluate(model, tokens, target_mask)
    if final_loss < best_loss:
        best_loss = final_loss
        best_step = args.steps
    return {
        "proxy": proxy,
        "qk_bits": qk_bits,
        "max_suffix_length": max_suffix_length,
        "mismatch_scale": mismatch_scale,
        "model_seed": model_seed,
        "fit_target_count": int(target_mask.sum().item()),
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "best_loss": best_loss,
        "best_step": best_step,
        "first_below_threshold": first_below,
        "ever_success": first_below >= 0 or best_loss < args.success_threshold,
        "final_success": math.isfinite(final_loss)
        and final_loss < args.success_threshold,
        "step_ms": elapsed * 1000.0 / args.steps,
    }


def run_gradient_shell(args: argparse.Namespace) -> list[Dict[str, object]]:
    device = torch.device(args.device)
    rows: list[Dict[str, object]] = []
    for qk_bits in args.qk_bits:
        for sequence_length in args.shell_sequence_lengths:
            for max_suffix_length in args.windows:
                for mismatch_scale in args.mismatch_scales:
                    for shell_seed in args.shell_seeds:
                        rows.extend(
                            _run_gradient_shell_cell(
                                args,
                                device=device,
                                qk_bits=qk_bits,
                                sequence_length=sequence_length,
                                max_suffix_length=max_suffix_length,
                                mismatch_scale=mismatch_scale,
                                shell_seed=shell_seed,
                            )
                        )
    return rows


def _run_gradient_shell_cell(
    args: argparse.Namespace,
    *,
    device: torch.device,
    qk_bits: int,
    sequence_length: int,
    max_suffix_length: int,
    mismatch_scale: float,
    shell_seed: int,
) -> list[Dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(
        700_000 + shell_seed
    )
    base_query = torch.randn(
        1,
        sequence_length,
        args.heads,
        qk_bits,
        generator=generator,
    ).to(device)
    base_key = torch.randn(
        base_query.shape,
        generator=generator,
    ).to(device)
    base_value = torch.randn(
        1,
        sequence_length,
        args.value_heads,
        args.value_bits,
        generator=generator,
    ).to(device)
    upstream = torch.randn(
        1,
        sequence_length,
        args.heads,
        args.value_bits,
        generator=generator,
    ).to(device)
    rows: list[Dict[str, object]] = []
    hard_output: Optional[Tensor] = None
    for proxy in args.proxies:
        query = base_query.detach().requires_grad_(True)
        key = base_key.detach().requires_grad_(True)
        value = base_value.detach().requires_grad_(True)
        output = rosa_soft_suffix_proxy(
            query,
            key,
            value,
            proxy=proxy,
            max_suffix_length=max_suffix_length,
            scale=args.scale,
            mismatch_scale=mismatch_scale,
        )
        if hard_output is None:
            hard_output = output.detach()
        elif not torch.equal(output.detach(), hard_output):
            raise AssertionError("proxy changed exact hard forward")
        gradients = torch.autograd.grad(
            (output * upstream).sum(),
            (query, key, value),
        )
        dimension_mode, suffix_mode = PROXY_CONFIGS[proxy]
        with torch.no_grad():
            _, _, probabilities = _proxy_state(
                query,
                key,
                max_suffix_length=max_suffix_length,
                scale=args.scale,
                mismatch_scale=mismatch_scale,
                dimension_mode=dimension_mode,
                suffix_mode=suffix_mode,
            )
            entropy = -(
                probabilities
                * probabilities.clamp_min(
                    torch.finfo(probabilities.dtype).tiny
                ).log()
            ).sum(dim=-1).mean()
        rows.append(
            {
                "proxy": proxy,
                "qk_bits": qk_bits,
                "sequence_length": sequence_length,
                "max_suffix_length": max_suffix_length,
                "mismatch_scale": mismatch_scale,
                "shell_seed": shell_seed,
                "query_gradient_rms": float(
                    gradients[0].float().square().mean().sqrt()
                ),
                "key_gradient_rms": float(
                    gradients[1].float().square().mean().sqrt()
                ),
                "value_gradient_rms": float(
                    gradients[2].float().square().mean().sqrt()
                ),
                "route_entropy": float(entropy),
            }
        )
    return rows


def run_length_competition(
    *,
    proxy: str,
    qk_bits: int,
    distractor_length: int,
    mismatch_scale: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Grow a short target suffix past a frozen longer wrong suffix."""

    device = torch.device(args.device)
    target_length = min(2 * distractor_length, args.competition_max_length)
    if target_length <= distractor_length:
        raise ValueError("competition target length must exceed distractor length")
    short_length = args.competition_short_length
    if not 1 <= short_length < distractor_length:
        raise ValueError("competition short length must be in [1, distractor length)")

    distractor_route = distractor_length + 4
    target_route = distractor_route + target_length + 8
    target_start = target_route - target_length
    sequence_length = target_route + 8
    query = torch.ones(
        1,
        sequence_length,
        1,
        qk_bits,
        device=device,
    )
    base_key = -torch.ones_like(query)
    base_key[
        :, distractor_route - distractor_length : distractor_route
    ] = 1.0
    if qk_bits > 1:
        base_key[:, target_start:target_route, :, 1:] = 1.0
    base_key[:, target_route - short_length : target_route, :, 0] = 1.0
    trainable_count = target_length - short_length
    trainable_bits = nn.Parameter(
        torch.full(
            (trainable_count,),
            -float(args.competition_initial_margin),
            device=device,
        )
    )
    value = -torch.ones(
        1,
        sequence_length,
        1,
        1,
        device=device,
    )
    value[:, target_start : target_route + 1] = 1.0
    dimension_mode, suffix_mode = PROXY_CONFIGS[proxy]

    def materialize_key() -> Tensor:
        key = base_key.clone()
        key[0, target_start : target_route - short_length, 0, 0] = (
            trainable_bits
        )
        return key

    initial_key = materialize_key()
    initial_output = rosa_soft_suffix_proxy(
        query,
        initial_key,
        value,
        proxy=proxy,
        max_suffix_length=target_length,
        scale=args.scale,
        mismatch_scale=mismatch_scale,
    )
    initial_loss = (initial_output[0, -1, 0, 0] - 1.0).square()
    (initial_gradient,) = torch.autograd.grad(initial_loss, trainable_bits)
    with torch.no_grad():
        _, raw_scores, probabilities = _proxy_state(
            query,
            initial_key,
            max_suffix_length=target_length,
            scale=args.scale,
            mismatch_scale=mismatch_scale,
            dimension_mode=dimension_mode,
            suffix_mode=suffix_mode,
        )
        initial_target_probability = float(
            probabilities[0, 0, -1, target_route]
        )
        initial_distractor_probability = float(
            probabilities[0, 0, -1, distractor_route]
        )
        initial_target_raw_score = float(
            raw_scores[0, 0, -1, target_route]
        )
        initial_distractor_raw_score = float(
            raw_scores[0, 0, -1, distractor_route]
        )

    optimizer = torch.optim.SGD(
        (trainable_bits,),
        lr=args.competition_learning_rate,
    )
    crossing_step = -1
    final_output = float(initial_output[0, -1, 0, 0].detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, args.competition_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = rosa_soft_suffix_proxy(
            query,
            materialize_key(),
            value,
            proxy=proxy,
            max_suffix_length=target_length,
            scale=args.scale,
            mismatch_scale=mismatch_scale,
        )
        loss = (output[0, -1, 0, 0] - 1.0).square()
        loss.backward()
        optimizer.step()
        final_output = float(output[0, -1, 0, 0].detach())
        if final_output > 0.0:
            crossing_step = step
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "proxy": proxy,
        "qk_bits": qk_bits,
        "short_length": short_length,
        "distractor_length": distractor_length,
        "target_length": target_length,
        "mismatch_scale": mismatch_scale,
        "initial_target_raw_score": initial_target_raw_score,
        "initial_distractor_raw_score": initial_distractor_raw_score,
        "initial_target_probability": initial_target_probability,
        "initial_distractor_probability": initial_distractor_probability,
        "initial_oldest_bit_gradient": float(initial_gradient[0]),
        "initial_newest_bit_gradient": float(initial_gradient[-1]),
        "initial_gradient_rms": float(
            initial_gradient.float().square().mean().sqrt()
        ),
        "crossing_step": crossing_step,
        "success": crossing_step >= 0,
        "matched_trainable_bits": int((trainable_bits > 0).sum().item()),
        "trainable_bits": trainable_count,
        "final_output": final_output,
        "elapsed_seconds": elapsed,
    }


def run_competition_matrix(args: argparse.Namespace) -> list[Dict[str, object]]:
    return [
        run_length_competition(
            proxy=proxy,
            qk_bits=qk_bits,
            distractor_length=distractor_length,
            mismatch_scale=mismatch_scale,
            args=args,
        )
        for proxy in args.proxies
        for qk_bits in args.competition_bits
        for distractor_length in args.competition_distractor_lengths
        for mismatch_scale in args.mismatch_scales
    ]


def _summarize_fits(runs: list[Dict[str, object]]) -> Dict[str, object]:
    summaries: Dict[str, object] = {}
    group_keys = sorted(
        {
            (
                str(run["proxy"]),
                int(run["qk_bits"]),
                int(run["max_suffix_length"]),
                float(run["mismatch_scale"]),
            )
            for run in runs
        }
    )
    for proxy, bits, window, mismatch_scale in group_keys:
        selected = [
            run
            for run in runs
            if run["proxy"] == proxy
            and run["qk_bits"] == bits
            and run["max_suffix_length"] == window
            and run["mismatch_scale"] == mismatch_scale
        ]
        success_steps = [
            int(run["first_below_threshold"])
            for run in selected
            if int(run["first_below_threshold"]) >= 0
        ]
        key = f"{proxy}:D{bits}:W{window}:lambda{mismatch_scale:g}"
        summaries[key] = {
            "runs": len(selected),
            "ever_successes": sum(bool(run["ever_success"]) for run in selected),
            "final_successes": sum(
                bool(run["final_success"]) for run in selected
            ),
            "median_final_loss": statistics.median(
                float(run["final_loss"]) for run in selected
            ),
            "median_best_loss": statistics.median(
                float(run["best_loss"]) for run in selected
            ),
            "median_success_step": (
                statistics.median(success_steps) if success_steps else None
            ),
            "mean_step_ms": statistics.mean(
                float(run["step_ms"]) for run in selected
            ),
        }
    return summaries


def _analytical_tables(args: argparse.Namespace) -> Dict[str, object]:
    one_mismatch_gates = []
    for bits in args.qk_bits:
        for mismatch_scale in args.mismatch_scales:
            one_mismatch_gates.append(
                {
                    "qk_bits": bits,
                    "mismatch_scale": mismatch_scale,
                    "mean": math.exp(-mismatch_scale / bits),
                    "sqrt": math.exp(-mismatch_scale / math.sqrt(bits)),
                }
            )
    lengths = sorted({0, 1, 2, 4, 8, 16, 32, *args.windows})
    suffix_utilities = []
    for length in lengths:
        raw = torch.tensor(float(length), dtype=torch.float64)
        row = {"length": length}
        for mode in _SUFFIX_MODES:
            if mode == "collision_lr":
                continue
            row[mode] = float(_suffix_score_utility(raw, mode).item())
        suffix_utilities.append(row)
    return {
        "one_mismatch_gates": one_mismatch_gates,
        "exact_suffix_utilities": suffix_utilities,
    }


def run_matrix(args: argparse.Namespace) -> Dict[str, object]:
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    fits = []
    if not args.shell_only and not args.competition_only:
        fits = [
            run_fit(
                proxy=proxy,
                qk_bits=qk_bits,
                max_suffix_length=window,
                mismatch_scale=mismatch_scale,
                model_seed=model_seed,
                args=args,
            )
            for proxy in args.proxies
            for qk_bits in args.qk_bits
            for window in args.windows
            for mismatch_scale in args.mismatch_scales
            for model_seed in args.model_seeds
        ]
    shell = (
        []
        if args.fit_only or args.competition_only
        else run_gradient_shell(args)
    )
    competition = (
        run_competition_matrix(args)
        if args.run_competition or args.competition_only
        else []
    )
    return {
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "proxy_configs": PROXY_CONFIGS,
        "hard_forward": "exact_longest_latest_for_all_proxies",
        "residual_hidden_size": args.hidden_size,
        "proxies": args.proxies,
        "qk_bits": args.qk_bits,
        "windows": args.windows,
        "mismatch_scales": args.mismatch_scales,
        "model_seeds": args.model_seeds,
        "shell_seeds": args.shell_seeds,
        "shell_sequence_lengths": args.shell_sequence_lengths,
        "steps": args.steps,
        "success_threshold": args.success_threshold,
        "target_mode": args.target_mode,
        "scale": args.scale,
        "analytical": _analytical_tables(args),
        "gradient_shell": shell,
        "length_competition": competition,
        "fits": fits,
        "fit_summaries": _summarize_fits(fits) if fits else {},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proxies",
        nargs="+",
        choices=tuple(PROXY_CONFIGS),
        default=list(PROXY_CONFIGS),
    )
    parser.add_argument("--qk-bits", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--windows", nargs="+", type=int, default=[2, 8, 32])
    parser.add_argument(
        "--mismatch-scales",
        nargs="+",
        type=float,
        default=[1.5, 3.0],
    )
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--shell-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--success-threshold", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument(
        "--shell-sequence-lengths",
        nargs="+",
        type=int,
        default=[16, 32, 64],
    )
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--motif-min", type=int, default=4)
    parser.add_argument("--motif-max", type=int, default=8)
    parser.add_argument(
        "--target-mode",
        choices=("any-candidate", "strict-longest-latest"),
        default="strict-longest-latest",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--shell-only", action="store_true")
    parser.add_argument("--run-competition", action="store_true")
    parser.add_argument("--competition-only", action="store_true")
    parser.add_argument(
        "--competition-bits",
        nargs="+",
        type=int,
        default=[2, 4, 8],
    )
    parser.add_argument(
        "--competition-distractor-lengths",
        nargs="+",
        type=int,
        default=[4, 8, 16],
    )
    parser.add_argument("--competition-short-length", type=int, default=1)
    parser.add_argument("--competition-max-length", type=int, default=32)
    parser.add_argument("--competition-initial-margin", type=float, default=0.25)
    parser.add_argument("--competition-learning-rate", type=float, default=1.0)
    parser.add_argument("--competition-steps", type=int, default=1500)
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exclusive_modes = sum(
        (args.fit_only, args.shell_only, args.competition_only)
    )
    if exclusive_modes > 1:
        raise ValueError(
            "--fit-only, --shell-only, and --competition-only are mutually exclusive"
        )
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
