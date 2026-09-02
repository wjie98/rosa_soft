"""Controlled initializers for state-attention research models."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from rosa_soft.initialization import (
    initialize_orthogonal_value_,
    initialize_partial_shared_qk_,
)


QK_INITIALIZATIONS = ("default", "partial_shared_orthogonal")
VALUE_INITIALIZATIONS = ("default", "orthogonal", "paired_output")


class _StateAttentionModel(Protocol):
    embedding: nn.Embedding
    recurrent: nn.GRUCell
    extra_recurrent: nn.ModuleList
    symbol_norm: nn.LayerNorm
    query: nn.Linear
    key: nn.Linear
    value: nn.Linear
    output: nn.Linear
    output_norm: nn.LayerNorm
    head: nn.Linear
    num_heads: int
    value_heads: int
    qk_bits: int
    value_bits: int


def _reset_module(module: nn.Module, seed: int) -> None:
    if any(parameter.is_cuda for parameter in module.parameters(recurse=False)):
        raise ValueError("controlled initialization must run before model.to(cuda)")
    reset_parameters = getattr(module, "reset_parameters", None)
    if reset_parameters is None:
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reset_parameters()


def _reset_model_modules(model: _StateAttentionModel, seed: int) -> None:
    modules = [
        model.embedding,
        model.recurrent,
        *model.extra_recurrent,
        model.symbol_norm,
        model.query,
        model.key,
        model.value,
        model.output,
        model.output_norm,
        model.head,
    ]
    for index, module in enumerate(modules):
        _reset_module(module, seed + 104_729 * (index + 1))


def _initialize_partial_shared_qk(
    model: _StateAttentionModel,
    *,
    seed: int,
    shared_bits: int,
    correlation: float,
    logit_median: float,
) -> None:
    initialize_partial_shared_qk_(
        model.query,
        model.key,
        num_heads=model.num_heads,
        qk_bits=model.qk_bits,
        shared_qk_bits=shared_bits,
        correlation=correlation,
        logit_median=logit_median,
        seed=seed,
        zero_bias=False,
    )


def _initialize_orthogonal_value(
    model: _StateAttentionModel,
    *,
    seed: int,
    logit_median: float,
) -> None:
    initialize_orthogonal_value_(
        model.value,
        num_value_heads=model.value_heads,
        value_bits=model.value_bits,
        logit_median=logit_median,
        seed=seed,
        zero_bias=False,
    )


def _pair_output_with_value(model: _StateAttentionModel) -> None:
    import math

    value_dim = int(model.value_bits)
    groups = model.num_heads // model.value_heads
    scale = 1.0 / math.sqrt(model.num_heads)
    output_blocks = []
    for query_head in range(model.num_heads):
        value_head = query_head // groups
        start = value_head * value_dim
        value_block = model.value.weight[start : start + value_dim]
        output_blocks.append(scale * value_block.transpose(0, 1))
    with torch.no_grad():
        model.output.weight.copy_(torch.cat(output_blocks, dim=1))


def initialize_state_attention_model(
    model: _StateAttentionModel,
    *,
    seed: int,
    qk_init: str,
    qk_shared_bits: int,
    qk_correlation: float,
    qk_logit_median: float,
    value_init: str,
    value_logit_median: float,
) -> None:
    """Reset all modules deterministically, then apply controlled overrides."""

    if qk_init not in QK_INITIALIZATIONS:
        raise ValueError(f"qk_init must be one of {QK_INITIALIZATIONS}")
    if value_init not in VALUE_INITIALIZATIONS:
        raise ValueError(f"value_init must be one of {VALUE_INITIALIZATIONS}")
    _reset_model_modules(model, int(seed))
    if qk_init == "partial_shared_orthogonal":
        _initialize_partial_shared_qk(
            model,
            seed=int(seed) + 30_000_001,
            shared_bits=int(qk_shared_bits),
            correlation=float(qk_correlation),
            logit_median=float(qk_logit_median),
        )
    if value_init != "default":
        _initialize_orthogonal_value(
            model,
            seed=int(seed) + 40_000_003,
            logit_median=float(value_logit_median),
        )
    if value_init == "paired_output":
        _pair_output_with_value(model)


__all__ = [
    "QK_INITIALIZATIONS",
    "VALUE_INITIALIZATIONS",
    "initialize_state_attention_model",
]
