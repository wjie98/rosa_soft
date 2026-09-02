"""Long-suffix, many-distractor extrapolation gate for ROSA estimators.

This benchmark isolates one question: can a backward estimator repair the
oldest decisive bit of a W-symbol query when the hard forward must choose
among every position in a long context?  The context is a real global key
sequence, not an independent table of suffix windows.

Aligned context blocks repeat the target trajectory.  Every non-target block
flips one content bit at its oldest position, so a solved query sees all of
them as strict length-(W-1) distractors.  Before training, the query flips the
same role-dependent bit and many distractors become exact matches.  The only
trainable scalar is shared by every probe; there is no value or readout
shortcut.

Training materializes only the requested final query rows, O(P*T*W), while
long-context evaluation packs each D-bit symbol into an integer and scans
candidate chunks exactly.  No candidate is removed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import temporal_quadratic_proxy  # noqa: E402
from benchmarks.fast_weight_proxy import (  # noqa: E402
    _count_sketch,
    _count_sketch_map,
    _symbol_interaction_features,
)
from benchmarks.suffix_kernel_proxy import (  # noqa: E402
    _quadratic_features as _suffix_quadratic_features,
    _suffix_feature_branches,
    _symbol_kernel_features as _suffix_symbol_features,
)
from benchmarks.temporal_quadratic_proxy import (  # noqa: E402
    DEFAULT_STATE_DIM as TEMPORAL_DEFAULT_STATE_DIM,
    _temporal_quadratic_fingerprints,
)
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    ROSA_SOFT_NULL_ROUTE_SCORE,
)
from rosa_soft.soft_reference import (  # noqa: E402
    _hard_sign,
    _hard_sign_with_softsign_vjp,
    _suffix_score_utility,
)


ESTIMATORS = (
    "production",
    "exact_relevant_bitflip",
    "state_quadratic_attention",
)
TEMPORAL_ESTIMATOR = "temporal_quadratic_attention"
SUFFIX_KERNEL_ESTIMATORS = {
    "suffix_sketch_raw_attention": "raw",
    "suffix_sketch_quadratic_attention": "quadratic",
    "suffix_sketch_level_quadratic_attention": "level_quadratic",
}
EXACT_SUFFIX_KERNEL_ESTIMATORS = {
    "exact_suffix_raw_attention": "raw",
    "exact_suffix_quadratic_attention": "quadratic",
    "exact_suffix_level_quadratic_attention": "level_quadratic",
}
ESTIMATOR_CHOICES = ESTIMATORS + (TEMPORAL_ESTIMATOR,) + tuple(
    SUFFIX_KERNEL_ESTIMATORS
) + tuple(EXACT_SUFFIX_KERNEL_ESTIMATORS)
VALUE_MODES = ("balanced_binary", "coherent_negative")


@dataclass(frozen=True)
class GateSpec:
    """Fixed symbols and layout shared by all matched estimators."""

    target_signs: Tensor
    target_values: Tensor
    target_blocks: Tensor
    fault_bits: Tensor
    window: int
    bits: int
    phase_bits: int
    content_bits: int
    value_bits: int
    value_mode: str
    seed: int
    train_context_length: int

    @property
    def probes(self) -> int:
        return self.target_signs.size(0)

    @property
    def target_positions(self) -> Tensor:
        return (self.target_blocks + 1) * self.window - 1

    def to(self, device: torch.device) -> "GateSpec":
        return GateSpec(
            target_signs=self.target_signs.to(device),
            target_values=self.target_values.to(device),
            target_blocks=self.target_blocks.to(device),
            fault_bits=self.fault_bits.to(device),
            window=self.window,
            bits=self.bits,
            phase_bits=self.phase_bits,
            content_bits=self.content_bits,
            value_bits=self.value_bits,
            value_mode=self.value_mode,
            seed=self.seed,
            train_context_length=self.train_context_length,
        )


@dataclass(frozen=True)
class DenseContext:
    key_signs: Tensor
    values: Tensor
    target_positions: Tensor


@dataclass(frozen=True)
class TemporalQuadraticContext:
    memory: Tensor
    normalizer: Tensor
    target_features: Tensor


@dataclass(frozen=True)
class SuffixKernelContext:
    memory: Tensor
    normalizer: Tensor
    target_features: Tensor
    route_kernel: str
    window: int
    mismatch_scale: float
    sketch_dim: int
    sketch_count: int
    sketch_seed: int


@dataclass(frozen=True)
class RouteState:
    output: Tensor
    selected_positions: Tensor
    max_suffix_lengths: Tensor
    target_suffix_lengths: Tensor
    max_distractor_suffix_lengths: Tensor
    exact_distractor_counts: Tensor
    strict_distractor_counts: Tensor


def _phase_width(window: int) -> int:
    return 0 if window == 1 else math.ceil(math.log2(window))


def _binary_signs(ids: Tensor, width: int) -> Tensor:
    if width == 0:
        return torch.empty(ids.numel(), 0, dtype=torch.float32)
    shifts = torch.arange(width, dtype=torch.int64)
    positive = ((ids.reshape(-1, 1) >> shifts) & 1).bool()
    return torch.where(positive, 1.0, -1.0)


def make_gate_spec(
    *,
    seed: int,
    probes: int,
    window: int,
    bits: int,
    value_bits: int = 8,
    value_mode: str = "balanced_binary",
    train_context_length: int,
) -> GateSpec:
    if probes < 1:
        raise ValueError("probes must be positive")
    if window < 1:
        raise ValueError("window must be positive")
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if value_bits < 1 or value_bits > 16:
        raise ValueError("value_bits must be in [1, 16]")
    if value_mode not in VALUE_MODES:
        raise ValueError(f"value_mode must be one of {VALUE_MODES}")
    phase_bits = _phase_width(window)
    content_bits = bits - phase_bits
    if content_bits < 1:
        raise ValueError(
            "bits must leave at least one content bit after phase encoding"
        )
    full_blocks = train_context_length // window
    if full_blocks < 3:
        raise ValueError("training context must contain at least three blocks")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    phase = _binary_signs(torch.arange(window), phase_bits)
    phase = phase.unsqueeze(0).expand(probes, -1, -1)
    content = 2 * torch.randint(
        2,
        (probes, window, content_bits),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32) - 1.0
    target_signs = torch.cat((phase, content), dim=-1)
    if value_mode == "coherent_negative":
        target_values = torch.ones(probes, value_bits)
    else:
        target_values = 2 * torch.randint(
            2,
            (probes, value_bits),
            generator=generator,
            dtype=torch.int64,
        ).to(torch.float32) - 1.0

    probe_ids = torch.arange(probes, dtype=torch.int64)
    fault_offsets = (17 * int(seed) + 3 * probe_ids).remainder(content_bits)
    fault_bits = phase_bits + fault_offsets

    # Keep every target in the first quarter of the 8K training context.  Long
    # evaluation therefore appends genuinely later competitors.
    early_blocks = max(1, full_blocks // 4)
    target_blocks = (
        104_729 * probe_ids + 8191 * int(seed)
    ).remainder(early_blocks)
    return GateSpec(
        target_signs=target_signs,
        target_values=target_values,
        target_blocks=target_blocks,
        fault_bits=fault_bits,
        window=int(window),
        bits=int(bits),
        phase_bits=phase_bits,
        content_bits=content_bits,
        value_bits=int(value_bits),
        value_mode=value_mode,
        seed=int(seed),
        train_context_length=int(train_context_length),
    )


def _flip_bits_for_blocks(spec: GateSpec, blocks: Tensor) -> Tensor:
    probe_ids = torch.arange(
        spec.probes,
        dtype=torch.int64,
        device=blocks.device,
    ).view(-1, 1)
    offsets = (
        blocks.view(1, -1) + 17 * spec.seed + 3 * probe_ids
    ).remainder(spec.content_bits)
    return offsets + spec.phase_bits


def materialize_context(
    spec: GateSpec,
    context_length: int,
    *,
    device: torch.device,
) -> DenseContext:
    if context_length < spec.train_context_length:
        raise ValueError("context_length cannot be shorter than training")
    positions = torch.arange(context_length, device=device)
    phases = positions.remainder(spec.window)
    blocks = torch.div(positions, spec.window, rounding_mode="floor")
    target_signs = spec.target_signs.to(device)
    keys = target_signs[:, phases].clone()

    flip_bits = _flip_bits_for_blocks(spec.to(device), blocks)
    should_flip = (
        phases.view(1, -1).eq(0)
        & blocks.view(1, -1).ne(spec.target_blocks.to(device).view(-1, 1))
    )
    flip_mask = torch.nn.functional.one_hot(
        flip_bits,
        num_classes=spec.bits,
    ).bool() & should_flip.unsqueeze(-1)
    keys = torch.where(flip_mask, -keys, keys)

    target_positions = spec.target_positions.to(device)
    if spec.value_mode == "coherent_negative":
        values = torch.full(
            (spec.probes, context_length, spec.value_bits),
            -1.0,
            dtype=keys.dtype,
            device=device,
        )
    else:
        value_rows = []
        target_values_cpu = spec.target_values.detach().cpu()
        for probe in range(spec.probes):
            generator = torch.Generator(device="cpu").manual_seed(
                50_000_017 + 1_000_003 * spec.seed + 97_409 * probe
            )
            row = 2 * torch.randint(
                2,
                (context_length, spec.value_bits),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.float32) - 1.0
            collisions = row.eq(target_values_cpu[probe]).all(dim=-1)
            row[collisions, 0] *= -1.0
            value_rows.append(row)
        values = torch.stack(value_rows).to(device=device, dtype=keys.dtype)
    rows = torch.arange(spec.probes, device=device)
    values[rows, target_positions] = spec.target_values.to(device)
    return DenseContext(
        key_signs=keys,
        values=values,
        target_positions=target_positions,
    )


def make_query_logits(
    spec: GateSpec,
    fault_logit: Tensor,
    *,
    logit_margin: float,
) -> Tensor:
    target = spec.target_signs.to(fault_logit.device)
    fault_mask = torch.zeros_like(target, dtype=torch.bool)
    rows = torch.arange(spec.probes, device=fault_logit.device)
    fault_mask[rows, 0, spec.fault_bits.to(fault_logit.device)] = True
    return torch.where(
        fault_mask,
        target * fault_logit,
        target * float(logit_margin),
    )


def _pack_signs(signs: Tensor) -> Tensor:
    if signs.size(-1) > 16:
        raise ValueError("packed benchmark codes support at most 16 bits")
    shifts = torch.arange(
        signs.size(-1),
        dtype=torch.int64,
        device=signs.device,
    )
    return ((signs > 0).to(torch.int64) << shifts).sum(dim=-1)


def _constant_symbol_feature_table(
    bits: int,
    *,
    mismatch_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Materialize the exact local feature once for every binary symbol."""

    symbol_ids = torch.arange(1 << bits, dtype=torch.int64, device=device)
    shifts = torch.arange(bits, dtype=torch.int64, device=device)
    symbols = torch.where(
        ((symbol_ids.view(-1, 1) >> shifts) & 1).bool(),
        torch.ones((), dtype=dtype, device=device),
        -torch.ones((), dtype=dtype, device=device),
    )
    logits = symbols.view(1, 1 << bits, 1, bits)
    return _suffix_symbol_features(logits, mismatch_scale)[0, 0]


def _constant_nested_suffix_levels(
    symbol_codes: Tensor,
    symbol_feature_table: Tensor,
    *,
    window: int,
    sketch_dim: int,
    branch_seed: int,
):
    """Yield fixed-width suffix levels without expanding every token to 2**D."""

    product_spectrum = None
    context_length = symbol_codes.size(1)
    for offset in range(window):
        factor_table = _count_sketch(
            symbol_feature_table,
            sketch_dim,
            int(branch_seed) + 97_409 * offset,
        )
        unshifted = factor_table[symbol_codes]
        if offset == 0:
            factor = unshifted
        else:
            factor = torch.zeros_like(unshifted)
            factor[:, offset:] = unshifted[:, : context_length - offset]
        spectrum = torch.fft.fft(factor, dim=-1)
        product_spectrum = (
            spectrum
            if product_spectrum is None
            else product_spectrum * spectrum
        )
        yield torch.fft.ifft(product_spectrum, dim=-1).real


def _aggregate_suffix_feature_group(
    features: Tensor,
    values: Tensor,
    target_positions: Tensor,
    *,
    quadratic: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    probes, context_length, input_dim = features.shape
    if quadratic:
        feature_dim = input_dim * (input_dim + 1) // 2
    else:
        feature_dim = input_dim
    value_dim = values.size(-1)
    memory = features.new_zeros(probes, feature_dim, value_dim)
    normalizer = features.new_zeros(probes, feature_dim)
    rows = torch.arange(probes, device=features.device)
    target = features[rows, target_positions]
    if quadratic:
        target = _suffix_quadratic_features(target)

    # Bound the temporary transformed tile to roughly four million scalars.
    chunk_size = max(1, min(context_length, (1 << 22) // (probes * feature_dim)))
    for start in range(0, context_length, chunk_size):
        end = min(start + chunk_size, context_length)
        chunk = features[:, start:end]
        if quadratic:
            chunk = _suffix_quadratic_features(chunk)
        memory.add_(torch.einsum("ptf,ptv->pfv", chunk, values[:, start:end]))
        normalizer.add_(chunk.sum(dim=1))
    return memory, normalizer, target


@torch.no_grad()
def _make_suffix_kernel_context(
    context: DenseContext,
    *,
    route_kernel: str,
    window: int,
    mismatch_scale: float,
    sketch_dim: int,
    sketch_count: int,
    sketch_seed: int,
) -> SuffixKernelContext:
    if route_kernel not in ("raw", "quadratic", "level_quadratic"):
        raise ValueError("unknown suffix route kernel")
    symbol_codes = _pack_signs(context.key_signs)
    symbol_table = _constant_symbol_feature_table(
        context.key_signs.size(-1),
        mismatch_scale=mismatch_scale,
        dtype=context.key_signs.dtype,
        device=context.key_signs.device,
    )
    memory_groups = []
    normalizer_groups = []
    target_groups = []
    if route_kernel == "raw":
        exact_local = symbol_table[symbol_codes]
        memory, normalizer, target = _aggregate_suffix_feature_group(
            exact_local,
            context.values,
            context.target_positions,
            quadratic=False,
        )
        memory_groups.append(memory)
        normalizer_groups.append(normalizer)
        target_groups.append(target)
    branch_scale = 1.0 / math.sqrt(sketch_count)
    for branch in range(sketch_count):
        branch_seed = int(sketch_seed) + 10_000_019 * branch
        levels = _constant_nested_suffix_levels(
            symbol_codes,
            symbol_table,
            window=window,
            sketch_dim=sketch_dim,
            branch_seed=branch_seed,
        )
        if route_kernel == "quadratic":
            compressed = context.key_signs.new_zeros(
                context.key_signs.size(0),
                context.key_signs.size(1),
                sketch_dim,
            )
            buckets, signs = _count_sketch_map(
                window * sketch_dim,
                sketch_dim,
                int(sketch_seed) + 70_000_027 + 3_000_017 * branch,
                context.key_signs.device,
            )
            for level_index, level in enumerate(levels):
                start = level_index * sketch_dim
                end = start + sketch_dim
                level_buckets = buckets[start:end]
                level_signs = signs[start:end].to(level.dtype)
                index = level_buckets.view(1, 1, -1).expand_as(level)
                compressed.scatter_add_(
                    -1,
                    index,
                    level * level_signs.view(1, 1, -1),
                )
            memory, normalizer, target = _aggregate_suffix_feature_group(
                compressed,
                context.values,
                context.target_positions,
                quadratic=True,
            )
            memory_groups.append(memory * branch_scale)
            normalizer_groups.append(normalizer * branch_scale)
            target_groups.append(target * branch_scale)
        else:
            for level_index, level in enumerate(levels):
                if route_kernel == "raw" and level_index == 0:
                    continue
                memory, normalizer, target = _aggregate_suffix_feature_group(
                    level,
                    context.values,
                    context.target_positions,
                    quadratic=route_kernel == "level_quadratic",
                )
                memory_groups.append(memory * branch_scale)
                normalizer_groups.append(normalizer * branch_scale)
                target_groups.append(target * branch_scale)
    return SuffixKernelContext(
        memory=torch.cat(memory_groups, dim=1),
        normalizer=torch.cat(normalizer_groups, dim=1),
        target_features=torch.cat(target_groups, dim=1),
        route_kernel=route_kernel,
        window=int(window),
        mismatch_scale=float(mismatch_scale),
        sketch_dim=int(sketch_dim),
        sketch_count=int(sketch_count),
        sketch_seed=int(sketch_seed),
    )


def _suffix_kernel_attention_carrier(
    query_logits: Tensor,
    suffix_context: SuffixKernelContext,
) -> tuple[Tensor, Tensor]:
    local_features = _suffix_symbol_features(
        query_logits.unsqueeze(2),
        suffix_context.mismatch_scale,
    )
    branches = _suffix_feature_branches(
        query_logits.unsqueeze(2),
        representation="sketch",
        fingerprint_length=suffix_context.window,
        mismatch_scale=suffix_context.mismatch_scale,
        sketch_dim=suffix_context.sketch_dim,
        sketch_count=suffix_context.sketch_count,
        sketch_seed=suffix_context.sketch_seed,
        local_features=local_features,
    )
    query_groups = []
    if suffix_context.route_kernel == "raw":
        query_groups.append(local_features[:, 0, -1])
    branch_scale = 1.0 / math.sqrt(suffix_context.sketch_count)
    for branch, levels in enumerate(branches):
        if suffix_context.route_kernel == "raw":
            groups = [
                level[:, 0, -1] * branch_scale for level in levels[1:]
            ]
        elif suffix_context.route_kernel == "level_quadratic":
            groups = [
                _suffix_quadratic_features(level[:, 0, -1]) * branch_scale
                for level in levels
            ]
        else:
            joined = torch.cat(tuple(level[:, 0, -1] for level in levels), dim=-1)
            compressed = _count_sketch(
                joined,
                suffix_context.sketch_dim,
                suffix_context.sketch_seed
                + 70_000_027
                + 3_000_017 * branch,
            )
            groups = [
                _suffix_quadratic_features(compressed) * branch_scale
            ]
        query_groups.extend(groups)
    query_features = torch.cat(query_groups, dim=-1)
    denominator = torch.einsum(
        "pf,pf->p",
        query_features,
        suffix_context.normalizer,
    ).clamp_min(1e-6)
    carrier = torch.einsum(
        "pf,pfv->pv",
        query_features,
        suffix_context.memory,
    ) / denominator.unsqueeze(-1)
    target_weight = torch.einsum(
        "pf,pf->p",
        query_features,
        suffix_context.target_features,
    ) / denominator
    return carrier, target_weight


def _suffix_lengths_from_codes(
    query_codes: Tensor,
    key_codes: Tensor,
    window: int,
) -> Tensor:
    probes, context_length = key_codes.shape
    running = torch.ones(
        probes,
        context_length,
        dtype=torch.bool,
        device=key_codes.device,
    )
    lengths = torch.zeros_like(running, dtype=torch.int64)
    for local_position in range(window - 1, -1, -1):
        shift = window - 1 - local_position
        match = torch.zeros_like(running)
        match[:, shift:] = key_codes[:, : context_length - shift].eq(
            query_codes[:, local_position].view(-1, 1)
        )
        running &= match
        lengths += running
    return lengths


def _route_state_from_lengths(
    lengths: Tensor,
    values: Tensor,
    target_positions: Tensor,
    window: int,
) -> RouteState:
    probes, context_length = lengths.shape
    max_lengths = lengths.amax(dim=-1)
    positions = torch.arange(context_length, device=lengths.device)
    selected = torch.where(
        lengths.eq(max_lengths.view(-1, 1)) & max_lengths.view(-1, 1).gt(0),
        positions.view(1, -1),
        -torch.ones((), dtype=torch.int64, device=lengths.device),
    ).amax(dim=-1)
    safe_selected = selected.clamp_min(0)
    value_bits = values.size(-1)
    output = values.gather(
        1,
        safe_selected.view(-1, 1, 1).expand(-1, 1, value_bits),
    ).squeeze(1)
    output = torch.where(
        selected.ge(0).view(-1, 1),
        output,
        torch.zeros_like(output),
    )

    target_mask = positions.view(1, -1).eq(target_positions.view(-1, 1))
    target_lengths = lengths.gather(1, target_positions.view(-1, 1)).squeeze(1)
    distractor_lengths = lengths.masked_fill(target_mask, -1)
    strict_length = max(window - 1, 0)
    return RouteState(
        output=output,
        selected_positions=selected,
        max_suffix_lengths=max_lengths,
        target_suffix_lengths=target_lengths,
        max_distractor_suffix_lengths=distractor_lengths.amax(dim=-1),
        exact_distractor_counts=(
            lengths.eq(window) & ~target_mask
        ).sum(dim=-1),
        strict_distractor_counts=(
            lengths.eq(strict_length) & ~target_mask
        ).sum(dim=-1),
    )


def dense_hard_route(
    query_logits: Tensor,
    context: DenseContext,
    window: int,
) -> RouteState:
    query_codes = _pack_signs(_hard_sign(query_logits))
    key_codes = _pack_signs(context.key_signs)
    lengths = _suffix_lengths_from_codes(query_codes, key_codes, window)
    return _route_state_from_lengths(
        lengths,
        context.values,
        context.target_positions,
        window,
    )


def _final_query_suffix_products(
    query_logits: Tensor,
    context: DenseContext,
    *,
    window: int,
    mismatch_scale: float,
) -> Tensor:
    query_symbols = _hard_sign_with_softsign_vjp(query_logits)
    probes, context_length, _ = context.key_signs.shape
    local_gates = []
    for local_position in range(window):
        shift = window - 1 - local_position
        valid_keys = context.key_signs[:, : context_length - shift]
        mismatch_rate = 0.5 * (
            1.0
            - query_symbols[:, local_position].unsqueeze(1) * valid_keys
        ).mean(dim=-1)
        gate = torch.exp(-float(mismatch_scale) * mismatch_rate)
        if shift:
            gate = torch.nn.functional.pad(gate, (shift, 0), value=0.0)
        local_gates.append(gate)
    gates = torch.stack(local_gates, dim=-1)
    return torch.flip(gates, dims=(-1,)).cumprod(dim=-1)


def _production_suffix_carrier(
    query_logits: Tensor,
    context: DenseContext,
    *,
    window: int,
    scale: float,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor]:
    suffix_products = _final_query_suffix_products(
        query_logits,
        context,
        window=window,
        mismatch_scale=mismatch_scale,
    )
    raw_suffix_scores = suffix_products.sum(dim=-1)
    candidate_scores = _suffix_score_utility(raw_suffix_scores)
    candidate_logits = candidate_scores * float(scale) - math.log(
        context.key_signs.size(1)
    )
    null_logits = candidate_logits.new_full(
        (candidate_logits.size(0), 1),
        ROSA_SOFT_NULL_ROUTE_SCORE * float(scale),
    )
    probabilities = torch.softmax(
        torch.cat((null_logits, candidate_logits), dim=-1),
        dim=-1,
    )
    carrier = (
        probabilities[:, 1:].unsqueeze(-1) * context.values
    ).sum(dim=1)
    target_probability = probabilities[:, 1:].gather(
        1,
        context.target_positions.view(-1, 1),
    ).squeeze(1)
    return carrier, target_probability


def _exact_suffix_kernel_carrier(
    query_logits: Tensor,
    context: DenseContext,
    *,
    route_kernel: str,
    window: int,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor]:
    suffix_products = _final_query_suffix_products(
        query_logits,
        context,
        window=window,
        mismatch_scale=mismatch_scale,
    )
    raw_scores = suffix_products.sum(dim=-1)
    if route_kernel == "raw":
        weights = raw_scores
    elif route_kernel == "quadratic":
        weights = raw_scores.square()
    elif route_kernel == "level_quadratic":
        weights = suffix_products.square().sum(dim=-1)
    else:
        raise ValueError("unknown exact suffix route kernel")
    denominator = weights.sum(dim=-1).clamp_min(1e-6)
    carrier = (
        weights.unsqueeze(-1) * context.values
    ).sum(dim=1) / denominator.unsqueeze(-1)
    target_weight = weights.gather(
        1,
        context.target_positions.view(-1, 1),
    ).squeeze(1) / denominator
    return carrier, target_weight


def _quadratic_attention_carrier(
    query_logits: Tensor,
    context: DenseContext,
    *,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor]:
    query_features = _symbol_interaction_features(
        query_logits[:, -1:].unsqueeze(2),
        mismatch_scale,
        max_degree=2,
    )[:, 0, 0]
    key_features = _symbol_interaction_features(
        context.key_signs.unsqueeze(2),
        mismatch_scale,
        max_degree=2,
    )[:, 0]
    weights = torch.einsum("bf,btf->bt", query_features, key_features)
    denominator = weights.sum(dim=-1).clamp_min(1e-6)
    carrier = (
        weights.unsqueeze(-1) * context.values
    ).sum(dim=1) / denominator.unsqueeze(-1)
    target_weight = weights.gather(
        1,
        context.target_positions.view(-1, 1),
    ).squeeze(1) / denominator
    dependency = query_logits.sum(dim=(1, 2), keepdim=False).unsqueeze(-1) * 0.0
    return carrier + dependency, target_weight


def _make_temporal_quadratic_context(
    context: DenseContext,
    *,
    window: int,
    state_dim: int,
) -> TemporalQuadraticContext:
    key_states = temporal_quadratic_proxy._temporal_suffix_states(
        context.key_signs.unsqueeze(2),
        window,
        state_dim,
    )[:, 0]
    key_states = temporal_quadratic_proxy._normalize_temporal_state(key_states)
    target_states = key_states.gather(
        1,
        context.target_positions.view(-1, 1, 1).expand(
            -1,
            1,
            state_dim,
        ),
    ).squeeze(1)
    target_features = temporal_quadratic_proxy._quadratic_polynomial_features(
        target_states
    )
    feature_dim = target_features.size(-1)
    memory = context.values.new_zeros(
        context.values.size(0),
        feature_dim,
        context.values.size(-1),
    )
    normalizer = context.values.new_zeros(
        context.values.size(0),
        feature_dim,
    )
    feature_chunk_size = 256
    for start in range(0, key_states.size(1), feature_chunk_size):
        end = min(start + feature_chunk_size, key_states.size(1))
        key_features = (
            temporal_quadratic_proxy._quadratic_polynomial_features(
                key_states[:, start:end]
            )
        )
        memory = memory + torch.einsum(
            "ptf,ptv->pfv",
            key_features,
            context.values[:, start:end],
        )
        normalizer = normalizer + key_features.sum(dim=1)
    return TemporalQuadraticContext(
        memory=memory,
        normalizer=normalizer,
        target_features=target_features,
    )


def _temporal_quadratic_attention_carrier(
    query_logits: Tensor,
    temporal_context: TemporalQuadraticContext,
    *,
    window: int,
    state_dim: int,
) -> tuple[Tensor, Tensor]:
    query_features = _temporal_quadratic_fingerprints(
        query_logits.unsqueeze(2),
        window,
        state_dim,
    )[:, 0, -1]
    denominator = torch.einsum(
        "pf,pf->p",
        query_features,
        temporal_context.normalizer,
    ).clamp_min(1e-6)
    carrier = torch.einsum(
        "pf,pfv->pv",
        query_features,
        temporal_context.memory,
    ) / denominator.unsqueeze(-1)
    target_weight = torch.einsum(
        "pf,pf->p",
        query_features,
        temporal_context.target_features,
    ) / denominator
    return carrier, target_weight


def hard_forward_with_proxy(
    estimator: str,
    spec: GateSpec,
    context: DenseContext,
    fault_logit: Tensor,
    *,
    logit_margin: float,
    scale: float,
    mismatch_scale: float,
    temporal_state_dim: int = TEMPORAL_DEFAULT_STATE_DIM,
    temporal_context: Optional[TemporalQuadraticContext] = None,
    suffix_sketch_dim: int = 32,
    suffix_sketch_count: int = 2,
    suffix_sketch_seed: int = 0,
    suffix_context: Optional[SuffixKernelContext] = None,
) -> tuple[Tensor, RouteState, Tensor]:
    if estimator not in ESTIMATOR_CHOICES:
        raise ValueError(f"estimator must be one of {ESTIMATOR_CHOICES}")
    query_logits = make_query_logits(
        spec,
        fault_logit,
        logit_margin=logit_margin,
    )
    hard = dense_hard_route(query_logits, context, spec.window)
    if estimator == "production":
        carrier, target_weight = _production_suffix_carrier(
            query_logits,
            context,
            window=spec.window,
            scale=scale,
            mismatch_scale=mismatch_scale,
        )
    elif estimator == "state_quadratic_attention":
        carrier, target_weight = _quadratic_attention_carrier(
            query_logits,
            context,
            mismatch_scale=mismatch_scale,
        )
    elif estimator == TEMPORAL_ESTIMATOR:
        if temporal_context is None:
            temporal_context = _make_temporal_quadratic_context(
                context,
                window=spec.window,
                state_dim=temporal_state_dim,
            )
        carrier, target_weight = _temporal_quadratic_attention_carrier(
            query_logits,
            temporal_context,
            window=spec.window,
            state_dim=temporal_state_dim,
        )
    elif estimator in EXACT_SUFFIX_KERNEL_ESTIMATORS:
        carrier, target_weight = _exact_suffix_kernel_carrier(
            query_logits,
            context,
            route_kernel=EXACT_SUFFIX_KERNEL_ESTIMATORS[estimator],
            window=spec.window,
            mismatch_scale=mismatch_scale,
        )
    elif estimator in SUFFIX_KERNEL_ESTIMATORS:
        if suffix_context is None:
            suffix_context = _make_suffix_kernel_context(
                context,
                route_kernel=SUFFIX_KERNEL_ESTIMATORS[estimator],
                window=spec.window,
                mismatch_scale=mismatch_scale,
                sketch_dim=suffix_sketch_dim,
                sketch_count=suffix_sketch_count,
                sketch_seed=suffix_sketch_seed,
            )
        carrier, target_weight = _suffix_kernel_attention_carrier(
            query_logits,
            suffix_context,
        )
    else:
        flipped_query = make_query_logits(
            spec,
            -fault_logit.detach(),
            logit_margin=logit_margin,
        )
        flipped = dense_hard_route(flipped_query, context, spec.window)
        fault_sign = torch.where(
            fault_logit.detach() > 0,
            torch.ones_like(fault_logit),
            -torch.ones_like(fault_logit),
        )
        slope = -fault_sign * (flipped.output - hard.output).detach()
        carrier = fault_logit * slope
        target_weight = torch.full(
            (spec.probes,),
            float("nan"),
            dtype=hard.output.dtype,
            device=hard.output.device,
        )
    output = hard.output + (carrier - carrier.detach())
    return output, hard, target_weight.detach()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_condition(
    spec: GateSpec,
    estimator: str,
    *,
    device: torch.device,
    steps: int,
    learning_rate: float,
    initial_fault_logit: float,
    logit_margin: float,
    scale: float,
    mismatch_scale: float,
    gradient_stall_tolerance: float,
    temporal_state_dim: int = TEMPORAL_DEFAULT_STATE_DIM,
    suffix_sketch_dim: int = 32,
    suffix_sketch_count: int = 2,
    suffix_sketch_seed: int = 0,
) -> dict[str, object]:
    context = materialize_context(
        spec,
        spec.train_context_length,
        device=device,
    )
    fault_logit = torch.nn.Parameter(
        torch.tensor(float(initial_fault_logit), device=device)
    )
    optimizer = torch.optim.Adam((fault_logit,), lr=float(learning_rate))
    temporal_context = None
    suffix_context = None
    context_build_started = time.perf_counter()
    if estimator == TEMPORAL_ESTIMATOR:
        temporal_context = _make_temporal_quadratic_context(
            context,
            window=spec.window,
            state_dim=temporal_state_dim,
        )
    elif estimator in SUFFIX_KERNEL_ESTIMATORS:
        suffix_context = _make_suffix_kernel_context(
            context,
            route_kernel=SUFFIX_KERNEL_ESTIMATORS[estimator],
            window=spec.window,
            mismatch_scale=mismatch_scale,
            sketch_dim=suffix_sketch_dim,
            sketch_count=suffix_sketch_count,
            sketch_seed=suffix_sketch_seed,
        )
    _synchronize(device)
    context_build_ms = (time.perf_counter() - context_build_started) * 1000.0
    initial_gradient = None
    max_gradient = 0.0
    first_exact_step = None
    executed_steps = 0
    stalled = False
    initial_state = None
    initial_target_weight = None
    started = time.perf_counter()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output, route, target_weight = hard_forward_with_proxy(
            estimator,
            spec,
            context,
            fault_logit,
            logit_margin=logit_margin,
            scale=scale,
            mismatch_scale=mismatch_scale,
            temporal_state_dim=temporal_state_dim,
            temporal_context=temporal_context,
            suffix_sketch_dim=suffix_sketch_dim,
            suffix_sketch_count=suffix_sketch_count,
            suffix_sketch_seed=suffix_sketch_seed,
            suffix_context=suffix_context,
        )
        if initial_state is None:
            initial_state = route
            finite_weights = target_weight[torch.isfinite(target_weight)]
            initial_target_weight = (
                float(finite_weights.mean()) if finite_weights.numel() else None
            )
        exact = route.selected_positions.eq(context.target_positions)
        if bool(exact.all()) and first_exact_step is None:
            first_exact_step = step
        loss = (output - spec.target_values.to(device)).square().mean()
        if step == steps or bool(exact.all()):
            break
        loss.backward()
        gradient = float(fault_logit.grad.detach())
        if initial_gradient is None:
            initial_gradient = gradient
        max_gradient = max(max_gradient, abs(gradient))
        if abs(gradient) <= gradient_stall_tolerance:
            stalled = True
            break
        optimizer.step()
        executed_steps += 1
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    with torch.no_grad():
        final_query = make_query_logits(
            spec,
            fault_logit,
            logit_margin=logit_margin,
        )
        final_state = dense_hard_route(final_query, context, spec.window)
    assert initial_state is not None
    return {
        "estimator": estimator,
        "initial_fault_logit": float(initial_fault_logit),
        "final_fault_logit": float(fault_logit.detach()),
        "initial_gradient": initial_gradient,
        "max_abs_gradient": max_gradient,
        "initial_target_proxy_weight": initial_target_weight,
        "first_exact_step": first_exact_step,
        "executed_steps": executed_steps,
        "stalled": stalled,
        "train_success": bool(
            final_state.selected_positions.eq(context.target_positions).all()
        ),
        "initial_route_accuracy": float(
            initial_state.selected_positions.eq(context.target_positions).float().mean()
        ),
        "final_route_accuracy": float(
            final_state.selected_positions.eq(context.target_positions).float().mean()
        ),
        "initial_target_suffix_mean": float(
            initial_state.target_suffix_lengths.float().mean()
        ),
        "initial_exact_distractor_mean": float(
            initial_state.exact_distractor_counts.float().mean()
        ),
        "final_target_suffix_mean": float(
            final_state.target_suffix_lengths.float().mean()
        ),
        "final_max_distractor_suffix_mean": float(
            final_state.max_distractor_suffix_lengths.float().mean()
        ),
        "train_ms": elapsed_ms,
        "ms_per_executed_step": elapsed_ms / max(executed_steps, 1),
        "context_build_ms": context_build_ms,
    }


def streaming_hard_evaluate(
    spec: GateSpec,
    fault_logit: float,
    *,
    context_length: int,
    chunk_size: int,
    logit_margin: float,
    device: torch.device,
) -> dict[str, object]:
    if context_length < spec.train_context_length:
        raise ValueError("evaluation context cannot be shorter than training")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    fault = torch.tensor(float(fault_logit), device=device)
    query = make_query_logits(spec, fault, logit_margin=logit_margin)
    query_codes = _pack_signs(_hard_sign(query))
    target_codes = _pack_signs(spec.target_signs.to(device))
    target_positions = spec.target_positions.to(device)
    target_blocks = spec.target_blocks.to(device)
    probe_ids = torch.arange(spec.probes, device=device).view(-1, 1)

    best_lengths = torch.zeros(spec.probes, dtype=torch.int64, device=device)
    best_positions = torch.full_like(best_lengths, -1)
    target_lengths = torch.full_like(best_lengths, -1)
    exact_distractors = torch.zeros_like(best_lengths)
    strict_distractors = torch.zeros_like(best_lengths)
    strict_length = max(spec.window - 1, 0)
    started = time.perf_counter()
    for start in range(0, context_length, chunk_size):
        end = min(start + chunk_size, context_length)
        endpoints = torch.arange(start, end, device=device)
        running = torch.ones(
            spec.probes,
            end - start,
            dtype=torch.bool,
            device=device,
        )
        lengths = torch.zeros_like(running, dtype=torch.int64)
        for local_position in range(spec.window - 1, -1, -1):
            shift = spec.window - 1 - local_position
            positions = endpoints - shift
            valid = positions.ge(0)
            safe_positions = positions.clamp_min(0)
            phases = safe_positions.remainder(spec.window)
            blocks = torch.div(
                safe_positions,
                spec.window,
                rounding_mode="floor",
            )
            key_codes = target_codes[:, phases]
            flip_offsets = (
                blocks.view(1, -1) + 17 * spec.seed + 3 * probe_ids
            ).remainder(spec.content_bits)
            flip_bits = flip_offsets + spec.phase_bits
            should_flip = (
                valid.view(1, -1)
                & phases.view(1, -1).eq(0)
                & blocks.view(1, -1).ne(target_blocks.view(-1, 1))
            )
            key_codes = key_codes ^ torch.where(
                should_flip,
                torch.ones_like(flip_bits) << flip_bits,
                torch.zeros_like(flip_bits),
            )
            running &= valid.view(1, -1) & key_codes.eq(
                query_codes[:, local_position].view(-1, 1)
            )
            lengths += running

        target_in_chunk = (
            target_positions.ge(start) & target_positions.lt(end)
        )
        if bool(target_in_chunk.any()):
            rows = target_in_chunk.nonzero(as_tuple=False).squeeze(1)
            columns = target_positions[rows] - start
            target_lengths[rows] = lengths[rows, columns]
        target_mask = endpoints.view(1, -1).eq(target_positions.view(-1, 1))
        exact_distractors += (lengths.eq(spec.window) & ~target_mask).sum(dim=-1)
        strict_distractors += (
            lengths.eq(strict_length) & ~target_mask
        ).sum(dim=-1)

        chunk_best = lengths.amax(dim=-1)
        chunk_latest = torch.where(
            lengths.eq(chunk_best.view(-1, 1)) & chunk_best.view(-1, 1).gt(0),
            endpoints.view(1, -1),
            -torch.ones((), dtype=torch.int64, device=device),
        ).amax(dim=-1)
        replace = chunk_best.gt(best_lengths) | (
            chunk_best.eq(best_lengths) & chunk_latest.gt(best_positions)
        )
        best_lengths = torch.where(replace, chunk_best, best_lengths)
        best_positions = torch.where(replace, chunk_latest, best_positions)
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    success = best_positions.eq(target_positions)
    constructed_distractors = context_length // spec.window - 1
    return {
        "context_length": int(context_length),
        "candidate_count": int(context_length),
        "constructed_aligned_distractors": int(constructed_distractors),
        "route_accuracy": float(success.float().mean()),
        "all_routes_correct": bool(success.all()),
        "target_suffix_mean": float(target_lengths.float().mean()),
        "max_suffix_mean": float(best_lengths.float().mean()),
        "exact_distractor_mean": float(exact_distractors.float().mean()),
        "strict_distractor_mean": float(strict_distractors.float().mean()),
        "scan_ms": elapsed_ms,
        "million_candidates_per_second": (
            spec.probes * context_length / max(elapsed_ms, 1e-9) / 1000.0
        ),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _summarize(
    records: Sequence[dict[str, object]],
    windows: Sequence[int],
    estimators: Sequence[str],
    evaluation_lengths: Sequence[int],
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for estimator in estimators:
        estimator_summary = {}
        for window in windows:
            selected = [
                record
                for record in records
                if record["estimator"] == estimator and record["window"] == window
            ]
            first_steps = [
                float(record["training"]["first_exact_step"])
                for record in selected
                if record["training"]["first_exact_step"] is not None
            ]
            extrapolation = {}
            for length in evaluation_lengths:
                evaluations = [
                    next(
                        item
                        for item in record["evaluations"]
                        if item["context_length"] == length
                    )
                    for record in selected
                ]
                extrapolation[str(length)] = {
                    "all_correct_fraction": _mean(
                        [float(item["all_routes_correct"]) for item in evaluations]
                    ),
                    "mean_route_accuracy": _mean(
                        [float(item["route_accuracy"]) for item in evaluations]
                    ),
                    "mean_exact_distractors": _mean(
                        [float(item["exact_distractor_mean"]) for item in evaluations]
                    ),
                    "mean_scan_ms": _mean(
                        [float(item["scan_ms"]) for item in evaluations]
                    ),
                }
            estimator_summary[str(window)] = {
                "runs": len(selected),
                "train_success_fraction": _mean(
                    [float(record["training"]["train_success"]) for record in selected]
                ),
                "mean_initial_gradient": _mean(
                    [
                        float(record["training"]["initial_gradient"])
                        for record in selected
                        if record["training"]["initial_gradient"] is not None
                    ]
                ),
                "mean_first_exact_step_on_success": _mean(first_steps),
                "extrapolation": extrapolation,
            }
        summaries[estimator] = estimator_summary
    return summaries


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    records = []
    evaluation_cache: dict[tuple[int, int, bool, int], dict[str, object]] = {}
    for window in args.windows:
        for seed in args.seeds:
            spec = make_gate_spec(
                seed=seed,
                probes=args.probes,
                window=window,
                bits=args.bits,
                value_bits=args.value_bits,
                value_mode=args.value_mode,
                train_context_length=args.train_context_length,
            ).to(device)
            for estimator in args.estimators:
                training = train_condition(
                    spec,
                    estimator,
                    device=device,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    initial_fault_logit=args.initial_fault_logit,
                    logit_margin=args.logit_margin,
                    scale=args.scale,
                    mismatch_scale=args.mismatch_scale,
                    gradient_stall_tolerance=args.gradient_stall_tolerance,
                    temporal_state_dim=getattr(
                        args,
                        "temporal_state_dim",
                        TEMPORAL_DEFAULT_STATE_DIM,
                    ),
                    suffix_sketch_dim=getattr(args, "suffix_sketch_dim", 32),
                    suffix_sketch_count=getattr(args, "suffix_sketch_count", 2),
                    suffix_sketch_seed=(
                        getattr(args, "suffix_sketch_seed", 0) + seed
                    ),
                )
                solved = float(training["final_fault_logit"]) > 0.0
                evaluations = []
                for context_length in args.eval_context_lengths:
                    cache_key = (window, seed, solved, context_length)
                    evaluation = evaluation_cache.get(cache_key)
                    if evaluation is None:
                        evaluation = streaming_hard_evaluate(
                            spec,
                            float(training["final_fault_logit"]),
                            context_length=context_length,
                            chunk_size=args.chunk_size,
                            logit_margin=args.logit_margin,
                            device=device,
                        )
                        evaluation_cache[cache_key] = evaluation
                    evaluations.append(dict(evaluation))
                records.append(
                    {
                        "seed": seed,
                        "window": window,
                        "phase_bits": spec.phase_bits,
                        "content_bits": spec.content_bits,
                        "estimator": estimator,
                        "training": training,
                        "evaluations": evaluations,
                    }
                )
    return {
        "schema_version": 1,
        "objective": (
            "repair the oldest decisive suffix bit at 8K and preserve exact "
            "latest-longest routing under 64K/1M many-distractor extrapolation"
        ),
        "scope": (
            "controlled final-query-row estimator gate; not a language-model "
            "quality or throughput benchmark"
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "estimators": list(args.estimators),
        "windows": list(args.windows),
        "seeds": list(args.seeds),
        "probes": args.probes,
        "bits": args.bits,
        "value_bits": args.value_bits,
        "value_mode": args.value_mode,
        "train_context_length": args.train_context_length,
        "eval_context_lengths": list(args.eval_context_lengths),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_fault_logit": args.initial_fault_logit,
        "logit_margin": args.logit_margin,
        "scale": args.scale,
        "mismatch_scale": args.mismatch_scale,
        "temporal_state_dim": getattr(
            args,
            "temporal_state_dim",
            TEMPORAL_DEFAULT_STATE_DIM,
        ),
        "suffix_sketch_dim": getattr(args, "suffix_sketch_dim", 32),
        "suffix_sketch_count": getattr(args, "suffix_sketch_count", 2),
        "suffix_sketch_seed": getattr(args, "suffix_sketch_seed", 0),
        "records": records,
        "summaries": _summarize(
            records,
            args.windows,
            args.estimators,
            args.eval_context_lengths,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATOR_CHOICES,
        default=list(ESTIMATORS),
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 32],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-bits", type=int, default=8)
    parser.add_argument(
        "--value-mode",
        choices=VALUE_MODES,
        default="balanced_binary",
    )
    parser.add_argument("--train-context-length", type=int, default=8192)
    parser.add_argument(
        "--eval-context-lengths",
        nargs="+",
        type=int,
        default=[8192, 65536, 1048576],
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--initial-fault-logit", type=float, default=-0.25)
    parser.add_argument("--logit-margin", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument(
        "--temporal-state-dim",
        type=int,
        default=TEMPORAL_DEFAULT_STATE_DIM,
    )
    parser.add_argument("--suffix-sketch-dim", type=int, default=32)
    parser.add_argument("--suffix-sketch-count", type=int, default=2)
    parser.add_argument("--suffix-sketch-seed", type=int, default=0)
    parser.add_argument("--gradient-stall-tolerance", type=float, default=1e-14)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(args.initial_fault_logit) or args.initial_fault_logit >= 0.0:
        raise ValueError("initial_fault_logit must be finite and negative")
    if not math.isfinite(args.logit_margin) or args.logit_margin <= 0.0:
        raise ValueError("logit_margin must be finite and positive")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if args.suffix_sketch_dim < 1 or (
        args.suffix_sketch_dim & (args.suffix_sketch_dim - 1)
    ):
        raise ValueError("suffix_sketch_dim must be a positive power of two")
    if args.suffix_sketch_count < 1:
        raise ValueError("suffix_sketch_count must be positive")
    if any(length < args.train_context_length for length in args.eval_context_lengths):
        raise ValueError("evaluation lengths cannot be shorter than training")
    if args.train_context_length not in args.eval_context_lengths:
        raise ValueError("evaluation lengths must include the training length")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    report = run_matrix(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.summary_only:
        print(json.dumps(report["summaries"], indent=2, allow_nan=False))
    else:
        print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
