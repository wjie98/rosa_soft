# RosaAnchor Developer Report

This document describes the public RosaAnchor training proxy and the hard
ROSA runtime shipped in this repository.

## Goals

ROSA is a discrete suffix-routing layer. The production inference path should
run the hard suffix automaton, while the training path needs a dense enough
surrogate gradient for Q/K/V. RosaAnchor is the current training proxy. Its
forward pass is intentionally aligned with hard ROSA suffix matching, but keeps
a soft candidate distribution so gradients can flow through Q/K and V.

The design constraints are:

- preserve soft gradients to V;
- keep the route score close to hard suffix matching as scale grows;
- avoid extra user-facing controls where a fixed rule is enough;
- expose telemetry so scale can be adjusted during training;
- keep inference state in `RosaRuntime`, not in Python `__del__`-managed raw
  pointers.

## Tensor Layout

`rosa_anchor_ops(query, key, value, ...)` uses:

- `query`: `[B, T, H, D]`
- `key`: `[B, T, H, D]`
- `value`: `[B, T, H_v, D_v]`

`H % H_v == 0`, so value heads are shared grouped-query style.

`D` is the Q/K bit width. The CUDA proxy supports `1 <= D <= 32`.
`RosaRuntime` supports packed hard runtime Q/K/V widths `1..8`.

## Quantization

The route bit for a scalar activation is:

```text
bit(x) = 1[x > 0]
sign(x) = +1 if x > 0 else -1
```

Forward values use sign bits. Backward uses a softsign-style STE derivative:

```text
d sign(x) / dx ~= 1 / (1 + |x|)^2
```

This keeps gradients useful near zero and naturally damps very large Q/K/V
activations.

## Candidate Set

For query token `i`, candidate `j` means "the value after the key prefix ending
at `j - 1`". Candidates are `1 <= j <= i`; `j = 0` is represented by a sink/null
slot with zero value contribution.

For each candidate, only the suffix window is scanned:

```text
L(i, j) = min(window_size, i + 1, j)
```

Offset `l = 0` compares `q_i` with `k_{j-1}`. Offset `l = 1` compares
`q_{i-1}` with `k_{j-2}`, and so on.

## Weighted Bit Mismatch

For a Q/K pair at offset `l`, the binary mismatch mask is:

```text
mismatch_d = bit(q_{i-l,d}) XOR bit(k_{j-1-l,d})
```

If all bits match, the pair match weight is exactly `1`.

For mismatched bits, RosaAnchor uses confidence-weighted Hamming cost. The
confidence of one scalar is:

```text
c(x) = tanh(|x|)
```

The per-bit mismatch weight is:

```text
w_d = 0.25 + 0.75 * min(c(q_{i-l,d}), c(k_{j-1-l,d}))
```

The pair match is:

```text
mu_l(i, j) = exp(-lambda * sum_d mismatch_d * w_d)
```

The lower bound `0.25` keeps mismatches meaningful even when activations are
near zero; high-confidence mismatches are penalized more heavily.

## Multiplicative Suffix Score

The suffix score uses the adaptive multiplicative recurrence:

```text
p_0 = 1
p_{l+1} = p_l * mu_l(i, j)
score(i, j) = sum_{l=0}^{L(i,j)-1} p_{l+1}
```

This is the key hard-aligned property: once a mismatch is strong, later suffix
positions are discounted multiplicatively. As scale grows and the window is
large enough, the highest-scoring route approaches the hard ROSA route. For
exact full-window evaluation, use `logit_epsilon=0`.

The match sharpness is derived from the current scale and window:

```text
lambda = log(window_size) + 1.5 * log(1 + |scale|)
```

This keeps the suffix recurrence sharper when the selector softmax is sharper.

## Candidate Logits

Each candidate logit is:

```text
z(i, j) = scale * (score(i, j) - sink_threshold + tie(i, j))
```

The fixed sink threshold is:

```text
sink_threshold = 0.5
```

The sink/null logit is `0`, and contributes no value. It is equivalent to an
attention sink slot used to represent no usable suffix match.

The recency tie-break is intentionally weak:

```text
tie(i, j) = 0.25 * j / max(i, 1)
```

It only breaks near-identical routes in favor of recent matches. It should not
be used as a major position-bias term.

The candidate distribution is:

```text
P(null | i) = exp(0) / Z
P(j | i) = exp(z(i, j)) / Z
Z = exp(0) + sum_{1 <= j <= i} exp(z(i, j))
```

The output is:

```text
out_{i,h,d} = sum_j P(j | i,h) * sign(value_{j,h_v,d})
```

The null slot contributes zero.

## Early Stop

When `logit_epsilon > 0`, a candidate suffix scan can stop early if the
remaining maximum possible logit mass is too small:

```text
|scale| * remaining_offsets * current_product <= logit_epsilon
```

Set `logit_epsilon=0` for exact suffix-window scoring. Nonzero values reduce
work on easy rows and should be validated on the target shape.

## Backward Path

Backward reuses saved packed Q/K bits and row softmax statistics when possible.
The gradient is accumulated through:

- the selector softmax;
- the multiplicative suffix recurrence;
- the confidence-weighted mismatch terms;
- V sign values through the softsign STE derivative.

Q/K directional damping is an optional guardrail:

```text
qk_damper_strength in [0, 1]
```

If a Q/K gradient would increase an already large activation magnitude, the
gradient is smoothly reduced between `|x| = 2` and `|x| = 4`. This is intended
to slow saturation without changing the forward function.

## Telemetry

`rosa_anchor_ops(..., return_telemetry=True)` returns `(output, telemetry)`.
The main fields are:

- `top_prob`: top probability including the null slot;
- `candidate_top_prob`: top probability among non-null candidates;
- `candidate_mass`: total non-null probability mass;
- `null_prob`: sink/null probability;
- `entropy_norm`: normalized row entropy;
- `effective_window_mean`: mean scanned suffix offsets;
- `truncated_fraction`: early-stop fraction.

Use telemetry probes periodically rather than on every hot training step.

## Scale Adjustment

The default operator scale is a shape-based estimate. For training, use
`RosaAnchorScaleController` with a target top probability. The recommended
starting target is `0.8`:

```python
from rosa_soft import RosaAnchorScaleConfig, RosaAnchorScaleController

controller = RosaAnchorScaleController(
    RosaAnchorScaleConfig(
        seq_len=ctx_len,
        qk_bits=qk_bits,
        window_size=window_size,
        target_top_prob=0.8,
        update_interval=100,
    )
)
```

At probe steps:

```python
output, telemetry = rosa_anchor_ops(
    q, k, v,
    window_size=window_size,
    scale=controller.scale,
    return_telemetry=True,
)
controller.observe(telemetry)
controller.advance()
```

The update is logit-space proportional control:

```text
scale <- scale * exp(rate * (logit(target_top_prob) - logit(ema(top_prob))))
```

This keeps tuning tied to a distribution-level indicator rather than raw
sequence length or activation magnitude.

## Parameter Guidance

`window_size` is the main semantic control. Larger windows make the proxy closer
to long-suffix ROSA behavior, but increase work. Use the smallest value that
captures the task's useful suffix length. Current CUDA backward supports
`window_size <= 512`.

`scale` controls selector sharpness. Do not push it to infinity during training;
that creates peaky gradients and can remove exploration. Use telemetry-guided
adjustment instead.

`logit_epsilon` controls early stop. Start with `0` for validation. Use
`1e-3` only after checking that `truncated_fraction` and task metrics are stable.

`qk_damper_strength` is optional. Start at `0`. If Q/K activations saturate and
route entropy collapses, try `0.1..0.3`.

## Current Status

The public release contains:

- CUDA RosaAnchor training proxy with forward/backward and telemetry;
- hard `RosaRuntime` with explicit lifetime management;
- compact runtime backend plus map backend for correctness comparison;
- RWKV7+ROSA overlap example;
- scale-controller training skeleton.

Release smoke results on the local test shape:

```text
RosaRuntime tests passed
compact runtime: about 1.5x faster than map backend on B=4,T=256,H=8,Hv=2,bits=4
RosaAnchor CUDA forward/backward smoke passed
```

These are smoke checks, not a replacement for downstream model validation.
