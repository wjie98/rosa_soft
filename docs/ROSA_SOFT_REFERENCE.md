# RosaSoft Reference Guide

`rosa_soft/soft_reference.py` is the semantic oracle for the CUDA operator.
It favors direct tensor equations over runtime efficiency.

## 1. Public Entry Points

```python
rosa_soft_reference(
    query,
    key,
    value,
    *,
    max_suffix_length=32,
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)

rosa_soft_varlen_reference(
    query,
    key,
    value,
    cu_seqlens,
    ...
)
```

With `dropout_p=0`, calls are deterministic and consume no PyTorch RNG state.
With dropout enabled, the public call draws one scalar seed only when a
backward graph is required. There is no public seeded or explicit-mask
variant.

## 2. Function Map

| Function | Meaning |
| --- | --- |
| `_hard_sign` | Binary `x > 0` quantization. |
| `_HardSignWithSoftsignVjp` | Hard numerical symbol with softsign derivative. |
| `_causal_route_mask` | Null plus all valid causal route columns. |
| `_pairwise_exact_symbol_match` | Exact all-bit Q/K equality for hard forward. |
| `_pairwise_soft_match_gates` | Normalized Hamming mismatch and exponential gate. |
| `_suffix_prefix_product_scores` | Complete expected matched-prefix length. |
| `_suffix_score_utility` | Fixed normalized square-root transform from raw suffix evidence to route score. |
| `_select_latest_longest_routes` | Exact hard winner with latest tie break. |
| `_route_probabilities` | Null-aware candidate-normalized dense softmax. |
| `_apply_attention_dropout` | Counter-based post-softmax inverted dropout. |
| `_build_vjp_carrier` | Probability-weighted signed value carrier. |
| `_HardForwardSoftVjpReference` | Exact forward plus surrogate custom VJP. |

These helpers are private. Tests may inspect them only when validating the
mathematical decomposition, not as a supported API.

## 3. Route Tensor Convention

Materialized route tensors have shape:

```text
[B, H, query_position, route_index]
```

Column zero is null. Column `a > 0` compares against key position `a - 1` and
routes to value position `a`.

For row `t`, valid columns are:

```text
0, 1, ..., t.
```

This shifted convention prevents a token from routing to itself using its own
key: the newest non-null route at row `t` matches key `t - 1` and returns
value `t`.

## 4. Exact Forward Construction

`_hard_route_forward` performs:

```text
exact local equality
  -> diagonal suffix prefix sums
  -> largest exact length
  -> largest route index among ties
  -> gather hard-signed value
```

The null column contains no local match. If the largest exact suffix length is
zero, the selected index is explicitly reset to zero.

## 5. Surrogate Construction

Backward rebuilds:

```text
local_match_gates
  = exp(-mismatch_scale * normalized_hamming)

raw_suffix_scores
  = sum of diagonal prefix products up to max_suffix_length

route_scores
  = (sqrt(2) + 1) * (sqrt(1 + raw_suffix_scores) - 1)

route_scores[..., 0]
  = 0.5

route_probabilities
  = softmax(route_scores * scale - nonnull_log(candidate_count))

dropped_probabilities
  = route_probabilities * Bernoulli(1 - dropout_p) / (1 - dropout_p)

vjp_carrier
  = sum of dropped_probabilities * hard-signed values with softsign VJP
```

`torch.autograd.grad(vjp_carrier, inputs, grad_output)` supplies the custom
backward. The returned forward tensor is never replaced by `vjp_carrier`.

## 6. Saved State

The reference custom function saves Q, K, value, and an empty-or-scalar dropout
seed plus static Python controls. It recomputes all route tensors and the mask
in backward. Saving a
quadratic probability or score matrix would make the reference easier to
step through but would needlessly increase activation memory.

The function is once differentiable. Higher-order derivatives are not part of
the contract.

## 7. Inspection

`rosa_soft.testing.inspect_rosa_soft` materializes development state:

```text
hard_output
selected_route_indices
exact_suffix_lengths
soft_suffix_scores
route_probabilities
```

The reported probabilities are pre-dropout. `soft_suffix_scores` deliberately
reports raw prefix-product evidence `S`; `route_scores` is the masked `U(S)`
used by softmax. Inspection is deterministic and does not sample a training
mask.

`rosa_soft.diagnostics.summarize_rosa_soft` reduces that state into scalar
health metrics. Neither module participates in the production hot path.

Inspection names describe the current estimator. The old `proxy_scores`,
random uniforms, sampled gates, and tiered scores no longer exist.

## 8. Reference Versus CUDA

| Property | PyTorch reference | CUDA |
| --- | --- | --- |
| Forward semantics | Exact hard ROSA | Exact hard ROSA |
| Backward support | Every causal route | Every causal route |
| Estimator | Minimal VJP with optional dropout | Same equations and mask hash |
| Route storage | Materialized quadratic tensors | Online scan/recompute |
| Accumulation | Reference dtype, FP32 for half/bfloat16 | FP32 |
| Intended use | Tests and small experiments | Training |

Parity tests compare hard output exactly and gradients within dtype-specific
tolerances.

## 9. Useful Invariants

Tests should assert:

1. Changing surrogate controls cannot change hard output.
2. At `dropout_p=0`, repeated calls do not consume RNG state and return equal
   VJPs.
3. `mismatch_scale -> infinity` moves raw suffix evidence toward exact suffix
   lengths and route scores toward their normalized square-root utility.
4. At `dropout_p=0`, every causal value candidate is structurally eligible
   for FP32 credit; finite precision may still underflow a tiny probability.
5. No route or gradient crosses a packed segment boundary.
6. The latest exact route wins hard ties.
7. Fixed seeds reproduce dropout VJPs; changing a seed never changes hard
   output.
8. The null value is always zero and value position zero receives no route
   gradient.

## 10. Deliberate Inefficiencies

The oracle allocates pairwise `[B,H,T,T,D]` broadcast intermediates and
quadratic score/probability tensors. Do not complicate it with CUDA-style
tiling or candidate pruning. Its job is to state the estimator clearly.

When memory becomes limiting, reduce the test shape or use CUDA. A second
chunked reference is only justified if it preserves an equally inspectable
equation-level oracle.
