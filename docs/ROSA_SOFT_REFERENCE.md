# Hard-Forward RosaSoft Reference and Kernel Map

`rosa_soft_reference` is the executable semantic oracle for
`rosa_soft`. It runs on CPU or CUDA without the compiled extension:

```bash
ROSA_BUILD_EXTENSION=0 pip install -e .
```

The reference intentionally materializes candidate and random tensors. Its
purpose is derivation, inspection, and parity testing, not long-context
throughput.

## 1. Public Surface

The training API has one semantic control and two backward controls:

```python
output = rosa_soft_reference(
    query_logits,
    key_logits,
    payload_logits,
    max_suffix_length=32,
    route_temperature=1.0,
    mismatch_penalty=3.0,
)
```

It always returns one tensor. Random generators, sampled uniforms, and
candidate state are not accepted by the public training function. Tests and
diagnostics use separate entry points:

```python
from rosa_soft.testing import inspect_rosa_soft

output, inspection = inspect_rosa_soft(
    query_logits,
    key_logits,
    payload_logits,
    generator=torch.Generator().manual_seed(0),
)
```

The testing hook `rosa_soft_reference_with_uniforms` accepts explicit uniforms
solely so CUDA can be checked against the exact same random sample.

## 2. Forward Path

The reference forward is implemented by these stages:

| Function | Responsibility |
| --- | --- |
| `_hard_sign` | Map positive logits to `+1` and all others to `-1`. |
| `_pairwise_exact_symbol_match` | Compare complete Q/K codes for every causal pair. |
| `_suffix_prefix_product_scores` | Accumulate consecutive exact matches along suffix diagonals. |
| `_select_latest_longest_routes` | Select longest suffix, latest route on ties, or null. |
| `_expand_payload_heads` | Apply grouped-query payload-head sharing. |
| `_gather_routed_payloads` | Return one hard payload symbol or exact zero. |

`_HardForwardSoftVjpReference.forward` calls only this hard pipeline. It saves
the original Q/K/payload tensors and one mismatch sample for backward, but no
surrogate output is evaluated in forward.

If gradients are disabled, `rosa_soft_reference` bypasses the custom
autograd function and does not sample random numbers. This keeps evaluation
deterministic and avoids changing the caller's RNG stream.

## 3. Backward Path

Backward reconstructs a differentiable surrogate with fixed stages:

| Function | Responsibility |
| --- | --- |
| `_hard_sign_with_softsign_vjp` | Exact hard sign value with softsign derivative. |
| `_pairwise_stochastic_match_gates` | Cubic mismatch values and hard-Hamming local VJP. |
| `_suffix_prefix_product_scores` | Multiplicative suffix score for every route. |
| `_masked_route_scores` | Route scores plus fixed null score `0.5`. |
| `_route_probabilities` | Plain route softmax divided by `route_temperature`. |
| `_build_vjp_surrogate` | Route VJP through hard payload and payload VJP through detached probabilities. |

`torch.autograd.grad` computes the reference VJP from this surrogate. The
custom autograd boundary returns those gradients for Q/K/payload while preserving
the unrelated hard forward value.

The estimator is first-order only. `_HardForwardSoftVjpReference.backward` is
marked `once_differentiable`; higher-order gradients are not part of the
contract.

## 4. Materialized Reference State

For shapes:

```text
Q, K: [B, T, H, D]
V:    [B, T, H_v, D_v]
```

the detached inspection state contains:

| Field | Shape |
| --- | --- |
| `exact_suffix_lengths` | `[B, H, T, T]` |
| `proxy_scores` | `[B, H, T, T]` |
| `route_scores` | `[B, H, T, T]`, reconstructed on access |
| `route_probabilities` | `[B, H, T, T]` |
| `valid_routes` | `[T, T]` |
| `selected_routes` | `[B, H, T]` |

The temporary mismatch uniforms still cost `O(B H T^2 D)` while an inspection
is built, but are released before the snapshot is returned. Persistent
inspection state is `O(B H T^2)`. The production CUDA path materializes
neither tensor class.

## 5. Fixed Mismatch Estimator

For each hard mismatching bit and one sample `u`, the numerical penalty is:

```text
alpha = 1 - 0.5 * u^3
```

The reference accumulates only two logical Hamming values per local pair:

```text
h = sum hard_mismatch
H = sum hard_mismatch * alpha
```

Its local numerical gate is:

```text
exp(-mismatch_penalty * H)
```

The detach construction in `_local_match_gate_with_hard_vjp` leaves that
value unchanged while making its local derivative proportional to:

```text
mismatch_penalty * exp(-mismatch_penalty * h)
```

No Q/K magnitude appears in `h` or `H`. Magnitude affects only the
fixed softsign derivative. This blocks a direct numerical confidence channel
while still giving logits a sign-boundary gradient.

CUDA needs the ratio between the hard-Hamming derivative and the numerical gate
when differentiating a suffix product. It evaluates the stable form directly:

```text
gate = exp(-lambda * H)
ratio = exp(-lambda * (h - H))
```

Because `H <= h`, `ratio` is in `(0,1]`. This avoids dividing by a tiny gate.

## 6. CUDA Mapping

The CUDA implementation maps reference tensors to recomputation:

| Reference operation | CUDA implementation |
| --- | --- |
| Hard Q/K signs | `pack_sign_bits_kernel`, one `int32` word per `(B,T,H)`. |
| Exact suffix matrix | `hard_forward_kernel` scans routes and keeps only `(best_length,best_route)`. |
| Random tensor | One seed plus counter reconstruction in `counter_uniform`. |
| Local mismatch tensor | `stochastic_local_match_gate` reconstructs samples and keeps only `h/H` accumulators. |
| Proxy score matrix | `proxy_suffix_score` evaluates one route in registers; the selected execution plan may cache one row. |
| Probability matrix | Stable block softmax reductions; probabilities are always consumed without a materialized matrix. |
| Q/K surrogate graph | Closed-form VJP in `accumulate_local_match_qk_vjp`. |
| Payload surrogate graph | Probability-weighted softsign derivative with FP32 atomics. |

Backward has exactly three compile-time execution plans:

| Plan | Shared-memory contents | Intended case |
| --- | --- | --- |
| `CacheRouteScores` | softmax stats, row `grad_output`, Q suffix, one score row | score row is the smaller reusable payload |
| `ReduceKeyGradInShared` | softmax stats, row `grad_output`, Q suffix, one K-gradient tile | K tile is the smaller reusable payload |
| `MinimalSharedMemory` | softmax stats only | neither optimized layout fits the portable 48 KiB limit |

`make_backward_shared_layout` computes every offset and the total byte count
for both host plan selection and device pointer binding. Derived route
utilities are deliberately recomputed; there is no utility cache. There are
three soft-backward template instances per dtype, not a Cartesian product of
runtime cache booleans.

The autograd context saves:

```text
Q, K, payload
packed Q bits
packed K bits
one int64 seed
max_suffix_length, route_temperature, mismatch_penalty
```

It does not save hard route matrices, proxy matrices, probabilities, random
samples, or telemetry.

## 7. Counter Randomness

CUDA assigns one deterministic counter to every possible mismatch sample:

```text
counter = ((((b * H + h) * T + q_pos) * T + k_pos) * D + bit)
```

`splitmix64(seed + (counter + 1) * stride)` produces 64 random-looking bits.
The high 24 bits are mapped to:

```text
u = (mantissa + 0.5) / 2^24
```

Thus `u` is strictly inside `(0,1)`. Repeated evaluation of the same local pair
reconstructs the same sample without memory traffic or mutable RNG state.

The current implementation uses a stable SplitMix counter hash rather than
PyTorch's internal Philox offset API. This keeps parity simple and decouples
the kernel from private generator internals. If a future integration requires
exact participation in a distributed Philox stream, the seed reservation can
change while retaining the same index contract.

## 8. Numerical Rules

The oracle and kernel share these edge rules:

- sign zero is `-1`;
- null output is positive zero;
- invalid future routes have probability zero;
- null score is exactly `0.5`;
- hard ties select the latest route;
- proxy ties have no recency bias;
- low temperature selects the proxy winner, which need not equal the hard
  latest-longest route at finite mismatch penalty;
- FP16/BF16 proxy arithmetic is promoted to FP32;
- a fixed seed reconstructs fixed samples, but CUDA atomic gradient reductions
  are numerically rather than bitwise deterministic;
- Q/K bit width is restricted to `1..32`;
- `route_temperature` and `mismatch_penalty` must be finite and positive;
- the CUDA inverse temperature and mismatch penalty must be positive normal
  FP32 values;
- `max_suffix_length / route_temperature` must fit in FP32, so a finite
  inverse temperature cannot overflow score logits at the configured horizon;
- `max_suffix_length` may exceed `T` and is clamped by each candidate's valid suffix.

These details are parity requirements, not implementation suggestions.

## 9. Validation Strategy

The tests are split by failure domain.

### Hard semantics

- compare suffix lengths to nested Python loops;
- force all-match ties and all-mismatch null rows;
- vary random samples, route temperature, and mismatch penalty while requiring identical
  forward output;
- test a row with more than 128 candidates to exercise CUDA block striding.

### Estimator invariants

- compare cubic gate values to the scalar formula;
- change Q/K magnitudes without changing signs and require equal proxy values;
- change random samples and require equal hard-Hamming local gradients;
- require route_temperature to change only route probabilities;
- require mismatch penalty to change leakage but not exact local matches;
- require soft payload credit on hard-null rows.

### Kernel parity

Tests reproduce the CUDA SplitMix counter tensor in Python, run the private
reference and CUDA entry points with the same seed, then compare:

```text
hard output
grad_query
grad_key
grad_payload
```

Coverage includes grouped heads, `D=1..32`, FP32, FP16, BF16, non-contiguous
inputs, singleton sequences, and static-control changes. Current reproduced
results and their environment belong in `validation/latest.json`, not in this
semantic specification.

### End-to-end fitting

`examples/fit_soft_reference.py` accepts:

```bash
python examples/fit_soft_reference.py --operator reference --device cuda
python examples/fit_soft_reference.py --operator cuda --device cuda
```

Different estimators consume different random streams, so their optimization
trajectories need not match step by step. The parity test, not identical fit
curves, establishes estimator equivalence. The fitting gate separately checks
whether dense surrogate credit can train the exact hard forward route.

## 10. Complexity and Performance

The reference intentionally has quadratic route and random memory. CUDA
reconstructs random values from counters and keeps persistent estimator state
linear in `T`. Compute remains quadratic because every valid causal route is
visited. This is a semantic requirement: RosaSoft trains sparse hard routing
with dense backward credit. Performance work may change tiling, reductions,
recomputation, and shared-memory plans, but not route support.

Use `benchmarks/rosa_soft.py` for measurements. Report the exact commit,
hardware, software stack, shape, dtype, warmup, repeat count, and data pattern;
do not carry old timings into this design document.

## 11. Removed Mechanisms

The rewrite removed these mechanisms from the operator:

| Removed mechanism | Reason |
| --- | --- |
| Dynamic route-temperature and mismatch-penalty schedules | Task-level policy, not operator semantics; coupling caused unstable ablations. |
| Window-derived mismatch penalty | Configured `W` can be a loose upper bound and is not actual suffix-tail amplification. |
| Q/K magnitude confidence | Creates a numerical side channel and duplicates STE conditioning. |
| Hard-winner margin in backward | Biases credit toward the current discrete route and suppresses exploration. |
| Recency score | Hard ties already resolve discretely; soft recency changes the proxy objective. |
| Optional mismatch width | The cubic range is now part of the fixed estimator. |
| Hard/soft payload modes | Distributed soft payload credit is retained because hard-selected payload failed the full-model fit gate. |
| Early-stop epsilon | Approximation policy was mixed into semantic scoring. |
| Q/K damper | Optimizer policy and saturation diagnosis belong outside the operator. |
| Hot-path telemetry | Candidate matrices and reductions damaged the production ABI. |

This is the main simplification: variability that defines a different
estimator is fixed; task-level tuning remains outside; only three useful
run-level controls remain public.

## 12. Engineering Boundary

Both backward implementations visit every causal route. There is no candidate
index, specialized `D/W` semantic branch, quadratic CUDA workspace, or stored
random tensor. Candidate pruning remains prohibited: an exact suffix index may
optimize hard forward, but it must never choose backward support.
