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
    query,
    key,
    value,
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
    query,
    key,
    value,
    generator=torch.Generator().manual_seed(0),
)
```

The testing hook `rosa_soft_reference_with_noise` accepts an explicit noise
tensor solely so CUDA can be checked against the exact same random sample.

## 2. Forward Path

The reference forward is implemented by these stages:

| Function | Responsibility |
| --- | --- |
| `_hard_sign` | Map positive logits to `+1` and all others to `-1`. |
| `_pairwise_exact_local_match` | Compare complete Q/K codes for every causal pair. |
| `_diagonal_suffix_sum` | Accumulate consecutive exact matches along suffix diagonals. |
| `_select_hard_actions` | Select longest suffix, latest action on ties, or null. |
| `_expand_value_heads` | Apply grouped-query value-head sharing. |
| `_gather_action_values` | Return one hard V symbol or exact zero. |

`_RosaSoftReferenceFunction.forward` calls only this hard pipeline. It saves
the original Q/K/V tensors and one mismatch sample for backward, but no
surrogate output is evaluated in forward.

If gradients are disabled, `rosa_soft_reference` bypasses the custom
autograd function and does not sample random numbers. This keeps evaluation
deterministic and avoids changing the caller's RNG stream.

## 3. Backward Path

Backward reconstructs a differentiable surrogate with fixed stages:

| Function | Responsibility |
| --- | --- |
| `_sign_with_softsign_jacobian` | Exact hard sign value with softsign derivative. |
| `_pairwise_proxy_local_match` | Cubic mismatch values and anchored VJP. |
| `_diagonal_suffix_sum` | Multiplicative suffix score for every action. |
| `_allocation_scores` | Candidate scores plus fixed null score `0.5`. |
| `_proxy_probabilities` | Plain action softmax divided by `route_temperature`. |
| `_surrogate_output` | Route VJP through hard V and value VJP through detached probabilities. |

`torch.autograd.grad` computes the reference VJP from this surrogate. The
custom autograd boundary returns those gradients for Q/K/V while preserving
the unrelated hard forward value.

The estimator is first-order only. `_RosaSoftReferenceFunction.backward` is
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
| `hard_lengths` | `[B, H, T, T]` |
| `proxy_scores` | `[B, H, T, T]` |
| `allocation_scores` | `[B, H, T, T]` |
| `probabilities` | `[B, H, T, T]` |
| `action_mask` | `[T, T]` |
| `hard_actions` | `[B, H, T]` |
| `mismatch_noise` | `[B, H, T, T-1, D]` |

This makes every estimator component observable, but its random tensor alone
costs `O(B H T^2 D)`. That cost is acceptable for a correctness oracle and
explicitly forbidden in the CUDA design.

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

The detach construction in `_relaxed_gate_with_hard_jacobian` leaves that value
unchanged while making its local derivative proportional to:

```text
mismatch_penalty * exp(-mismatch_penalty * h)
```

No Q/K magnitude appears in `h` or `H`. Magnitude affects only the
fixed softsign derivative. This blocks a direct numerical confidence channel
while still giving logits a sign-boundary gradient.

CUDA needs the ratio between the anchored derivative and the numerical gate
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
| Exact suffix matrix | `hard_forward_kernel` scans candidates and keeps only `(best_length,best_action)`. |
| Random tensor | One seed plus counter reconstruction in `counter_uniform`. |
| Local mismatch tensor | `local_gate` reconstructs samples and keeps only `h/H` accumulators. |
| Proxy score matrix | `proxy_score` evaluates one action in registers; the selected execution plan may cache one row. |
| Probability matrix | Stable block softmax reductions; probabilities are always consumed without a materialized matrix. |
| Q/K surrogate graph | Closed-form VJP in `accumulate_qk_vjp`. |
| V surrogate graph | Probability-weighted softsign derivative with FP32 atomics. |

Backward has exactly three compile-time execution plans:

| Plan | Shared-memory contents | Intended case |
| --- | --- | --- |
| `ScoreCached` | softmax stats, row `grad_output`, Q suffix, one score row | score row is the smaller reusable payload |
| `KeyReduced` | softmax stats, row `grad_output`, Q suffix, one K-gradient tile | K tile is the smaller reusable payload |
| `Generic` | softmax stats only | neither optimized layout fits the portable 48 KiB limit |

`make_backward_shared_layout` computes every offset and the total byte count
for both host plan selection and device pointer binding. Derived action
utilities are deliberately recomputed; there is no utility cache. There are
three soft-backward template instances per dtype, not a Cartesian product of
runtime cache booleans.

The autograd context saves:

```text
Q, K, V
packed Q bits
packed K bits
one int64 seed
max_suffix_length, route_temperature, mismatch_penalty
```

It does not save hard action matrices, proxy matrices, probabilities, random
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
- invalid future actions have probability zero;
- null score is exactly `0.5`;
- hard ties select the latest action;
- proxy ties have no recency bias;
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
- vary random samples, route_temperature, and lambda while requiring identical
  forward output;
- test a row with more than 128 candidates to exercise CUDA block striding.

### Estimator invariants

- compare cubic gate values to the scalar formula;
- change Q/K magnitudes without changing signs and require equal proxy values;
- change random samples and require equal anchored local gradients;
- require route_temperature to change only action probabilities;
- require lambda to change mismatch leakage but not exact local matches;
- require soft V credit on hard-null rows.

### Kernel parity

Tests reproduce the CUDA SplitMix counter tensor in Python, run the private
reference and CUDA entry points with the same seed, then compare:

```text
hard output
grad_query
grad_key
grad_value
```

Coverage includes grouped heads, `D=1..32`, FP32, FP16, BF16, non-contiguous
inputs, singleton sequences, and static-control changes. The recorded FP32
probe has bit-exact forward output and `5.96e-8` maximum absolute gradient
error.

### End-to-end fitting

`examples/fit_soft_reference.py` accepts:

```bash
python examples/fit_soft_reference.py --operator reference --device cuda
python examples/fit_soft_reference.py --operator cuda --device cuda
```

With the default 1000-step repeated-motif task:

| Operator | Final hard CE | Accuracy | Time per step |
| --- | ---: | ---: | ---: |
| Reference | `4.78e-4` | `100%` | `12.73 ms` |
| CUDA, current seed 0 | `3.83e-4` | `100%` | `3.13 ms` |
| CUDA, current seed 2 | `3.39e-4` | `100%` | `3.02 ms` |
| Historical bitflip | `1.61e-4` | `100%` | not remeasured |

Different estimators consume different random streams, so their optimization
trajectories need not match step by step. The parity test, not identical fit
curves, establishes mathematical equivalence. Model/data seed 1 remains a hard
failure case: the current CUDA run ends near `0.175` CE and `87.5%` accuracy.
In contrast, four perturbation seeds with fixed model/data seed 0 all reach
`100%`, so that failure is not attributed only to mismatch RNG variance.

## 10. Complexity and Performance

The reference intentionally has quadratic candidate and random memory. CUDA
reconstructs random values from counters and keeps persistent estimator state
linear in `T`. The completed dense-kernel pass changed time, not support:

| RTX 3070 FP32 | `T=256` | `T=512` | `T=1024` | `T=1152` | `T=1536` | `T=2048` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pre-pass CUDA train step | `3.018` | `10.349` | `36.684` | not recorded | not recorded | not recorded |
| Before cache simplification | `1.500` | `5.450` | `19.196` | `28.104` | `49.935` | not recorded |
| Three-plan isolated probe | `1.482` | `5.610` | `19.354` | `24.745` | `51.970` | not recorded |
| Final long-run probe | `1.528` | `6.002` | `20.426` | `26.132` | `52.390` | `93.387` |
| Peak operator MiB | `0.118` | `0.235` | `0.469` | `0.528` | `0.704` | `0.938` |

Benchmark shape: `B=1,H=4,D=8,H_v=2,D_v=8,W=32`, 30 warmups and
100 measured forward/backward steps for the final row. The earlier isolated
rows used 20 warmups and 150 measured steps. Prolonged benchmark runs changed
the observed boost clock enough to move absolute times by roughly 5%; the
reliable plan-level result is that `T=1152` improved from `29.307 ms` under the
old planner to `24.745 ms` in the immediate three-plan A/B. Runtime remains
quadratic because every valid causal action is still visited.

The final soft kernel has no local-memory spills. On SM86,
`ScoreCached/KeyReduced/Generic` use `60 / 56-62 / 64` registers depending on
dtype. On SM75 they use `64-66 / 66 / 72`. Hard forward uses 26 registers.

## 11. Removed Mechanisms

The rewrite removed these mechanisms from the operator:

| Removed mechanism | Reason |
| --- | --- |
| Dynamic route_temperature and lambda schedules | Task-level policy, not operator semantics; coupling caused unstable ablations. |
| Window-derived lambda | Configured `W` can be a loose upper bound and is not actual suffix-tail amplification. |
| Q/K magnitude confidence | Creates a numerical side channel and duplicates STE conditioning. |
| Hard-winner margin in backward | Biases credit toward the current discrete route and suppresses exploration. |
| Recency score | Hard ties already resolve discretely; soft recency changes the proxy objective. |
| Optional mismatch width | The cubic range is now part of the fixed estimator. |
| Hard/soft V modes | Soft distributed V credit is retained because hard-selected V failed the full-model fit gate. |
| Early-stop epsilon | Approximation policy was mixed into semantic scoring. |
| Q/K damper | Optimizer policy and saturation diagnosis belong outside the operator. |
| Hot-path telemetry | Candidate matrices and reductions damaged the production ABI. |

This is the main simplification: variability that defines a different
estimator is fixed; task-level tuning remains outside; only three useful
run-level controls remain public.

## 12. Executed Simplification Results

The ten-step simplification pass produced these decisions:

1. **Baseline inventory.** The old planner had 14 reachable cache
   combinations and up to eight soft-kernel template combinations per dtype.
   The starting suite had 101 passing tests.
2. **Utility cache removed.** A second-pass utility recomputation replaced the
   per-action shared array. FP32 parity remained intact; short rows improved
   slightly and long rows stayed within run-to-run variation.
3. **Query-cache branch made static.** The hot-loop runtime boolean was moved
   into compile-time plan features. This was primarily a control-flow cleanup,
   with timing changes below 1%.
4. **Grad-output branch made static.** Row `grad_output` caching also became a
   compile-time property. The temporary Boolean product exposed the template
   explosion and motivated one explicit plan enum.
5. **Three plans replace budgets.** `ScoreCached`, `KeyReduced`, and `Generic`
   replaced the 6 KiB score budget, 12 KiB K budget, and nested dispatch.
   There are now nine soft kernels total across three dtypes and zero spills.
   A proposed `W=1` Key-plan exclusion was rejected after it regressed
   `T=1536` from `3.235` to `3.452 ms`; payload size remains the only
   tie-breaker between plans that fit.
6. **Shared layout centralized.** One host/device layout function computes all
   offsets and sizes. Deterministic parity covered all three plans, including a
   forced `Generic` case; Compute Sanitizer reported zero errors.
7. **Stable derivative ratio.** The analytical ratio above replaced the
   `1e-30` floor. Lambda sweeps from `0.1` through `20` matched the reference.
   A same-clock A/B found only `0.3-0.7%` timing difference.
8. **Production boundaries fixed.** Raw ops now clamp large windows before
   int conversion, enforce one CUDA device, use a CUDA device guard, reject
   nonrepresentable FP32 controls, declare first-order-only autograd, and
   provide fake kernels for full-graph compilation.
9. **Historical mechanisms audited.** The hot path has no telemetry, Q/K
   damper, confidence route weight, winner margin, recency term, dynamic
   schedule, or truncated suffix scan. Python and dispatcher signatures are
   locked by tests.
10. **Dense semantics revalidated.** Both backward passes visit every causal
    action. There is no candidate index, specialized `D/W` kernel, quadratic
    workspace, or stored random tensor. The production suite passes 116 tests
    on SM75 and SM86; historical research tests are isolated from this count.
    The 1000-step CUDA fit reaches hard CE `3.83e-4` at 100% accuracy on model
    seed 0.

Candidate pruning remains prohibited: an exact suffix index may optimize hard
forward, but it must never choose backward support.
