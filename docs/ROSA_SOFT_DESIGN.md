# RosaSoft Operator Report

This document defines the current `rosa_soft_reference` and
`rosa_soft` contract. The two implementations have the same mathematics:
the PyTorch version is the inspectable oracle and the CUDA version reconstructs
the fixed stochastic estimator without materializing its quadratic random
state.

## 1. Contract

RosaSoft is a training operator for hard ROSA routing. It is not a soft
replacement for the hard suffix automaton.

```python
output = rosa_soft(
    query_logits,
    key_logits,
    payload_logits,
    max_suffix_length=32,
    route_temperature=1.0,
    mismatch_penalty=3.0,
)
```

The public controls are deliberately limited to:

| Parameter | Default | Role |
| --- | ---: | --- |
| `max_suffix_length` | `32` | Maximum suffix length used by hard and proxy scores. |
| `route_temperature` | `1.0` | Backward route-allocation temperature. |
| `mismatch_penalty` | `3.0` | Backward mismatch leakage and hard-Hamming local VJP scale. |

There is no dynamic schedule, context-derived scaling, confidence weighting,
winner margin, early stop, optional perturbation width, payload-gradient mode, or
telemetry branch in the training operator.

## 2. Tensor Layout

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `query_logits` | `[B, T, H, D]` | Q symbol logits. |
| `key_logits` | `[B, T, H, D]` | K symbol logits. |
| `payload_logits` | `[B, T, H_p, D_p]` | Routed payload symbol logits. |

`H % H_p == 0`; each payload head is shared by `H / H_p` route heads.
`1 <= D <= 32` so each CUDA Q/K code fits in one packed word. CUDA supports
FP32, FP16, and BF16 inputs and accumulates gradients in FP32.

## 3. Exact Hard Forward

Every Q/K/payload scalar uses the same hard sign:

```math
s(x)=
\begin{cases}
+1,&x>0\\
-1,&x\le 0
\end{cases}
```

For query row `i`, causal route `a` is in `1..i`. Its most recent symbol pair
is `q_i` and `k_{a-1}`. The exact suffix length is

```math
L(i,a)=\max\left\{
\ell\le\min(W,i+1,a):
s(q_{i-r})=s(k_{a-1-r})\ \forall r<\ell
\right\}.
```

Equality here means equality of the entire `D`-bit code. The selected route has
the largest suffix length, with the latest route winning ties:

```math
a_i^*=\max\operatorname*{argmax}_{1\le a\le i}L(i,a).
```

If every route has length zero, null route `0` is selected and the output is
exact positive zero. Otherwise:

```math
y_{i,h}=s(v_{a_i^*,\,h_v}).
```

This is the only forward value returned during training. `route_temperature`,
`mismatch_penalty`, and random samples cannot change it.

## 4. Fixed Straight-Through Map

Backward substitutes a softsign Jacobian for every hard sign:

```math
\hat s(x)=\operatorname{stopgrad}(s(x)-r(x))+r(x),
\qquad
r(x)=\frac{x}{1+|x|},
```

```math
\frac{\partial \hat s}{\partial x}
=\frac{1}{(1+|x|)^2}.
```

Forward still evaluates to exactly `s(x)`. This map is fixed rather than
configurable because changing it changes the estimator itself.

## 5. Mismatch Exploration

For a Q/K pair and bit `d`, the hard mismatch is

```math
\delta_d=\frac{1-s(q_d)s(k_d)}{2}\in\{0,1\}.
```

Each `(b,h,q_pos,k_pos,d)` receives an independent uniform sample `u_d`.
It converts each mismatching bit to one penalty:

```math
\alpha_d=1-\frac{1}{2}u_d^3.
```

The penalty is in `[0.5,1]`. Matching bits contribute exactly zero:

```math
h=\sum_d\delta_d,\qquad
H=\sum_d\delta_d\alpha_d.
```

The numerical local match used by the suffix proxy is

```math
\mu=e^{-\lambda H}.
```

Independent samples are allowed to reorder non-exact candidates. Exact matches
remain exactly `1`, and every mismatch remains below `1`; therefore the
perturbation cannot alter the hard forward route and converges to hard matching
as `mismatch_penalty` grows.

## 6. Hard-Hamming Local VJP

Using the relaxed penalties directly in backward would make the local gradient
scale random and high variance. The estimator therefore uses a stop-gradient
hard-Hamming correction:

```math
c=\operatorname{stopgrad}\left(e^{-\lambda(h-H)}\right),
```

```math
\mu=
\exp\left[
-\lambda\left(
\operatorname{stopgrad}(H)
+c\hat h-\operatorname{stopgrad}(c\hat h)
\right)
\right],
```

where `hat h` is the STE Hamming expression. Its numerical value remains
`exp(-lambda H)`, while its local derivative follows the hard Hamming
distance:

```math
\frac{\partial\mu}{\partial\hat h}
=-\lambda e^{-\lambda h}.
```

The CUDA VJP needs the ratio between this hard-Hamming derivative and the
numerical gate. It evaluates that ratio without a numerical floor:

```math
\rho=e^{-\lambda(h-H)}.
```

Since `H` is at most `h`, `0 < rho <= 1`. This is algebraically the same as
`exp(-lambda*h)/mu` without dividing by a possibly tiny numerical gate.

## 7. Multiplicative Suffix Score

For route `a`, let `mu_r(i,a)` be the local match at suffix offset `r`.
The proxy score is

```math
p_0=1,\qquad
p_{r+1}=p_r\mu_r,
```

```math
R(i,a)=\sum_{r=0}^{\min(W,i+1,a)-1}p_{r+1}.
```

An early mismatch suppresses all longer products. This mirrors longest-prefix
termination more closely than summing independent local similarities.

## 8. Route Allocation

The null route has a fixed score `0.5`. Every valid non-null route uses its
proxy suffix score directly:

```math
z(i,0)=0.5,\qquad z(i,a)=R(i,a).
```

Invalid future routes have score negative infinity. Backward probabilities
are exactly

```math
P(a|i)=\operatorname{softmax}_a\left(\frac{z(i,a)}{\tau}\right).
```

There is no hard-winner bonus or recency term. Lower route_temperature sharpens the
existing proxy ranking but also suppresses gradients to losing routes; it
does not increase the maximum suffix horizon. Because finite mismatch
perturbations can make the proxy winner differ from the hard latest-longest
route, the zero-temperature proxy limit is not a hard-route equivalence
guarantee.

## 9. Route and Payload VJP

The surrogate used only to define the VJP separates route and payload paths:

```math
\tilde y_i^{route}
=\sum_{a>0}P(a|i)\operatorname{stopgrad}(s(v_a)),
```

```math
\tilde y_i^{payload}
=\sum_{a>0}\operatorname{stopgrad}(P(a|i))\hat s(v_a).
```

Q/K receive the softmax route gradient against every hard payload symbol.
Payload logits receive probability-weighted softsign credit, including when
hard forward selected null. The custom autograd function exposes only the hard
output, so neither surrogate term nor its probabilities are visible forward.

## 10. CUDA Random Reconstruction

The PyTorch oracle materializes
`u[B,H,T,T-1,D]` for inspection. CUDA saves one scalar seed and reconstructs
each sample from:

```text
counter = ((((b * H + h) * T + q_pos) * T + k_pos) * D + bit)
```

A stable 64-bit SplitMix hash maps `(seed,counter)` to a 24-bit open-interval
uniform. The same counter is recomputed whenever a local gate is revisited.
CUDA therefore saves Q/K packed words and one seed, not an
`O(B H T^2 D)` random tensor.

The hash is an implementation detail, but counter stability is part of the
test contract: the test suite reconstructs the samples in Python and compares
the complete VJP. CUDA Q/K/payload gradients use atomic accumulation, so repeated
runs with one seed are numerically reproducible but can differ in their last
floating-point bits.

The backward kernel has three compile-time execution plans:

| Plan | Reused shared state |
| --- | --- |
| `CacheRouteScores` | row gradient, Q suffix, and all row scores |
| `ReduceKeyGradInShared` | row gradient, Q suffix, and one K-gradient tile |
| `MinimalSharedMemory` | softmax reduction statistics only |

Host plan selection and device pointer binding use the same
`BackwardSharedLayout` calculation. The two optimized plans are considered
only when their complete layout fits a portable 48 KiB shared-memory limit;
when both fit, the smaller score-row or K-tile payload wins. This leaves three
auditable variants per dtype rather than independently toggling each cache.

## 11. Complexity Boundary

The CUDA kernel currently enumerates all causal routes:

```text
hard forward: O(B H T^2 min(W,T))
soft backward: O(B H T^2 min(W,T) D)
persistent estimator state: O(B H T)
```

Packed Q/K comparisons make hard forward inexpensive, and counter
reconstruction removes quadratic random memory. It does not remove quadratic
candidate compute. This operator is a correct dense training kernel, not the
long-context suffix-automaton runtime.

The dense candidate set is part of the training semantics. RosaSoft is meant
to use a dense approximate gradient to train a sparse hard inference route.
Restricting backward to a bounded candidate set would bias credit toward
already-retrieved routes. A route omitted by the current index would receive
no signal to become retrievable, so indexing and learning could enter a
self-reinforcing collapse.

The default backward must therefore evaluate every valid causal route. It
must not use top-k pruning, ANN/LSH retrieval, sampled negatives, hard-winner
neighborhoods, score thresholds, early termination, or a suffix automaton to
select gradient support. An exact suffix index may optimize hard forward only.
Sparse-gradient research, if any, must remain a separate non-default operator
and be compared against the dense VJP.

The optimization target is redundant computation inside the dense estimator:
online softmax, route tiling with suffix halos, exact diagonal recurrences,
local-gate reuse, checkpoint/recompute tradeoffs, and block-local gradient
reductions. These preserve full support even though the asymptotic candidate
set remains `T^2`.

## 12. Diagnostics

Training returns only the output tensor. For small probes:

```python
from rosa_soft.diagnostics import summarize_rosa_soft
from rosa_soft.testing import inspect_rosa_soft

output, inspection = inspect_rosa_soft(q, k, v)
summary = summarize_rosa_soft(inspection)
```

The detached inspection contains hard lengths, proxy scores, route scores,
route probabilities, masks, and selected routes. Mismatch uniforms are
temporary and are released before the detached snapshot is returned. This
explicit separation prevents diagnostics from adding allocations or branches
to the CUDA hot path.

## 13. Validation

The current suite covers:

- hard suffix lengths against a naive implementation;
- null behavior and latest-route tie breaking;
- forward independence from all soft controls and random samples;
- cubic mismatch values and hard-Hamming local VJPs;
- magnitude-independent numerical proxy values;
- static route-temperature and mismatch-penalty separation;
- grouped payload heads, non-contiguous tensors, `D=32`, and singleton rows;
- FP32, FP16, and BF16 CUDA paths;
- exact counter reconstruction and PyTorch/CUDA VJP parity;
- all three shared-memory plans;
- non-current CUDA devices, raw-op parameter boundaries, and large windows;
- full-graph forward and AOT backward through `torch.compile`.

Current reproduced test, fitting, and benchmark results are recorded with
their environment in `validation/latest.json`. Historical estimator evidence
is isolated under `docs/research/`.
