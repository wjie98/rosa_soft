# ROSA and RosaSoft Technical Report

This document explains why `rosa_soft` separates a hard ROSA forward path from
a stochastic surrogate backward path.

The short version:

- `RosaRuntime` is the stateful hard suffix-automaton inference path.
- `rosa_soft` is the dense CUDA training operator.
- `rosa_soft_reference` is its PyTorch semantic oracle.
- Training and inference observe the same hard Q/K/V route output.
- Softness exists only in the custom backward estimator.

## Design Ladder

The current operator came from a sequence of partial solutions:

| Step | Useful property | Blocking problem |
| --- | --- | --- |
| Hard ROSA | Exact discrete suffix retrieval and compact inference state. | Q/K comparisons and route changes have no ordinary gradient. |
| Bitflip perturbation | Measures route sensitivity while keeping hard forward. | Requires many hard re-evaluations and has high variance. |
| Soft dynamic programming | Gives dense credit to candidate suffixes. | Quadratic state, sequential recurrence, and a hackable soft forward. |
| Suffix attention | Maps suffix comparison to GPU-friendly candidate rows. | Additive evidence lets older matches compensate for an early mismatch. |
| RosaSoft | Hard forward plus multiplicative stochastic suffix credit in backward. | Dense training still enumerates all causal candidates. |

## 1. Hard ROSA

ROSA turns Q/K activations into symbols:

```math
s(x)=
\begin{cases}
+1,&x>0\\
-1,&x\le 0.
\end{cases}
```

For a query suffix ending at `i` and a key suffix ending at `j`, the hard
matching length follows:

```math
L_{i,j}^{hard}=
\begin{cases}
1+L_{i-1,j-1}^{hard},&s(q_i)=s(k_j)\\
0,&\text{otherwise}.
\end{cases}
```

Inference chooses a historical value associated with the longest exact suffix.
A suffix automaton can maintain this state without storing an attention-style
matrix. The training difficulty is that symbolization and winner selection are
step functions.

## 2. Bitflip Perturbation

Let `F(B,V)` be hard ROSA from packed route bits `B` and values `V`. Flipping
one bit `u` gives a directional signal:

```math
\Delta_u=
\left\langle
g,\,
F(\operatorname{flip}_u(B),V)-F(B,V)
\right\rangle.
```

This is a strong local oracle because it asks the actual discrete operator what
would change. It also scales with the number of probes and inherits large
route jumps. The historical tiny fitting task showed that hard-forward
learning is possible, but bitflip is not an economical full-model primitive.

## 3. Soft Dynamic Programming

A differentiable local match `m(i,j)` can define:

```math
S_{i,j}=m(i,j)(1+S_{i-1,j-1}),
```

followed by a softmax over candidates. This gives every candidate a gradient in
one graph, but introduces three problems:

- `O(T^2)` score state;
- diagonal recurrence that is difficult to tile;
- a soft weighted-value forward that can carry information unavailable to
  hard inference.

The last issue is fundamental. If training can exploit probability tails or
value amplitudes, it need not learn a useful discrete route.

## 4. Suffix Attention

Suffix attention flattens a recent window into features and scores them with an
attention-like dot product:

```math
\phi_q(i)=[q_i,\eta q_{i-1},\ldots,\eta^{W-1}q_{i-W+1}],
```

```math
s_{i,j}=\frac{\langle\phi_q(i),\phi_k(j)\rangle}{\sqrt{WD}}.
```

This is parallel and kernel-friendly. Its score is additive, however, so a
mismatch at the most recent symbol can be offset by older similarities. Hard
longest-suffix matching terminates at that mismatch. The objective is
therefore structurally different even before hard inference is considered.

## 5. Current Path: RosaSoft

RosaSoft keeps:

- hard Q/K/V signs in forward;
- exact longest-suffix selection with latest-action ties;
- a finite suffix window for the dense training oracle;
- dense candidate credit only in backward;
- multiplicative local gates so an early mismatch suppresses later offsets;
- stochastic exploration that cannot alter the hard forward;
- soft distributed V gradients.

### Hard Forward

For row `i`, action `a` compares `q_i` to `k_{a-1}` and continues backward
until the first unequal complete symbol or `max_suffix_length`. The action with the
longest exact suffix wins; latest action resolves equal lengths. If every
candidate mismatches immediately, the null action returns exact zero.

The returned value is one hard signed `V[a]`. No soft score, probability, or
weighted value is visible to the model.

### Straight-Through Boundary

Backward uses:

```math
\hat s(x)=\operatorname{stopgrad}
\left(s(x)-\frac{x}{1+|x|}\right)
+\frac{x}{1+|x|},
```

with derivative:

```math
\frac{\partial\hat s}{\partial x}=\frac{1}{(1+|x|)^2}.
```

Numerically `hat s(x)` is still the hard sign.

### Independent Mismatch Exploration

For hard mismatch bit `delta_d`, sample an independent `u_d` and use:

```math
\alpha_d=1-\frac{u_d^3}{2}.
```

The relaxed Hamming distance is:

```math
H=\sum_d\delta_d\alpha_d.
```

The local numerical gate is:

```math
\mu=e^{-\lambda H}.
```

Every exact symbol match remains `1`. Every mismatching symbol is strictly
below `1`. Non-exact candidates may reorder, which is intentional exploration;
only the exact/non-exact boundary must remain invariant.

A detach-based Jacobian anchor uses the local derivative scale:

```math
\lambda e^{-\lambda h},
\qquad h=\sum_d\delta_d,
```

rather than a random exponential scale based on `H`.

### Multiplicative Suffix Score

For action `a`, let `mu_r` be the local gate at suffix offset `r`:

```math
p_0=1,\qquad p_{r+1}=p_r\mu_r,
```

```math
R(i,a)=\sum_r p_{r+1}.
```

An early mismatch is present in every later product. This is the key property
missing from additive suffix attention.

### Candidate Distribution

The null score is fixed at `0.5`; every valid action uses `R(i,a)`:

```math
P(a|i)=\operatorname{softmax}_a
\left(\frac{z(i,a)}{\text{route_temperature}}\right).
```

There is no recency term or current-winner bonus in backward. Temperature
controls how widely route credit is distributed. It does not change the hard
forward or the maximum learnable suffix length.

`mismatch_penalty` controls mismatch leakage and its anchored local Jacobian.
It is independent of route_temperature and `max_suffix_length`. A configured window may be
only a loose upper bound, so deriving lambda from that bound is unjustified.

### Value Credit

Q/K route gradients use probabilities against detached hard V symbols. V
gradients use detached probabilities and the softsign Jacobian. Consequently,
non-winning actions and hard-null rows can still train V, while no weighted
value leaks into forward.

## 6. Public Controls

The operator intentionally exposes only:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `max_suffix_length` | `32` | Hard and proxy suffix horizon. |
| `route_temperature` | `1.0` | Backward action-allocation route_temperature. |
| `mismatch_penalty` | `3.0` | Backward mismatch leakage and Jacobian scale. |

These are fixed run-level values. The operator does not schedule or infer them
from context length, code width, training step, diagnostics, or each other.

Removed mechanisms include magnitude confidence, winner margins, recency
biases, optional perturbation width, hard/soft V modes, early-stop epsilon, Q/K
dampers, and hot-path telemetry. Each either changed the estimator, duplicated
optimizer policy, or added state without sufficient evidence.

## 7. Reference and CUDA

The PyTorch reference materializes candidate matrices and independent samples:

```text
mismatch_uniform: [B,H,T,T-1,D]
```

CUDA stores one seed and reconstructs each `u` from the index
`(b,h,q_pos,k_pos,bit)`. A local gate keeps only `h` and `H`
accumulators. This removes quadratic random memory while preserving exact
sample-level parity with the reference.

The current CUDA kernel still scans every causal action. Counter randomness
solves memory state, not candidate complexity:

```text
persistent estimator state: O(B H T)
dense candidate compute:    O(B H T^2 W D)
```

This full action support is intentional. RosaSoft's advantage is precisely
that it trains a sparse hard route with dense proxy gradients. Pruning the
backward candidate set according to the current hard route, a learned index,
top-k scores, ANN/LSH retrieval, thresholds, or sampled negatives changes the
estimator. In particular, an undiscovered route can receive no gradient,
remain undiscovered, and create a self-reinforcing collapse.

Hard `RosaRuntime` uses bounded compact suffix state for inference, and an exact
index may optimize training forward. Neither may choose the actions included
in the default backward pass. Long-context training optimization must preserve
all causal actions while reducing redundant work through online softmax,
dense tiling, exact diagonal recurrences, caching, and checkpointed
recomputation.

## 8. Validation

The test suite covers hard suffix semantics, latest ties, null rows,
static-control isolation, cubic mismatch gates, Jacobian anchoring, dense
low-rank route discovery, exact
diagonal-recurrence adjoints, all three CUDA execution plans, grouped heads,
`D=32`, non-contiguous inputs, FP16/BF16, and exact counter reconstruction.

With identical reconstructed samples, CUDA is bit-exact in forward and the
recorded FP32 maximum absolute VJP error is `5.96e-8`.

On the default repeated-motif fit at 1000 steps:

| Method | Final hard CE | Accuracy |
| --- | ---: | ---: |
| Historical bitflip | `1.61e-4` | `100%` |
| Current PyTorch reference | `4.78e-4` | `100%` |
| Current CUDA kernel, seed 0 | `3.83e-4` | `100%` |
| Current CUDA kernel, seed 2 | `3.39e-4` | `100%` |

On an RTX 3070 with FP32 `B=1,H=4,D=8,H_v=2,D_v=8,W=32`, CUDA
forward+backward takes `1.528/6.002/20.426 ms` at `T=256/512/1024` in the
final long-run probe. This is about 42-49% faster than the pre-pass dense
kernel while retaining every causal candidate. Quadratic growth remains and
does not justify reducing gradient support.

## 9. Runtime Contract

Training:

```python
y = rosa_soft(
    q,
    k,
    v,
    max_suffix_length=128,
    route_temperature=1.0,
    mismatch_penalty=3.0,
)
```

Inference:

```python
with RosaRuntime(
    num_heads,
    num_value_heads,
    qk_bits=4,
    value_bits=4,
    max_suffix_length=128,
) as rt:
    out, endpos = rt.update(q, k, v, return_packed=False)
```

`RosaRuntime` uses the same finite suffix horizon, latest-match tie rule, and
exact null value as the training forward. Latest end positions are propagated
only through suffix classes reachable within that horizon, so repeated-symbol
updates are `O(max_suffix_length)` instead of quadratic in accumulated context.
One per-instance executor serializes state transitions, asynchronous work, and
close. The current packed runtime stores one byte per head, so Q/K and payload
widths are `1..8`; the training Q/K contract remains `1..32`. Packed calls
ignore byte bits above the declared widths.
