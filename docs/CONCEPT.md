# RosaSoft Concept

RosaSoft trains an exact discrete ROSA router without exposing a soft value
path in forward. Its design has two deliberately different semantics:

- forward is the deployment behavior;
- backward is one dense credit-assignment approximation with optional
  attention dropout.

The separation is implemented by a custom VJP. It is not a soft forward
relaxation.

## 1. Inputs

For dense batches:

```text
query   Q: [B, T, H, D]
key     K: [B, T, H, D]
value V: [B, T, Hv, Dv]
```

`H` must be divisible by `Hv`. Each value head is shared by `H / Hv` query
heads. Packed variable-length inputs remove `B` and use `cu_seqlens` to define
independent segments.

Every Q, K, and value bit is quantized by

```text
sign(x) = +1 if x > 0 else -1.
```

Zero therefore belongs to the negative symbol.

## 2. Exact Hard Forward

At query position `t`, route `a` is either:

- `a = 0`: the null route, whose value is the all-zero vector;
- `1 <= a <= t`: match the query history ending at `t` against the key
  history ending at `a - 1`, then return value `V[a]`.

For a non-null route, its exact suffix length is

```text
L(t,a) = max l such that
         sign(Q[t-r]) == sign(K[a-1-r])
         for every r in [0, l),
```

bounded by `max_suffix_length` and both sequence boundaries. Equality means
all `D` bits match for that head and position.

Forward selects the non-null route with largest `L(t,a)`. Equal lengths select
the latest route, meaning largest `a`. If every non-null route has length
zero, forward selects null. The returned value is exactly:

```text
0                         for null
sign(V[a])                for a selected non-null route.
```

No probability, soft symbol, or surrogate score contributes to this value.

## 3. Symbol VJP

Backward evaluates hard symbols numerically but assigns them the derivative of
softsign:

```text
z(x) = sign(x)
dz/dx := 1 / (1 + |x|)^2.
```

This straight-through rule keeps the forward algebra binary while reducing
the gradient of already-saturated logits. It is applied to Q, K, and value
logits.

## 4. Local Match Gate

For one query/key symbol pair, normalized Hamming mismatch is

```text
m = (1 / 2D) * sum_i (1 - z(q_i) z(k_i)).
```

Because `z` is numerically `+1/-1`, `m` is exactly the mismatch fraction in
forward arithmetic. The deterministic local match gate is

```text
g = exp(-mismatch_scale * m).
```

An exact symbol match has `g = 1`; a mismatch has `0 < g < 1`. Normalizing by
`D` makes `mismatch_scale` describe a mismatch fraction rather than a raw
bit count.

## 5. Soft Suffix Score

For route `(t,a)`, define the product for suffix length `l`:

```text
P_l(t,a) = product from r=0 to l-1 of g(t-r, a-1-r).
```

The route score is the expected matched-prefix length:

```text
F(t,a) = sum from l=1 to W of P_l(t,a),
```

where `W` is clipped by `max_suffix_length` and sequence boundaries.

This complete prefix sum is important. Every deeper suffix position receives
credit before it becomes part of the current hard winner. Frontier-only or
winner-only gradients remove that discovery path.

As `mismatch_scale` approaches infinity, `g` approaches the exact local
match indicator and `F` approaches the exact suffix length. At finite
penalty, a near match may receive more backward credit than a shorter exact
match. That is intentional exploration in the VJP; it cannot change forward.

## 6. Dense Route Distribution

The fixed null score is

```text
F(t,0) = 0.5.
```

For `N_t` valid non-null candidates, logits are

```text
logit(t,0) = 0.5 * scale
logit(t,a) = F(t,a) * scale - log(N_t),  a > 0.
```

The route probabilities are a softmax over null and every valid causal route.
Subtracting `log(N_t)` treats all non-null candidates as one aggregate
hypothesis, so merely increasing context length does not multiply their prior
mass.

`scale` only controls competition among scores that already exist. A larger
scale sharpens routing but does not create support for a longer suffix. Unlike
dot-product attention, the mismatch is already normalized by `D`, so the
default does not apply an implicit `1 / sqrt(D)`.

## 7. Value Carrier

Let the post-softmax dropout multiplier be

```text
d(t,a) = Bernoulli(1 - dropout_p) / (1 - dropout_p).
```

The differentiable carrier used only for the VJP is

```text
y_soft(t) = sum_a d(t,a) * p(t,a) * z(V[a]),
```

with a zero value for null. The custom autograd function returns the exact
hard value in forward and applies

```text
J(y_soft)^T * grad_output
```

in backward. Q/K receive route-discovery credit and values receive
distributed probability-weighted credit from the same route distribution.
The dropout mask is never visible in forward. Its inverted scaling preserves
the deterministic carrier in expectation.

## 8. Irreducible Contract

The production estimator contains only:

1. hard binary symbols with softsign VJP;
2. normalized Hamming mismatch;
3. one exponential local gate;
4. one complete suffix prefix-product sum;
5. one candidate-normalized dense softmax;
6. optional standard attention dropout;
7. one distributed value carrier.

It has no mismatch perturbation, antithetic branch, hard-tier wrapper, bounded
residual, adaptive schedule, confidence gate, recency prior, top-k candidate
set, or soft forward value. Dropout stores one scalar seed and reconstructs
route decisions by index; it does not save a quadratic mask.

Every valid causal route is still scored and normalized in backward. Dropout
can mask a route's carrier contribution for one sample, but it does not define
or prune the candidate set. A sparse hard forward does not justify a
structurally sparse training gradient: undiscovered routes need support before
they can become winners.

## 9. Static Controls

The public controls are intentionally limited:

| Control | Default | Effect |
| --- | ---: | --- |
| `max_suffix_length` | `32` | Hard and surrogate suffix horizon. |
| `scale` | `1.0` | Multiplicative backward attention-logit scale. |
| `dropout_p` | `0.0` | Post-softmax inverted attention dropout in backward. |
| `mismatch_scale` | `3.0` | Leakage and gradient scale of local mismatches. |

These values are explicit run-level hyperparameters. The operator does not
derive them from `D`, `T`, the configured window, the active suffix length, or
the training step.

For long near-match discovery, an overly large mismatch scale can make
products vanish before the model repairs later suffix positions. Changing the
attention-logit `scale` cannot repair that loss of support: `scale` changes
route competition, while `mismatch_scale` changes whether a near-match score
exists.

The default `3.0` is a conservative choice for the default
`dropout_p=0` estimator. A matched synthetic fitting probe favored `3.0`
without dropout and `9.0` with `dropout_p=0.1`; neither value was uniformly
better. Keep non-default values explicit and calibrate them with the training
recipe rather than embedding a schedule in the operator.

## 10. Complexity Boundary

The semantic backward is dense in routes:

```text
route support: O(B * H * T^2)
suffix work:   bounded by max_suffix_length
```

The PyTorch oracle materializes quadratic route tensors for clarity. CUDA
visits the same routes but recomputes local state and uses online reductions,
so dense gradient support does not imply an `O(B H T^2)` persistent
workspace. Counter-based dropout also needs only scalar saved state, not a
quadratic mask. Candidate pruning would change the estimator and is not a
kernel optimization.
