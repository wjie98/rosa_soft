# Dense Unbounded V1 Freeze

`rosa-soft-dense-unbounded-v1` is the smallest deterministic semantic
reference for the dense RosaSoft estimator. It is deliberately separate from
the production package and kernel work.

The frozen entry point is:

```python
from benchmarks.dense_unbounded_v1 import rosa_soft_dense_unbounded_v1
```

Its public signature contains only `query`, `key`, and `value`. Constants are
fixed at `scale=1`, `mismatch_scale=3`, `null_score=0.5`, and `dropout_p=0`.

## Contract

The forward pass is exact discrete ROSA:

1. Quantize Q, K, and V with `+1` for values greater than zero and `-1`
   otherwise.
2. Compute exact causal aligned-suffix lengths over the entire available
   history. There is no suffix window.
3. Select the longest route, breaking equal-length ties in favor of the latest
   route.
4. Return that route's binary V; route zero is the exact zero null value.

The backward pass differentiates one dense carrier without changing the hard
forward:

```text
m[t,a] = mean_d (1 - q[t,d] k[a-1,d]) / 2
g[t,a] = exp(-3 m[t,a])
S[t,a] = g[t,a] * (1 + S[t-1,a-1])
U(S)    = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
z[t,a] = U(S[t,a]) - log(N_t)       for a > 0
z[t,0] = 0.5
p[t,:] = softmax(z[t,:])
carrier[t] = sum_a p[t,a] * binary_v[a]
```

The binary symbols use the softsign derivative
`d sign(x) / dx := 1 / (1 + abs(x))^2` only in the VJP. Every causal route is
included; there is no top-k, sampling, candidate filtering, winner detach, or
QK/V role split.

## Why This Is Minimal

The snapshot excludes attention dropout, varlen packing, CUDA schedules,
checkpointing, kernel tiling, and experimental score branches. Those features
can be tested against this baseline but cannot redefine it. A changed formula
must receive a new versioned module and tag.

This is a biased surrogate for a discontinuous operation, not an exact
derivative. Its purpose is to preserve the simplest estimator already shown to
train while retaining exact hard forward semantics.

## Verification

At the freeze point:

- The dedicated CPU test plus the related dense/top-two/dual reference tests
  passed: `97 passed`.
- FP64 output is bitwise equal to production
  `rosa_soft_reference(max_suffix_length=T, scale=1, dropout_p=0,
  mismatch_scale=3)`; Q/K/V gradients agree within `2e-14`.
- On GPU0, an RTX 3070 (`sm_86`), FP32/FP16/BF16 hard outputs are bitwise equal
  to that production call and every gradient is finite. Maximum FP32 absolute
  gradient error was `4.4703484e-08`; FP16 and BF16 errors were zero in the
  smoke test.
- A length-70 semantic test proves that hard routing is not truncated at 32.
- Sign-preserving rescaling cannot change the forward output.

The machine-readable record is
`validation/dense_unbounded_v1_freeze.json`.

## Content Identity

```text
c7e79c4bde7caeb881543297f7f28dfc4d1de57757fed530ed586a3bdaf9c849  benchmarks/dense_unbounded_v1.py
97f2f35c1ac13d9346d28bccfdb07690eb573bcab58c3d39fcdd2bcbd143acd3  tests/test_dense_unbounded_v1.py
```

The base repository commit is `01c9b0b`. The annotated tag points to a detached
snapshot commit, so creating the freeze does not move or clean the active
`main` worktree.
