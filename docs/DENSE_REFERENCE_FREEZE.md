# RosaSoft Dense Reference Freeze

`rosa-soft-dense-reference-v1` freezes the current exact-hard-forward,
dense-surrogate-backward implementation as the comparison baseline for future
estimators. It is a reference implementation, not a claim that the surrogate
is the exact derivative of discrete ROSA routing.

## Frozen contract

- Forward uses binary `sign` symbols, exact causal latest-longest suffix
  routing, latest-position tie breaking, and an exact zero null value.
- Backward scans every valid causal candidate. Candidate pruning, sampled
  candidate gradients, and bounded candidate sets are outside this reference.
- Suffix evidence is transformed by
  `U(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1)` before the dense softmax proxy.
- Public controls remain keyword-only:
  `max_suffix_length=32`, `scale=1.0`, `dropout_p=0.0`, and
  `mismatch_scale=3.0`.
- `dropout_p` has PyTorch semantics: it is the probability of dropping a
  post-softmax backward weight. It never changes the hard forward result.
- Dense and packed-varlen layouts, grouped value heads, all nonempty Q/K/V
  gradient masks, FP32/FP16/BF16, autocast, checkpointing, and `torch.compile`
  remain part of the supported surface.

The compiled CUDA extension exports only these four operator schemas:

1. `rosa_soft::hard_forward`
2. `rosa_soft::hard_forward_varlen`
3. `rosa_soft::surrogate_vjp_masked`
4. `rosa_soft::surrogate_vjp_varlen_masked`

The former research-only indexed hard-forward and factorized diagonal-VJP raw
operators are deliberately absent. Their measurements remain historical
evidence under `docs/research` and `validation`, but they are not linked into
the frozen package.

## Reproduction snapshot

The extension was rebuilt for `sm_75;sm_86` with PyTorch `2.11.0+cu128` and
CUDA toolkit `12.8`, then tested on physical GPU 1, an RTX 2080 Ti (`sm_75`).

- Full suite: `500 passed, 2 skipped` in 15.26 seconds.
- Fresh-process schema check: all four production schemas present; both removed
  research schemas absent.
- Fat binary: 10,130,880 bytes, SHA-256
  `e72c244e091c75080e33f7013e30a96d949f2ce0d1b4805e13f6754857e9d30b`.
- Resource audit: 115 kernel instances per architecture, at most 144 registers,
  and no nonzero stack or local-memory allocation.

### Hard-forward fitting

`examples/fit_soft_reference.py` was run for seeds 0 through 7, 1,000 steps,
`dropout_p=0`, and `mismatch_scale=3`. Six of eight runs finished below
`1e-3`; all six collision-free runs did so. The two remaining runs finished
within `0.001` of the conditional-entropy floor imposed by collisions in their
hard features. Final losses by seed were:

```text
0: 0.00021315  1: 0.61642039  2: 0.00028860  3: 0.00036597
4: 0.00023598  5: 0.00029076  6: 0.00031390  7: 0.13898043
```

This reproduces the earlier result: near-zero fitting is reached whenever the
hard representation can separate the targets; the two failures are structural,
not unexplained optimizer failures.

### Recall gates

- Shortcut-free associative recall passed `8/8`; success steps were
  `[31, 38, 25, 32, 37, 47, 38, 25]`, exactly matching the previous record.
- Contextual reset-RNN recall was intentionally launched twice with identical
  settings. The launches passed `3/4` and `4/4`; mean validation exact accuracy
  was `0.995117` and `0.998047`. This records the known atomic/numerical basin
  sensitivity instead of reporting only the better launch.

### Runtime spot check

With `B=1, H=4, D=8, Hv=2, Dv=8, W=32`, FP32 QKV backward, 30 warmups, and
200 timed repeats on the RTX 2080 Ti:

| Layout | T=256 | T=512 | T=1024 |
| --- | ---: | ---: | ---: |
| Dense | 0.674 ms | 1.621 ms | 5.442 ms |
| Varlen, segment 256 | 1.035 ms | 1.480 ms | 2.872 ms |

These are smoke-test timings, not a cross-version performance guarantee.

## Freeze policy

This tag may receive documentation or reproducibility corrections only. Any
new estimator, candidate approximation, stochastic-gradient method, route
semantics, or incompatible kernel factorization must use a separate module or
operator name. That keeps this version available as an exact-hard,
dense-gradient control.

Known boundaries remain: backward is `O(T^2)`, CUDA atomic accumulation is not
bitwise deterministic, defaults are empirical rather than universal, and no
realistic full-model pretraining run has yet established scaling behavior.
