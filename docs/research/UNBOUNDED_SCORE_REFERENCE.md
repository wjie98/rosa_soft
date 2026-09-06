# Unbounded ROSA Score Reference

This record freezes the research conclusion for constructing a dense surrogate
distribution when hard ROSA uses an unlimited suffix length. It describes the
PyTorch oracle in `benchmarks/dual_score_reference.py`; it does not change the
public production operator or prescribe a CUDA implementation.

## Decision

The reference keeps three explicitly selectable backward distributions:

1. `discovery`: the current normalized-Hamming, square-root production score,
   extended to the complete suffix;
2. `information`: a random-collision-calibrated score with finite-sample
   variance control;
3. `dual`: a fixed probability mixture of those independently normalized
   distributions.

All three execute exactly the same unlimited, binary, latest-longest hard
forward. The research default is `dual` with `evidence_power=0.25` and
`information_weight=0.5`. This is the best candidate in the tests below, but
it is not yet promoted to the public production estimator: controlled suffix
repair still exposes wrong-direction gradients that only exact bitflip fixes.

## 1. Non-Negotiable Invariants

- The forward sees only hard `{-1,+1}` Q/K/V symbols.
- No soft score, probability, magnitude, or dropout mask changes the forward.
- A query selects the longest exact historical suffix and breaks ties toward
  the latest endpoint.
- Route zero is a real null route with value zero.
- Suffix length is unlimited. There is no training window `W` in this oracle.
- The backward evaluates every causal candidate. It does not prune, sample, or
  stop gradients to difficult routes.
- Packed sequences are independent; neither hard state nor dropout coordinates
  cross a segment boundary.
- `dropout_p` follows PyTorch attention semantics: it is the post-softmax
  probability of dropping a backward weight.

These constraints separate the semantic problem from kernel engineering. A
faster implementation is valid only if it reproduces this contract.

## 2. Historical Mechanisms Audited

| Mechanism | Purpose | Decision |
| --- | --- | --- |
| Raw suffix sum `S=sum product(g)` | Give every depth a dense gradient | Keep only as an intermediate; linear growth lets a slightly longer route dominate too quickly |
| Normalized square-root utility | Compress suffix growth while keeping `U(1)=1` | Keep in `discovery`; it remains the strongest simple production compromise |
| Log and lower power utilities | Spread credit among close candidates | Reject as default; at very large `N` they need impractically long exact suffixes |
| `h/D` local Hamming normalization | Keep local gate scale stable across symbol width | Keep in `discovery` |
| Fixed null score `0.5` | Preserve a learnable null competitor | Keep in `discovery` |
| Candidate prior `-log(N)` | Prevent random non-null mass from growing merely because context grows | Keep in every mode |
| Exact hard tiers or frontier-only tails | Approximate winner ordering more literally | Reject; they remove credit from suffix positions that must improve before reaching the frontier |
| Dynamic temperature, lambda, or null schedules | Adapt sharpness to context or training phase | Reject; no robust gain justified the extra policy and scaling ambiguity |
| Per-mismatch perturbation and antithetic sampling | Approximate discrete counterfactuals | Do not use in this deterministic reference; variance and cost are unfavorable |
| Post-softmax dropout | Add standard attention-style exploration | Keep as an optional backward regularizer; it does not reduce oracle work |
| Old collision likelihood ratio using `h/D` | Calibrate random routes | Retain only as a historical control; it underperformed directed repair and exact-bitflip alignment |
| Exact full bitflip | Measure the real one-bit counterfactual | Keep as an oracle, not as the scalable estimator |
| Sparse/filtered bitflip | Avoid evaluating unaffected counterfactuals | Separate research line; it is not the dense score reference |
| ARM, DisARM, Bernoulli mean field | Seek an unbiased discrete estimator | Reject as primary ROSA training estimators because one-bit suffix jumps produce excessive variance |
| Learned Transformer or low-rank recurrent proxy | Learn route behavior indirectly | Separate research line; adds parameters/state and does not preserve a parameter-free exact contract |
| Hard candidate pruning | Reduce quadratic work | Forbidden here; dense credit is the reason this reference exists |

The old tests and documents remain useful evidence, but only mechanisms in the
first section define this oracle.

## 3. Hard Semantics

For query row `t` and non-null route `a`, route `a` compares against the key
ending at `a-1` and retrieves the value at `a`. Let hard symbols be `q`, `k`
and `v`. The exact local match is

```text
e[t,a] = 1 when q[t] == k[a-1] in every bit, else 0.
```

The unlimited diagonal recurrence is

```text
L[t,a] = e[t,a] * (1 + L[t-1,a-1]).
```

The hard route is the largest `a` among routes with maximal positive `L[t,a]`.
If every length is zero, route zero is selected. The output is exactly
`sign(v[a])`, or zero for null. Score mode and all score parameters are absent
from this computation.

## 4. Discovery Distribution

For symbol width `D`, define the Hamming mismatch count

```text
h[t,a] = 0.5 * sum_d (1 - q[t,d] * k[a-1,d]).
g_d[t,a] = exp(-lambda * h[t,a] / D).
```

The dense suffix evidence follows the same diagonal shape as the exact DP:

```text
S[t,a] = g_d[t,a] * (1 + S[t-1,a-1]).
```

Equivalently, `S` is the sum of all soft suffix survival products ending at
`(t,a)`. It supplies credit at every suffix depth, including depths that do not
yet affect the hard winner. Production then applies

```text
U(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1).
```

`U(0)=0`, `U(1)=1`, and `U(S)=Theta(sqrt(S))`. For a row with `N` candidates:

```text
candidate_logit = scale * U(S) - log(N)
null_logit      = scale * 0.5.
P_discovery     = softmax([null, candidates]).
```

This geometry is good for discovery because a near match remains competitive,
but its random-background null mass depends on `D`, and exact evidence grows
only as the square root of suffix length.

## 5. Information Distribution

The information branch deliberately uses the integer mismatch count without
dividing by `D`:

```text
g_i[t,a] = exp(-lambda * h[t,a]).
```

Under independent balanced random bits,

```text
h ~ Binomial(D, 1/2)
z = E[g_i] = ((1 + exp(-lambda)) / 2)^D.
```

For a candidate with `A` available suffix lengths, let

```text
G_l = product_{r=0}^{l-1} g_i[t-r,a-r].
```

The score implemented by the oracle is

```text
I_beta = log(
    sum_{l=1}^{A} G_l * z^(-beta*l)
    / sum_{l=1}^{A} z^((1-beta)*l)
).
```

The candidate and null logits are

```text
candidate_logit = I_beta - log(N)
null_logit      = 0.
P_information   = softmax([null, candidates]).
```

This normalization has an exact property under the declared random null:

```text
E[G_l] = z^l
E[exp(I_beta)] = 1.
```

Consequently, averaging candidate weights with `-log(N)` has expected
unnormalized mass one, independent of `N` and `D`. This is an expectation
identity, not an independence claim; overlapping suffixes remain correlated.

`beta` is a collision-correction fraction, not a classical power posterior.
The mismatch term keeps its full `lambda`, while only `-log(z)` evidence is
tempered. This is intentional. A true power-likelihood variant that also
scaled the mismatch penalty was tested and reached about `0.5243` bitflip
cosine at its best, below `0.5353` for the retained formula.

Full correction (`beta=1`) is formally calibrated but extremely heavy-tailed:
rare exact long strings dominate its expectation. In finite 8K to 1M samples
at `D=8`, observed non-null mass was only about `0.06..0.10`. Values
`0.1..0.5` avoid that failure. `beta=0.25` is retained because it gave perfect
8-seed contextual validation while preserving substantially faster exact-match
evidence growth than the square-root score for normal symbol widths.

The implementation evaluates `I_beta` in log space:

```text
r[t,a] = log(g_i[t,a]) - beta*log(z)
         + logaddexp(0, r[t-1,a-1]).
```

It then subtracts the corresponding log geometric normalizer. This remains
finite for long exact suffixes where a value-domain product would overflow.

## 6. Dual Distribution

The two branches are normalized independently and mixed in probability space:

```text
P_dual = (1-rho) * P_discovery + rho * P_information.
```

The default is `rho=0.5`. Mixing logits or raw scores was rejected because
their units and growth laws differ; one branch would silently set the other's
effective temperature. Probability mixing guarantees valid normalization and
preserves nonzero dense support from both credit geometries.

The static controls have deliberately narrow meanings:

| Control | Meaning |
| --- | --- |
| `mismatch_scale` | Shared Hamming penalty `lambda`; discovery divides its mismatch count by `D`, information does not |
| `scale` | Discovery logit scale only; information is already in natural log-evidence units |
| `evidence_power` | Fraction `beta` of random-collision log evidence credited per suffix step |
| `information_weight` | Probability-mixture coefficient `rho`; ignored by either single mode |
| `dropout_p` | Probability of dropping a final post-softmax backward route weight |

Multiplying `I_beta` by `scale` would generally destroy
`E[exp(I_beta)]=1`, so information intentionally has no additional temperature.

A hierarchical alternative was also tested: information controlled only total
non-null mass while discovery ranked candidates. Its best exact-bitflip cosine
was about `0.511`, below the retained probability mixture, and it degraded as
information weight increased. It is therefore not implemented.

## 7. Backward Data Flow

The public research function is a custom autograd operation:

```text
Q/K/V logits
  -> exact hard signs
  -> unlimited exact diagonal DP
  -> latest-longest route
  -> exact hard output                         [forward result]

saved Q/K/V logits + upstream dO
  -> hard signs with softsign VJP
  -> all-pairs mismatch counts
  -> selected dense score recurrence(s)
  -> independently normalized route probabilities
  -> optional probability mixture
  -> optional post-softmax dropout
  -> weighted hard V carrier
  -> autograd replay                          [Q/K/V surrogate VJP]
```

The sign forward is hard while its declared local derivative is

```text
d softsign(x) / dx = 1 / (1 + abs(x))^2.
```

This keeps forward/train/inference semantics aligned. The value carrier also
uses hard values with the same VJP, so value magnitude cannot leak through the
forward. Dropout is applied after the final distribution and affects only this
carrier.

## 8. Validation Results

### Semantic and numerical checks

The focused suite currently has 106 passing tests. It covers:

- scalar DP and log-evidence oracles;
- exact random-background normalization by exhaustive enumeration;
- hard parity with the production reference;
- suffixes longer than 32;
- null and latest-tie semantics;
- dense and packed-varlen parity, including empty segments;
- grouped value heads and non-contiguous tensors;
- partial Q/K/V gradient masks and the singleton connected-zero case;
- deterministic post-softmax dropout replay;
- FP16/BF16 on physical GPU0, an RTX 3070.

The low-precision hard output was bit-exact with FP32. Every tested gradient was
finite; FP16/BF16 gradient cosine against the corresponding FP32 calculation
was at least `0.999996` in the direct probe.

### Exact one-bit counterfactual alignment

The matched matrix contains 96 cells: eight seeds, `T in {4,6,8}`, and
`D in {1,2,4,8}`. Every estimator receives the same hard input, value, loss
weight, and exact full-bitflip oracle.

| Estimator | Mean cosine | Sign agreement | Useful top-k |
| --- | ---: | ---: | ---: |
| Linear suffix sum | 0.50629 | 0.80122 | 0.36489 |
| Production square root | 0.50363 | 0.80832 | 0.37139 |
| Old collision LR | 0.43491 | 0.77168 | 0.35312 |
| Information, beta=.25 | 0.52447 | 0.83356 | 0.38889 |
| Dual, beta=.25/rho=.5 | **0.53529** | 0.83243 | 0.38520 |

The width breakdown explains why mixing helps:

| D | Square-root cosine | Information cosine | Dual cosine |
| ---: | ---: | ---: | ---: |
| 1 | 0.58964 | 0.55712 | 0.58633 |
| 2 | 0.66737 | 0.66660 | 0.67560 |
| 4 | 0.37545 | 0.43733 | 0.43649 |
| 8 | 0.16518 | 0.28105 | 0.27754 |

Information is most valuable where a wider symbol genuinely carries stronger
collision evidence; discovery protects the narrow-symbol regime.

### Random-background null scaling

With suffix length 32 and four probes per cell, information remains near 0.5
non-null mass as the candidate count grows:

| D | Discovery at 8K | Information at 8K | Discovery at 1M | Information at 1M |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7794 | 0.5018 | 0.7762 | 0.5002 |
| 2 | 0.5795 | 0.5022 | 0.5784 | 0.5001 |
| 4 | 0.5011 | 0.5012 | 0.5014 | 0.5002 |
| 8 | 0.4748 | 0.4846 | 0.4752 | 0.5000 |

The dual mass is exactly halfway between these columns at `rho=0.5`; it trades
perfect null calibration for the discovery branch's gradient geometry.

### Exact-suffix capacity

The smallest ideal exact suffix that beats null at default parameters is:

| Candidate count | Discovery, any D | Information D=1 | D=2 | D=4 | D=8 | D=16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 24 | 48 | 23 | 11 | 4 | 1 |
| 1M | 48 | 78 | 38 | 18 | 8 | 3 |
| 100M | 78 | 106 | 52 | 25 | 12 | 5 |

This is the intended width-aware behavior. `D=1` has little information per
symbol and can be worse than discovery; `D>=4` scales much better.

### Shortcut-free contextual recall

On the 8-seed, 500-step contextual gate:

| Estimator | Passed | Mean/min exact validation | Median first exact | ms/step |
| --- | ---: | ---: | ---: | ---: |
| Unbounded discovery | 6/8 | 0.99365 / 0.97656 | 127 | 21.56 |
| Unbounded information | 5/8 | 0.97461 / 0.90234 | 134 | 22.27 |
| Unbounded dual | **8/8** | **1.0 / 1.0** | 112 | 26.87 |

The optimized single-branch implementation is about 15 percent faster than
the first prototype, which accidentally evaluated both branches. Dual costs
about 25 percent more than discovery in this Python/autograd benchmark.

`beta=0.1` also passed 8/8 and reached first exact earlier (median 102.5), but
its minimum validation accuracy was 0.9922 rather than 1.0. The tiny bitflip
mean advantage did not justify replacing the more reliable `beta=0.25`.

### Controlled long-suffix repair

At `D=8`, 8K training context, and 64K/1M evaluation:

| W | Production | Information | Dual | Exact relevant bitflip |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.625 | 0.750 | 0.750 | 1.000 |
| 4 | 0.500 | 0.500 | 0.500 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32 | 1.000 | 1.000 | 1.000 | 1.000 |

Every successfully repaired model kept the same exact hard route at 1M. The
failed `W=2/4` runs had a wrong initial gradient sign, not a vanishing gradient.
At `D=4`, information/dual also regressed one `W=2` seed relative to production
(`0.875` versus `1.0`). Stronger or better-calibrated evidence cannot infer an
exact discrete counterfactual when random routed values assign misleading
credit among many distractors.

## 9. Complexity and Kernel Contract

The materialized PyTorch oracle intentionally favors auditability:

- hard forward: `O(B*H*T^2*D)` work and `O(B*H*T^2)` materialized state;
- dense surrogate: the same asymptotic work/state, plus value accumulation;
- final-query research specialization: `O(P*T*W*D)` work and `O(P*T*W)` state;
- packed varlen: the sum of each segment's independent quadratic cost.

A production kernel may stream tiles and reduce saved state, but must preserve:

1. one exact all-pairs mismatch count shared by selected score branches;
2. unlimited diagonal continuation across tile boundaries;
3. FP32 score/softmax accumulation for FP16/BF16 inputs;
4. one online softmax state for each independently normalized branch;
5. probability-space mixing only after both normalizations;
6. the same dropout counter for the final mixed route weight;
7. dense Q/K/V credit for every causal route;
8. no materialized `T^2` tensor in the optimized implementation.

For `D<=32`, a kernel can use popcount for the hard mismatch and lookup tables
for `exp(-lambda*h/D)` and `-lambda*h`. It must still apply the declared manual
surrogate derivative; an integer lookup cannot itself carry the Q/K VJP.
Information needs a stable log recurrence (`logaddexp`/softplus) or an exactly
equivalent rescaling scheme. Its route-length normalizer depends only on
`D`, `lambda`, `beta`, and available suffix length and can be precomputed in
`O(T)` storage.

## 10. Known Limits

- This is a deterministic surrogate, not an unbiased derivative of discrete
  latest-longest routing.
- Random-null calibration assumes independent balanced bits. Learned bit
  imbalance or Q/K correlation can shift it. A detached per-head background
  estimate is a possible future experiment, not part of this minimal oracle.
- Dual requires two normalized recurrences and therefore more compute/state.
- The controlled `W=2/4` failures prove that score geometry alone does not
  solve every value-mediated credit conflict.
- Natural-text collisions, shared causal trunks, and multi-layer ROSA training
  remain required before production promotion.
- The Python custom backward replays autograd and is intentionally not a
  `torch.compile(fullgraph=True)` contract. Compiled support belongs to the
  eventual registered operator, not this readable oracle.

## 11. Files and Reproduction

Core files:

- `benchmarks/dual_score_reference.py`
- `benchmarks/dual_score_ablation.py`
- `benchmarks/dual_score_null_scaling.py`
- `tests/test_dual_score_reference.py`
- `tests/test_dual_score_ablation.py`
- `tests/test_dual_score_null_scaling.py`

Primary validation records:

- `validation/dual_score_ablation_v2.json`
- `validation/dual_score_contextual_optimized_v1.json`
- `validation/dual_score_contextual_beta010_rho05_v1.json`
- `validation/dual_score_long_suffix_v1.json`
- `validation/dual_score_long_suffix_beta025_v1.json`
- `validation/dual_score_long_suffix_d4_v1.json`
- `validation/dual_score_null_scaling_v1.json`
- `validation/dual_score_null_scaling_beta1_v1.json`

Focused tests on physical GPU0:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m pytest -q \
  tests/test_dual_score_reference.py \
  tests/test_dual_score_ablation.py \
  tests/test_dual_score_null_scaling.py \
  tests/test_contextual_estimator_recall.py \
  tests/test_long_suffix_extrapolation.py
```

Null scaling through one million candidates:

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/dual_score_null_scaling.py \
  --device cuda:0 \
  --candidate-counts 8192 65536 1048576 \
  --symbol-dims 1 2 4 8 \
  --suffix-length 32 --probes 2 --seeds 0 1 \
  --evidence-power 0.25 --information-weight 0.5 \
  --json-out validation/dual_score_null_scaling_v1.json
```

The reference is complete enough to serve as a numerical and semantic oracle.
The evidence does not yet justify replacing the frozen production estimator.
