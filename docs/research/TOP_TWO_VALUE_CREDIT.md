# ROSA Soft Winner Competition and Value Credit

## Scope

This note covers research-only estimators in:

- `benchmarks/top_two_value_reference.py`
- `benchmarks/top_two_value_ablation.py`
- `benchmarks/top_two_long_suffix.py`

They all execute the same exact, unlimited-suffix hard ROSA forward. Only the
custom backward differs. None is wired into the production operator.

## Dense baseline

For route logits `z_a`, binary route values `v_a`, and upstream gradient `G`,
the ordinary softmax carrier is

```text
p_a = exp(z_a) / sum_b exp(z_b)
y   = sum_a p_a v_a
```

Its score credit is

```text
dL/dz_a = p_a (<G,v_a> - sum_b p_b <G,v_b>)
         = sum_b p_a p_b <G,v_a-v_b>.
```

Thus dense attention is already a weighted collection of pairwise route
competitions. Low-probability candidates contribute little, while retaining a
nonzero repair path if the current winner or runner-up is wrong.

The continuous weighted value is only a backward carrier. It is not exposed by
the hard forward and need not itself be a valid binary code.

## Explicit top two

`top2` keeps the two largest logits and renormalizes them. It is a biased,
sparse-support score estimator:

```text
p_top2 = softmax(top2(z)); p_else = 0.
```

It approaches winner/runner-up competition when the score distribution is
already reliable, but prevents every omitted route from receiving direct score
credit. The null route also commonly occupies one of the two slots early in
training, so "top two routes" need not mean the two best non-null suffixes.

## Winner versus aggregate rest

For a chosen winner `w`, ordinary attention has an exact hierarchical form:

```text
r_b = softmax(z_rest)_b
p_w = sigmoid(z_w - logsumexp(z_rest))
c   = sum_{b != w} r_b v_b
y   = p_w v_w + (1-p_w)c.
```

`winner_rest` detaches `r` inside `c`, while retaining gradients through `p_w`.
This preserves the numerical carrier and gives every finite rest candidate a
gradient through `logsumexp(rest)`, but removes competition inside the rest
set. `hard_winner_rest` chooses the exact hard ROSA route as `w`; `winner_rest`
chooses the largest soft logit.

This decomposition provides a clean test of the claim that only the current
winner boundary matters. It is not an unbiased reformulation of dense
attention because the `(1-p_w) dc` term is intentionally removed.

## QK and V roles

The reference computes the two first-order paths separately:

```text
QK carrier: p(Q,K) * stop_gradient(v)
V carrier:  stop_gradient(p) * v(V)
```

Their gradients are exactly those of the ordinary dense carrier. The split
prevents a V-gradient ablation from silently changing QK credit.

`selected_value` replaces the dense V carrier with the exact hard-selected
value. This reproduces the old QK/V-detach idea: QK still sees the soft binary
values, while only the selected V receives value credit.

## Guarantees

All variants preserve:

1. Exact unlimited-suffix hard ROSA output in training forward.
2. No soft value leakage to the model-visible output.
3. Dense candidate support for `dense`, `winner_rest`, and
   `hard_winner_rest`.
4. Exact equality between `split_dense` and the full-horizon production
   reference VJP, up to floating-point evaluation order.
5. O(T^2) dense score work in the reference; no candidate filtering is used.

`top2` deliberately does not preserve dense gradient support.

## Current evidence

The exact bitflip-oracle matrix covers `T={4,6,8}`, QK widths
`{1,2,4,8}`, V widths `{1,2,4,8}`, and eight seeds (384 cells per
estimator).

| Estimator | Mean cosine | Sign agreement | Missed useful bitflips |
|---|---:|---:|---:|
| dense production | 0.4975 | 0.8017 | 0.00% |
| winner/rest, soft winner | 0.3722 | 0.6983 | 0.00% |
| winner/rest, hard winner | 0.4564 | 0.7710 | 0.00% |
| explicit top two | 0.4963 | 0.8184 | 3.07% |
| dual score research control | 0.5344 | 0.8296 | 0.00% |

Across these initial states, the top two routes contain 82.3% of total mass,
but the top two non-null candidates contain only 33.4%. The soft winner is the
null route in 86.8% of rows. Explicit top-two therefore often reduces to null
versus one suffix candidate.

The non-null top-two binary-value collision rate is 56.3% at V1, 28.5% at V2,
7.5% at V4, and 0% in this V8 sample. Wider V reduces exact value collisions,
but bitflip alignment is not monotonic in V width, so collision is not the only
source of estimator error.

On the paired 8-seed, 600-step single-sequence fit:

| Estimator | Ever below 1e-3 | Final below 1e-3 | Median final loss |
|---|---:|---:|---:|
| dense production | 6/8 | 6/8 | 0.000639 |
| selected V | 7/8 | 6/8 | 0.000590 |
| explicit top two | 6/8 | 6/8 | 0.000659 |
| top two + selected V | 6/8 | 6/8 | 0.000696 |
| dual score research control | 7/8 | 7/8 | 0.000627 |
| exact bitflip | 5/8 | 5/8 | 0.000533 |

Explicit top-two can reduce loss faster on individual failed dense seeds, but
does not improve aggregate reliability and can fail on seeds solved by dense.

On the QK8/V4 shortcut-free contextual recall gate (eight paired seeds, 300
steps):

| Estimator | Passed | Mean validation exact | Minimum validation exact |
|---|---:|---:|---:|
| dense production | 7/8 | 0.991 | 0.949 |
| selected V | 4/8 | 0.738 | 0.195 |
| explicit top two | 0/8 | 0.579 | 0.352 |
| top two + selected V | 0/8 | 0.342 | 0.113 |
| dual score research control | 8/8 | 0.999 | 0.996 |
| exact all-bit bitflip | 0/8 | 0.346 | 0.113 |

The top-two and selected-V shortcuts improve some single-sequence fits but do
not survive a contextual task. Exact all-bit bitflip also performs poorly here,
consistent with its high-variance, discontinuous credit rather than with an
implementation mismatch.

## Long-suffix gate

The balanced-binary long gate trains at 8K with `W={1,2,4,8,32}` and evaluates
the exact hard route at 8K, 64K, and 1M. Across four seeds per W:

| Estimator | Training successes | Notable failure |
|---|---:|---|
| dense production | 18/20 | one seed each at W2 and W4 |
| winner/rest, soft winner | 7/20 | seed-sensitive gradient sign |
| winner/rest, hard winner | 20/20 | none in this one-bit gate |
| explicit top two | 10/20 | 0/4 at W32 |
| dual score research control | 19/20 | one seed at W4 |
| exact relevant-bit bitflip | 20/20 | none |

Every solved 8K condition remains exactly correct at 1M. At W32, explicit
top-two has exactly zero initial gradient in all four seeds: its two retained
routes are tied wrong distractors affected identically by the decisive bit, and
the useful target is outside the retained support. This is a deterministic
counterexample to hard top-two truncation.

## Null-route controls

Two additional score controls test the candidate prior:

- `no_prior` removes the non-null `-log(N)` shift relative to null.
- `hard_null_gate` masks null in the backward whenever hard ROSA already has a
  non-null match.

Neither improves the 384-cell oracle (`0.496` and `0.445` cosine versus `0.497`
for production). On contextual recall they pass only 4/8 and 3/8 runs. On the
balanced long gate they reproduce production's 18/20 success pattern.

An adversarial coherent-value long gate exposes a real boundary: null mass can
reverse the dense gradient at W1-W8 because moving probability from wrong
values toward null helps the soft carrier without finding the target. Both
null controls repair that gate to 20/20, while the production control only
solves W32. Since removing null worsens contextual reliability, this boundary
does not justify another production branch or a tuned prior coefficient.

## Provisional decision

Keep dense softmax, the existing candidate prior, and dense V credit as the
minimal production reference. Retain top-two, winner/rest, null controls, and
selected-V only as research controls. A replacement should not be promoted
unless it preserves repair support and improves both multi-seed contextual and
long-suffix gates without condition-specific tuning.
