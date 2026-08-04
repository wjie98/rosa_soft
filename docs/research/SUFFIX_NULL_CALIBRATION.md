# Suffix Utility And Null Calibration

This record separates three questions that are easy to conflate:

1. how a differentiable suffix proxy assigns local Q/K credit;
2. how suffix evidence grows with matched length;
3. how one null hypothesis competes with a growing candidate set.

Every training proxy below keeps the exact longest/latest hard forward. Only
the backward carrier changes.

## 1. Normalized Power Family

The tested utility family is

```text
U_alpha(S) = ((1 + S)^alpha - 1) / (2^alpha - 1), alpha > 0
U_0(S)     = log(1 + S) / log(2)
```

All members satisfy `U(0)=0` and `U(1)=1`. The production square-root utility
is `alpha=1/2`, because `1/(sqrt(2)-1)=sqrt(2)+1`:

```text
U_sqrt(S) = (sqrt(2)+1) (sqrt(1+S)-1)
```

Lower alpha spreads gradient more evenly among close competitors. It can
therefore repair a short undertrained route, but compresses the evidence gap
between genuinely short and long exact matches.

At `D=4`, `mismatch_scale=3`, and 500 directed-competition steps:

| Utility | Frozen distractor | Repaired target |
| --- | ---: | ---: |
| linear / alpha=.75 | 8 | 16 |
| square root | 16 | 32 at step 173 |
| alpha=.25 | 32 | 64 at step 424 |
| logarithm | 32 | 64 at step 95 |

That local win did not transfer cleanly. Across 48 ordinary fitting cells,
square root passed `32/48` and alpha=.25 passed `30/48`. More importantly, at
`scale=1` and `N=10^8`, the ideal exact suffix length needed to overcome the
candidate prior grows approximately as follows:

| Utility | Required exact length |
| --- | ---: |
| linear | 19 |
| square root | 78 |
| alpha=.25 | 439 |
| logarithm | 496,000 |

The square-root utility remains the production compromise. There is no new
alpha parameter in the public operator.

## 2. Candidate-Count Correction

For a row with `N` valid non-null routes, production subtracts `log(N)` from
every non-null logit. The resulting non-null partition is an average candidate
weight, not a sum that grows mechanically with context length.

Removing this correction makes random-background non-null recall approach one
as `N` grows. It is therefore part of the estimator semantics, not an optional
large-context heuristic.

With `mismatch_scale=3`, `scale=1`, and a fixed null score of `0.5`, Monte Carlo
random-background recall was approximately:

| Q/K bits | Random-background non-null mass |
| ---: | ---: |
| 1 | 0.775 |
| 2 | 0.578 |
| 4 | 0.501 |
| 8 | 0.475 |

Increasing W beyond 8 changes these values little because random prefix
products decay geometrically. The fixed null score is therefore already well
calibrated for the default `D=4`; making it dynamic would add policy without a
demonstrated training gain.

## 3. Collision Likelihood Ratio

Under independent uniform hard bits, the mismatch count is
`h ~ Binomial(D, 1/2)`. For the local gate

```text
g = exp(-mismatch_scale * h / D)
```

the random-null mean is known exactly:

```text
z0 = E[g] = ((1 + exp(-mismatch_scale/D)) / 2)^D
```

Normalize each gate as `r=g/z0`. Every prefix product has expectation one
under the random-collision null. The research route uses the log of the
uniform mixture over valid prefix products, a deterministic dense VJP with no
Bernoulli estimator and no added sampling variance. Null logit zero and the
same `-log(N)` candidate correction then have a direct likelihood-ratio
interpretation.

This improves theoretical large-candidate capacity and passed `35/48` broad
fitting cells versus production's `32/48`. In shortcut-free contextual recall,
both passed `8/8` seeds; median first-exact step was `73.5` for collision LR
and `78.5` for production.

It still fails the decisive directional gate: collision LR passed `7/12`
long-versus-short competition cells versus production's `9/12`, and some long
routes gave the newest repair bit a wrong-direction gradient. A calibrated
likelihood ratio is not automatically a good credit-assignment geometry.

## 4. Decision

- Keep normalized square root, fixed `null_score=0.5`, and `-log(N)` in
  production.
- Keep power/log and collision LR in research benchmarks only.
- Reconsider collision LR only after it passes directed long-route repair,
  broad multi-seed fitting, shortcut-free contextual recall, and candidate
  calibration together.
- Do not spend CUDA specialization or public API surface on a proxy that has
  not passed those semantic gates.

Raw null-calibration output is stored in
`validation/null_calibration_ablation.json`. Reproduction commands are listed
in `benchmarks/README.md`. Power-family competition and fitting outputs are in
`validation/suffix_power_*.json`. Collision-LR competition, fitting, and
contextual recall outputs are stored in the correspondingly named
`validation/collision_lr_*.json` and
`validation/contextual_collision_lr_*.json` files.
