# Historical RosaSoft Temperature Ladder

> Historical research record. The private script used for this sweep is not
> shipped and targets a deleted prototype API. The decisions remain design
> evidence; current validation uses the production tests and fitting example.
> Current production uses the reciprocal attention convention
> `scale = 1 / route_temperature`.

## 1. Decision Under Test

Temperature changes only the backward action distribution:

```text
P(a) = softmax(R(a) / route_temperature)
```

It does not change candidate order, the hard action, or the maximum suffix
window. The experiment tests whether it changes:

1. the final reachable hard suffix length;
2. only the number of optimization steps needed to reach that length;
3. route discovery when the target starts at different proxy ranks;
4. the effect of the fixed null score.

No route_temperature schedule or value derived from D, T, W, training step, or
diagnostics exists in the operator.

## 2. Predeclared Grid

The final static grid is exactly:

```text
route_temperature = {0.5, 1, 2}
```

`route_temperature=1` is the natural baseline because one additional exact suffix
term adds approximately one score unit and therefore one unit of softmax
log-odds. Values outside this grid are not part of this calibration.

## 3. Controlled Construction

For each target suffix length:

```text
L = {1, 2, 4, 8, 16, 32}
T = 10L + 1
D = 4
W = L
mismatch_penalty = 1
```

Nine non-overlapping candidate blocks are constructed. The target begins with
hard length `L-1`; one trainable Q logit at the oldest suffix position must
cross zero before the target reaches exactly `L`.

The initial target proxy rank is fixed to:

| Rank class | Exact non-null rank | Higher-score distractors |
| --- | ---: | ---: |
| `top1` | 1 | 0 |
| `top4` | 4 | 3 |
| `outside_top8` | 9 | 8 |

All route_temperature and null runs share identical Q/K/V tensors, initial logits,
and fixed mismatch samples. Eight perturbation seeds are crossed with every
configuration. The route utility remains active until success, but success is
recorded only when the hard action is the target and its hard suffix length is
exactly L. Soft values never enter forward.

## 4. 200-Step Ladder

The optimizer is static SGD with learning rate `0.1`. Each cell contains
`successes / 24` followed by the conditional mean success step.

| L | route_temperature `0.5` | route_temperature `1` | route_temperature `2` |
| ---: | ---: | ---: | ---: |
| 1 | `24/24 @ 7.0` | `24/24 @ 12.7` | `24/24 @ 25.0` |
| 2 | `24/24 @ 9.0` | `24/24 @ 20.4` | `24/24 @ 51.4` |
| 4 | `24/24 @ 11.3` | `24/24 @ 30.1` | `24/24 @ 94.4` |
| 8 | `24/24 @ 16.2` | `24/24 @ 45.2` | `24/24 @ 151.3` |
| 16 | `24/24 @ 29.3` | `24/24 @ 80.5` | `0/24` |
| 32 | `24/24 @ 67.4` | `16/24 @ 146.9` | `0/24` |

The finite-budget result strongly favors lower route_temperature, but it does not
yet distinguish convergence speed from reachability.

## 5. L=32 Long Budget

At 1000 steps every route_temperature reaches the same hard result:

| Temperature | Hard success | Final target length | Mean success step | Maximum step |
| ---: | ---: | ---: | ---: | ---: |
| `0.5` | `24/24` | `32` | `67.4` | `112` |
| `1` | `24/24` | `32` | `183.0` | `257` |
| `2` | `24/24` | `32` | `482.0` | `573` |

Temperature did not change the reachable suffix length in this task. It did
change optimization time by about `7.1x` between `0.5` and `2`, so it is not
an operationally irrelevant parameter.

Initial rank also did not change final reachability:

| Rank | route_temperature `0.5` | route_temperature `1` | route_temperature `2` |
| --- | ---: | ---: | ---: |
| top 1 | `109.9` steps | `255.1` steps | `570.6` steps |
| top 4 | `52.5` steps | `168.5` steps | `470.3` steps |
| rank 9 | `39.9` steps | `125.3` steps | `405.3` steps |

Rank 1 is not automatically easiest. Losing actions with negative route
utility can supply useful gradients, while a nearly saturated top action can
have a smaller softmax Jacobian. Rank alone is therefore not a sufficient
route_temperature diagnostic.

## 6. Cross-Task Route Check

The earlier hard-null and near-null route matrix was extended with
`route_temperature=0.5`. All comparisons below use the same 96 configurations and
paired perturbation seeds per route_temperature.

| Scenario | route_temperature `0.5` | route_temperature `1` | route_temperature `2` |
| --- | ---: | ---: | ---: |
| Hard-null success | `62/96` | `48/96` | `36/96` |
| Hard-null mean step | `15.2` | `17.5` | `22.4` |
| Near-null success | `96/96` | `94/96` | `92/96` |
| Near-null mean step | `3.5` | `6.2` | `12.5` |

This second task also favors lower route_temperature. It still does not establish
that `0.5` is safe for full-model training with much deeper initial ranks.

## 7. Temperature and Null

The full ladder crosses:

```text
route_temperature = {0.5, 1, 2}
null_score = {0, 0.5, 1}
```

There are 1296 paired rows. At every route_temperature, all three null values produce
exactly the same success/failure outcome for every matched
`(L, rank, perturb_seed)` cell.

| Temperature | null `0` | null `0.5` | null `1` |
| ---: | ---: | ---: | ---: |
| `0.5` | `144/144 @ 23.5` | `144/144 @ 23.4` | `144/144 @ 23.3` |
| `1` | `136/144 @ 50.7` | `136/144 @ 50.6` | `136/144 @ 50.5` |
| `2` | `96/144 @ 80.8` | `96/144 @ 80.5` | `96/144 @ 80.1` |

For `L>=8`, null probability is approximately zero and null score has no
material effect. At `L=1`, null changes allocation and gradient scale, but its
direction depends on the task: higher null is slightly faster for the
persistent utility ladder and was worse for route creation. Null cannot be
selected from only one positive-route objective.

## 8. Decisions

1. Keep route_temperature static. Do not derive or schedule it from D, T, or W.
2. Change the production default from `2` to the natural baseline `1`.
3. Retain the public parameter because its training-time effect is material.
4. Stop broad route_temperature tuning. Future ablations use only `{0.5, 1, 2}`.
5. Treat `0.5` as a promising training-recipe candidate, not a new default,
   until full-model fitting and deeper-rank route discovery are measured.
6. Keep the internal null score at `0.5`. Joint calibration shows no reason to
   change it without a null-retention objective.

## 9. Reproduction Status

The old local command is not supported by the current package. Recreate this
sweep only as a new test or experiment against `rosa_soft.testing`, translating
the historical controls to `scale = 1 / route_temperature` and
`mismatch_scale = mismatch_penalty`.
