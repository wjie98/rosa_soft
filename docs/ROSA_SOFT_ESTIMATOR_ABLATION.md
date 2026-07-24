# RosaSoft Estimator Ablation Matrix

> Historical research record. The private `experiments/` workspace is not
> shipped with the package, and its original scripts target deleted prototype
> APIs. Results below inform the fixed production estimator but are not part of
> the current executable validation suite.

## 1. Scope

This matrix tests estimator choices without changing the production operator.
Every variant keeps:

- exact hard Q/K/V signs in forward;
- causal shifted actions, finite suffixes, latest hard ties, and hard null;
- softsign STE for Q/K/V;
- the hard-Hamming Jacobian anchor;
- multiplicative suffix scores;
- dense probabilities over every causal action;
- separated route and value VJPs.

The public `rosa_soft` ABI and CUDA formulas remain unchanged while the
ablation is running.

## 2. Mean-Matched Shape Factor

Sampled mismatch penalties use:

```text
alpha_p(u) = 1 - c_p u^p
c_p = (p + 1) / 8
```

Therefore every shape has:

```text
E[alpha_p] = 0.875
```

| Shape | `p` | `c_p` | Penalty range |
| --- | ---: | ---: | --- |
| Linear | 1 | 0.25 | `[0.75, 1]` |
| Quadratic | 2 | 0.375 | `[0.625, 1]` |
| Cubic, current | 3 | 0.5 | `[0.5, 1]` |

Each shape is crossed with:

- `single`: use `alpha_p(u)`;
- `antithetic`: average gates from `alpha_p(u)` and `alpha_p(1-u)`.

This is a strict `3 x 2` factorial comparison. Single and antithetic runs use
the same data, initialization, and uniform stream.

| Estimator ID | Mismatch penalty | Branches | Paired comparison |
| --- | --- | ---: | --- |
| `linear_single` | `1 - 0.25u` | 1 | `linear_antithetic` |
| `linear_antithetic` | `1 - 0.25u`, `1 - 0.25(1-u)` | 2 | `linear_single` |
| `quadratic_single` | `1 - 0.375u^2` | 1 | `quadratic_antithetic` |
| `quadratic_antithetic` | `1 - 0.375u^2`, `1 - 0.375(1-u)^2` | 2 | `quadratic_single` |
| `cubic_single` | `1 - 0.5u^3` | 1 | `cubic_antithetic` |
| `cubic_antithetic` | `1 - 0.5u^3`, `1 - 0.5(1-u)^3` | 2 | `cubic_single` |

Within one shape, single and antithetic have exactly the same expected gate:
`E[g(u)] = E[(g(u) + g(1-u))/2]`. The branch comparison therefore changes
variance and finite-sample geometry, not the expected gate. Across shapes only
the expected penalty is matched. Their expected exponential gates are not
identical because `exp(-lambda alpha)` is nonlinear; this limitation is
reported rather than hidden.

Two deterministic controls are separate from the factorial:

| Control | Purpose |
| --- | --- |
| `cubic_expected` | Removes random exploration while exactly matching the expected one-bit cubic gate for each lambda. |
| `hard_hamming` | Uses `exp(-lambda*h)` numerically and in the anchored Jacobian. |

## 3. Experimental Layers

| Layer | What it measures | Primary metrics |
| --- | --- | --- |
| Geometry/VJP | Local estimator alignment and perturbation variance | hard probability, hard NLL, effective actions, gradient noise scale/cosine |
| Hard-null route, `h=D` | Ability to create a route from an all-bit mismatch | success rate, steps to sign crossing, first Q gradient |
| Near-null route, `h=1` | Same test with exactly one target mismatch | success rate, steps to sign crossing, first Q gradient |
| Hard-forward fit | Joint Q/K/V optimization and discrete collapse | hard CE, hard accuracy, success rate, time to thresholds |
| Null retention | Avoiding a spurious route when null is correct | positive recall, false-route rate, joint success |
| Held-out tasks | Whether a result generalizes beyond the tiny motif | associative recall, longer suffix fitting, joint Q/K/V fitting |

Model/data seeds and perturbation seeds are crossed, not conflated. Deterministic
controls run once per model seed.

## 4. Static Parameter Matrix

The broad calibration grid is:

| Factor | Values |
| --- | --- |
| `D` | `2, 4, 8, 16` for geometry; `2, 4, 8` for route tests |
| `T` | `16/32, 64/128` |
| `W` | `1, 4/8, 32` where valid |
| `mismatch_penalty` | `0.5, 1, 2, 3, 5` |
| `route_temperature` | `1, 2, 4` |
| `null_score` | `0, 0.5, 1` for geometry; add `2` for route boundaries |
| Symbol scenarios | IID/null-heavy, repeated exact suffixes, one-bit-corrupted suffixes |

Geometry calibration reports a Pareto surface. It must not select parameters by
hard NLL alone because a sharper hard approximation can suppress exploration.
The final route_temperature-only ladder is documented separately in
`ROSA_SOFT_TEMPERATURE_LADDER.md`.

## 5. Predeclared Decisions

The antithetic branch is retained only if, on held-out experiments:

- paired hard-fit success improves by at least 5 percentage points;
- the paired difference is significant at `p < 0.05`;
- the gain appears in at least two task families or materially reduces
  convergence time;
- no task family has a material regression.

Otherwise the second branch is deleted. Its measured PyTorch cost is about
`4%`; the eventual CUDA decision must also remeasure kernel time and registers.

Cubic is replaced only if a mean-matched alternative improves paired success
by at least 5 percentage points without reducing model-seed coverage or hard
accuracy on held-out shapes.

Static lambda, route_temperature, and null values are selected per declared model
regime. No dynamic or window-derived schedule is introduced into the operator.

## 6. Executed Results

### 6.1 Geometry/VJP Pilot

The paired pilot contains 288 configurations at `D=4/8`, `T=16`, `W=4/8`,
three symbol scenarios, and three data seeds.

| Variant | Q/K gradient noise scale | Hard probability |
| --- | ---: | ---: |
| Cubic single | `2.569e-3` | `0.28904` |
| Cubic antithetic | `5.661e-4` | `0.28900` |
| Cubic expected | `0` | `0.28910` |

Antithetic averaging reduces local gradient noise by about `4.5x`, but the
absolute noise is already small and mean route geometry is unchanged.

### 6.2 Full `3 x 2` Fitting Pilot

The fit pilot contains 96 paired 1000-step trajectories:

```text
6 variants x 8 model seeds x 2 perturbation seeds
```

| Variant | CE `<1e-3` | Model seeds with any success | 100% hard accuracy | Step time |
| --- | ---: | ---: | ---: | ---: |
| Linear single | `7/16` | `4/8` | `7/16` | `11.714 ms` |
| Linear antithetic | `8/16` | `4/8` | `8/16` | `12.276 ms` |
| Quadratic single | `6/16` | `3/8` | `7/16` | `11.657 ms` |
| Quadratic antithetic | `7/16` | `4/8` | `7/16` | `12.221 ms` |
| Cubic single | `8/16` | `6/8` | `9/16` | `11.898 ms` |
| Cubic antithetic | `9/16` | `6/8` | `9/16` | `12.323 ms` |

Across all three shapes, antithetic has `24/48` successes and single has
`21/48`. Among 11 discordant pairs, antithetic wins 7 and single wins 4; the
two-sided exact paired result is not significant. The second branch is still a
deletion candidate.

Across both branch modes, cubic has `17/32` successes, linear `15/32`, and
quadratic `13/32`. Cubic currently has the best model-seed coverage and must not
be removed based on this pilot.

### 6.3 Hard-Null Boundary Calibration

The valid boundary run contains 768 cases with independent Q bits, initial
logit `-0.1`, and no initial exact route.

| `D` | lambda `0.5` | lambda `1` | lambda `2` | lambda `3` |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `56.3%` | `81.3%` | `87.5%` | `75.0%` |
| 4 | `75.0%` | `75.0%` | `12.5%` | `0%` |
| 8 | `50.0%` | `0%` | `0%` | `0%` |

Other aggregate effects in this controlled task:

| Factor | Result |
| --- | --- |
| Context | `T=16: 55.2%`, `T=64: 30.2%` |
| Suffix | `W=1: 27.1%`, `W=4: 58.3%` |
| Temperature | `1: 47.9%`, `2: 37.5%` |
| Null score | `0.5: 43.8%`, `1: 41.7%` |
| Branch | single and antithetic both `42.7%`; all 384 pairs agree on success |

This task exposed why the former lambda `3`
cannot create an all-bit-mismatched route at `D>=4`. It does not imply that
lambda `0.5` is universally optimal; already-near matches have a different
Hamming distribution.

### 6.4 Near-Null Boundary Calibration

The matched near-null run also contains 768 cases, but every initial target is
exactly one bit from a hard route.

| `D` | lambda `0.5` | lambda `1` | lambda `2` | lambda `3` |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `62.5%` | `100%` | `100%` | `87.5%` |
| 4 | `100%` | `100%` | `100%` | `100%` |
| 8 | `100%` | `100%` | `100%` | `100%` |

Other aggregate effects:

| Factor | Result |
| --- | --- |
| Context | `T=16: 97.9%`, `T=64: 93.8%` |
| Suffix | `W=1: 91.7%`, `W=4: 100%` |
| Temperature | `1: 96.9%`, `2: 94.8%` |
| Null score | `0.5: 96.9%`, `1: 94.8%` |
| Branch | single and antithetic both `95.8%`; all 384 pairs agree on success |

The reversal relative to the hard-null task is decisive: lambda sensitivity is
controlled by the current Hamming distance and competing-route margin, not by
`D` alone. `D` changes the expected distance distribution under an
initialization, but it is not itself a sufficient lambda schedule.

### 6.5 Null-Score Boundary Calibration

The null extremes use `cubic_single`; the already measured `0.5/1` rows are
paired with new `0/2` rows using identical geometry and perturbation seeds.
There are 192 matched configurations per scenario and null value.

| Scenario | null `0` | null `0.5` | null `1` | null `2` |
| --- | ---: | ---: | ---: | ---: |
| Hard-null, `h=D` | `44.8%` | `43.8%` | `41.7%` | `36.5%` |
| Near-null, `h=1` | `97.9%` | `96.9%` | `94.8%` | `89.6%` |

Success is monotone non-increasing with null score in every paired grid cell:
there are no cases where a higher null score succeeds and a lower one fails.
This is route-discovery evidence only. A higher hard-null gradient RMS did not
improve sign crossing, so gradient magnitude alone is not a valid null-score
selection metric.

Null score cannot be finalized until the complementary null-retention task is
run. Setting it to zero may improve route creation while also encouraging
spurious routes on examples whose correct action is null.

### 6.6 Temperature Length Ladder

A 432-row controlled ladder crosses target lengths `1/2/4/8/16/32`, exact
initial target ranks `1/4/9`, eight perturbation seeds, and static route_temperatures
`0.5/1/2`. At a 200-step budget lower route_temperature is much faster. In a
1000-step `L=32` confirmation, all three route_temperatures reach `24/24` hard
successes and final length 32:

| Temperature | Mean success step |
| ---: | ---: |
| `0.5` | `67.4` |
| `1` | `183.0` |
| `2` | `482.0` |

Temperature changes optimization time, not the reachable suffix length in this
controlled task. Its `7.1x` convergence-step effect is material, so the public static
control is retained. The production default changes from `2` to `1`.

A 1296-row route_temperature/null cross has identical paired success outcomes for
null scores `0/0.5/1`. Null is negligible once suffix scores are large and
remains task-dependent for weak, short routes. See
`ROSA_SOFT_TEMPERATURE_LADDER.md` for the complete matrix.

### 6.7 Held-Out Antithetic Decision

The held-out run crosses 20 data/initialization seeds with two perturbation
seeds. Associative recall trains four independent historical routes. The
long-suffix task extends a length-4 hard route through three mismatches to
length 16. Single and antithetic variants share data, initialization, and the
base uniform stream.

| Task | Cubic single | Cubic antithetic | Mean success step, single / antithetic |
| --- | ---: | ---: | ---: |
| Associative recall | `32/40` | `32/40` | `53.56 / 53.69` |
| Long suffix | `40/40` | `40/40` | `70.25 / 74.95` |

All 80 paired success outcomes agree, so the paired success difference is zero
with exact `p=1`. On jointly successful long-suffix runs, antithetic is faster
in 10 pairs, slower in 28, and tied in 2; the two-sided sign test is
`p=0.00510` in the direction of single being faster. The predeclared retention
threshold is not met. The production estimator therefore deletes the second
branch; the research variant remains available to reproduce the decision.

### 6.8 Null Retention

The paired task shares one trainable bit between a positive row and a null
row. Establishing the required positive route deliberately creates a false
route in the null row; success requires preserving the positive route while a
private null-row bit removes the false route.

| Null score | Joint success | Mean success step |
| ---: | ---: | ---: |
| `-1` | `40/40` | `7.45` |
| `0` | `40/40` | `7.80` |
| `0.5` | `40/40` | `7.80` |
| `1` | `40/40` | `8.15` |
| `2` | `40/40` | `10.13` |
| `3` | `40/40` | `15.83` |
| `4` | `40/40` | `32.55` |
| `5` | `40/40` | `77.45` |
| `6` | `0/40` | not reached |

`0.5` is inside a broad stable plateau. Scores above about `4` suppress route
learning sharply, so the internal null score remains the static constant
`0.5`; no null schedule is justified.

### 6.9 Numerical-Gate Decision

Twenty held-out seeds compare the sampled cubic gate, its exact expected gate,
and the hard-Hamming numerical gate. All use the same hard-Hamming Jacobian.

| Task | Sampled cubic | Cubic expectation | Hard Hamming |
| --- | ---: | ---: | ---: |
| Associative | `16/20 @ 53.63` | `16/20 @ 53.69` | `16/20 @ 55.13` |
| Long suffix | `20/20 @ 68.25` | `20/20 @ 76.15` | `20/20 @ 127.10` |
| Null retention | `20/20 @ 7.85` | `20/20 @ 7.80` | `20/20 @ 7.90` |

The sampled single branch is retained. Deterministic expectation remains a
debugging control; hard Hamming is too slow on staged long-suffix discovery.

### 6.10 Jacobian-Soft Decision

The hard, midpoint, and fully relaxed Jacobian distances use identical hard
forward values and sampled numerical gates.

| Task | Hard | Midpoint | Relaxed |
| --- | ---: | ---: | ---: |
| Associative | `32/40 @ 53.56` | `32/40 @ 51.28` | `32/40 @ 50.19` |
| Long suffix | `40/40 @ 70.25` | `40/40 @ 73.03` | `40/40 @ 78.48` |
| Null retention | `40/40 @ 7.83` | `40/40 @ 7.65` | `40/40 @ 7.95` |

Midpoint and relaxed are faster on the short task, but slower on long suffixes.
Against hard, their long-suffix paired sign-test p-values are `0.00107` and
`1.43e-5`. Production keeps the simpler hard-distance Jacobian.

### 6.11 Value-Credit Decision

A joint task starts both the target Q route and target V symbol with wrong
bits. It measures structural hard routing and V learning together.

| Workload | Distributed soft V | Hard-selected V |
| --- | ---: | ---: |
| One association | `14/40` (`35%`) | `30/40` (`75%`) |
| Four competing associations | `2/40` (`5%`) | `16/40` (`40%`) |

The one-association difference is stable at learning rates `0.1`, `0.25`, and
`0.5`. In the four-association run hard-selected wins 14 discordant pairs,
loses none, and has exact `p=0.000122`. Distributed V spreads the target value
across candidates and can destroy route identity in this direct construction.

The required full-model fit gate reverses the production decision:

| Value credit | Lambda | Final hard CE | Accuracy |
| --- | ---: | ---: | ---: |
| Hard selected | `0.5` | `0.157` | `93.75%` |
| Hard selected | `1` | `0.172` | `93.75%` |
| Hard selected | `3` | `0.157` | `93.75%` |
| Distributed, reference | `0.5` | `0.157` | `93.75%` |
| Distributed, reference | `1` | `3.84e-4` | `100%` |
| Distributed, reference | `2` | `5.50e-4` | `100%` |
| Distributed, reference | `3` | `4.78e-4` | `100%` |
| Distributed, CUDA | `3` | `3.83e-4` | `100%` |

Hard-selected V also reduced a `D_v=64`, `T=256/512` kernel probe by about
`17%/14%`, but performance cannot override the fit regression. Production
therefore retains distributed V credit. Hard-selected V remains a useful
research control; a future constrained V estimator must pass both the route
identity probe and full-model fitting.

### 6.12 Static Lambda/Temperature Decision

The held-out Q/K cross uses 20 data seeds, two perturbation seeds, fixed null
`0.5`, and the hard Jacobian. V is fixed in these three tasks, so their route
results are independent of the V-credit choice.

| Temperature | Lambda | Associative | Long suffix | Null retention |
| ---: | ---: | ---: | ---: | ---: |
| `0.5` | `0.5` | `32/40 @ 18.84` | `10/40 @ 420.0` | `40/40 @ 5.40` |
| `0.5` | `1` | `32/40 @ 28.13` | `0/40` | `40/40 @ 4.93` |
| `1` | `0.5` | `32/40 @ 49.28` | `40/40 @ 9.53` | `40/40 @ 9.10` |
| `1` | `1` | `32/40 @ 53.56` | `40/40 @ 70.25` | `40/40 @ 7.83` |
| `2` | `0.5` | `28/40 @ 92.32` | `40/40 @ 3.75` | `40/40 @ 17.00` |
| `2` | `1` | `28/40 @ 94.68` | `40/40 @ 5.40` | `40/40 @ 13.43` |

Low route_temperature is not generally easier: it locks onto the current shorter
suffix competitor and can remove the gradient needed to establish a longer
route. High route_temperature helps that construction but regresses short
associative discovery. This selects static `route_temperature=1`.

The controlled route tasks favor lambda `0.5`, but the full-model fit at
lambda `0.5` fails with either V estimator. Lambda `1` fits under the PyTorch
random stream but not the seed-0 CUDA counter stream; lambda `3` fits both
implementations and also reaches `100%` on CUDA seed 2. The conservative
production default therefore remains `mismatch_penalty=3`. This is a default, not
a D/T/W formula: active Hamming distance and competition geometry explain why
different tasks reverse the ranking.

## 7. Current Decisions

- Keep the cubic single-branch numerical gate.
- Keep stochastic exploration.
- Delete antithetic averaging from the production estimator.
- Fix the internal null score at `0.5`.
- Keep the hard-distance Jacobian anchor.
- Retain distributed soft V credit; hard-selected V failed full-model fitting.
- Do not derive lambda from `D` or configured `W`. Select one static value per
  declared initialization regime using its observed Hamming-distance
  distribution.
- Use static `mismatch_penalty=3` and `route_temperature=1` as production defaults.
- Retain the public route_temperature control: it did not change final reachable
  length in the isolated ladder, but changes optimization speed and can lock
  out longer routes in competitive contexts.
- Stop broad route_temperature tuning. Any further static comparison is restricted
  to `0.5/1/2`; no D/T/W-derived schedule is allowed.
- Stratify calibration by T. Candidate normalization reduced route gradients
  and success from short to long contexts; this does not justify an in-operator
  dynamic schedule.

## 8. Remaining Decision Gates

The estimator-level gates in this matrix are closed. The remaining external
validation is full-model pretraining at realistic depth and context, with
active Hamming distance, candidate entropy, hard-route recall, and suffix
length reported. Such runs may motivate explicit per-run parameter choices,
but not an in-operator adaptive schedule.

## 9. Reproduction Status

The original local scripts are intentionally excluded from the public branch
and no longer match the production API. A new experiment must port the desired
variant onto `rosa_soft.testing`, pin its random samples, and compare its VJP
against `rosa_soft_reference`; the obsolete commands are not a supported
validation path.

Invalid shared-logit and all-failure `-0.5` pilots are discarded and must not
be used for success-rate conclusions.
