# Historical RosaSoft Estimator Ablation Matrix

> Historical research record. The private `experiments/` workspace is not
> shipped with the package, and its original scripts target deleted prototype
> APIs. Sections 1 through 7 document estimators that existed before the
> deterministic-minimal reduction; they do not describe current production
> code. Section 8 records that historical reduction, section 9 records the
> static-default correction after dropout entered the operator, and section 12
> records the current suffix-score geometry.
>
> Sections 1 through 6.12 predate mismatch-fraction normalization, hard-tier
> proxy scores, and candidate-normalized null allocation. Their absolute
> lambda and null-score conclusions must not be applied to the current
> estimator. The normalized tiered-prototype recalibration begins in section
> 7; the current default correction is in section 9.
>
> Terminology mapping for the current API: `scale = 1 / route_temperature`,
> `mismatch_scale = mismatch_penalty`, and training `payload` is now `value`.
> The post-softmax attention-dropout candidate recorded here was subsequently
> promoted to the optional production control `dropout_p`; the measurements
> below remain the historical evidence, not a current API description.

## 1. Scope

This matrix tests estimator choices without changing the production operator.
Every variant keeps:

- exact hard Q/K/V signs in forward;
- causal shifted actions, finite suffixes, latest hard ties, and hard null;
- softsign STE for Q/K/V;
- the hard-Hamming Jacobian target;
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
| `hard_hamming` | Uses `exp(-lambda*h)` numerically and in the target Jacobian. |

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
`TEMPERATURE_LADDER.md`.

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
`TEMPERATURE_LADDER.md` for the complete matrix.

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
branch. The antithetic result remains a historical record; its obsolete local
script is not shipped and cannot be run through the current API unchanged.

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
historical estimator default therefore remained `mismatch_penalty=3` at that
stage. This value does not apply to the post-rewrite estimator in section 7.

## 7. Tiered-Prototype Recalibration

The then-current tiered prototype changed three coupled definitions:

```text
h_bar = hard_mismatch_count / D
H_bar = stochastic_mismatch_mass / D
local_gate = exp(-lambda * H_bar)

tail = max(stochastic_suffix_score - exact_suffix_length, 0)
tiered_score = exact_suffix_length + 0.5 * tail / (1 + tail)

nonnull_logit = tiered_score / temperature - log(candidate_count)
```

These changes invalidate the old `lambda=3` scale and the old null-score
interpretation.

### 7.1 Structural Factorial

A `2 x 2 x 2` seed-0 pilot crossed hard tiers, candidate normalization, and
mismatch-fraction normalization for 600 repeated-motif steps. The useful
interaction was hard tiers plus candidate normalization; mismatch fractions
needed a new lambda calibration rather than direct reuse of `3`.

With all three prototype components enabled, the reference lambda sweep gave:

| Lambda | Final hard CE |
| ---: | ---: |
| `3` | `0.157` |
| `6` | `0.157` |
| `9` | `3.10e-4` |
| `12` | `4.26e-4` |
| `16` | `0.157` |
| `24` | `0.157` |

This is non-monotone. Increasing lambda is not a generic cure for gradient
leakage or long suffixes.

### 7.2 Fixed Estimator Shape

Single-variable four-seed comparisons retained the existing fixed choices:

| Variant | `<1e-3` successes |
| --- | ---: |
| cubic + softsign, `temperature=1` | `3/4` |
| `temperature=0.5` | `2/4` |
| `temperature=2` | `2/4` |
| no stochastic mismatch perturbation | `2/4` |
| linear perturbation | `2/4` |
| identity sign STE | `2/4` |

Tier residual spans `0.5/0.75/1.0` later gave `2/4`, `1/4`, and `0/4` on
matched reference streams. Null scores from `-2` through `3` moved which
stream succeeded but did not create a robust plateau. The fixed cubic,
softsign, residual span `0.5`, null score `0.5`, and temperature `1` remain.

### 7.3 Prototype Lambda

Reference-only seed 0 favored both `9` and `12`, but the CUDA counter stream
exposed a robustness difference. On eight CUDA model/perturbation seeds:

| Lambda | CE `<1e-3` | 100% accuracy |
| ---: | ---: | ---: |
| `3` | `1/8` | `2/8` |
| `6` | `2/8` | `2/8` |
| `9` | `3/8` | `3/8` |
| `12` | `1/8` | `3/8` |

That prototype selected `mismatch_penalty=9`. It was an empirical compromise
across two random implementations, not a D/T/W formula or a universal
optimum. Section 9 explains why it is not the current default.

### 7.4 D/T/W Matrix

At `T=16,W=8`, 600-step reference/CUDA fits with lambda `9` gave:

| D | Reference `<1e-3` | CUDA `<1e-3` |
| ---: | ---: | ---: |
| `4` | `2/4` | `2/4` |
| `8` | `2/4` | `3/4` |
| `16` | `2/4` | `2/4` |
| `32` | `2/4` | `2/4` |

There is no monotone D collapse in this tiny fit, but success is only partial.
At `T=16,D=4`, windows `4/8/16` gave `2/4`, `3/4`, and `2/4` reference
successes under the then-tested lambda `12`; a larger configured horizon was
not monotonically easier.

At `T=32,W=16`, lambda `9`, both implementations and both `D=4/32` gave
`0/4` successes. This longer repeated-motif configuration is a documented
failure boundary, not a passing operator gate.

The historical, no-longer-shipped `estimator_stability.py` matrix crossed
`D=1/4/8/16/32`, `T=16/32/64`, `W=4/16/32`, and eight fixed perturbation
seeds. At lambda `9`, its worst relative seed RMS is `0.0676` and minimum
cosine to the mean gradient is `0.9951`. Directional noise is controlled, but
mean Q/K gradient norm falls from `0.378..1.836` at `D=4` to
`0.00475..0.00635` at `D=32` on random codes. Mismatch-fraction normalization
stabilizes the scalar definition; it does not remove the scarcity of exact
collisions in high-dimensional random codes.

### 7.5 Final Estimator Checks

A final controlled matrix targeted the remaining plausible deletions rather
than adding new objectives.

Flat, tiered, and hierarchical candidate priors produced identical outcomes
when every candidate occupied exact tier zero. That control shows the prior
implementation itself does not create recall; it does not invalidate hard
tiering once candidates have different exact suffix lengths.

Restricting suffix credit to the current frontier improved the short
association probe to `16/20`, but failed the long-suffix construction `0/8`
and remained stuck at hard suffix length `4`. A conditional-tail variant also
gave `0/8`. Frontier-weighted hybrids reached at most `13/20` on the short
probe and did not establish a robust long-route result. The complete
multiplicative tail is therefore necessary: it gives every suffix position a
path to improve before that position becomes the current hard frontier.

On the same long construction, the prototype full-tail estimator reached
`8/8` at `temperature=2, lambda=9` in `11..48` steps. Lowering lambda to `6`
also reached `8/8`, but in `19..75` steps. This confirms that the long-suffix
failure of the frontier variants was structural, not just an unlucky static
lambda.

A paired positive/null sweep used 40 seeds for every combination of
`temperature in {0.5,1,2}` and `lambda in {6,9}`. Fixed null score `0.5`
passed all `40/40` positive cases and produced zero false routes in null
cases. Higher null scores `3..5` introduced positive-case failures. No
learned or context-dependent null calibration is justified.

Temperature remains task-sensitive. The short discovery task favors `0.5`;
the explicit long-suffix competition favors `2`. Temperature sharpens or
widens credit among routes but does not itself create longer support. Static
`1` remains the least specialized default, with `0.5/1/2` reserved for
declared run-level comparisons.

These checks retain the single cubic branch. Antithetic sampling is not
needed, and replacing the cubic shape did not supply evidence strong enough
to justify another public control.

### 7.6 Historical Decision

At this point the prototype retained cubic sampling and exact hard tiers.
That decision was later superseded by the reduction in section 8.

## 8. Deterministic-Minimal Reduction

The final reduction tested which mechanisms were actually carrying useful
training signal. All variants retained exact hard forward, every causal
candidate, candidate normalization, distributed value credit, and static
controls.

### 8.1 Multi-Seed Fitting

Twelve development seeds followed by twenty held-out seeds gave:

| Estimator | Near-zero fits | Route tasks | Relative step cost |
| --- | ---: | ---: | ---: |
| Sampled cubic plus hard tier | `17/32` | `32/32` | baseline |
| Deterministic raw Hamming gate, null `0` | `19/32` | `32/32` | about `5x..7x` faster |
| Sampled raw Hamming gate | `20/32` | `32/32` | slower than deterministic |
| Deterministic raw Hamming gate, null `0.5` | `17/32` | `32/32` | about `5x..7x` faster |

The one-seed fitting difference between sampled and deterministic raw gates
was insufficient to retain RNG, random state, or estimator variance. Null
`0.5` was retained despite the small fit-count difference because it gives
the intended hard-limit separation: no exact match has score `0`, while the
first exact match has score at least `1`.

### 8.2 Symbol VJP

Identity STE looked locally aligned on some tiny probes but failed the actual
fitting gate: it reached `4/12` versus `8/12` for softsign under the minimal
raw estimator. Softsign therefore remains the only symbol VJP.

An exhaustive 90-configuration hard-bit-flip comparison measured:

| Estimator | Cosine | Sign agreement | Useful-coordinate agreement |
| --- | ---: | ---: | ---: |
| Sampled tier prototype | `0.5179` | `0.8304` | `0.5383` |
| Deterministic raw plus softsign | `0.5214` | `0.8056` | `0.5249` |

The minimal estimator missed no coordinate that the exact bit-flip oracle
marked useful. The differences do not justify the prototype's extra state.

The broader current executable matrix spans 144 configurations and 4320 Q/K
coordinates. It reports cosine `0.4844`, sign agreement `0.8642`, and misses
`6/720` oracle-useful flips (`0.83%`). This is the maintained diagnostic
baseline; the earlier 90-configuration comparison above remains the paired
prototype-versus-minimal ablation.

### 8.3 Long-Suffix Signal

An adversarial construction placed a target route one bit from the desired
long suffix and a decoy at exact length `W-1`. The bounded hard-tier residual
made the target gradient decay exponentially and become nearly zero by
`W=16..32`. The raw expected-prefix score retained more than four orders of
magnitude more gradient at `W=16`.

This establishes the key simplification: exact tiers are useful for hard
forward but harmful in the discovery VJP. The backward score is now the raw
complete expected-prefix sum.

The same probe also separated the two static controls in current terminology:

- increasing `scale` sharpens existing route competition;
- lowering `mismatch_scale` keeps near-match prefix products alive.

At `W=32`, `mismatch_scale=9` could still make the target signal tiny while
`mismatch_scale=1` retained it. Attention-logit scale cannot substitute for
mismatch leakage.

### 8.4 Four-Way Hard-Forward Fitting Pilot

The deterministic reduction was reopened after the single-seed fit gate was
found to hide two stable failure basins. A current executable benchmark
compares four VJPs while requiring bitwise-identical hard output:

1. the deterministic production surrogate;
2. independent per-mismatch cubic perturbations without antithetic or tiers;
3. exhaustive single-bit hard counterfactuals for Q, K, and value;
4. inverted dropout on post-softmax route probabilities in the backward
   carrier.

The fourth variant follows standard attention-dropout placement:

```text
p_drop = p_route * Bernoulli(1 - r) / (1 - r)
```

It does not mask logits, renormalize surviving routes, or alter the hard
forward. At `r=0`, its VJP is exactly the deterministic VJP.

The paired RTX 2080 Ti pilot used eight model/data seeds, matched noise seeds,
1000 AdamW steps, `D=4`, `T=16`, `W=8`, static `scale=1`,
`mismatch_scale=9`, and success threshold CE `<1e-3`.

| Backward estimator | Ever `<1e-3` | Final `<1e-3` | Median success step | PyTorch step |
| --- | ---: | ---: | ---: | ---: |
| Deterministic | `6/8` | `6/8` | `442.0` | `9.81 ms` |
| Mismatch random | `6/8` | `6/8` | `435.5` | `10.26 ms` |
| Exact bitflip | `6/8` | `5/8` | `386.0` | `11.79 ms` |
| Attention dropout, `r=0.1` | `7/8` | `7/8` | `424.0` | `9.69 ms` |

The aggregate counts hide important behavior:

- deterministic training failed model seeds 1 and 6;
- mismatch randomness rescued seed 6 but regressed seed 4;
- attention dropout rescued seed 6 without regressing a deterministic
  success in this matrix, but still failed seed 1;
- exact bitflip reached `<1e-3` on seed 6 and later escaped to CE `1.455`;
  final-only reporting would incorrectly classify its exploration ability.

The dropout-rate check was not a broad tuning exercise:

| Dropout rate | Ever `<1e-3` | Final `<1e-3` |
| ---: | ---: | ---: |
| `0` | `6/8` | `6/8` |
| `0.05` | `6/8` | `6/8` |
| `0.1` | `7/8` | `7/8` |
| `0.2` | `6/8` | `6/8` |

This is not a broad optimum. To separate model geometry from stochastic luck,
the two deterministic failures were each crossed with eight independent noise
seeds at rate `0.1`:

| Estimator | Model seed 1 | Model seed 6 | Combined | Median success step |
| --- | ---: | ---: | ---: | ---: |
| Mismatch random | `0/8` | `5/8` | `5/16` | `554` |
| Attention dropout | `1/8` | `6/8` | `7/16` | `790` |

Only four paired noise cases disagreed: dropout won three and mismatch
randomness won one. This sample is too small for a statistical superiority
claim. It does show that post-softmax dropout has real, though probabilistic,
basin-escape behavior and can sometimes supply a useful direction that the
tested mismatch perturbation does not.

The reported latency is for research-only PyTorch implementations. The
mismatch prototype materializes `O(BHT^2D)` random values and dropout
materializes `O(BHT^2)` values. A fused kernel can reconstruct either with a
counter-based RNG, but dropout needs one draw per route rather than one per
route-bit. No production-kernel decision should use these PyTorch timings as
its final cost estimate.

### 8.5 Softmax-Proportional High-Dropout Probe

A 2026-07-30 follow-up tested whether choosing retained routes from the
current softmax distribution makes `dropout_p=0.9` usable. This is a research
estimator only; production attention-dropout semantics were not changed.

Two detached sampling laws were tested. For non-null conditional probability
`r_i` and target keep count `K=(1-dropout_p)N`:

1. PPS-Bernoulli uses `pi_i=min(1,c*r_i)`, with `c` chosen so
   `sum_i pi_i=K`, then weights a retained route by `1/pi_i`.
2. Categorical sampling draws `max(1,floor(K))` routes with replacement from
   `r_i`, then weights route counts by `1/(sample_count*r_i)`.

The sampling distribution, inclusion probabilities, and sampled decisions
are all stop-gradient. Each law was crossed with:

- dense VJP, which retains the complete softmax score Jacobian;
- selected VJP, which stops score gradients for routes not selected.

Both return the exact hard ROSA forward. Dense variants preserve the
deterministic soft-carrier VJP in expectation. Selected variants deliberately
omit unselected score derivatives and are biased.

The first matrix used the same RTX 2080 Ti fitting geometry as Section 8.4,
with two stochastic streams for each of eight model/data seeds. Results are
aggregated over 16 trajectories:

| Estimator | Ever `<1e-3` | Final `<1e-3` | Median success step | Median final CE |
| --- | ---: | ---: | ---: | ---: |
| Standard dropout, `p=0.1` | `12/16` | `12/16` | `412.5` | `3.44e-4` |
| Standard dropout, `p=0.9` | `10/16` | `10/16` | `416.0` | `5.08e-4` |
| PPS, `p=0.9`, dense VJP | `11/16` | `10/16` | `442.0` | `3.87e-4` |
| PPS, `p=0.9`, selected VJP | `10/16` | `9/16` | `460.0` | `3.89e-4` |
| Categorical, `p=0.9`, dense VJP | `8/16` | `8/16` | `420.5` | `3.32e-3` |
| Categorical, `p=0.9`, selected VJP | `10/16` | `10/16` | `414.5` | `4.61e-4` |

PPS looked competitive in the first `6/8` run but fell to `4/8` final
successes under the second random stream. One dense PPS trajectory and one
selected PPS trajectory crossed the threshold and later returned to CE near
`0.19`. Categorical selected VJP similarly changed from `6/8` to `4/8`.
The aggregate result does not support a reliable improvement over standard
`p=0.9`, and all `p=0.9` variants remained below the `p=0.1` baseline.

Short rows matter in this matrix. At `T=16`, standard or PPS `p=0.9` retains
at least one non-null route in only about `54%` of nonempty rows. Initial
softmax-proportional allocation raised that average to only about `55%`;
initial scores were usually too flat for probability weighting to help.
Categorical sampling guarantees one route but consequently retains
`16/136=11.8%` of all non-null row-route pairs, an effective dropout rate of
about `0.882`.

A four-seed `T=64` control reduced that boundary effect. Categorical retained
`189/2080=9.09%`, an effective dropout rate of about `0.909`. No estimator
reached CE `<1e-3` in 1000 steps, so final loss was compared:

| Estimator | Median final CE |
| --- | ---: |
| Standard dropout, `p=0.1` | `0.324` |
| Standard dropout, `p=0.9` | `0.365` |
| PPS, `p=0.9`, dense VJP | `0.365` |
| Categorical, `p=0.9`, selected VJP | `0.365` |

These prototypes still compute the complete dense route distribution before
sampling, so their measured latency is not evidence of kernel speedup.
Computational savings would additionally require a cheap detached proposal
or sampled partition estimate. The training result does not justify that
kernel work yet: selecting from the current softmax protects current winners
but weakens the broad exploration that made uniform attention dropout useful.

The executable PPS and categorical branches were removed after this negative
gate. This section retains the experiment and result; it does not describe
currently selectable benchmark estimators.

### 8.6 High-Dropout Stabilization

A second 2026-07-30 probe tested five ways to make a nominal
`dropout_p=0.9` useful without changing the exact hard forward:

1. fixed-count, stratified query-row sampling of the complete soft VJP;
2. the same row sampling applied only to the residual above a dense
   suffix-length-one carrier;
3. fixed-count systematic route sampling with Horvitz-Thompson weights;
4. the same route sampling with per-row self-normalized weights;
5. estimator 1 with an exact dense VJP every eighth training call.

For estimator 2, let `C_W` be the complete soft carrier, let `C_1` use only
the local length-one match score, and let `m_q` be a detached stratified row
weight. One row is sampled uniformly from each contiguous stratum and is
weighted by that stratum's width, so `E[m_q]=1`. The estimator is:

```text
C_hat(q) = C_1(q) + m_q * (C_W(q) - C_1(q))
```

This is an exact control variate for the deterministic soft VJP:
`E[C_hat]=C_W`. It is not an extra training objective and introduces no soft
value into forward. `dropout_p` has PyTorch's drop-probability meaning; the
number of sampled rows per `(batch, head)` is
`max(1, round((1-dropout_p)*T))`.

The first screen used `T=16`, `W=8`, two heads, `mismatch_scale=9`, 1000
steps, eight model/data seeds, and matched stochastic seeds. Only candidates
with independent value were advanced to a second stochastic stream:

| Candidate, all at `p=0.9` | Matched final | Noise 17 final | Combined final |
| --- | ---: | ---: | ---: |
| 1. Stratified row VJP | `5/8` | `5/8` | `10/16` |
| 2. Length-one control variate | `6/8` | `7/8` | `13/16` |
| 3. Fixed-count route HT | `7/8` | `4/8` | `11/16` |
| 4. Self-normalized route weights | `2/8` | not run | rejected |
| 5. Dense every eighth call | `5/8` | not run | rejected |
| Standard attention dropout, `p=0.1` | `6/8` | `6/8` | `12/16` |
| Standard attention dropout, `p=0.9` | `5/8` | `5/8` | `10/16` |

Estimator 2 reached the threshold at least once in `14/16` runs and ended
below it in `13/16`. Its median final CE was `3.74e-4` and median first
success step was `428.5`. Candidate 1 did no better than ordinary
high-rate dropout. Candidate 3 was sensitive to the random stream. Candidate
4 changed the gradient scale and null competition. Candidate 5 added dense
work without improving its underlying row estimator.

A fixed-input, fixed-upstream-gradient Monte Carlo check separated estimator
variance from fitting luck. It used 512 samples at `T=8`, `W=6`, and
`p=0.9`; normalized variance is
`E[||g-Eg||^2] / ||g_dense||^2`:

| Estimator | Normalized variance | Median sample cosine | Negative cosine |
| --- | ---: | ---: | ---: |
| Standard attention dropout | `11.83` | `0.110` | `28.9%` |
| Stratified row VJP | `7.50` | `0.179` | `7.6%` |
| Length-one control variate | `0.90` | `0.943` | `0/512` |
| Fixed-count route HT selected VJP | `1.73` | `0.657` | `0.2%` |

The selected-route HT mean remained `22.8%` from the dense gradient after
512 samples. Sampling attention values with HT weights is not sufficient to
recover the complete softmax score Jacobian after unselected score paths are
detached. By contrast, the retained control variate is also covered by an
exhaustive test over all stratified row selections; its mean VJP matches the
dense reference to float64 tolerance.

The `T=64` check did not converge to CE `<1e-3` for any method in 1000 steps.
Across matched and noise-17 streams, median final CE was `0.36466` for the
control variate, `0.36462` for standard `p=0.1`, and `0.36456` for standard
`p=0.9`. This length is therefore an inconclusive quality gate, not evidence
of a long-context improvement.

The PyTorch prototype still builds `C_W` for every row and is intentionally
not faster. A fused backward can sample rows before the suffix recurrence,
compute `W=1` for all rows, and compute lengths `2..W` only for selected
rows. Under a uniform per-length cost approximation, its suffix-depth work
fraction is:

```text
1/W + keep_fraction * (W - 1)/W
```

That fraction is `23.4%` for the tested `T=16,W=8`, `20.7%` for
`T=64,W=8`, and about `12.8%` for large `T,W=32,p=0.9`. These are bounds for
the soft suffix recurrence only; local pairwise gates, the length-one
carrier, softmax/value work, and the exact hard forward remain.
The prototype stores `O(BHT)` row uniforms for simple autograd replay. A CUDA
kernel should instead reconstruct each selected-row draw from a counter keyed
by dropout seed, logical batch index, head, and stratum.

Decision: retain only estimator 2 as the research-only `suffix_dropout`
benchmark option. Delete candidates 1, 3, 4, and 5, along with the earlier
PPS/categorical implementations. Do not add it to the production API or CUDA
kernel until a larger fitting task demonstrates a quality/compute tradeoff
and a row-selective kernel realizes actual savings.

### 8.7 Estimator-Shape Decision

This reduction kept only:

- hard signs with softsign VJP;
- normalized Hamming mismatch;
- deterministic `exp(-mismatch_scale * mismatch_rate)` gates;
- the raw complete suffix prefix-product sum;
- fixed null score `0.5` and non-null `-log(candidate_count)`;
- distributed soft value credit;
- the then-tested controls `scale=1` and `mismatch_scale=9`;
- every causal candidate in backward.

It removed:

- sampled mismatch perturbations and RNG state;
- antithetic branches and cubic shape controls;
- exact hard-tier and bounded-residual wrappers;
- separate numerical and Jacobian gates;
- all adaptive schedules.

At that point, the four-way pilot did not change the static controls.
Attention dropout became the first stochastic candidate because it improved
the pilot, preserved the estimator mean, used a standard placement, and was
cheaper to kernelize than per-mismatch perturbation. It was subsequently
implemented as the optional production `dropout_p` control. Exact bitflip
remains an oracle/control, not a scalable training implementation.

## 9. Current Static-Default Correction

The `mismatch_scale=9` choice above was calibrated before hard tiers and
mismatch perturbations were removed. The deterministic-minimal reduction
retained `9` without a matched `3` versus `9` default comparison, so that
inheritance was reopened after `dropout_p` entered the CUDA operator.

An earlier matched RTX 2080 Ti rerun used one production CUDA trajectory per
model/data seed `0..7`, 1000 AdamW steps, `D=4`, `T=16`, `W=8`, `scale=1`,
and success threshold CE `<=1e-3`:

| `mismatch_scale` | `dropout_p` | Ever successful | Final successful |
| ---: | ---: | ---: | ---: |
| `3` | `0` | `7/8` | `7/8` |
| `9` | `0` | `6/8` | `6/8` |
| `3` | `0.1` | `6/8` | `6/8` |
| `9` | `0.1` | `7/8` | `7/8` |

Neither scale dominates across dropout settings. Because production defaults
to `dropout_p=0`, the public default is restored to `mismatch_scale=3`.
`9` remains a valid explicit experiment value, especially for the tested
`dropout_p=0.1` recipe, but it is not a universal default and must not be
selected implicitly.

### 9.1 Production Recheck and Atomic Variability

A 2026-07-31 recheck used the current production CUDA operator on an RTX
2080 Ti. The fitting geometry remained `D=4`, `T=16`, `W=8`, two heads,
1000 AdamW steps, and CE `<=1e-3`. Each of eight model/data seeds was run
four times. Nonzero-dropout runs used dropout seeds `0,17,29,43`;
`dropout_p=0` repeated the same nominally RNG-free launch.

| Configuration | Ever successful | Final successful | Any success by model seed | Majority success by model seed | Median final CE | Median success step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mismatch_scale=3, dropout_p=0` | `27/32` | `26/32` | `7/8` | `7/8` | `2.95e-4` | `412` |
| `mismatch_scale=3, dropout_p=0.1` | `26/32` | `26/32` | `8/8` | `6/8` | `3.17e-4` | `406` |
| `mismatch_scale=3, dropout_p=0.9` | `26/32` | `25/32` | `8/8` | `6/8` | `3.61e-4` | `416.5` |
| `mismatch_scale=9, dropout_p=0.1` | `25/32` | `25/32` | `7/8` | `6/8` | `3.83e-4` | `426` |

`Any success` asks whether at least one of four launches solved a model seed.
`Majority success` requires at least three. Dropout widened the set of
reachable basins: at `mismatch_scale=3`, seeds 1 and 7 each succeeded under
some dropout streams. It did not improve per-run success. The default and
`dropout_p=0.1` both ended at `26/32`; `dropout_p=0.9` ended at `25/32` and
had one trajectory cross the threshold before rebounding to CE `0.012`.
The prior `mismatch_scale=9, dropout_p=0.1` recipe also fell to `25/32`.

The repeated `dropout_p=0` result exposed an important CUDA distinction.
The path consumes no RNG, but global FP32 atomic accumulation is not bitwise
deterministic. A dedicated repeat of model seed 7 succeeded in `5/8` launches,
with final CE split between roughly `2.4e-4` and `0.139`. The dense reference
solved the same seed. At initialization, CUDA/reference query and key
parameter gradients had relative L2 errors of only `2.47e-7` and `2.07e-7`,
respectively, with cosine `1.0`; Q/K hard signs remained identical through
200 lockstep steps. The small accumulation difference was later amplified by
hard sign crossings. The CUDA correctness suite still passed `84` tests with
one hardware skip, including the deterministic-algorithm guard. This is
optimization-path sensitivity within the documented VJP tolerance, not a
hard-forward mismatch, but single-run fitting counts are not stable enough to
select defaults.

The shortcut-free associative-recall gate used eight seeds, four
associations, `D=8`, `W=1`, and at most 250 steps:

| `dropout_p` at `mismatch_scale=3` | Passed | Median success step | Slowest |
| ---: | ---: | ---: | ---: |
| `0` | `8/8` | `31.5` | `37` |
| `0.1` | `8/8` | `30.5` | `40` |
| `0.9` | `8/8` | `39.0` | `88` |

High dropout preserved direct historical route discovery, but increased the
tail of training time.

A four-model-seed `T=64,W=8` fit did not reach CE `1e-3` under any
configuration in 1000 steps:

| `dropout_p` at `mismatch_scale=3` | Matched median final CE | Noise-17 median final CE |
| ---: | ---: | ---: |
| `0` | `0.272` | not applicable |
| `0.1` | `0.321` | `0.352` |
| `0.9` | `0.320` | `0.343` |

This small longer-sequence check favors the default but remains
under-converged and seed-sensitive.

Finally, an interleaved, warmed production microbenchmark used dense QKV
backward, `B=1`, four heads, `D=8`, `D_v=8`, and `W=32`. Each cell is the
median of three 50-iteration measurements:

| `T` | `dropout_p=0` | `dropout_p=0.1` | `dropout_p=0.9` |
| ---: | ---: | ---: | ---: |
| `64` | `0.410 ms` | `0.402 ms` | `0.463 ms` |
| `128` | `0.440 ms` | `0.449 ms` | `0.461 ms` |
| `256` | `0.733 ms` | `0.756 ms` | `0.758 ms` |

At `T=256`, nonzero dropout added about 3% and increased measured operator
allocation from `0.14844 MiB` to `0.14893 MiB`. The production implementation
computes every suffix and route before applying dropout, so `dropout_p=0.9`
costs essentially the same as `0.1`; it is not a compute-sparsity mechanism.

Decision: keep `dropout_p=0, mismatch_scale=3` as the conservative production
default. `dropout_p=0.1` remains a valid explicit exploration control because
it expanded any-run model-seed coverage, but it did not improve aggregate
success. `dropout_p=0.9` is trainable on the tiny tasks and passed recall, but
its higher variance, slower tail, lack of speedup, and weaker `T=64` result do
not justify a recommended production recipe. Future fitting comparisons must
repeat `dropout_p=0` launches as well as dropout seeds.

### 9.2 Failure Anatomy and Benchmark Correction

A follow-up analyzed the failed trajectories at individual target positions
instead of treating final CE as an opaque success count. For this tiny model,
the downstream network receives:

```text
embedding(current_token) + output(flatten(hard_routed_values))
```

The embedding has no positional or contextual input. Therefore rows with the
same current token and the same hard routed value bits are exactly
indistinguishable to every downstream parameter. Grouping supervised rows by
that hard feature and computing empirical `H(target | hard_feature)` gives a
model-independent lower bound on CE for the current discrete state.

Representative production CUDA runs were:

| Run | Final CE | Hard-feature entropy | Excess CE | Conflict |
| --- | ---: | ---: | ---: | --- |
| `T=16`, seed 1, `dropout_p=0` | `0.154814` | `0.154033` | `0.000781` | 1 group, 2 rows |
| `T=16`, seed 1, `dropout_p=0.1`, noise 29 | `0.616434` | `0.616131` | `0.000303` | 2 groups, 6 rows |
| `T=16`, seed 1, `dropout_p=0.1`, noise 43 | `0.000236` | `0` | `0.000236` | none |
| `T=64`, seed 0, `dropout_p=0` | `0.517467` | `0.517117` | `0.000350` | 3 groups, 48/60 rows |
| `T=64`, seed 2, `dropout_p=0` | `0.107924` | `0.107169` | `0.000755` | 1 group, 10/57 rows |

The seed-1 plateau has a particularly direct explanation. Query positions 9
and 10 both have current token 7 and end with the same two-head routed value,
but their targets are 7 and 4. Their best possible contribution to the
9-target mean loss is:

```text
2 * log(2) / 9 = 0.154033
```

The measured CE converges to that value. Across the inspected failures, the
conflicts came from routing to the same raw value-token signature, not from
different value tokens quantizing to the same value bits. The fitting script
now reports both collision classes explicitly.

The cancellation is visible before the optimizer. At the failed seed-1
checkpoint, the two conflicting rows had:

| Parameter path | Per-row gradient cosine | Residual norm after summing |
| --- | ---: | ---: |
| Q/K | `-0.952` | `18.2%` of the sum of norms |
| value | `-0.963` | `14.2%` |
| output/readout | approximately `-1.000` | `0.038%` |

The remaining Q/K signal is then divided across all nine targets, diffuse
soft routes, and the softsign VJP. This explains why a discrete collision can
remain stable even though the soft carrier differs between the two rows.

An exhaustive check at a failed checkpoint found 18 individually useful
Q/K bit flips. The production tensor-level direction had cosine `0.160` with
their hard linearized loss deltas; the parameter-level cosine was `0.375`.
The production Q/K parameter-gradient norm was `0.00399`, versus `0.0524`
for exact bitflip. This is a real proxy weakness, but exact bitflip did not
solve the optimization problem: four nominal repeats for each of seeds 1 and
7 all failed, ending near `0.15448` and `0.13933`. Its mean step time was
about `12.3 ms`, compared with roughly `3.4 ms` for the production CUDA
operator. A one-bit output counterfactual is not a realizable independent
parameter update after shared projections aggregate positions.

Several targeted interventions were rejected:

- per-head QK RMS normalization kept softsign derivatives active but did not
  improve the hard seeds;
- copied or fully shared Q/K projections moved failures between seeds and
  remained `0/4` at `T=64`;
- changing the sign-VJP tail from `(1+|x|)^-2` to `(1+|x|)^-1` rescued seed 1
  but regressed other trajectories, ending `6/8`; exponent `1.5` ended `5/8`;
- exact bitflip failed both selected hard seeds despite its better local
  direction;
- sampling half the supervised token rows demonstrated basin escape, but its
  matched eight-seed result was only `5/8`; sampling one row was too noisy.

These negatives do not support another production mechanism or parameter.
They do show that stochastic asymmetry can break a cancellation, which is
consistent with attention dropout expanding any-run seed coverage without
improving aggregate per-run success.

The target-mask definition was also found to confound the old long-sequence
result. `historical_target_mask` asks whether any correct historical suffix
candidate exists. It does not ask whether exact token-equality ROSA's
longest/latest rule selects that candidate. The raw token automaton's route
accuracy was:

| Length | Seeds | Raw longest/latest route accuracy |
| --- | --- | --- |
| `T=16` | `0,2,5,6` | `0.917, 0.900, 0.900, 0.889` |
| `T=64` | `0,1,2,3` | `0.667, 0.717, 0.842, 0.839` |

Thus, part of the old fitting task rewards Q/K recodings that intentionally
depart from token equality. A strict mask that retains only targets selected
by raw longest/latest routing improved `T=16` to `7/8`, and produced one
success out of four at `T=64`; the latter remained an optimization challenge,
with median CE about `0.193`. The original mask remains useful as a hard
combinatorial fitting stress test, but it must not be described as a pure
semantic-recall gate.

Decision:

1. keep the production VJP and static defaults unchanged;
2. require hard-feature entropy, excess CE, and collision class in future
   repeated-motif reports;
3. keep the existing any-candidate stress task separate from a strict
   longest/latest task;
4. include trained-failure checkpoints in gradient-alignment studies instead
   of relying only on random tensors;
5. require realistic contextual-residual pretraining tests in addition to the
   token-only fitting task.

### 9.3 Trained Checkpoints and Contextual Recall

The five follow-up actions above were implemented and run on 2026-07-31 on an
RTX 2080 Ti. Production defaults remained `dropout_p=0` and
`mismatch_scale=3`; nonzero dropout was passed explicitly.

At trained repeated-motif checkpoints, the hard-feature entropy floor still
separated representational collisions from optimizer failure:

| Seed | Final CE | Entropy floor | Excess CE | Tensor QK cosine | Shared-parameter cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.154476` | `0.154033` | `0.000444` | `0.5198` | `-0.0322` |
| 7 | `0.000226` | `0` | `0.000226` | `0.4443` | `0.3585` |

The first row is already at the best loss permitted by its hard features.
Its positive tensor-space alignment does not survive aggregation through the
shared Q/K projections. This confirms both conclusions: the observed plateau
is primarily a discrete feature collision, and the production estimator still
has a parameter-space bias worth measuring. Exact bit flips are a diagnostic
oracle here, not a drop-in optimizer.

Target modes were then paired on seeds `0,2,10,27,29`, chosen because their
masks actually differ:

| Target mode | Collision-free runs | Mean final CE | Mean tensor cosine | Mean parameter cosine |
| --- | ---: | ---: | ---: | ---: |
| `any-candidate` | `2/5` | `0.337655` | `0.2760` | `0.1154` |
| `strict-longest-latest` | `4/5` | `0.135956` | `0.3130` | `0.1308` |

Strict targets remove rows whose desired continuation conflicts with the raw
longest/latest automaton and materially reduce ambiguity. They do not make
the learned hard state collision-free by construction: strict seed 27 ended
at CE `0.678826` with an entropy floor of `0.678765`.

A second gate trains an embedding plus resettable GRU residual, Q/K/value
projections, RosaSoft, and a readout from scratch. Each episode stores four
cue/payload pairs, clears the recurrent state, then issues four queries.
Paired episodes have complementary payloads but exactly identical post-reset
tokens and residuals. Consequently a residual-only predictor has bit loss
`log(2)` and exact accuracy at most `0.5`; in practice all three no-recall
ablations were near random.

One paired four-seed run produced:

| Recipe | Passed runs | Mean validation exact | Minimum exact | Mean step |
| --- | ---: | ---: | ---: | ---: |
| `dropout_p=0` | `2/4` | `0.993164` | `0.988281` | `12.408 ms` |
| `dropout_p=0.1` | `4/4` | `1.000000` | `1.000000` | `12.435 ms` |

For both recipes, every query used an earlier hard route and every
complementary pair produced a different hard routed value. Mean exact
accuracy was only `0.0527/0.0771` with the hard route zeroed,
`0.0693/0.0625` with the current value substituted, and `0.0596` for the
trained residual-only model. Thus the successful output cannot come from a
soft forward leak, a post-reset residual shortcut, or a current-token code.

The CUDA VJP uses global atomics, so repeated launches can enter different
hard-symbol basins even at `dropout_p=0`; another launch reached `4/4`.
Therefore this small matrix supports `dropout_p=0.1` as an explicit
exploration control, but not a default change. The production default remains
zero until larger models, more launch replicates, and longer contextual
sequences show a stable held-out advantage.

## 10. Remaining Research Gates

Dropout is now an optional production control. Its implementation has fixed
seed PyTorch/CUDA parity, counter-based mask reconstruction without a stored
`O(BHT^2)` mask, packed-varlen coverage, compile/checkpoint integration, and
a shortcut-free associative-recall gate. These engineering results do not
make nonzero dropout a training default. Before changing `dropout_p=0` or
revising `mismatch_scale` again, require:

1. a larger paired model-seed matrix with independent noise replicates;
2. long-suffix competition, associative recall, and null-retention tasks;
3. realistic full-model pretraining with exact hard-route recall and learned
   suffix length reported alongside loss;
4. matched static `mismatch_scale` comparisons under each declared dropout
   recipe.

These runs should also report active mismatch rate, route entropy, and
sensitivity to explicit `scale` and `mismatch_scale` choices. They may
motivate different run-level defaults, but not hidden adaptive schedules. A
dropout schedule is out of scope unless a fixed probability first clears
these gates.

## 11. Reproduction Status

The obsolete stochastic and tier scripts were removed because they targeted
deleted private APIs. Current executable checks are:

```text
tests/test_soft_reference.py
tests/test_soft_cuda.py
tests/test_soft_varlen.py
tests/test_discrete_gradient_alignment.py
tests/test_estimator_fit_ablation.py
tests/test_trained_fit_alignment.py
tests/test_validation_cli.py
examples/fit_soft_reference.py
examples/associative_recall_gate.py
examples/contextual_rnn_recall_gate.py
benchmarks/discrete_gradient_alignment.py
benchmarks/estimator_fit_ablation.py
benchmarks/trained_fit_alignment.py
```

Any future estimator experiment must be a separately named research path and
compare against the deterministic `rosa_soft_reference` without changing the
default operator.

## 12. Current Suffix-Score Geometry

The raw complete prefix sum retained in section 8 is now an intermediate
evidence value, not the final route score. Production computes

```text
S = sum_l product_{r < l} exp(-mismatch_scale * mismatch_rate_r)
U(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
```

and sends `U(S)` to candidate-normalized softmax. The Q/K VJP includes
`U'(S)=(sqrt(2)+1)/(2 sqrt(1+S))`. Hard forward remains exact
latest-longest ROSA and never reads this score.

This choice came from a separately named 2x2 ablation, not from the obsolete
stochastic prototypes above. `h/sqrt(D)` was rejected because it caused
width-dependent exponential saturation; normalized Hamming `h/D` remains.
The square-root suffix utility passed all 18 directed long-versus-short
competition cells versus 12/18 for raw evidence, remained neutral on the
small exact-bitflip alignment matrix, and had no persistent multi-seed fitting
regression. See `docs/research/SURROGATE_GRADIENT_DIRECTIONS.md` section 17
and `validation/suffix_proxy_ablation.json` for the current evidence.
