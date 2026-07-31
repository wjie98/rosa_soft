# Next-Generation Surrogate Gradients for RosaSoft

Status: research proposal, 2026-07-31

The validated production baseline is frozen at commit `86194a5`. This document
does not change the operator contract or recommend replacing the production
estimator before controlled experiments.

## 0. Updated Research Decision

The independent reviews changed the initial ranking:

1. **Build a globally consistent stochastic hard-bit oracle.** This is the
   only proposed reference that preserves shared Q/K bits, exact suffix jumps,
   null, and latest ties under a declared differentiable expectation.
2. **Build an exact winner-aware margin-edit target on tiny cases.** This is
   the most promising ROSA-specific task-directed estimator, but its solver is
   the research problem.
3. **Test dense-surrogate plus sampled counterfactual residual correction.**
   This preserves the current discovery path on every step and can use hard
   evidence without becoming a sparse-gradient operator.
4. **Treat winner marginals as a semantic ablation, not the default
   successor.** Their mean-field independence invents impossible routes and
   faithful winner differentiation removes some useful deep-suffix credit.
5. **Test route pullbacks and parameter-space response fitting only after the
   first three establish the dominant approximation boundary.**

The production estimator remains unchanged. The immediate objective is to
construct better judges before constructing another production proxy.

## 1. The Problem Must Be Stated Before Choosing an Estimator

For Q/K routing, RosaSoft contains the composition

```text
continuous logits
    -> hard signs
    -> exact latest-longest suffix automaton
    -> one hard route
    -> downstream loss.
```

As a deterministic function of the Q/K logits, this loss is locally constant
away from sign boundaries. Its mathematical derivative is zero almost
everywhere and undefined at a boundary. Therefore, a useful nonzero "gradient"
must differentiate a different, explicitly declared object.

There are four coherent choices:

| Class | Object being differentiated | Typical methods |
| --- | --- | --- |
| Stochastic hard objective | Expected loss under random hard bits or routes | ARM, DisARM, REINFORCE, PSA |
| Smoothed deterministic objective | A continuous relaxation of signs, suffixes, or winner selection | STE, ReinMax-style relaxations, differentiable DP |
| Discrete target update | A projected, proximal, or loss-augmented target structure | SPIGOT, I-MLE, AIMLE, direct loss minimization |
| Zeroth-order hard objective | A convolution or finite difference of the actual hard loss | evolution strategies, randomized smoothing |

The current RosaSoft VJP is in the second class. It is a useful dense
credit-assignment rule, not the true derivative of the hard operator. "More
accurate" is meaningless unless the target in the second column is named.

This distinction also prevents a common category error: ReinMax, GOSTE, ARM,
and DisARM assume a stochastic discrete variable. The production ROSA route is
a deterministic latest-longest decision, not a categorical sample from its
current softmax carrier.

## 2. Frozen Baseline and Observed Failure Modes

The current estimator is:

1. exact hard Q/K/V signs in forward;
2. softsign VJP for each sign;
3. `exp(-mismatch_scale * mismatch_rate)` for a local symbol pair;
4. a sum of suffix prefix products, interpreted as expected suffix length;
5. a candidate-normalized softmax over that scalar length;
6. an optional post-softmax attention dropout;
7. a dense value carrier used only by backward.

This has five distinct approximation boundaries:

1. **Quantizer geometry.** Softsign supplies a local slope that the hard sign
   does not have. Its magnitude depends on the logit margin.
2. **Suffix geometry.** The expected prefix length compresses a full
   distribution over suffix lengths into one scalar.
3. **Winner geometry.** Softmax over expected lengths is not the probability
   that a candidate wins a latest-longest competition.
4. **Parameter geometry.** Tensor-level Q/K pseudo-gradients are aggregated
   through shared projection weights. Locally useful activation directions can
   cancel or reverse in the realizable parameter subspace.
5. **Value carrier.** Every candidate value receives backward credit even
   though forward observes one selected value. Route-estimator changes also
   change nonselected-value training unless V is frozen or its VJP is ablated
   separately.

The fourth problem has preliminary measured evidence. On the trained seed-1
checkpoint, the current estimator had combined tensor-space cosine `0.5198`
against exhaustive one-bit counterfactual changes, but shared Q/K parameter
cosine `-0.0322`. The final cross entropy `0.154476` was almost entirely
explained by a hard-feature conditional-entropy floor of `0.154033`, so this
case also demonstrates why optimization failure must be separated from the
current checkpoint's feature collision. That entropy is conditional on the
current routed features; changing Q/K can change the floor. The small
checkpoint sample motivates further tests but does not establish a universal
parameter-space failure.

The present route construction introduces three empirical corrections:

- route `scale`;
- fixed `null_score = 0.5`;
- `-log(nonnull_count)` candidate normalization.

They compensate for properties of a softmax over expected lengths. They are
not consequences of the latest-longest automaton.

## 3. What the Literature Actually Offers

### 3.1 Approximate chain rules

[ReinMax](https://arxiv.org/abs/2304.08612) interprets straight-through
estimation as a first-order numerical approximation and uses a Heun-style
second-order correction without Hessians. The 2026
[Generalized and Optimal Straight-Through Estimators](https://openreview.net/forum?id=5LZQfHUVER)
places ST, FouST, DARN, and related estimators in one approximate-chain-rule
family and optimizes variance subject to explicit bias constraints.

These results are useful for a stochastic binary quantizer or categorical
route distribution. They do not justify applying a categorical estimator to a
deterministic ROSA winner that was not sampled from that distribution.

### 3.2 Stochastic hard estimators and local expectations

[ARM](https://arxiv.org/abs/1807.11143),
[DisARM](https://arxiv.org/abs/2006.10680), and
[Local Expectation Gradients](https://arxiv.org/abs/1503.01494) estimate
gradients of expected losses under stochastic discrete variables. DisARM uses
antithetic coupling and Rao-Blackwellization; local expectation gradients
analytically integrate one discrete variable while sampling the rest.
[Path Sample-Analytic estimators](https://arxiv.org/abs/2006.03143) combine
sampled binary paths with an analytic linearization and explicitly accept a
small approximation bias. Discrete
[Stein control variates](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a5a5b0ff87c59172a13342d428b1e033-Abstract-Conference.html)
reduce score-function variance without additional target-function
evaluations.

These methods are valuable as a correctness oracle for a declared noisy-bit
ROSA objective. They normally require extra hard evaluations, and the
variance can still be severe for all Q/K bits of a long sequence. The
previously rejected "antithetic mismatch branch" in RosaSoft is not an
ablation of ARM or DisARM: it perturbed a deterministic proxy, whereas these
methods couple samples from a stochastic hard objective.

### 3.3 Structured target methods

[SPIGOT](https://arxiv.org/abs/1805.04658) pulls a downstream gradient back to
a feasible structured pseudo-target. It avoids pretending that a hard argmax
has an ordinary Jacobian.

[I-MLE](https://arxiv.org/abs/2106.01798) differentiates discrete exponential
families using perturb-and-MAP and only requires a MAP solver.
[AIMLE](https://arxiv.org/abs/2209.04862) adapts the finite-difference target
step to trade gradient density against bias. These are relevant to ROSA only
after its exact structure and score features are defined. Treating the
existing deterministic route as if it were already an exponential-family
sample would again be a target mismatch.

[Differentiation of black-box combinatorial solvers](https://arxiv.org/abs/1912.02175)
and
[differentiable perturbed optimizers](https://arxiv.org/abs/2002.08676)
make the same solver boundary explicit: they require a linear structured
score and ordinary or perturbed optimization calls. A correct ROSA use must
solve a joint bit-and-route target, not only perturb a post-hoc route index.

[Randomly perturbed direct loss minimization](https://proceedings.mlr.press/v139/indelman21a.html)
uses ordinary and loss-augmented structured predictions to obtain a
task-directed update. Its requirement is the same: ROSA needs a declared
structured score and a tractable loss-augmented solver before the result
applies.

[Differentiable dynamic programming](https://proceedings.mlr.press/v80/mensch18a.html)
and [Fenchel-Young losses](https://www.jmlr.org/papers/v21/19-021.html)
provide principled regularized structured prediction. SparseMAP is less
attractive as a default here because RosaSoft deliberately needs dense route
discovery.

### 3.4 Policy targets and hard-loss calibration

The 2026
[DAPS policy-search formulation](https://openreview.net/forum?id=wJhhCmbFzY)
uses a KL-regularized nonparametric target and automatic step-size control.
Its useful idea for ROSA is not its VAE application; it is replacing an
arbitrary proxy step by a trust-region target whose effective sample size or
KL movement is controlled.

[Guided evolution strategies](https://proceedings.mlr.press/v97/maheswaranathan19a.html)
combine a surrogate-gradient subspace with hard function evaluations. This is
particularly relevant to the observed gap between activation-space and
shared-parameter-space alignment.

Finally, dense backward support for a sparse hard route is independently
supported by
[Dense Backpropagation for sparsely gated MoE](https://proceedings.mlr.press/v262/panda24a.html).
That work does not solve the ROSA automaton, but it supports retaining dense
route discovery rather than pruning candidates during training.

## 4. Candidate A: Latest-Longest Winner Marginals

This is the cleanest semantic alternative to the current route softmax, but it
is not automatically a better training estimator. It differentiates current
winner probability more faithfully while deleting some of the deliberate
look-ahead credit that lets RosaSoft establish a longer suffix before that
length changes the selected route.

### 4.1 Keep a local match probability

For candidate route `a` and suffix offset `r`, let

```text
G[a,r] in [0,1]
```

be a differentiable probability-like value for a complete symbol match. A
diagnostic `A0` prototype should reuse the current exponential Hamming gate.
This isolates the route-distribution change and adds no new hyperparameter.
It is not yet a production candidate because that gate equals exactly one on
an exact symbol match.

Interpret local matches as independent Bernoulli variables for the purpose of
the VJP only. Define the survival probability of suffix length `L_a`:

```text
S_a(0) = 1
S_a(l) = product_{r=0}^{l-1} G[a,r]
         = P(L_a >= l)
```

For a query row `t`, the candidate-specific horizon is

```text
w_{t,a} = min(max_suffix_length, a)
```

because `1 <= a <= t` and the key-side boundary is binding. Define the
suffix-length mass as

```text
p_a(l) = S_a(l) * (1-G[a,l]),  0 <= l < w_a
p_a(w_a) = S_a(w_a)
p_a(l) = 0,                    otherwise.
```

The complete CDF definition is

```text
C_a(k) = 0,                k < 0
C_a(k) = 1-S_a(k+1),       0 <= k < w_a
C_a(k) = 1,                k >= w_a.
```

### 4.2 Differentiate the actual tie rule

ROSA chooses the greatest suffix length and, on a tie, the latest candidate.
For candidate indices increasing with recency, its winner marginal is

```text
P(R = a) =
    sum_{l=1}^{w_a}
        p_a(l)
        * product_{b<a} C_b(l)
        * product_{b>a} C_b(l-1).
```

Earlier candidates may tie `a`, because `a` is later and wins that tie.
Later candidates must be strictly shorter. The null probability is

```text
P(R = 0) = product_b p_b(0).
```

Under the mean-field independence assumption, these probabilities are
normalized by construction.

### 4.3 Why this is materially different

The current estimator computes

```text
softmax_a(E[L_a]).
```

The proposed estimator computes an approximation to

```text
P(a = latest_argmax_b L_b).
```

In general these are not equal. The proposed form:

- encodes latest-tie semantics exactly;
- derives the null mass instead of assigning `null_score = 0.5`;
- accounts for the number of candidates without `-log(N)`;
- needs no route temperature or `scale`;
- preserves positive support for every valid candidate when all relevant
  local gates are strictly inside `(0,1)`;
- converges to the exact one-hot hard route when local gates converge to exact
  match indicators.

This removes three explicit route controls in one conceptual change. It does
not remove context-length sensitivity: for independent candidates,
`P(R=0)=product_b(1-G[b,0])` can collapse exponentially as context grows. With
the current gate, even a one-bit mismatch at large `D` can give `G` close to
one. The existing candidate normalization may therefore encode a useful prior
correction rather than merely compensate for a poor softmax.

### 4.4 Risks and implementation shape

The candidate-length variables are not truly independent: different routes
share query and key symbols. This is a mean-field VJP, not an exact
probabilistic model. It is nevertheless aligned with the order statistic and
tie rule that ROSA actually executes.

The dependence error can be first-order, not a small tail effect. With
`D=W=1`, one random query bit, and two identical fixed key bits, the globally
consistent route masses can be

```text
(null, old, recent) = (0.5, 0, 0.5).
```

The independent-candidate formula instead gives

```text
(null, old, recent) = (0.25, 0.25, 0.5).
```

It assigns mass to an impossible old route: identical keys must either both
match or both mismatch the shared query bit, and the recent route wins a
match tie. Repeated motifs, which are central to suffix retrieval, make this
failure especially relevant.

The current gate creates an important boundary failure. With one-symbol
candidates ordered from old to recent,

```text
P(R = first) = G_1 * product_{b>1}(1-G_b).
```

If two later candidates have exact gates `G_b=1`, this mass and every
first-order derivative that could revive it are zero. Null mass has the same
problem when multiple exact first symbols exist. An exact maximum-length,
latest candidate can also make the whole route distribution deterministic.
This is semantically valid for fixed Bernoulli probabilities but violates
RosaSoft's broad route-discovery goal when those probabilities themselves are
only proxies for changeable Q/K bits.

Therefore:

- `A0` with the frozen gate is a route-geometry diagnostic;
- a production `A1` needs strictly interior local probabilities, an explicit
  dense exploration mixture, or a justified nonlocal target update;
- any such fix must be counted as part of the estimator rather than hidden as
  a numerical clamp.

There is a second, more fundamental loss of support even when every gate is
strictly interior. Consider two candidates with horizons `(1,2)`. Let

```text
x = G[old,0]
y = G[recent,0]
z = G[recent,1].
```

Because the second candidate is more recent,

```text
P(R=0)      = (1-x)(1-y)
P(R=old)    = x(1-y)
P(R=recent) = y.
```

The deeper gate `z` disappears entirely. Once the recent candidate matches one
symbol, it already beats the old candidate by the tie rule; extending it to
length two cannot affect the current output. This is correct differentiation
of the current winner distribution, but it contradicts RosaSoft's complete
prefix-credit principle. Positive route mass also does not imply a nonzero
carrier gradient when candidate values or upstream utilities cancel.

Candidate A therefore asks a real empirical question:

> Is faithful current-route credit more valuable than speculative credit for
> suffix lengths that do not yet affect the current route?

It must not be described as preserving the current dense suffix estimator.

Products can underflow, and exact matches create zero CDF entries. The
prototype must use float64 diagnostics and a zero-aware product implementation
rather than blindly applying `log(clamp(C))`. Tests must include exact zero,
exact one, long suffixes, and tied candidates.

The probability hard limit is also distinct from a useful gradient limit.
With the frozen gate, exact hard winner probabilities require
`mismatch_scale -> infinity`; merely increasing Q/K magnitudes does not change
the gate's numerical mismatch count. At that limit, derivatives on mismatches
vanish while exact-match derivatives can scale with `mismatch_scale`. There is
no finite, dense Jacobian limit.

A direct PyTorch prototype and its autograd tape may materialize
`[B,H,T,T,W]` state. That is acceptable only for estimator validation. A
custom forward could process one suffix length at a time:

1. update candidate survival values;
2. form the current PMF and CDF;
3. run exclusive prefix and suffix products across candidates;
4. accumulate each candidate's winner mass;
5. discard length-local state.

This is not yet a complete VJP design. An analytic reverse still needs
length-local candidate adjoints, and a naive Q/K bit scatter costs
`O(BHT^2WD)`. Factoring it requires at least a local-edge adjoint field or a
new cross-axis schedule.

The candidate products also run across up to `T` routes for every suffix
length, rather than only `W` suffix gates per route. Separate forward and
reverse ordered scans are a real synchronization boundary. Long rows cannot
keep several `T`-float arrays in shared memory; multi-CTA rows require tiled
scan carries, checkpoints, or global scratch. Even one FP32 `[B,H,T,T]` field
is 1 GiB at `B=1,H=16,T=4096`.

Exact zeros require a division-free custom reverse. `log(clamp(C))` would add
spurious mass and destroy the hard limit, while ordinary division loses the
valid derivative of a product containing exactly one zero. A possible
research recurrence for one length is an associative lower-triangular
transform,

```text
P' = P * C_a(l)
J' = J * C_a(l-1) + utility_a * p_a(l) * P,
```

whose reverse can propagate product adjoints without division. This still
needs proof, byte accounting, packed-varlen planning, and an FP32 numerical
test. Candidate A must earn all of that complexity through estimator results.

## 5. Candidate B: Task-Directed Route Pullback

Winner marginals describe which routes are plausible, but not which alternate
route the downstream task wants. A structured target can add that information
without exposing a soft value in forward.

For hard output `y` and upstream gradient `g_y`, define the local first-order
cost of switching to route `a`:

```text
c_a = <g_y, hard_value[a] - y>.
```

Two coherent targets are:

### Euclidean SPIGOT target

```text
z_target = projection_simplex(z_hard - eta * c)
score_adjoint = z_hard - z_target.
```

This adjoint is with respect to a declared structured score. SPIGOT alone
does not define how route-simplex credit reaches suffix matches or shared Q/K
bits.

### KL trust-region target

Given a dense base marginal `mu`, use

```text
mu_target[a] proportional to mu[a] * exp(-c_a / eta)
```

Then minimize the explicit pulled-back loss

```text
L_target(x) =
    -sum_a stopgrad(mu_target[a]) * log(mu_a(x)).
```

For `mu=softmax(logits)`, the logit adjoint is `mu-mu_target`. For an arbitrary
winner-marginal function `mu(x)`, that shortcut is invalid: autograd must
apply the full Jacobian of `mu(x)` to `-mu_target/mu`.

Choose `eta` by a fixed target KL or effective sample size, not a hidden
training-step schedule. Candidate A is one possible base marginal, but its
mean-field and zero-credit failures remain present.

Advantages:

- the direction is downstream-task-aware;
- the update is bounded by a declared trust region;
- no soft route value is returned in forward;
- the target can remain dense.

Risks:

- Euclidean projection often becomes sparse and can damage discovery;
- a KL target cannot discover support absent from its base distribution;
- this only fixes route credit, not the local sign VJP;
- computing all candidate values remains dense.
- the linear cost `c_a` may rank forced-route downstream losses incorrectly;
- route-simplex targets do not by themselves produce a feasible joint
  bit-and-route target.

This should be tested both with the frozen route distribution and with
Candidate A. It should not be merged with additional entropy losses or
temperature schedules. Freeze V in the first attribution test, then compare
the linear `c_a` ranking with exact downstream losses from forced-route
outputs. The KL construction is inspired by policy-search trust regions; it
is not a direct implementation of DAPS unless routes are sampled, advantages
are estimated, and a weighted-likelihood projection is actually performed.

## 6. Candidate C: A Globally Consistent Stochastic Hard-ROSA Oracle

This is the highest-priority correctness tool. Let `s` be one global
assignment of every Q/K bit and let `R(s)` run exact latest-longest ROSA on
that assignment. Parameterize independent bit probabilities by

```text
pi_i(x) = sigmoid(x_i / tau)
q_tau(s | x) = product_i Bernoulli(s_i; pi_i(x)).
```

For a fixed upstream vector `u`, define the smoothed hard-operator scalar

```text
Phi_tau(x;u) = E_s[u^T Y_hard(R(s))].
```

The exact local-expectation derivative is

```text
d Phi_tau / d x_i =
    pi_i'(x_i)
    * E_{s_-i}[
        u^T (Y_hard(s_i=+1,s_-i)
             - Y_hard(s_i=-1,s_-i))
      ].
```

This preserves every shared-bit correlation, latest tie, null decision, and
suffix jump. Candidate A does not: it samples candidate match events
independently after shared bits have already been discarded.

`Phi_tau` is an exact VJP oracle for the expected hard operator with fixed
`u`. It is not the gradient of the full expected downstream loss unless the
downstream model and loss are reevaluated inside each hard sample. Both
objects are useful, but reports must not mix them.

Candidate estimators for this declared objective are:

1. exhaustive global-bit enumeration on very small cases;
2. local-expectation/Rao-Blackwellized coordinate estimates;
3. ARM or DisARM with globally shared hard-bit samples;
4. REBAR-like control variates using the current dense surrogate;
5. antithetic Gaussian smoothing of Q/K logits or projection parameters.

Full enumeration grows as `2^(number of relevant bits)`. Realistic exact
cases are about `T<=4,D<=2` for one head, or a single query row with a
restricted relevant history. `T=6,D=4` is not an exhaustive test. Larger
cases require Monte Carlo route frequencies and conditional-coordinate
checks.

This is initially a diagnostic oracle, not a production default. It should
first test repeated keys, repeated suffixes, null transitions, and latest
ties, because those expose independence errors with the fewest bits. Use
common random numbers and antithetic samples.

[Automatic differentiation of programs with discrete randomness](https://proceedings.neurips.cc/paper_files/paper/2022/hash/43d8e5fc816c692f342493331d5e98fc-Abstract-Conference.html)
shows that unbiased stochastic derivatives can be propagated through general
discrete random programs. It is a useful theoretical model for a stochastic
ROSA automaton, although its current tooling and event representation are not
a direct PyTorch production path.

Costs and risks:

- two or more exact hard evaluations per estimate;
- high variance in the full Q/K bit space;
- stochastic training and deterministic deployment are different objectives;
- a noise scale is unavoidable and must be tied to observed sign margins or
  a target hard-bit transition rate.
- as `tau -> 0`, the distribution becomes deterministic and its useful
  gradient vanishes away from sign boundaries; exact hard behavior and a
  globally nonzero derivative cannot both hold in that limit.

ReinMax, GOSTE, ARM/DisARM, and Rao-Blackwellized estimators become legitimate
comparisons inside this explicitly stochastic formulation. PSA remains a
biased analytic approximation. None is a drop-in replacement for the
deterministic production VJP.

## 7. Candidate D: Shared-Parameter Hard-Response Calibration

The measured tensor/parameter alignment gap suggests that improving only the
operator-local VJP may have a ceiling. Periodically calibrate the realizable
Q/K projection update using the actual hard loss.

Let `g_s` be the ordinary surrogate parameter gradient. Build a small
subspace from:

- normalized `g_s`;
- a short history of Q/K optimizer directions;
- one or two random directions orthogonal to that history.

For each selected direction `u`, measure the finite-radius response

```text
rho_epsilon(u) =
    (L_hard(theta + epsilon*u) - L_hard(theta - epsilon*u))
    / (2*epsilon).
```

Fit the lowest-norm vector in this subspace that matches the measured
responses, then blend or trust-region project the surrogate gradient toward
it. These are secants across sign-boundary events, not derivatives of the
locally constant deterministic loss. A fixed noise/radius distribution would
define a smoothed objective; direction-specific event radii instead define a
regularized response-fitting heuristic. This is Guided-ES-inspired, not the
Guided ES estimator itself.

`epsilon` should be selected by a discrete event target:

- Q/K hard-bit transition fraction;
- hard-route transition fraction;
- or quantiles of predicted sign-boundary crossing distances.

Raw parameter norm is the wrong scale because only boundary crossings can
change the hard route.

This directly tests jointly realizable shared-parameter moves and is
estimator-agnostic. With `m` directions, one calibration costs at least `2m`
full forwards, plus any epsilon search, direction/history storage,
orthogonalization, parameter overlays, and distributed synchronization.
Route-transition targeting also needs a research inspection surface because
the production CUDA API does not return route indices.

Implement this outside the operator as a no-grad training-loop harness. Start
with one surrogate direction over an epsilon grid, report forward-equivalent
overhead per calibration period, and retain the dense gradient's orthogonal
component and all value gradients.

## 8. Candidate E: Winner-Aware Margin-Weighted Minimum Edit

This is the strongest ROSA-specific target proposal. It treats Q/K bits and
the exact route as one feasible structure instead of independently repairing
candidate paths.

Let

```text
Z = {(s,r) : s is one global Q/K bit assignment and r = R(s)}
```

and let `x` be the continuous Q/K logits. The base MAP problem is

```text
(s_0,r_0) = argmax_{(s,r) in Z} <x,s>.
```

It is exactly the deterministic forward: `s_0=sign(x)` and `r_0=R(s_0)`.
Given a downstream route cost `c_r`, define the loss-augmented target

```text
(s_eta,r_eta) =
    argmax_{(s,r) in Z} <x,s> - eta*c_r.
```

The pseudo-gradient

```text
g_x = (s_0 - s_eta) / eta
```

pushes a gradient-descent step toward the globally feasible target bits.
Flipping bit `i` loses `2*|x_i|` in the base score, so the solver naturally
uses sign margins as edit costs. It must also establish the target suffix,
suppress any longer competitor, and obey the latest tie rule.

This is a legitimate black-box/loss-augmented solver target once `c_r` and the
solver are declared. It does not require pretending that independent route
matches are a probability model.

Why it may help:

- current prefix products can make a distant mismatch exponentially suppress
  all deeper credit;
- repair cost is additive and directly measures distance in the hard state
  space;
- it generalizes one-bit counterfactuals to coordinated multi-bit suffix
  formation;
- it is naturally task-directed.
- the returned bit assignment is globally consistent by construction;
- exact winner and tie constraints decide whether extending the target or
  breaking a competitor is cheaper.

Why it is difficult:

- one Q/K bit participates in many candidate paths;
- making one route win can also require suppressing later or longer
  competitors;
- solving the exact target can become a weighted MaxSAT/MIP problem;
- a feasible activation-bit target may still be unrealizable by one update of
  shared projection parameters;
- exact global minimum repair is a coupled combinatorial problem.
- the local linear route cost `c_r` may not match the true downstream loss.

Tiny cases should use exhaustive global-bit search or weighted MaxSAT and
serve as the oracle. A scalable approximation can first enumerate target
`(route,length)` pairs, satisfy their required equality constraints with
margin-weighted union/find components, and add competitor-breaking
constraints only when extension cannot win. The earlier raw path score

```text
sum_{r<l} Hamming(sign(Q[t-r]), sign(K[a-1-r]))
```

is only an independent-path heuristic; it neither guarantees that the route
wins nor chooses mutually compatible Q/K endpoints.

Because this target is generally sparse, a production version would likely
use it as a correction or periodic target alongside a dense discovery
baseline. First test exact tiny targets, one-step hard-loss descent, and
shared-parameter realizability before designing the scalable approximation.

## 9. Candidate F: Dense Bitflip Control-Variate Correction

The existing exhaustive bitflip diagnostic supplies more information than a
scalar alignment score. It defines a discrete one-bit counterfactual target
for every Q/K coordinate. Exhaustively evaluating it is too expensive, while
sampling it alone would produce the sparse gradient that this project
explicitly avoids.

Use the current dense surrogate as a control variate instead.

Let:

```text
d_i = <grad_output, Y_hard(bit i flipped) - Y_hard(base)>
s_i = -sign(x_i) * current_logit_gradient_i
I_i = indicator that coordinate i was sampled
p_i = nonzero inclusion probability of coordinate i.
```

Estimate the response in the direction of the opposite sign:

```text
r_hat_i = s_i + (I_i / p_i) * (d_i - s_i)
g_hat_i = -sign(x_i) * r_hat_i.
```

Then

```text
E[r_hat_i] = d_i.
```

Every unsampled coordinate still receives the dense surrogate. Sampling
corrects only the residual between the exact counterfactual and the cheap
proxy. This is materially different from sampled sparse backpropagation.

Why it is attractive:

- hard forward remains exact;
- dense route-discovery support remains present on every step;
- the correction targets a measured proxy error rather than adding another
  relaxation;
- variance depends on the residual `d-s`, which should be smaller than the
  raw bitflip signal when the surrogate is useful;
- the current exhaustive bitflip implementation already supplies a tiny-case
  oracle.

The construction is related in spirit to control-variate estimators such as
[REBAR](https://proceedings.neurips.cc/paper_files/paper/2017/hash/ebd6d2f5d60ff9afaeda1a81fc53e2d0-Abstract.html),
but its declared target is ROSA's one-bit counterfactual field, not a
stochastic-latent-variable likelihood.

Important limitations:

- the full bitflip field is itself a finite-difference pseudo-gradient, not
  the derivative of deterministic ROSA;
- `d_i` is exact for the hard operator under the fixed upstream linearization,
  not the exact downstream loss after a flip;
- independent activation-bit flips may not be jointly realizable through
  shared Q/K projection parameters;
- inverse-probability correction can have high variance when `p_i` is small;
- a naive sampled flip still reruns a hard suffix scan;
- an optimizer step can cross multiple sign boundaries, where one-bit
  additivity is inaccurate.
- `s_i` and `d_i` can have very different scales even when their cosine is
  useful; a fitted control-variate scale must use held-out or lagged samples
  if unbiasedness is claimed.

The first experiment should use uniform stratified sampling by batch, head,
query/key role, and sequence region. Do not begin with learned importance
sampling. Report the exact residual variance and compare:

```text
current dense surrogate
sampled bitflip alone
dense surrogate plus sampled residual correction
full bitflip on tiny cases.
```

If the hybrid works, a kernel can recompute only the hard routes whose suffix
paths contain a sampled Q/K symbol. Counter-based sampling and fixed per-stratum
budgets avoid storing a dense mask. The dense baseline remains unchanged.

An analogous operator-level randomized-smoothing correction is also possible.
For a random direction `epsilon`,

```text
delta_h =
    <grad_output,
     Y_hard(x + sigma*epsilon) - Y_hard(x - sigma*epsilon)>
    / (2*sigma)
```

gives a hard-output directional secant. Subtracting the corresponding
surrogate directional response forms a control variate. This probes
coordinated changes, but its high-dimensional variance is likely worse than
stratified bitflip residuals and it should be tested second.

## 10. Local Match Models to Test Later

Do not change the local gate in the first winner-marginal experiment. After
route geometry is isolated, compare only these models:

1. **Frozen exponential Hamming gate.**
   `G = exp(-mismatch_scale * mismatch_rate)`.
2. **Independent logistic bit noise.**
   For bit probabilities `p_q` and `p_k`, agreement is
   `p_q*p_k + (1-p_q)*(1-p_k)`.
3. **Coupled-threshold agreement.**
   With a shared random threshold and CDF `F`,
   `P(bit_q = bit_k) = 1 - |F(q)-F(k)|`.

The coupled-threshold model has an attractive hard limit: same-sign saturated
logits agree with probability one and opposite signs agree with probability
zero. It also gives a probabilistic meaning to the local derivative. However,
it introduces a CDF scale, pair correlations remain approximate, and products
over `D` bits may vanish. It is therefore not the first experiment.

Models 2 and 3 are strictly interior for finite, nonsaturated logits and can
repair the boundary-support failure of `A0`. They also allow magnitude to
encode confidence and therefore reopen scale hacking. A useful `A1` experiment
should compare ordinary logits with per-symbol RMS-normalized logits and count
that normalization as part of the estimator.

Raw complete-symbol probabilities also create a severe `D*W` calibration
problem. If two same-sign bits each have confidence `0.99`, their independent
agreement probability is `0.9802`; over `D=32` bits this is about `0.527`, and
survival through 32 such symbols is about `1.3e-9`. A normalized geometric
mean avoids that exact collapse but ceases to be the complete-symbol
probability of the declared independent-bit model. `A1` therefore trades the
old route controls for a bit-noise scale and a normalization choice; it does
not eliminate calibration by construction.

The stochastic hard-bit oracle in Candidate C should use the same bit
probabilities as `A1`. Its Monte Carlo route frequencies and gradients then
measure the error caused specifically by Candidate A's independence
assumption.

Scale-hacking checks are mandatory for every local VJP. Multiplying Q/K logits
by a positive constant leaves all hard behavior unchanged. The test matrix
must report how much the surrogate direction changes under multipliers
`0.25, 1, 4, 16`. A stop-gradient RMS normalization or tangent-space
projection is justified only if this test exposes a real failure.

## 11. Strict Experimental Matrix

### Phase 0: estimator contract

Every candidate must pass:

1. bit-exact hard forward equality with production;
2. no soft value or probability visible in forward;
3. latest-tie and null-route unit tests;
4. complete support and zero-gradient maps, with boundary zeros reported
   rather than silently clamped;
5. declared limiting behavior;
6. deterministic replay, or explicit stochastic seed semantics;
7. finite outputs at exact zeros, exact ones, and long suffixes.

Dense discovery remains a production-project constraint, but it is not a
mathematical correctness condition for an oracle or structured target.
Research candidates should be judged by escape probability, hard-response
descent, and training behavior. A sparse target can qualify only as a
correction alongside a dense baseline.

Candidate A additionally requires:

- winner masses sum to one;
- exhaustive enumeration of independent local-match Bernoulli variables
  agrees with the analytic marginal for tiny cases;
- the zero-temperature/local-hard limit is exactly the production route;
- permutations that preserve route order do not alter mass;
- explicit measurement of deeper suffix gates that are irrelevant to the
  current winner and therefore receive zero gradient.

Candidate C additionally requires:

- exact enumeration on globally shared bits for tractable shapes;
- repeated-key and repeated-suffix correlation cases;
- ARM/DisARM/local-expectation estimates compared at equal hard-evaluation
  budgets;
- separate labels for operator-local fixed-upstream and full downstream-loss
  objectives.

### Phase 1: local gradient diagnostics

Use:

```text
D in {2, 4, 8, 16, 32}
T in {4, 8, 16, 64}
W in {1, 4, 8, 32}
```

Report:

- bias, variance, cosine, and sign agreement against the estimator's declared
  objective;
- cosine against exhaustive bit-flip deltas as a separate heuristic metric;
- gradient support and effective candidate count;
- Q/K sign-margin distribution;
- actual bit and route changes after one virtual optimizer step;
- operator-local hard response and exact downstream-loss change after that
  step, reported separately;
- tensor-space and shared-parameter-space alignment.
- route-only attribution with V frozen, followed by selected-only versus
  distributed value-carrier ablations.

### Phase 2: training behavior

The minimum estimator set is:

| ID | Estimator |
| --- | --- |
| B0 | frozen production estimator |
| A0 | latest-longest winner marginals with frozen local gate |
| A1 | winner marginals with the best strictly interior local model |
| B1 | task-directed pullback on the frozen distribution |
| AB | winner marginals plus KL/SPIGOT pullback |
| C0 | exact globally shared-bit stochastic oracle on tiny cases |
| C1 | ARM/DisARM/local-expectation estimates of the same objective |
| D1 | best dense VJP plus periodic hard parameter calibration |
| E0 | exact winner-aware margin-edit target on tiny cases |
| F1 | dense VJP plus sampled exact-bitflip residual correction |

Minimum tasks:

- strict latest-longest fitting;
- any-candidate fitting;
- hard-null route creation;
- long-suffix discovery with an early mismatch;
- contextual reset-RNN recall;
- collision-free and deliberately collision-limited datasets.

Use at least eight model seeds and multiple launch repeats where CUDA
accumulation order can affect basin selection. Always report:

- hard-feature conditional entropy;
- excess loss above that entropy floor;
- route and bit collision class;
- any-run and per-run success;
- steps to threshold;
- runtime and peak memory.

### Phase 3: decision gates

A production replacement must:

1. preserve exact hard forward behavior;
2. remove or justify every new scalar control;
3. improve shared-parameter hard-descent rate, not only tensor cosine;
4. improve at least one collision-free fitting/recall result without a
   material regression elsewhere;
5. retain dense discovery;
6. avoid `O(BHT^2W)` persistent tape and provide an explicit byte-capped
   checkpoint/recompute plan rather than treating a quadratic field as free;
7. survive logit-rescaling and long-context tests.

Do not optimize a CUDA kernel before a PyTorch estimator clears these gates.

## 12. Recommended Execution Order

1. Build `C0`, the exact globally consistent stochastic-bit oracle, on
   genuinely enumerable shapes (`T<=4,D<=2` or a restricted query row).
2. At equal hard-evaluation budgets, compare local expectation, ARM, DisARM,
   a current-surrogate control variate, B0, and the existing deterministic
   bitflip field.
3. Build `E0`, the exact winner-aware margin-edit solver, by exhaustive search
   or weighted MaxSAT on the same tiny structures. Measure feasible target
   rate and paired one-step hard descent after shared-parameter aggregation.
4. Implement `A0` only as a float64 semantic ablation. Prove normalization and
   tie handling, then quantify impossible-route mass, boundary zeros,
   candidate-count sensitivity, and lost deep-suffix credit before training.
5. Compare one strictly interior `A1` against Monte Carlo frequencies from
   the identical global-bit model. Stop if total-variation or gradient error
   grows systematically with `D*W`, repeated motifs, or rescaling.
6. Prototype `F1` with uniform stratified samples and report residual variance
   before any learned sampling law or kernel work.
7. Run route-target and carrier attribution with V frozen first. Compare the
   plain carrier, explicit KL target, SPIGOT, and exact margin-edit target.
8. Test shared-parameter response fitting last, beginning with one direction
   and a fixed epsilon grid.
9. Only surviving estimators enter multi-seed fitting and contextual recall.
10. Design a kernel only after an estimator clears the semantic, optimization,
    and cost gates.

## 13. Checks Completed for This Proposal

The analytic Candidate A winner formula was compared with exhaustive
enumeration for 200 random independent-gate cases with one to four candidates
and varying horizons from one to three. Maximum probability and normalization
error in float64 was `7.77e-16`. This validates the formula under its
independence assumption, not the assumption itself.

The exact-gate support counterexample was also evaluated with autograd. For
three one-symbol candidates with gates `(0.7,1,1)`, the first candidate had
zero mass and gradient `(0,0,0)`. Replacing the later gates by `0.999`
restored only `7e-7` mass, demonstrating both the exact boundary failure and
the severe near-boundary conditioning.

Three independent reviews covered:

- winner probability, tie/null semantics, and hard-limit mathematics;
- stochastic-estimator objective matching and missing structured methods;
- PyTorch tape, CUDA scan, numerical zero, packed-varlen, and calibration
  costs.

Their corrections are incorporated above. No production source or operator
behavior changed.

The Candidate A hypothesis is precise:

> Replacing `softmax(E[suffix length])` with a differentiable marginal of the
> actual latest-longest winner will improve immediate route credit enough to
> compensate for losing speculative complete-prefix credit.

This is falsifiable. The parameter-removal benefit is real only if the
resulting context-length prior and zero-gradient regions remain trainable.
