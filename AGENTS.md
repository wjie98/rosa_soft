# ROSA Development Constraints

These constraints apply to the entire repository.

## Frozen Dense Reference

The `rosa-soft-dense-reference-v1` tag is the frozen baseline for exact hard
forward with dense surrogate backward. Treat its public `rosa_soft` and
`rosa_soft_varlen` semantics as immutable. Correctness and reproducibility
fixes may be backported without changing equations, defaults, candidate
support, RNG semantics, or tie/null behavior.

New estimators, sparse training methods, diagonal execution prototypes, and
alternative hard indexes must use a new module or operator name. Do not add
research-only schemas, kernels, switches, or hidden dispatch back to the
frozen extension. Historical experiments belong in `benchmarks/` and
`docs/research/`, outside the package and build graph.

## RosaSoft Training Contract

RosaSoft exists to train a sparse hard ROSA route with dense credit
assignment. The default training operator must preserve both sides:

- forward is exact hard ROSA;
- backward evaluates every valid causal route and differentiates the full
  dense Q/K proxy distribution;
- values receive probability-weighted credit from the same dense
  distribution.

Dense backward support is a semantic requirement, not an implementation
accident. Do not replace it with top-k candidates, ANN/LSH lookup, sampled
negatives, hard-winner neighborhoods, score thresholds, early termination, or
suffix-index candidate pruning. Such changes bias the route gradient toward
the model's current routing state, can make undiscovered routes receive no
signal, and can create self-reinforcing collapse.

An exact hard suffix index may optimize forward only. It must never determine
which routes participate in the default backward pass.

## Allowed Kernel Optimizations

Optimize the dense estimator without changing its support or formulas.
Preferred directions include:

- online softmax reductions over all causal routes;
- FlashAttention-style dense route tiling;
- exact finite-window diagonal recurrences and their exact adjoints;
- shared-memory or register caching and checkpoint/recompute tradeoffs;
- packed symbols, warp reductions, and block-local gradient accumulation;
- `needs_input_grad` specialization that skips unused Q/K/value work;
- precision changes that retain FP32 score/normalizer accuracy and pass parity.

Every valid causal route must still be visited. Numerical underflow is a
floating-point issue; structurally masking or omitting a route is not
allowed.

Dense compute does not require quadratic persistent state. The production
CUDA path computes each route's normalized Hamming mismatch, raw suffix
evidence, and transformed route score on demand; it must not store an
`O(B H T^2 D)` bit-comparison tensor. A new
`O(B H T^2)` score or adjoint workspace also requires an explicit memory
analysis and a checkpointing justification. The PyTorch reference may
materialize these tensors because it is the correctness oracle, not the
production kernel.

Current shape-matched measurements do not justify a dedicated `D=8` template:
it doubled backward kernel instances without improving latency. Likewise, a
full-row packed score cache selected from total token count increased
register/shared-memory pressure and regressed tested varlen shapes. This does
not prohibit the retained Q-only bounded tail cache, which uses average
segment length and recomputes every older score. Do not reintroduce the broad
choices without a new shape matrix that demonstrates a consistent win.

Private Q/K sign-bit buffers are head-major: dense uses `[B,H,T]` and packed
varlen uses `[H,N]`. Public packed inputs and gradients remain token-major
`[N,H,D]`. Dense K VJP accumulates in FP32 `[B,H,D,T]`, then returns a
contiguous public `[B,T,H,D]` tensor. This costs one additional
key-gradient-sized FP32 temporary at the final transpose, but coalesces
route-lane atomics and won the tested K, QK, and QKV matrices. Packed K VJP
still accumulates directly into `[N,H,D]`; extending the dense accumulator
layout to varlen regressed measured QK/QKV paths by `3.6%..26.1%`. Do not
repeat that change without solving the occupancy and transpose costs on a
representative gradient-mask matrix.

The current row kernel maps lanes to route candidates. Its suffix loop has no
data-dependent mismatch exit, so assigning a whole warp to one candidate
reduces useful route parallelism. A warp prefix scan is appropriate only as
part of a diagonal-recurrence kernel that also solves row-softmax staging,
finite-window correction, reverse adjoints, and workspace bounds. Do not add
a warp-per-route path to the current kernel or restrict the public suffix
window to multiples of 32 based on scan convenience.

The finite-window affine reverse formula is exact algebraically but loses
FP32 precision on long exact matches because it subtracts full-diagonal
correction terms. The cancellation-resistant research oracle sums only the
at-most-`W` real local contributions with compensated accumulation. Any
future diagonal production kernel must match that local oracle on the
`N=4096,W=4096` exact-match gate; do not restore the subtractive reverse as
the sole parity target.

The tested multi-stage diagonal VJP is a research control, not a production
plan. It passed parity but was slower on both sm_75 and sm_86 and materializes
an `O(B H T^2)` FP32 local-gate tensor. A block-local/ring replacement must
remove that workspace and pass the cross-GPU promotion gate before selection.
Likewise, the exact hard diagonal index is valuable on collapsed all-match
codes but regresses high-entropy codes. Do not select it from `T`, `D`, or
`W` alone; automatic use requires an asynchronous, validated code-density
signal that does not affect backward support.

The retained private fast paths are narrowly gated:

- dense Q-only utility caching requires `Dv >= 32` and a shared-memory fit;
- cooperative route-utility/value tiling requires a Q or K gradient,
  `Dv >= 64`, a sufficiently long dense row or average packed segment, and a
  shared-memory fit; packed QK-only starts at `Dv >= 128`;
- dense and packed value-only route/value tiling requires `Dv >= 32` and a
  shared-memory fit, independent of whether dense scores are cached or
  recomputed;
- dense K aggregation requires an enabled K gradient and `D <= 8`; K/KV may
  select it at `T >= 512`, while a simultaneous Q gradient raises the gate to
  `T >= 1024`; the complete shared layout must fit;
- packed K aggregation requires an enabled K gradient, `D <= 8`, average and
  local segment length at least 256, and a shared-memory fit;
- packed Q-only score caching requires average segment length at least 256 and
  stores only the most recent 1024 exact scores; every older score is
  recomputed, never omitted.

These thresholds are measured execution choices, not public semantics.
Broadening them requires all-gradient-mask A/B measurements and reference VJP
parity.

New execution plans are calibrated offline. Promotion requires bit-exact hard
parity (or the documented VJP tolerance), no tested latency ratio above 1.05,
at least 5% mean speedup, and no more than 64 MiB extra scratch in the stated
matrix. Do not add first-call timing, device-name allowlists, or hidden
environment overrides to the training operator.

Dense score-plan selection is conservative. Recompute is considered only for
`T >= 4096`; value-only backward additionally requires `T >= 64 W`, because
its second score scan is a larger fraction of total work. The host then
queries active blocks for the exact cache and recompute kernel instances,
including their complete dynamic shared layouts, and compares total launch
waves over the actual dense row count. Recompute is selected only when it
passes the work gate and reduces launch waves; equal waves prefer cache. These
SM75-calibrated constants are private execution thresholds, not estimator
semantics. Do not use total packed token count as a dense-cache proxy.

Cooperative temporary route-utility tiles are warp-owned: each warp produces
and consumes its own 32 route slots, with synchronization before consumption
and before slot reuse. The full dense Q-only utility cache is block-produced,
uses a block barrier before cross-warp consumption, and is not overwritten
between route tiles. Changes to either ownership rule require CUDA racecheck
and synccheck coverage, not only numerical parity.

The backward CTA size remains 128 threads. Global 64- and 256-thread builds
both won isolated masks but regressed other dense or packed paths by much
larger margins. Do not add a block-size ladder without a representative
dense/packed, all-mask matrix and a simpler selection law than the rejected
layout/mask/length heuristics.

## Packed Variable-Length Contract

`rosa_soft_varlen` and `rosa_soft_varlen_reference` operate on packed
`[N,H,D]` query/key and `[N,H_v,D_v]` value tensors. `cu_seqlens` defines
complete semantic boundaries:

- hard routes and suffix products never cross a segment;
- null and non-null candidate normalization is computed per local row;
- empty segments are valid;
- local value position zero is never a route candidate;
- enabled gradients never cross a segment.

With `dropout_p=0`, dense and packed execution are deterministic functions of
their local sequence contents. With dropout enabled, the same seed and local
batch/sequence, head, query, and route indices must reconstruct the same mask.
Do not substitute total `N` for a segment's local length, infer a segment's
competition set from neighboring packed storage, or add persistent segment
IDs/a public `max_seqlen` argument without measured need. Offset validation
must not introduce a host synchronization in the CUDA hot path.

## Sparse Experiments

A sparse-gradient experiment must be a separate, explicitly named research
operator and remain off by default. It cannot replace the dense reference or
the main CUDA training path. Before it can be considered, it must measure
gradient bias against the dense VJP, route discovery recall, multi-seed
collapse rate, and hard-forward fitting quality.

Global-bit stochastic research must sample each semantic Q/K bit once and
reuse that assignment everywhere the bit participates. Per-pair or per-suffix
resampling is a different model and cannot be reported as an ARM/DisARM
estimate of globally shared hard ROSA. Keep the production dense VJP, a
stochastic expected-hard VJP, deterministic bitflip fields, and structured
margin-edit targets labeled as different optimization objects.

Do not kernelize ARM/DisARM, mean-field winner marginals, sampled bitflip
residuals, or margin-edit targets from tiny-oracle results alone. A candidate
must first improve shared-parameter and shortcut-free contextual training at
matched hard-evaluation cost. In particular, lower variance relative to the
full bitflip field is not sufficient: on coordinated suffix edits that field
can be identically zero, and sampling fewer residual coordinates can succeed
only by preserving zero-mean exploration noise from the dense baseline.

The shared-projection and shortcut-free contextual gates have now been run.
At `tau=0.5`, the two-parameter projection fit succeeded in `16/16` runs for
production, mean-field, exact expectation, ARM, DisARM, and margin edit, but
production needed a median 64 steps versus 6 for mean-field/exact expectation.
On four 1000-step contextual seeds, production, production with dropout,
mean-field, ARM, and DisARM all passed; their median first-exact steps were
`92.5, 85, 103, 132, 163.5`. ARM/DisARM used four antithetic pairs per step,
and mean-field was slower than production in the instrumented prototype.
Reducing ARM/DisARM to one pair retained `4/4` final success but worsened
median first-exact steps to `296` and `441`, respectively.
These results keep all alternative estimators research-only: none improved
the contextual gate at matched cost. Reopening kernel work requires a new
shared-parameter regime where an alternative improves robustness or sample
efficiency, not another activation-space cosine result.

## Internal-Language Experiments

Stateful symbol generators, fixed phase/outline symbols, multi-hop feedback,
and dormant-bit growth are research model components. They are not RosaSoft
operator semantics and must not add public controls or alter the production
VJP.

Keep three claims separate in every result: an explicit hard codebook is
representable, a learned initialization routes correctly, and correctness is
retained after surrogate-gradient training. The strict grammar gate shows
these are not interchangeable: full 2/4/8-position oracle codes all route at
100%, while four-position learned codes pass only 2/4 seeds and eight-position
codes pass 0/4 under the current primary matrix.

An internal-language recall gate must exclude trainable value/readout and
residual shortcuts or retain explicit zero/current/shuffled interventions.
For multi-hop claims, the first hard routed value must be the only
assignment-dependent input to the next query; report each hop's route and
value accuracy, not final output alone.

Conflict-driven bit activation is an external structural edit when it reads
continuation labels. Report it as such, and report hard key conflicts
separately from query/key self-alignment. A conflict-free key codebook does
not imply correct routing when read and write heads have drifted apart. Do not
turn conflict entropy, alignment, reconstruction, or split margins into a
weighted loss bundle without a separate design decision.

## Operator Simplicity

The public RosaSoft controls remain:

- `max_suffix_length`;
- `scale`;
- `dropout_p`;
- `mismatch_scale`.

`scale` is the static multiplicative attention-logit scale with default `1.0`.
It is the reciprocal of the removed temperature convention. Do not derive it
from D, T, W, active suffix length, training step, or diagnostics. ROSA scores
are already based on a D-normalized mismatch rate, so there is no implicit
`1 / sqrt(D)` default.

`dropout_p` is standard post-softmax inverted attention dropout with default
`0.0`. It applies to the route weights used by the backward carrier and
therefore affects Q/K/value VJPs; hard forward must never read dropout state.
The implementation may store one scalar seed and reconstruct each route mask
with a counter-based RNG, but must not materialize or save an
`O(B H T^2)` random tensor. `dropout_p=0` must be an exact deterministic path
that does not advance RNG state. This excludes intentional sampling only:
the CUDA VJP uses global FP32 atomics and is not bitwise reproducible across
launches. Repeated fitting runs can therefore enter different hard-symbol
basins even when `dropout_p=0`; production training comparisons must include
repeated launches rather than treating one run per model seed as deterministic.

`mismatch_scale` is a static mismatch control with default `3.0`. Do not derive
it from configured D, T, or W: those bounds do not determine the active
Hamming distance or suffix competition. Local mismatch counts are normalized
by D before this scalar is applied. The default is the conservative
`dropout_p=0` choice from the current matched fitting probe, not a universal
optimum. `9.0` remains an explicit experiment value and must not silently
replace the default based on results from the removed tiered/random estimator.
The internal null score is fixed at `0.5`.

Null calibration must retain the non-null `-log(candidate_count)` correction:
without it, random-background recall converges to one as context grows. Under
uniform independent hard bits, `D=4`, `mismatch_scale=3`, and `scale=1`, the
fixed `0.5` null score already gives approximately `0.5` random-background
recall. Power/log suffix utilities and a collision-likelihood-ratio route were
tested as exact-hard-forward research controls. Stronger concavity repaired
some long-route competition but reduced large-candidate capacity; collision LR
improved broad fitting but lost directed long-route competition and sometimes
assigned the newest bit credit in the wrong direction. Keep both research-only
unless one replacement passes fitting, directed competition, contextual
recall, and large-candidate calibration together.

The backward first computes raw expected matched-prefix evidence `S` from
deterministic exponential mismatch gates, then uses the fixed route score
`U(S)=(sqrt(2)+1)(sqrt(1+S)-1)`. This transform is monotone, unbounded, and
calibrated by `U(0)=0,U(1)=1`; its derivative must be included coherently in
the Q/K VJP. Do not make it configurable or replace it with exact hard suffix
tiers, bounded residuals, sampled perturbations, antithetic branches, or
separate numerical/Jacobian gates. The fixed concavity is retained because it
improved long-versus-short route discovery without weakening hard forward.
The null route is one hypothesis against all non-null routes, so non-null
logits include the fixed candidate-count prior correction.

Distributed soft value credit is part of the validated estimator.
Hard-selected value won a direct structural probe but failed the
repeated-motif full-model fit for all tested mismatch scales, so it must
remain research-only unless a replacement passes both gates.

The repeated-motif fit is a combinatorial estimator stress test, not a pure
recall benchmark. Its historical-target mask requires a correct candidate to
exist, but does not require exact token-equality ROSA to select that candidate
under longest/latest routing. Every reported run must therefore include the
hard-feature conditional-entropy lower bound and the loss above that bound.
Do not attribute a plateau to value/readout optimization when the bound
already explains it, and do not use this fit alone to claim semantic route
learning. Direct recall claims require the shortcut-free associative gate or
an explicitly strict longest/latest target mask.

The two repeated-motif target modes must remain explicit:

- `any-candidate` asks whether any correct historical continuation exists and
  is the harder combinatorial recoding stress test;
- `strict-longest-latest` retains only rows where the raw token ROSA route
  already selects the target and is the cleaner automaton-recall probe.

Gradient-alignment claims must include trained checkpoints, tensor-space
alignment, shared Q/K parameter-space alignment, and the hard-feature entropy
floor. Random-tensor bitflip probes alone do not explain a trained plateau.

The contextual reset-RNN gate must preserve its causal proof: paired episodes
have complementary payloads, post-reset query inputs and residuals are
exactly identical, and zero-route, current-value, and residual-only ablations
cannot solve the pair. Require actual hard routes into earlier history and a
different hard routed value across complementary pairs. A payload position is
a canonical diagnostic route, not the only valid route: contextual values and
multiple heads may carry the same useful history through another earlier
slot.

Do not reintroduce dynamic schedules, window-derived mismatch scales,
confidence weighting, winner margins, recency bias, configurable perturbation
modes, value-gradient modes, early-stop epsilon, Q/K dampers, or hot-path
telemetry without new evidence and an explicit design decision.

The CUDA VJP must honor `ctx.needs_input_grad`. Disabled gradients may use
zero-length internal outputs, but every enabled gradient must remain
numerically equivalent to the complete dense VJP for the same dropout mask.
Value-only backward may skip route-utility and suffix-adjoint work that cannot
affect value credit; it may not prune route probabilities.

## Training and Runtime Boundary

The training and inference surfaces intentionally have different bit-width
contracts:

- `rosa_soft` and `rosa_soft_reference` accept Q/K widths in `1..32`;
- `RosaRuntime` is a packed one-byte deployment subset and accepts Q/K and
  payload widths in `1..8`.

Do not make the Runtime state machine wider merely to mirror the training
operator. A wider deployment format needs a separate memory, serialization,
and 100M-token scaling analysis. Runtime state may grow with context today;
the finite suffix horizon bounds matching work, not total automaton storage.

For asynchronous Runtime updates, tensors accepted by the Python API must be
snapshotted before control returns to the caller. Background work must never
borrow mutable caller storage. The pending-work queue must have explicit
backpressure, and every queued operation must observe terminal `FAILED` or
`CLOSED` state before entering native code.

## Validation Gate

Any RosaSoft kernel change must preserve:

- bit-exact hard forward output and latest-route tie behavior;
- full causal candidate support in backward;
- PyTorch/CUDA VJP parity for `dropout_p=0` and fixed dropout seeds within
  documented tolerances;
- FP32, FP16, BF16, grouped-head, `D=32`, singleton, and non-contiguous tests;
- packed-varlen boundaries, empty segments, counter parity, and unequal lengths;
- invalid CUDA offsets in isolated subprocesses and metadata overflow guards;
- all seven nonempty Q/K/value `needs_input_grad` masks, including dropout;
- repeated-motif fitting and a route-discovery test across multiple seeds;
- benchmark reporting for time, peak memory, registers, and spills.

The public, unseeded `rosa_soft` call must support
`torch.compile(..., backend="aot_eager", fullgraph=True)`; the packed-varlen
call must satisfy the same gate. Inductor or CUDA Graph support requires a
separate explicit test before documentation may claim it. Tests against a
private seeded helper do not satisfy this requirement. Integration coverage
must include autocast/GradScaler and activation checkpoint replay. CUDA
kernels that use global atomic accumulation must participate in PyTorch
deterministic-algorithm checks and must not claim bitwise determinism. Kernel
parity must include long configured suffix horizons, not only the smallest
execution-plan boundary.

The dense PyTorch implementation is the semantic oracle. Performance gains do
not justify silently changing the gradient estimator.
