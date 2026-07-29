# ROSA Development Constraints

These constraints apply to the entire repository.

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
CUDA path computes each route's normalized Hamming mismatch and suffix score
on demand; it must not store an `O(B H T^2 D)` bit-comparison tensor. A new
`O(B H T^2)` score or adjoint workspace also requires an explicit memory
analysis and a checkpointing justification. The PyTorch reference may
materialize these tensors because it is the correctness oracle, not the
production kernel.

Current shape-matched measurements do not justify a dedicated `D=8` template:
it doubled backward kernel instances without improving latency. Likewise,
using total packed token count to select the dense score-cache plan increased
register/shared-memory pressure and regressed tested varlen shapes. Do not
reintroduce either choice without a new shape matrix that demonstrates a
consistent win.

Dense packed Q/K symbols are head-major `[B,H,T]`. Dense K VJP accumulates in
FP32 `[B,H,D,T]`, then returns a contiguous public `[B,T,H,D]` tensor. This
costs one additional key-gradient-sized FP32 temporary at the final transpose,
but coalesces route-lane atomics and won the tested K, QK, and QKV matrices.
Packed varlen deliberately remains token-major `[N,H]` with direct
`[N,H,D]` K accumulation. Extending the dense accumulator layout to varlen
regressed measured QK/QKV paths by `3.6%..26.1%`; do not repeat that change
without solving the occupancy and transpose costs on a representative
gradient-mask matrix. Packed-varlen register pressure remains an optimization
target and must be reported for every estimator change.

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
that does not advance RNG state.

`mismatch_scale` is a static mismatch control with default `3.0`. Do not derive
it from configured D, T, or W: those bounds do not determine the active
Hamming distance or suffix competition. Local mismatch counts are normalized
by D before this scalar is applied. The default is the conservative
`dropout_p=0` choice from the current matched fitting probe, not a universal
optimum. `9.0` remains an explicit experiment value and must not silently
replace the default based on results from the removed tiered/random estimator.
The internal null score is fixed at `0.5`.

The backward route score is the raw expected matched-prefix length obtained
from deterministic exponential mismatch gates. Do not wrap it in exact hard
suffix tiers, bounded residuals, sampled perturbations, antithetic branches,
or separate numerical/Jacobian gates. Those mechanisms weakened long-suffix
credit or added variance and implementation state without a validated gain.
The null route is one hypothesis against all non-null routes, so non-null
logits include the fixed candidate-count prior correction.

Distributed soft value credit is part of the validated estimator.
Hard-selected value won a direct structural probe but failed the
repeated-motif full-model fit for all tested mismatch scales, so it must
remain research-only unless a replacement passes both gates.

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
