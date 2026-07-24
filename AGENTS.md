# ROSA Development Constraints

These constraints apply to the entire repository.

## RosaSoft Training Contract

RosaSoft exists to train a sparse hard ROSA route with dense credit
assignment. The default training operator must preserve both sides:

- forward is exact hard ROSA;
- backward evaluates every valid causal route and differentiates the full
  dense Q/K proxy distribution;
- payload logits receive probability-weighted credit from the same dense
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
- counter-based random reconstruction;
- shared-memory or register caching and checkpoint/recompute tradeoffs;
- packed symbols, warp reductions, and block-local gradient accumulation;
- precision changes that retain FP32 score/normalizer accuracy and pass parity.

Every valid causal route must still be visited. Numerical underflow is a
floating-point issue; structurally masking or omitting a route is not
allowed.

Dense compute does not require quadratic persistent state. The production
CUDA path must reconstruct mismatch randomness with a counter-based RNG and
keep only local `h/H` accumulators; it must not store an
`O(B H T^2 D)` random tensor. A new `O(B H T^2)` score or adjoint workspace
also requires an explicit memory analysis and a checkpointing justification.
The PyTorch reference may materialize these tensors because it is the
correctness oracle, not the production kernel.

## Sparse Experiments

A sparse-gradient experiment must be a separate, explicitly named research
operator and remain off by default. It cannot replace the dense reference or
the main CUDA training path. Before it can be considered, it must measure
gradient bias against the dense VJP, route discovery recall, multi-seed
collapse rate, and hard-forward fitting quality.

## Operator Simplicity

The public RosaSoft controls remain:

- `max_suffix_length`;
- `route_temperature`;
- `mismatch_penalty`.

`route_temperature` is a static backward optimization control with default `1.0`.
Do not derive it from D, T, W, active suffix length, training step, or
diagnostics. Broad tuning is closed; controlled comparisons use only
`0.5`, `1.0`, and `2.0`.

`mismatch_penalty` is a static mismatch control with default `3.0`. Do not derive
it from configured D, T, or W: those bounds do not determine the active
Hamming distance or suffix competition. The internal null score is fixed at
`0.5`.

Distributed soft payload credit is part of the validated estimator.
Hard-selected payload won a direct structural probe but failed the
repeated-motif full-model fit for all tested mismatch penalties, so it must
remain research-only unless a replacement passes both gates.

Do not reintroduce dynamic schedules, window-derived mismatch penalties,
confidence weighting, winner margins, recency bias, configurable perturbation
modes, payload-gradient modes, early-stop epsilon, Q/K dampers, or hot-path
telemetry
without new evidence and an explicit design decision.

## Validation Gate

Any RosaSoft kernel change must preserve:

- bit-exact hard forward output and latest-route tie behavior;
- full causal candidate support in backward;
- fixed-counter PyTorch/CUDA VJP parity within documented tolerances;
- FP32, FP16, BF16, grouped-head, `D=32`, singleton, and non-contiguous tests;
- repeated-motif fitting and a route-discovery test across multiple seeds;
- benchmark reporting for time, peak memory, registers, and spills.

The dense PyTorch implementation is the semantic oracle. Performance gains do
not justify silently changing the gradient estimator.
