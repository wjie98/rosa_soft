# RosaSoft Design

This document describes the current implementation. The normative equations
are in [CONCEPT.md](CONCEPT.md).

## 1. Module Boundaries

| Module | Responsibility |
| --- | --- |
| `soft_contract.py` | Shared shapes, dtypes, scalar defaults, and validation. |
| `soft_reference.py` | Pure PyTorch semantic oracle. |
| `soft.py` | Public CUDA wrappers and custom-autograd boundary. |
| `csrc/export.cpp` | PyTorch dispatcher schemas and registrations. |
| `csrc/rosa_soft.cpp` | Native validation, allocation, and CUDA dispatch. |
| `csrc/cuda/rosa_soft_kernels.cu` | Hard forward and dense surrogate VJP. |
| `testing.py` | Development-only materialized inspection state. |
| `diagnostics.py` | Summaries derived from inspection state. |

The public API does not expose estimator modes, seeds, random tensors, cache
plans, or diagnostics.

## 2. Forward Data Flow

Dense CUDA forward:

```text
Q/K logits
  -> pack sign bits into int32 [B,H,T]
  -> scan causal routes and exact suffixes
  -> latest-longest route index
V logits
  -> sign selected value
  -> hard output [B,T,H,Dv]
```

Forward also returns packed Q/K symbols to the private autograd wrapper.
Saving these avoids quantizing Q/K again in backward. The packed tensors are
linear in input size, not quadratic in route count.

The no-gradient public path calls hard forward directly and discards packed
symbols.

Packed variable-length forward stores private Q/K sign bits as head-major
`[H,N]` tensors for coalesced route-lane reads. Public query/key/value tensors
remain token-major, and the same state machine runs independently inside every
`cu_seqlens` segment.

## 3. Autograd State

The dense custom autograd function saves:

```text
Q, K, V, packed_Q, packed_K, dropout_seed
max_suffix_length, scale, dropout_p, mismatch_scale
```

The packed variant additionally saves `cu_seqlens`.

It saves one scalar RNG seed only when `dropout_p > 0`; it does not save route
probabilities, suffix scores, random samples, winners, or a quadratic
workspace. Backward recomputes route scores and reconstructs dropout decisions
from route indices. This is the
appropriate checkpoint boundary because route state is much larger than the
inputs for long sequences.

`ctx.needs_input_grad` is converted to a three-bit mask:

```text
bit 0: Q
bit 1: K
bit 2: value
```

CUDA skips work and allocation that cannot affect requested gradients.
Disabling a gradient must not alter the route distribution used by another
enabled gradient.

## 4. Backward Data Flow

For each batch, head, and query row, CUDA performs:

1. Visit null and every valid non-null causal route.
2. Reconstruct normalized Hamming mismatch from packed Q/K symbols.
3. Evaluate exponential gates and the complete raw suffix prefix sum `S`.
4. Transform non-null evidence with
   `U(S)=(sqrt(2)+1)(sqrt(1+S)-1)`.
5. Compute online softmax statistics using the fixed null score and
   non-null candidate correction.
6. Apply post-softmax inverted dropout to each route probability.
7. Accumulate probability-weighted value credit.
8. For Q/K gradients, compute dropped route utility from `grad_output` and
   signed value, then apply the exact adjoint of route softmax, `U`, the suffix
   recurrence, exponential gate, and softsign VJP. The additional chain factor
   is `U'(S)=(sqrt(2)+1)/(2 sqrt(1+S))`.

The implementation may scan a row more than once to avoid storing all route
scores. Recompute is an implementation choice; the visited candidate set and
equations are fixed.

## 5. CUDA Execution Plans

The backward kernel has two internal shared-memory plans:

- `RecomputeSuffixScores` for shapes where caching is not profitable or does
  not fit;
- `CacheSuffixScores`, which stores one row of raw suffix evidence plus the
  row query symbols and `grad_output` when the complete shared layout fits.

Both plans compute the same VJP. They are not estimator modes and never alter
candidate support. Dense CUDA first applies a measured work gate: recompute
is considered at `T >= 4096`, and value-only backward also requires
`T >= 64 W`. It then queries active blocks for the exact dtype, gradient-mask,
cooperative mode, and complete dynamic shared layout, converts occupancy into
total grid waves for the actual row count, and recomputes only when the launch
needs fewer waves. Equal waves prefer cache. Plan choice stays private because
it depends on compiled kernel resources, not model semantics. Packed varlen
always uses the recompute base plan because segment lengths vary inside one
launch; its bounded Q-only tail cache is a separate exact reuse layer, not a
full-row plan.

Dense K gradients use a head-major FP32 accumulation layout to coalesce
route-lane atomics, followed by a transpose to public `[B,T,H,D]`. For
`D <= 8`, overlapping K contributions may first be aggregated in a
`[D, blockDim + W - 1]` block-local FP32 tile. K/KV selects this path at
`T >= 512`; QK/QKV waits until `T >= 1024`, where the saved atomics outweigh
the tile and synchronization cost. Packed variable-length K gradients remain
token-major because the dense layout increased register pressure and regressed
measured SM75 kernels. Their block-local aggregation requires `D <= 8`, average
and local segment length at least 256, an enabled K gradient, and a shared-memory
fit. All other shapes use direct atomics.

Several narrowly gated paths remove avoidable value-dimension serialization
without changing route support:

- Q-only backward with `Dv >= 32` caches one route utility per candidate.
  Groups of 8 lanes cooperate at `Dv=32`; groups of 16 cooperate at larger
  dimensions.
- value-only backward with `Dv >= 32` caches one route tile of probabilities,
  then flattens route and value dimensions so adjacent lanes update adjacent
  value elements.
- Dense and packed Q/K paths use the same cooperative utility tile for
  sufficiently long sequences with `Dv >= 64`. Packed QK-only requires
  `Dv >= 128`; other packed masks start at 64.
- Packed Q-only backward with average segment length at least 256 caches the
  most recent 1024 exact route scores. Earlier routes are still visited and
  their scores are recomputed in the second pass.

Every path is selected only when its complete shared layout fits the portable
48 KiB limit. Small or unsupported shapes retain the generic path. These are
execution plans, not estimator variants. The backward CTA remains a fixed 128
threads: 64 and 256 won conflicting subsets of dense and packed masks, so a
runtime block-size ladder was rejected.

Temporary cooperative utility tiles are warp-owned: one warp produces and
consumes 32 route slots and synchronizes before consumption and reuse. The
full dense Q-only utility cache is produced across the block, synchronized
before cross-warp reads, and never reuses a route slot. Representative dense
full-cache, packed Q-only, and packed QKV paths pass both CUDA racecheck and
synccheck.

The row kernel accumulates raw FP32 Q/K/value symbol adjoints. A single
finalize kernel then applies each input's softsign Jacobian once; for dense K
it also transposes the head-major accumulator into public layout. This removes
repeated Jacobian loads from the route/suffix hot loops without changing the
surrogate.

All score, normalizer, and gradient accumulation arithmetic is FP32. Public
gradients are cast back to the input dtype in Python. `dropout_p=0` consumes no
RNG; otherwise Python creates one scalar seed and CUDA reconstructs each mask
with a counter hash over batch/sequence, head, query, and route indices. Global
atomics can still make floating-point accumulation order nondeterministic, so
PyTorch deterministic-algorithm mode remains guarded.

## 6. Reference Mapping

The reference deliberately materializes the same computation:

```text
pairwise hard symbols
  -> pairwise mismatch rates
  -> local gates [B,H,T,T]
  -> raw suffix evidence [B,H,T,T]
  -> normalized square-root route scores [B,H,T,T]
  -> masked candidate-normalized probabilities [B,H,T,T]
  -> optional post-softmax attention dropout
  -> value carrier
```

Its custom forward computes exact hard output. Its custom backward rebuilds
the carrier under `torch.enable_grad()` and asks PyTorch for the VJP. This
makes it slower and more memory hungry than CUDA but easy to inspect and
differentiate.

## 7. Dense and Packed Equivalence

For every nonempty segment, packed execution must equal running the dense
operator on that segment alone. With dropout, equivalence also requires the
same seed and the segment's sequence index:

- route indices are local;
- suffix recurrences stop at the segment start;
- candidate counts use the local row;
- local position zero is null-only and cannot be returned as a value;
- values and gradients never cross a boundary.

Empty packed segments are valid and contribute no work.

## 8. Numerical Behavior

Hard output is bit exact across supported floating dtypes because only signs
and exact symbol equality determine routing.

Backward parity is numerical:

- FP32 is the reference accumulation precision;
- FP16 may underflow very small long-tail route gradients;
- loss scaling recovers part of that tail;
- BF16 retains wider exponent range but lower mantissa precision.

Use FP32 projection logits when retaining the widest dense support is more
important than projection bandwidth.

The product of many gates can underflow. This is not structural pruning: the
route is still visited. A lower `mismatch_scale` is the semantic control for
long near-match credit; kernel code must not silently clamp, threshold, or
drop the route.

## 9. Why Removed Mechanisms Stay Removed

| Removed mechanism | Reason |
| --- | --- |
| Per-mismatch random perturbation | Added RNG state and VJP variance without a held-out fitting advantage over its deterministic reduction. |
| Antithetic second branch | Doubled local estimator work; measured gain was insufficient. |
| Cubic perturbation shape | Only existed to parameterize random mismatch mass. |
| Exact hard-tier wrapper | Made long-suffix gradients decay sharply and duplicated hard-state computation in backward. |
| Separate numerical and derivative gates | Produced an incoherent surrogate and more kernel state. |
| Dynamic scale or mismatch scale | Introduced policy and extra failure modes without a robust D/T/W law. |
| Candidate top-k or suffix-index pruning | Removes discovery gradient and changes the training objective. |

## 10. Optimization Rules

Kernel work may be reduced by:

- online reductions;
- route/suffix tiling;
- packed-symbol operations and warp collectives;
- recompute versus bounded shared caching;
- gradient-mask specialization;
- coalesced FP32 accumulation;
- launch fusion when numerical parity is retained.

It may not be reduced by omitting valid routes. Any new cache must account for
bytes, occupancy, registers, spills, and the shapes where it is selected.

The current row kernel assigns one lane to one route candidate. Its suffix
loop has a static bound and does not branch on mismatch content. Assigning a
whole warp to one candidate therefore loses route parallelism and is not a
valid optimization by itself. Warp prefix scans become relevant only after a
complete diagonal-recurrence redesign; that research result and its staging
constraints are recorded in
[KERNEL_OPTIMIZATION.md](research/KERNEL_OPTIMIZATION.md).

## 11. Required Validation

A semantic or kernel change must pass:

- hard forward and latest-tie tests;
- dense PyTorch/CUDA output and all Q/K/value gradient-mask parity at
  `dropout_p=0` and fixed nonzero-dropout seeds;
- FP32, FP16, BF16, grouped heads, `D=32`, singleton, and non-contiguous
  cases;
- packed segment isolation, empty segments, and unequal lengths;
- full-graph `torch.compile` with the `aot_eager` backend, autocast,
  GradScaler, and RNG-preserving checkpoint integration;
- exhaustive small-problem hard-bit-flip alignment;
- multi-seed fitting, trained-checkpoint alignment, and route-discovery
  probes;
- contextual reset-RNN recall with identical post-reset residuals,
  complementary targets, and zero-route/current-value/residual-only
  ablations;
- latency, memory, registers, and spill reporting on the target GPU.
