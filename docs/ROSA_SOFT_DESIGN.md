# RosaSoft Design

This document maps the maintained implementation. Mathematical definitions
are in [CONCEPT.md](CONCEPT.md), and the integration surface is in
[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md).

## 1. Module Boundaries

| Module | Responsibility |
| --- | --- |
| `soft_contract.py` | Shared public validation and scalar defaults. |
| `soft_reference.py` | Materialized PyTorch semantic oracle. |
| `soft.py` | CUDA custom-autograd wrapper. |
| `sam.py` | Synchronous SAM validation API and hard value gather. |
| `initialization.py` | Optional model projection initialization. |
| `csrc/export.cpp` | Dispatcher schemas and registrations. |
| `csrc/rosa_soft.cpp` | Native checks, allocations, and schedule dispatch. |
| `csrc/rosa_sam_core.h` | Pure C++ online suffix automaton. |
| `csrc/rosa_sam.cpp` | PyTorch custom-class binding. |
| `csrc/cuda/rosa_soft_kernels.cu` | Hard scan, row VJP, and packed VJP. |
| `csrc/cuda/rosa_soft_vjp_common.cuh` | Shared VJP math and finalization. |
| `csrc/cuda/rosa_soft_streaming_kernels.cu` | Long fixed-length tiled VJP. |
| `csrc/cuda/rosa_soft_block_diagonal_kernels.cu` | Gated SM80+ block VJP. |

`testing.py` and `diagnostics.py` materialize development information; neither
is imported by the hot path. Research and archived Runtime sources are not in
the extension build graph.

## 2. Hard Forward

Dense CUDA flow:

```text
Q/K logits [B,T,H,D]
  -> sign-pack int32 symbols [B,H,T]
  -> scan every causal key end and its exact full suffix
  -> choose globally longest, then latest
V logits [B,T,Hv,Dv]
  -> gather successor position and hard-sign
  -> hard output [B,T,H,Dv]
```

Matching has no configured horizon. The hard dispatcher schema deliberately
has no `max_suffix_length`; that scalar is sent only to backward. The packed
variable-length kernel applies the same scan independently inside every
segment.

The forward also returns private packed Q/K symbols to autograd. Reusing them
avoids quantizing Q/K again in backward and costs only linear state. A no-grad
call discards them after hard output.

The hard CUDA scan is simple and exact but quadratic. It is a training and
validation implementation, not the package's answer to long-context inference.

## 3. Autograd State

Dense autograd saves:

```text
Q, K, V, packed_Q, packed_K, dropout_seed
max_suffix_length, scale, dropout_p, mismatch_scale
```

Packed autograd additionally saves `cu_seqlens`. The dropout seed is empty
when `dropout_p=0` and scalar otherwise. Scores, probabilities, masks, route
winners, and suffix adjoints are recomputed instead of saved.

`ctx.needs_input_grad` becomes a Q/K/V bit mask. A disabled input may use an
empty internal gradient output, but it must not change probabilities or credit
used by another enabled input.

## 4. Backward Flow

For each batch, head, query row, and causal route, CUDA:

1. Reconstructs normalized Hamming mismatch from packed signs.
2. Evaluates the exponential local match gate.
3. Computes the complete finite-horizon suffix evidence `S`.
4. Applies `U(S)=(sqrt(2)+1)(sqrt(1+S)-1)`.
5. Merges null and non-null logits into online softmax statistics.
6. Reconstructs post-softmax dropout from the scalar seed and route indices.
7. Accumulates probability-weighted value credit.
8. Applies the softmax, utility, suffix, gate, and softsign adjoints for Q/K.

The utility derivative is

```text
U'(S) = (sqrt(2) + 1) / (2 * sqrt(1 + S)).
```

The implementation may make multiple complete route sweeps. Recompute avoids
quadratic persistent state without changing candidate support.

## 5. CUDA Schedules

Fixed-length backward has three internal schedules:

| Condition | Schedule |
| --- | --- |
| SM80+, `T>=4096`, `W=32`, `Dv=64` | Block-diagonal TF32 path. |
| SM75 `T>=4096`; SM80+ Q/K `T>=512`; SM80+ V-only `T>=2048` | 32-row tiled streaming. |
| Other fixed shapes | Row-owned cache/recompute. |

Packed input uses the row-owned variable-length implementation. Unsupported
block shapes fall through to streaming, and unsupported streaming shapes fall
through to the generic row kernel. Selection is private and cannot alter the
equations.

### Tiled streaming

One CTA owns 32 query rows, one warp owns a row, and lanes own route candidates.
The first complete sweep computes row softmax statistics. The second sweep
recomputes scores and fuses route utility, Q/K credit, and V accumulation.
Suffix horizons advance in exact chunks; no score matrix is stored.

### Block diagonal

The gated block schedule owns adjacent query rows and routes. It computes the
same finite-window recurrence on tile diagonals and uses TF32 WMMA only for
matrix contractions. Scores, normalizers, dropout, recurrence state, and
accumulators remain FP32. Its documented parity tolerance accounts for TF32
rounding; candidate support remains exact.

### Row owned

Short rows select between exact score recomputation and a bounded shared cache
when the whole layout fits. Cooperative value and route-utility tiles, K
aggregation, and packed tail caches are execution details. Every older route
is still recomputed when it is outside a cache.

All schedules use `rosa_soft_vjp_common.cuh` for constants, score transforms,
online softmax primitives, counter dropout, and the final softsign conversion.
CUDA translation units include the header, never another `.cu` file.

## 6. Layout And Precision

Public dense tensors are token-major `[B,T,H,D]`. Private packed signs are
head-major `[B,H,T]`, and packed-varlen signs are `[H,N]`, so neighboring route
lanes read neighboring symbols.

Dense K gradients may accumulate in FP32 `[B,H,D,T]` for coalesced atomics,
then transpose to public layout. Other enabled gradients also accumulate in
FP32 and are cast to input dtype after finalization.

Global atomics can change summation order across launches. CUDA VJPs are
numerically reproducible within tolerance, not bitwise deterministic. The
operator reports this through PyTorch deterministic-algorithm checks.

## 7. Dense And Packed Equivalence

For each nonempty packed segment, execution must equal running the dense
operator on that segment alone:

- route positions are local;
- suffix recurrence stops at segment start;
- candidate count and null competition use the local row;
- dropout counters include semantic sequence and local route indices;
- output and gradients never cross a boundary.

Empty segments perform no work and remain valid.

## 8. SAM Validation Path

The SAM is independent of CUDA hard scanning:

```text
Python logits
  -> int32 sign symbols
  -> synchronous CPU staging
  -> one RosaSuffixAutomaton per sequence/head
  -> local matched key end
  -> optional Python successor-value gather
```

The pure C++ automaton stores a vector of states and a vector of sparse linked
edges. `match_then_append` makes the causal order explicit. Standard SAM
extension creates at most one new state and one clone per appended symbol;
clones copy their sparse outgoing edges. Latest occurrence timestamps are
updated directly along suffix links for clarity.

This path deliberately excludes value history, async workers, locking, page
management, persistent identifiers, and serialization. Its expected routes
come from the independent O(T^2) DP in `tests/test_sam.py`, not from CUDA or a
second automaton.

## 9. Optimization Boundary

Semantics-preserving work includes:

- online reductions and exact tiling;
- bounded cache/recompute tradeoffs;
- packed symbols and warp collectives;
- gradient-mask specialization;
- coalesced accumulation and launch fusion;
- simplifying validation code without changing causal order.

The following change the objective and are excluded from production:

- top-k, sampling, or approximate candidate indexes;
- hard-neighborhood-only gradients;
- dynamic temperature/mismatch schedules;
- score thresholds or content-dependent backward termination;
- stochastic mismatch perturbation branches;
- hard-window truncation.

## 10. Validation

Changes must retain:

- SAM/DP equality for exhaustive, random, periodic, tie, chunked, and varlen
  cases;
- bit-exact hard CUDA/reference/SAM output;
- hard independence from all surrogate controls;
- all seven nonempty Q/K/V gradient masks;
- fixed-seed dropout parity;
- FP32, FP16, BF16, grouped heads, D32, non-contiguous and singleton inputs;
- packed empty segments and invalid-offset handling;
- compile, autocast, GradScaler, and checkpoint integration.

Kernel promotion additionally needs target-device latency, peak memory,
register, spill, racecheck, and synccheck evidence.
