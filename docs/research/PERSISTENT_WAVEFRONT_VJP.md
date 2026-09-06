# Persistent Block-Wavefront Soft-DP VJP

## Status

This document defines the research path for a persistent block-wavefront
implementation of the frozen dense RosaSoft surrogate. It does not change the
hard forward, candidate set, score equation, dropout semantics, or public
controls. The existing production row, tiled-streaming, and block-diagonal
schedules remain the comparison baseline until this path clears the complete
correctness and performance gate.

The eight-step finite-window prototype was completed on 2026-09-02 and remains
under `benchmarks/`; its best full schedule is still about twice as slow as
the finite production path at long sequence lengths on sm86. The later exact
unbounded diagonal-group replay is a separate estimator and schedule. Its
dense implementation is now linked into the production extension behind the
separately named `rosa_soft_unbounded` API. It does not alter the frozen
finite-window `rosa_soft` estimator or its defaults.

## 1. Coordinates And Ownership

Let `I` be a query tile, `J` a key/route tile, and `L` the square tile size.
Only the strict causal triangle is evaluated. Tiles are processed by

```text
wave = I + J.
```

Within one wave, every tile row `I` and every tile column `J` occurs at most
once. Consequently one CTA is the sole writer for each participating query
block and key/value block during that wave. The same property holds in reverse
wave order. A grid-wide barrier between waves therefore permits ordinary
coalesced load-add-store accumulation for Q, K, and V without cross-CTA
atomics.

The local suffix recurrence depends only on `(q-1,k-1)`. Its natural global
state index is

```text
delta = q - k,  1 <= delta < T.
```

Instead of storing separate top and left tile faces, the implementation keeps
one carry per global `delta`. Tile segments belonging to the same wave cover
disjoint `delta` ranges. A segment reads its incoming carry, scans its cells,
and writes its final carry back to the same index.

Ordinary CUDA shared memory is private to one CTA. A boundary produced by a
different CTA must pass through a small global-memory buffer, expected to stay
resident in L2, unless a later architecture-specific block-cluster schedule is
used. Correctness must never depend on the same physical CTA receiving the
next logical tile.

## 2. Frozen Soft DP

For one local Q/K comparison,

```text
m_n = popcount(q_n xor k_n)
x_n = exp(-mismatch_scale * m_n / D).
```

The exact finite-window suffix evidence is

```text
S_n = x_n + x_n x_(n-1) + ...  (at most W products).
```

It obeys

```text
P_n = product(x_(n-W) ... x_n), or 0 before W+1 gates exist
S_n = x_n * (1 + S_(n-1)) - P_n.
```

Because hard signs make every `m_n` an integer,

```text
P_n = exp(-mismatch_scale / D * sum(m_(n-W) ... m_n)).
```

The rolling mismatch sum must use integer state. Division by an outgoing gate
and subtraction of long floating-point log prefixes are prohibited.

## 3. CTA Affine Scan

Once `P_n` is known, every endpoint is an affine transform of the previous
score:

```text
f_n(s) = a_n * s + b_n
a_n = x_n
b_n = x_n - P_n.
```

Composition is associative:

```text
(a_r,b_r) o (a_l,b_l)
    = (a_r*a_l, a_r*b_l + b_r).
```

The first implementation uses `L=16` and divides a 256-thread CTA into sixteen
16-lane subwarps. A subwarp owns one local diagonal segment. It computes an
integer mismatch prefix and an affine prefix in four shuffle stages. Thirty-one
tile diagonals are covered in two batches. Inactive lanes use the affine
identity `(1,0)`.

This differs from the rejected warp-per-candidate schedule. Lanes emit many
neighboring route endpoints from one shared diagonal state; no warp is spent
recomputing the complete suffix of one candidate.

A prefix-history ring indexed by the absolute key coordinate supplies the
`W+1`-old cumulative mismatch count. `W<L` is supported exactly: all subwarp
lanes execute the same safe shuffle, and only the final `min(L,W+1)` lanes
publish history. This prevents both partial-mask shuffle deadlock and repeated
ring-slot writes. The reverse path retains a direct exact control for `W<L`;
the affine halo path is used for `W>=L`.

## 4. Persistent Forward Wave

A fixed resident cooperative grid executes all waves:

```text
for wave = 0 .. 2*N-2:
    process all causal tiles with I+J=wave
    grid.sync()
```

Each tile:

1. stages packed Q/K and the required value/gradient blocks;
2. reconstructs all local gates;
3. runs the segmented affine DP scan;
4. computes `grad_output @ sign(V)^T`;
5. updates the online `(maximum, normalizer, utility_numerator)` state for each
   query row;
6. publishes diagonal score and mismatch-prefix carries.

For fixed `I`, increasing wave visits `J` from left to right. This is exactly
the route order used by the current row-owned online softmax. The null item is
initialized before the first non-null tile. Counter-based dropout remains a
pure function of `(seed,batch,head,row,route)` and is independent of schedule.

## 5. Persistent Reverse Wave

After a grid barrier, waves run in descending order. Saved row statistics are
sufficient to reconstruct every route probability in any order. A tile
recomputes its finite-window scores from at most `W` Q/K halo gates; it does
not save quadratic forward activations.

The exact reverse state per global diagonal contains

```text
next_score_vjp
future_correction_sum
correction_vjp_ring[W+1].
```

The tile consumes successor state, scans each local diagonal backwards, emits
one `dL/dlog(x)` for every local gate, and publishes predecessor state. A gate
is finalized only by the tile containing that gate. This ownership rule is
required for atomics to be removed; a tile must not scatter adjoints into a
predecessor halo.

Within one reverse wave:

- query blocks are unique, so Q accumulation is conflict-free;
- key blocks are unique, so K accumulation is conflict-free;
- route/value blocks are unique, so V accumulation is conflict-free;
- diagonal carry ranges are disjoint.

The algebraic finite-window reverse is acceptable for the frozen `W=32` path
only after random, all-match, one-mismatch, and alternating-pattern parity.
Larger windows must also compare against the cancellation-resistant local
contribution oracle.

## 6. Synchronization Choices

The correctness prototype launches one kernel per wave. Kernel boundaries are
the global synchronization primitive and make boundary bugs easy to isolate.

The optimized implementation uses

```text
cooperative_groups::this_grid().sync()
```

with no more CTAs than can be resident concurrently. Every CTA participates in
every barrier, including waves where it owns no tile. The launch must query
cooperative-launch support and occupancy and retain an exact fallback.

An ordinary kernel in which resident CTAs spin on readiness flags is invalid:
waiting consumers can occupy all SMs while unscheduled producers never run.
A future barrier-free version must use a fixed persistent worker pool and a
ready queue, not arbitrary block-level polling.

## 7. State And Complexity

For fixed `W`, persistent state is linear in sequence length:

```text
row softmax state       O(B*H*T)
forward diagonal carry O(B*H*T)
mismatch prefix ring   O(B*H*T*W)
reverse carry ring     O(B*H*T*W)
Q/K/V gradients        O(input size)
```

No score, probability, dropout, or gate-adjoint matrix of size `T^2` is kept.
The dense candidate work remains `O(B*H*T^2)`. The number of global wave
barriers is `O(T/L)` for each direction.

## 8. Implementation And Promotion Plan

1. Implement a one-launch-per-wave finite-window score prototype.
2. Prove sampled score parity with the explicit prefix-product oracle.
3. Add null, utility, dropout, and online row-softmax state.
4. Add reverse diagonal carries and compare complete `dlog(g)` results.
5. Prove wave-level Q/K/V write-set disjointness and replace atomics with
   ordinary accumulation.
6. Replace host launch synchronization with a cooperative persistent kernel.
7. Add the subwarp affine scan and regular matrix contractions, then evaluate
   Tensor-Core promotion only where dimensions and accuracy justify it.
8. Validate `T`, `W`, `B*H`, dtype, gradient masks, dropout, random/all-match
   patterns, fitting behavior, Compute Sanitizer, and both sm75 and sm86.

Promotion requires estimator parity, unchanged exact hard output, no quadratic
workspace, and a stable end-to-end gain over both production streaming and
block-diagonal schedules. A score-only or isolated suffix win is insufficient.

## 9. Execution Results

All eight planned steps are implemented:

1. Multi-launch forward waves compute exact finite-window scores.
2. Sequential and affine scores match the explicit prefix-product oracle.
3. Null, candidate prior, dropout, utility, and online softmax are fused.
4. Reverse diagonal carries match the standalone `dlog(g)` oracle.
5. Q/K/V ownership is atomics-free within a wave, including serialized GQA
   head groups.
6. Cooperative persistent forward and reverse schedules use occupancy-bounded
   resident grids and `grid.sync()`.
7. A 16-lane affine scan replaces per-cell suffix scans for `W>=16`; utility
   uses a shared-memory blocked matrix contraction.
8. Boundary, dtype, dropout, structured-pattern, grouped-head, fitting,
   sanitizer, memory, and performance gates were run on GPU 0.

The complete research test file has 475 passing cases. It covers `D=1/8/32`,
all seven gradient masks, `W=1/15/16/17/31/32/33/64/65`, random and structured
symbols, FP32/FP16/BF16, dropout, irregular `B=2,H=6,Hv=3,Dv=65`, and both
schedules. The hard-route fitting control reaches exactly zero MSE at step 7
for both schedules. Compute Sanitizer `memcheck`, `synccheck`, and `racecheck`
report zero errors or hazards for `W=1/32`, dropout, and both schedules.

The remaining validation gap is sm75. This run was intentionally kept on GPU
0 (RTX 3070, sm86); the occupied RTX 2080 Ti was not used.

## 10. Performance Findings

Representative measurements use PyTorch 2.11.0+cu128, CUDA 12.8, RTX 3070,
`B=1,H=4,Hv=2,D=8,Dv=64,W=32`, FP16, and all gradients:

| T | production | multi-launch | persistent |
|---:|---:|---:|---:|
| 512 | 0.742 ms | 2.927 ms (3.94x) | 3.593 ms (4.84x) |
| 1024 | 2.060 ms | 5.451 ms (2.65x) | 6.433 ms (3.12x) |
| 2048 | 6.819 ms | 14.491 ms (2.13x) | 14.937 ms (2.19x) |
| 4096 | 20.317 ms | 40.995 ms (2.02x) | 40.585 ms (2.00x) |

The isolated score scan validates the requested affine optimization. At
`T=512/1024`, affine takes 0.802x/0.827x the sequential scan time, a 17-20%
gain. The tiled utility contraction takes 0.226x/0.241x/0.320x the scalar
multi-launch stats time at `T=512/1024/2048`; it is retained. Skipping utility
when only V is requested reduces the 4096-token full VJP from about 32.0 ms to
24.5 ms. Avoiding unused reverse-DP state in that path also reduces measured
operator workspace from 4.44 MiB to 2.38 MiB.

Persistent stats are consistently faster than per-wave host launches, but the
full persistent reverse is slower below roughly 4096 tokens. Its fixed
resident grid underfills short waves and every GQA group pays all grid
barriers. It is retained as a correctness and scaling control, not a production
candidate. The multi-launch path is the current best wavefront implementation.

No Tensor Core path is promoted here. The affine suffix recurrence is not a
matrix multiply, while the regular utility contraction is already several
times faster than its scalar control. Tensor Core promotion should be evaluated
only for Q/K/V contractions in an end-to-end kernel and must beat the current
production block-diagonal schedule, not an isolated microkernel.

## 11. Unbounded Diagonal-Group Replay

The finite block-wavefront path above is not the only useful checkpoint
decomposition. An exact **unbounded** soft DP can avoid all tile-edge state by
using its invariant global diagonal

```text
delta = query_position - key_position,  1 <= delta < T.
```

For one diagonal, with `x` denoting the soft match gate,

```text
S[q,delta] = x[q,delta] * (1 + S[q-1,delta]).
```

There is no finite-window correction and no dependency between different
diagonals. A warp owns one complete diagonal, advances in 32-position chunks,
and carries one affine transform in registers. The implementation processes a
fixed group of `G` diagonals at a time and reuses a `[B,H,T,G]` score tensor for
the next group. `G` is a replay granularity, not a suffix limit: every one of
the `T-1` causal diagonals is eventually visited.

This is the useful checkpoint analogy. The complete triangular DP is never
saved. Instead, a diagonal group is reconstructed whenever a later phase needs
it. The only state retained across groups is the per-row softmax boundary

```text
(maximum, normalizer, expected_utility), shape [B,H,T,3].
```

## 12. Forward And Reverse Passes

The first pass runs from the largest diagonal group to the smallest:

1. Scan the group's exact unbounded suffix scores.
2. Form `utility = grad_output @ sign(value)^T` for the group's routes.
3. Merge null and non-null logits into the online row softmax statistics.

The reverse pass replays each group:

1. Recompute suffix scores into the score workspace.
2. Recompute utility into a second workspace.
3. If value gradients are requested, reconstruct route probabilities from the
   score tile plus the saved row maximum and normalizer, then compute
   `dV = P^T @ grad_output`. This runs before the score workspace is mutated.
4. Form each raw-score VJP on demand from probability, utility, and the saved
   expected utility, then reverse-scan each diagonal using

   ```text
   dS[q] = dR[q] + x[q+1] * dS[q+1]
   dlog(x[q]) = S[q] * dS[q].
   ```

5. Overwrite the score workspace with `dlog(x)` and contract it into Q/K.

The row boundary state plays the same role as FlashAttention's online softmax
statistics: it is sufficient to reconstruct an exact probability tile during
backward. Materializing probabilities is deliberately avoided. An experiment
that overwrote the utility tile with probability inside the diagonal reverse
kernel increased the T=2048 all-gradient latency from about `5.07 ms` to
`6.94 ms`; the strided extra store increased reverse-kernel time from roughly
`1.14 ms` to `3.05 ms` per operation under the profiler.

Utility must be evaluated twice. The first evaluation is needed to finish the
row expectation; the second occurs only after that expectation is known and
can therefore form the exact softmax VJP. Retaining utility between those
passes would restore quadratic storage.

## 13. Memory And Work

For Q/K gradients, peak scratch is

```text
score workspace                  B*H*T*G FP32
utility workspace                B*H*T*G FP32
row softmax state                3*B*H*T FP32
```

The V-only path needs one group workspace. Returned Q/K/V gradients are
separate output storage. Thus scratch is `O(B*H*T*G)` and is linear in `T` for
a fixed `G`; it is not described as linear if `G` itself grows with `T`.

Dense work remains quadratic in sequence length. Unlike setting the existing
finite-window production kernel to `W=T`, the diagonal recurrence performs
constant work per candidate rather than rescanning up to T suffix elements per
candidate.

## 14. Regular Matrix Contractions

The suffix recurrence itself stays on CUDA cores. Two stateless contractions
use matrix-oriented schedules on SM80+:

- Utility is traversed as ordinary `16 x 16 (row, route)` tiles intersecting
  the current diagonal band. Four warps share one `grad_output` tile and each
  computes one neighboring route tile, then scatter the selected results back
  to `(row, delta)` storage. No enclosing off-band matrix is evaluated.
- The value VJP owns `16` routes per CTA. Four warps reconstruct a `16 x 16`
  probability tile from saved row statistics and replayed scores, then apply
  `P^T @ grad_output` for 64 value channels at a time. It accumulates all row
  tiles and all GQA heads before one conflict-free writeback. It neither reads
  the utility workspace nor stores a probability matrix.

FP32 accumulation is retained. SM75 or small-value-dimension cases use the
CUDA-core controls. A Tensor-Core Q/K contraction was also implemented and
removed: D is only 1--32, and two separate Q/K launches did not give a stable
end-to-end win over the fused CUDA-core kernel.

## 15. Load-Balancing Experiment

Pairing a diagonal of length `L` with one of length `C-L` gives equal total
work and is mathematically valid when the affine carry is reset at the join.
It was tested by pairing the shortest and longest diagonals inside every group.

It is not retained in the current multi-CTA schedule. Independent CTAs are
already dynamically assigned by the GPU, adjacent diagonals inside one CTA
have nearly equal lengths, and pairing halves the number of runnable warps.
For a score group at `T=8192,H=1`, the long-diagonal case regressed from about
`0.196 ms` to `0.328 ms`; the complete `T=2048,G=256` VJP changed from about
`20.03 ms` to `20.72 ms` before the later matrix optimizations. Pairing remains
relevant to a future fixed-worker persistent queue, where worker count rather
than global CTA scheduling creates a real tail bubble.

## 16. Unbounded Replay Results

Measurements use GPU 0, an RTX 3070 (sm86), PyTorch 2.11.0+cu128,
`B=1,H=4,Hv=2,D=8,Dv=64`, FP16, no dropout, and all gradients. The production
`W=T` column evaluates the same unbounded estimator through the old general
finite-window implementation. The production `W=32` column is a different,
finite estimator and is shown only as the practical latency reference.

| T | production W=32 | production W=T | replay G=256 | replay G=1024 | G=256 peak |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.723 ms | 3.138 ms | 0.491 ms | 0.401 ms | 4.40 MiB |
| 1024 | 1.933 ms | 14.803 ms | 1.240 ms | 1.028 ms | 8.80 MiB |
| 2048 | 6.596 ms | 87.453 ms | 3.866 ms | 3.178 ms | 17.59 MiB |
| 4096 | 19.342 ms | 590.892 ms | 14.416 ms | 11.798 ms | 35.19 MiB |

At T=2048 the G=256 replay path is about 22.6 times faster than the old
unbounded control; at T=4096 it is about 41.0 times faster. G=1024 is faster
but raises T=4096 scratch from about 35.2 MiB to 131.2 MiB. The production API
therefore keeps private G=256 as the conservative time/memory point rather
than exposing another tuning control.

Across these profile cases, maximum absolute Q/K error against the unbounded
production oracle is below `4.2e-7`. Maximum value-gradient error is below
`3.5e-4`, caused by the gated TF32 value contraction. The focused replay and
scheduling suites have 743 passing cases, including all gradient masks,
FP32/FP16/BF16, dropout,
GQA, D=1/8/32, irregular tile boundaries, structured collisions, and a hard
route fitting case that reaches exact zero loss. Compute Sanitizer memcheck,
synccheck, and racecheck report no errors or hazards on sm86. The complete
extension also compiles for sm75 without hard-DP spills; sm75 runtime
performance and sanitizer coverage remain open because that device was not
available. The sm86 cubin reports 23 registers for
score replay, 40 for diagonal reverse, 48 for TF32 utility, 52 for TF32 dV,
and 38 for Q/K accumulation, with no stack or local-memory spills.

The dense replay implementation now lives in
`rosa_soft/csrc/cuda/rosa_soft_unbounded_kernels.cu`. The benchmark wrapper
still exposes group size and intermediate state for experiments; the public
operator does not. Packed varlen has the same exact unbounded semantics but
currently uses the general segment-local fallback rather than the optimized
group replay.

## 17. Blocked Workspace

The original replay workspace was logically `[B,H,T,G]`. A diagonal-owned
warp consequently wrote lane-adjacent rows with a physical stride of `G`,
while row-oriented utility, stats, dV, and symbol consumers wanted neighboring
rows and diagonals together. The retained layout is

```text
[BH, ceil(G/4), ceil(T/32), 32 rows, 4 diagonals].
```

One CTA's four diagonal warps now write a `32 x 4` micro-tile. Consumers use
the same address function, so no transpose tensor or extra kernel exists.
Against the previous row-major replay at G=256, representative T=512/1024/
2048/4096 latency changed from about `0.568/1.534/5.145/18.738 ms` to
`0.491/1.240/3.866/14.416 ms`. The largest case improves by about 23% with
unchanged asymptotic memory and no register spill.

Moving probability and raw-score-VJP pointwise work into the Tensor-Core
utility writeback was also tested. It raised utility register pressure and
made T=2048/4096 roughly 2--4% slower, so it was removed. Utility remains a
pure contraction and the diagonal reverse remains a pure recurrence.

## 18. Persistent And Static Scheduling

The diagonal scheduling study isolates five one-launch schedules: ordinary
hardware CTA queuing, serial longest+shortest pairing, fixed strided workers,
an atomic persistent work queue, and fixed workers over balanced pairs. It
covers T=1024/4096/8192, H=1/4, and random/equal/periodic symbols. All 45
correctness cases are exact.

Across 18 performance rows, the best atomic persistent configuration
(`8 x SM` blocks) has median ratio `1.026` to the ordinary hardware queue and
wins only 2 rows. Serial pairing has median `1.030` and wins 6 mostly short
rows. Fixed balanced workers have median `1.081`. At T=8192 the ordinary queue
wins five of six rows; the one persistent win is only about 2.4%.

This is not attributed merely to kernel launch overhead:

1. Every schedule in this microbenchmark is already one kernel launch.
2. CUDA's block scheduler is itself a dynamic queue without a global atomic
   per diagonal.
3. Adjacent diagonals inside a replay group differ in length by at most G;
   serial long/short pairing halves immediately runnable warps.
4. Low fixed-worker counts underfill the GPU, while high counts converge to
   the hardware schedule plus loop or queue overhead.
5. A complete G=256, T=4096 VJP issues 134 kernels, but profiling measures
   about 16.6 ms of GPU work versus 1.54 ms of CPU submission. The CPU keeps
   the device queue populated.
6. Capturing the complete operation in a CUDA Graph changes `13.63 ms` to
   `13.50 ms`, an empirical launch-removal upper bound of about 1% on this
   shape.

A cooperative persistent full VJP cannot simply delete all remaining cost.
Score, utility, row statistics, dV, reverse DP, and symbol accumulation have
real producer/consumer barriers. Replacing launches with `grid.sync()` keeps
those barriers, constrains the whole kernel to the occupancy of its heaviest
phase, and forces short phases to use the same resident grid. The already
implemented finite cooperative control exhibits this underfill.

## 19. Consumer-Side Fusion

The T=4096 all-gradient profile attributes approximately 34.8% of GPU time to
the two utility contractions, 18.0% to dV, 17.7% to symbol accumulation,
10.5% to row statistics, 9.5% to score replay, and 9.4% to reverse DP. This
explains why a score-only scheduling win has little end-to-end leverage.

Score and utility are independent before row-stat consumption, so an explicit
two-stream fork/join was tested to overlap CUDA-core/EXP score work with the
Tensor-Core utility contraction. It regressed T=1024 from about 1.24 ms to
1.88 ms and T=4096 from 14.42 ms to 15.75 ms. Per-group event dependencies,
L2/bandwidth contention, and utility occupancy cost more than the hidden score
time. The branch was removed. A heterogeneous CTA kernel would remove event
traffic but not the resource contention or final phase barrier; it is not
carried in production without contrary end-to-end evidence.

## 20. Exact Unlimited Hard DP

The same diagonal geometry is useful for hard ROSA. One warp walks one full
Q/K diagonal in 32-cell chunks. A prefix maximum finds the nearest preceding
mismatch, so an equal endpoint's exact unlimited suffix length is its distance
from that mismatch. A 64-bit atomic priority

```text
(match_length << 32) | successor_route
```

reduces all diagonals into one winner per query row. This is not the historical
W-limited prototype: no hard window exists. Dense and packed-varlen paths use
only an `O(BHT)` winner array and `O(BHT^2)` work. Inputs below 512 tokens keep
the lower-overhead direct scan; larger inputs use DP.

On sm86 with B=1,H=4,D=8,Hv=2,Dv=64, FP16, all outputs are bit-exact with the
compact SAM and the former direct CUDA scan. At T=4096, production DP takes
about 0.268 ms on random symbols, 0.345 ms on periodic-64 symbols, and 0.673 ms
on all-equal symbols. The old direct scan measured about 0.188/64.88/108.28 ms
on those cases. Random input pays about 0.08 ms; repetitive inputs improve by
roughly 188x and 161x. Dense, empty-segment varlen, null, successor, and latest
tie semantics are covered. Memcheck, racecheck, and synccheck all report zero
errors or hazards.
