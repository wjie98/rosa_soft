# Persistent Macro-Tile RosaSoft VJP

## Status And Scope

This document records the second-generation wavefront research path and the
slab-replay design that was later promoted as the fixed-shape
`rosa_soft_unbounded` CUDA backward. The macro-wavefront schedules remain
research-only; the production path is the exact diagonal-slab decomposition
described in section 9. Packed varlen currently retains its exact segment-local
fallback.

The design has four non-negotiable properties:

1. Every causal non-null route is evaluated. There is no candidate pruning.
2. The local suffix recurrence is exact and unbounded in sequence length.
3. The hard forward remains the existing exact, unlimited ROSA operator.
4. Only `tile_size` is a schedule parameter. Sequence length is an input and
   the persistent grid size is selected from CUDA occupancy.

The exploratory implementations remain under `benchmarks/`. The promoted
implementation is `rosa_soft/csrc/cuda/rosa_soft_unbounded_kernels.cu`; it
passes the independent PyTorch oracle and exact memory-bounded replay tests.
A dense score tensor is returned only by a benchmark compatibility interface
used to validate the recurrence.

## 1. Exact Problem

For batch/head pair `z`, query position `q`, and key end `k < q`, define

```text
g[z,q,k] = exp(-mismatch_scale * popcount(Q[z,q] xor K[z,k]) / D)
S[z,q,k] = g[z,q,k] * (1 + S[z,q-1,k-1])
```

with `S=0` outside the matrix. The corresponding route index is `a=k+1`;
route zero is null. This is the unlimited form of the dense suffix evidence.
It visits the strict lower causal triangle and has `O(BHT^2)` arithmetic.

The transformed route score and logit are

```text
R(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
logit(null) = scale * 0.5
logit(q,k) = scale * R(S[q,k]) - log(q)
```

because row `q` has exactly `q` non-null candidates. Online row state is the
FlashAttention-style triple

```text
(row_max, row_normalizer, row_utility_numerator).
```

Dropout is applied to post-softmax route weights during backward and is
reconstructed from the existing counter-based RNG. It never changes the
score recurrence, row maximum, or row normalizer.

## 2. Coordinates

Use a fixed microtile size

```text
P = 32.
```

One square macro tile has

```text
L = n * P,  n >= 1.
```

`L` is called `tile_size` at the Python boundary. Supported compiled variants
are a small set such as `{32, 64, 96, 128}`. `tile_size=0` asks the dispatcher
to choose a variant. Explicit values are useful only for reproducible
research and must be positive multiples of 32.

For `N=ceil_div(T,L)`, macro tile `(I,J)` covers

```text
q in [I*L, min((I+1)*L,T))
k in [J*L, min((J+1)*L,T)).
```

Only cells with `k < q` are valid. A tile is relevant when `J <= I`. The DP
predecessor of every cell is `(q-1,k-1)`, so macro tiles are scheduled by

```text
wave = I + J.
```

Tiles in one wave have distinct row and column blocks. That fact is useful
for conflict-free row statistics and gradient ownership, but it is not by
itself a synchronization mechanism.

## 3. Parameters: T, G, And L

There are deliberately no independent `G` or replay-group controls.

```text
T       runtime sequence length
L       only tuning dimension
G(L)    SM_count * max_resident_CTA(kernel_variant(L))
```

`G` is computed with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` after the
kernel variant and dynamic shared-memory size are known. Persistent variants
launch exactly `G` CTAs and use grid-stride task ownership. It is never a user
knob. Cooperative variants check `cooperativeLaunch`; schedules without a
grid-wide barrier use an ordinary occupancy-bounded launch.

Removing a separately tuned replay group is important. Replay granularity is
derived internally from `L`, the memory budget, and the number of causal
macro tiles. It must not change equations or candidate coverage.

## 4. Macro-Tile State

One CTA owns one logical macro tile at a time. It does not hold an `L x L`
FP32 score matrix in shared memory. Instead, it advances through `P x P`
microtiles and retains:

```text
packed Q and K slices                    O(L)
row online-softmax state                 O(L)
incoming/outgoing diagonal DP carries    O(L)
two P x P work buffers                   O(P^2)
local replay checkpoints                 O(L^2/P)
```

The first implementation may stage only the current Q/K microtile if staging
all `2L` symbols lowers occupancy. This is a compile-time variant, not a new
public parameter.

For an `L x L` tile, the exact forward DP boundary has `2L-1` diagonal carry
values. Row statistics have `3L` values. Arrays are stored structure-of-arrays
and padded from `2L-1` to `2L`, so a warp can transfer 32 consecutive FP32
values in one 128-byte transaction.

Conceptually:

```text
registers, diagonal order
       d0 d1 d2 ...
          | fixed permutation
          v
shared edge_by_delta[2L]       contiguous by global delta
shared row_max[L]
shared row_sum[L]
shared row_utility[L]
          | warp-striped stores
          v
global checkpoint[batch_head, tile, field, contiguous_index]
```

Shared transpose arrays use a padded minor dimension such as `[warps][33]`
where it removes bank conflicts. All global boundary loads and stores must be
full-warp contiguous except at sequence edges.

## 5. Forward Schedule

### 5.1 Correctness baseline

The baseline launches one kernel per macro wave. Every CTA processes one tile
and a kernel boundary publishes all dependencies. A debug entry point may
write every raw score so the tile result can be compared cell-for-cell with
the unlimited PyTorch recurrence.

This baseline establishes:

- causal coordinates and null-route offset;
- partial tiles;
- incoming diagonal carries;
- local microtile traversal;
- row update order;
- bit packing and mismatch gates.

It is not a performance result.

### 5.2 Cooperative persistent baseline

A fixed occupancy-bounded cooperative grid executes all macro waves:

```text
for wave in 0 .. 2*N-2:
    for task in tiles_on_wave strided by G:
        process(task)
    grid.sync()
```

This removes host launch overhead while retaining an easy-to-audit global
barrier. Short waves underfill the grid, so it is only the first persistent
schedule.

### 5.3 Ready-queue schedule

The target schedule replaces per-wave barriers with a bounded persistent work
queue. Each tile has a dependency count. Producers publish boundary records
with release semantics, decrement successor counters, and enqueue a successor
when its count reaches zero. Consumers acquire the task before reading edges.

Only the occupancy-bounded `G` workers may spin. An arbitrary non-cooperative
grid is forbidden because waiting blocks can prevent producers from becoming
resident. Forward completion and the forward-to-reverse phase change retain a
grid-wide barrier.

The implemented queue is an exact control, not the preferred schedule. It
removes short-wave barriers, but global head/tail traffic and idle-worker
polling make it slower than multi-launch at the tested sizes.

### 5.4 Persistent row streams

The ready queue pays for a global head, reservation tail, publication flag,
two dependency decrements, and successor insertion around every tile. Merely
holding several claimed-but-unpublished queue positions hides part of the
publication latency, but does not remove this traffic. On sm75 at `T=8192`,
eight pending positions reduced score time from `69.18 ms` to `65.25 ms`,
still `6.2x` slower than multi-launch.

The retained local-signal experiment removes the central queue. One logical
task owns a macro row and advances its columns in order:

```text
forward/stats: (I,0), (I,1), ... (I,I)
reverse:       (I,I), (I,I-1), ... (I,0)
```

The same-row dependency is therefore local state. Forward and statistics poll
only whether row `I-1` has reached column `J`; reverse polls whether row `I+1`
has reached `J`. A row advances monotonically, so one progress integer per row
is sufficient and per-tile completion flags are unnecessary. Synchronization
state is `B * H * ceil(T/L)` integers instead of a triangular tile array.

A fixed occupancy-bounded CTA keeps four logical rows and executes any row
whose one external predecessor is ready. Logical rows use static grid-stride
ownership, so there is no assignment counter and a row cannot be duplicated.
Progress publication uses device-scope release/acquire atomics; relaxed polling
is followed by one acquire per warp before boundary data is consumed. Reading
a later progress value is valid because the requested boundary was already
published and all row progress is monotone.

Four slots are an internal constant, not a public schedule parameter. The
1/2/4/8-slot score ablation differed by about 1% even at 16K, so exposing the
choice would add tuning surface without a stable payoff.

Signal placement was also ablated directly on sm75. Three otherwise identical
row-stream kernels used (1) the compact integer array above, (2) one integer
per 32-byte L2 sector, or (3) the sign bit of each tile checkpoint's unused
last word as a forward/reverse phase token. The embedded variant is valid: a
tile record has `2L` words but only `2L-1` reachable deltas, release stores
publish the token after the payload, and an acquire load precedes payload
consumption. It nevertheless distributes polling across the much larger
checkpoint working set and complicates phase ownership.

At `B=1,H=1,T=8192,L=32`, compact/padded/embedded score times were
`6.292/6.407/6.453 ms`; full VJP times were
`273.163/274.773/275.709 ms`. The H=4 results were effectively tied. Nsight
Compute reported more than 99.98% L2 lookup hits for every layout and no RMW
atomic sectors: device-scope `atomic_ref` loads/stores compile to ordered
memory operations rather than atomic read-modify-write instructions here.
Thus tight packing causes neither a correctness dependency nor a measurable
false-sharing bottleneck on this workload. The retained implementation keeps
the compact `B * H * ceil(T/L)` array and removes both experimental branches.
The complete ablation is recorded in
`validation/row_signal_layout_ablation_sm75.json`.

For FP16 `B=1,H=4,Hv=2,D=8,Dv=64,L=32`, row-stream score time was
`3.36/8.39/33.07 ms` at `T=4096/8192/16384`, versus multi-launch
`4.08/10.61/74.50 ms`. After progress compression, complete VJP took
`163.35/486.06 ms` at 4K/8K versus multi-launch `173.66/526.76 ms`.
Replacing tile flags reduced the corresponding operator peaks from
`11.438/39.127 MiB` to `11.314/38.629 MiB`; latency was effectively unchanged.
The schedule is therefore a useful exact
control and a validated local-synchronization technique, but it does not beat
the synchronization-free folded or slabbed paths end to end.

### 5.5 Static short-long folding

The suffix recurrence is a set of independent global diagonals. For diagonal
`d=q-k`, its length is `T-d`. Pair it with diagonal `T-d`, whose length is `d`:

```text
long diagonal d                 length T-d
short diagonal T-d             length d
                                           --------
packed warp itinerary          length T
```

Except for the single self-paired middle diagonal when `T` is even, every warp
therefore receives exactly `T` cells. A warp scans both pieces with one
segmented affine scan and resets carry at the join. There is no queue, global
barrier, or recursive suffix loop.

The complete folded VJP has four schedules:

```text
1. checkpoint prepass

   diagonal d:       [================ long ================]
   diagonal T-d:     [== short ==]
   packed warp:      [================][==]  total T
                         | write exact macro-edge checkpoints

2. tile-local online statistics

   every causal macro tile independently replays from its checkpoint and
   writes L local (max, normalizer, utility-numerator) triples

3. deterministic row reduction

   each query row merges its tile-local triples in increasing key-tile order,
   starting from the null-route state

4. reverse

   Q/K: use the same length-T global-diagonal pairs; replay one L-cell
        checkpoint segment at a time and retain reverse carry in registers.

   V:   every causal macro tile is an independent grid-stride task and
        accumulates its exact contribution with FP32 atomics.
```

The checkpoint prepass makes all statistics and V tiles independent. It adds
a cheap packed-symbol recurrence pass, but removes `2N-1` imbalanced launch or
barrier phases. Q/K folded reverse uses FP32 atomics because different
diagonals share query and key positions. V retains the existing 32-query tile
reduction before each atomic update; a tested whole-column shared accumulator
was removed because its initialization and occupancy loss outweighed fewer
atomics.

This task granularity matters as much as equal-work pairing. For
`B*H=4,T=2048,L=32` on sm75, occupancy permits `G=816` CTAs. Pairing whole
macro rows or columns creates only 128 tasks, so 688 CTAs immediately exit.
The implemented statistics and V schedules instead expose 8,320 causal-tile
tasks, while each paired-diagonal schedule exposes 4,096 tasks. The fixed grid
can therefore remain occupied and consume multiple grid-stride tasks per CTA.

For `Dv >= 32`, statistics and Q/K reverse use the same 8/16-lane subgroup
utility reduction as the maintained operator. A subgroup reads contiguous
value features and cooperatively evaluates one route dot product; multiple
subgroups evaluate adjacent routes. This replaces one strided, serial `Dv`
loop per candidate without adding a precision mode, public control, or global
workspace. Smaller value dimensions retain the direct scalar loop.

`tile_size=0` selects `L=32` for this schedule. Explicit `{64,96,128}` variants
remain correctness and tuning controls, but every current sm75 performance
case favored 32.

### 5.6 Measured result

On an idle RTX 2080 Ti, PyTorch 2.11/CUDA 12.8, `B=1`, `H=4`, `Hv=2`, `D=8`,
`Dv=64`, FP16, and `L=32`, median CUDA-event timings were:

```text
T       multi VJP   folded VJP   folded/multi   replay G256   folded/replay
256        6.37 ms      0.57 ms       0.09        0.28 ms         2.03
512       13.40 ms      0.94 ms       0.07        0.84 ms         1.11
1024      30.01 ms      2.89 ms       0.10        3.00 ms         0.96
2048      68.36 ms     11.26 ms       0.16       11.80 ms         0.95
4096     175.03 ms     43.83 ms       0.25       46.91 ms         0.93
```

The ready-queue control was slower than multi-launch despite beating the
full-grid barrier control. This distinguishes removal of the barrier from the
larger gain due to static, equal-work folding.

The final column is decisive: folding plus fine-grained tile work solves the
CTA-load-imbalance problem inside the macro-wavefront design. Replay is still
best at `T <= 512`, where its extra launches are cheap, but folded crosses over
near `T=1024` and is 6-8% faster through `T=4096`. The replay path reconstructs
scores twice, uses a contiguous `O(BHTG)` workspace, and gives each Q/K/V
output a single accumulation owner. The folded macro path performs more score
replay and uses atomics, but exposes enough uniform work and has a faster V
schedule. Larger macro tiles reduce checkpoint traffic but lose enough
occupancy and local efficiency that `L=32` still wins.

Static folding is therefore a validated compute candidate, but this exact
implementation is not production-eligible because its workspace remains
quadratic. This made slabbed replay with `O(BHT)` live state the next useful
step; adding more queue, CTA-count, or tile-size controls could not address the
memory bound.

The first slabbed implementation now exists as
`benchmarks/slabbed_checkpoint_replay.py`. It establishes the required memory
bound without changing the estimator and now beats both folded and maintained
replay on the measured sm75 matrix. That fixed-512 file remains the historical
control; its simplified design was subsequently specialized and promoted into
the production unbounded translation unit.

The gradient-mask split at `T=2048` confirms that neither reverse branch is a
single removable bottleneck:

```text
requested gradient       folded L32       replay G256
Q/K only                     7.55 ms          5.09 ms
V only                       6.70 ms          7.67 ms
Q/K/V                       11.31 ms         11.92 ms
```

These rows are not additive because each call also computes shared row
statistics. At this length the measured operator scratch is 5.20 MiB for
folded L32 and 17.59 MiB for replay G256. Folded checkpoints and tile-local
statistics are initially smaller, but scale as `T^2/L`; replay scratch scales
as `T*G`. Thus the speed win belongs to replay and the short-sequence memory
win belongs to folded, while only replay satisfies the required asymptotic
workspace bound.

## 6. Microtile DP

Inside one `P x P` microtile, cells are grouped by local diagonal. For one
diagonal, gates form affine maps

```text
f_i(s) = g_i * s + g_i.
```

Composition is associative:

```text
(a_r,b_r) o (a_l,b_l)
    = (a_r*a_l, a_r*b_l + b_r).
```

A warp uses an inclusive affine prefix scan to evaluate one diagonal segment.
Each lane owns one complete packed symbol comparison, not one bit. This keeps
the popcount local and uses warp shuffles only for the recurrence. The segment
length is at most 32, so there is no branch-divergent suffix loop.

Macro tiles larger than 32 consist of a fixed microtile traversal. Carries
between microtiles stay in shared memory. Only macro-tile edges are published
globally, reducing synchronization and global edge traffic by approximately
`L/32` relative to a 32-wide global tiling.

### 6.1 CTA-local diagonal packing

Global long/short-diagonal pairing is not the only load-balancing level. A
square `32 x 32` microtile contains diagonal-segment lengths

```text
1, 2, ..., 31, 32, 31, ..., 2, 1.
```

These 63 segments contain exactly 1024 cells and can be statically packed
into 32 warp itineraries: keep the length-32 segment alone, pair each copy of
length `l` with one copy of length `32-l`, and pair the two length-16
segments. Each itinerary uses a segmented affine scan and explicitly resets
both forward and reverse carries at the segment boundary. Incoming and
outgoing carries remain indexed by the original global `delta`; packing is
only an execution permutation and must not join two recurrences.

This mapping is already implemented by `packed_diagonal_lane()` in the
macro-wavefront score and statistics paths. It is stronger than global
persistent scheduling because it removes lane-level tail waste without adding
a queue, global atomics, or a grid barrier. The packed statistics path writes
the resulting scores into a padded shared row-major tile before row-owned
softmax and value work. What remains pending is an isolated same-kernel A/B
against an unpacked 63-segment schedule; scan-only throughput is not an
acceptable promotion result.

The optimization is less relevant to slab replay. A slab CTA follows global
`delta` chains through rectangular 32-row chunks, so all lanes are active
except in the first or final chunk of a chain. CTA-local packing should be
tested there only after profiling confirms material tail waste; it must not
add square-tile bookkeeping to the regular long-sequence path by default.

## 7. Online Softmax And Utility

Rows are consumed in increasing key order. Each microtile transforms raw
scores, applies the candidate prior, and merges local statistics into the row
triple. The numerically stable merge is

```text
m = max(m_old, m_tile)
l = l_old * exp(m_old-m) + l_tile * exp(m_tile-m)
u = u_old * exp(m_old-m) + u_tile * exp(m_tile-m).
```

The null route initializes every valid row. Utility is

```text
dot(grad_output[q], sign(value[k+1])).
```

GQA value heads are mapped once per query head. The final implementation must
not repeat the entire suffix wavefront for every query-to-value head group.

## 8. Reverse Replay

The backward pass needs saved row statistics and enough exact DP state to
reconstruct each macro tile. It runs tiles in reverse dependency order:

1. load the forward macro-edge checkpoint;
2. replay local gates and scores in `P x P` microtiles;
3. reconstruct route probability from saved row `(max, normalizer)`;
4. accumulate `dV` before overwriting local score storage;
5. compute transformed-score VJP;
6. reverse-scan the recurrence along local diagonals;
7. accumulate Q/K log-gate credit;
8. publish predecessor adjoint edges.

For the unlimited recurrence

```text
dS[q,k] = dR[q,k] + g[q+1,k+1] * dS[q+1,k+1]
dlog(g[q,k]) = S[q,k] * dS[q,k].
```

The fixed permutation used by forward is reversed for incoming adjoint edges.
After a forward checkpoint is consumed, its allocation may be overwritten by
the reverse edge. This halves boundary storage without changing lifetime.

## 9. Memory Modes

Saving every macro-tile edge costs approximately

```text
4 * B * H * T^2 / L bytes
```

for one dominant FP32 edge field, plus linear row state and small metadata.
Tile-parallel statistics temporarily add approximately

```text
6 * B * H * T^2 / L bytes.
```

The combined peak is much smaller than an FP32 score matrix but remains
quadratic for fixed `L`; therefore it is not eligible for production by
itself.

The implemented exact mode uses the recurrence's stronger decomposition:
different global deltas `q-k` are independent DP chains. A slab holds one
contiguous range of diagonals. Production selects its capacity automatically
from `B*H`, `T`, the required workspaces, a 1 GiB dominant-workspace budget,
and a hard maximum of 8192 diagonals. Capacity is private storage granularity,
not a suffix window; there is no exposed `C`, `G`, or tile parameter.
Contiguous deltas keep all four score/reverse warps in a CTA at nearly equal
chain lengths and remove the former low/high-band coordinate mapping.

The exact data flow is:

1. initialize each row with the null route;
2. scan every diagonal in the current slab and materialize exact raw scores;
3. build route utilities in `16 x 16` row/delta tiles and merge online row
   softmax statistics;
4. discard the slab and repeat until every causal candidate has contributed;
5. traverse slabs again, reconstructing the same scores and utilities;
6. accumulate `dV`, reverse each unlimited diagonal recurrence, overwrite the
   score slab with log-gate VJPs, and owner-reduce Q/K gradients.

Three exact hot-path restrictions are retained:

1. stats launches only rows reachable by the current slab;
2. value owners launch only routes that can receive a candidate from it, while
   fully noncausal utility tiles return before loading V or `grad_output`;
3. each value-owner candidate computes its probability and 64-bit
   `grad_output` base once, then broadcasts both from shared memory across all
   value features.

The reverse pass deliberately rebuilds every slab with the same regular code
path. Reusing the final forward slab would remove one boundary replay, but
would couple workspace lifetime to traversal direction for little asymptotic
benefit. A packed triangular utility-grid decoder was also rejected: normalized
gains were only about `0.4%..2.5%`, insufficient to justify duplicate host and
device indexing logic.

There is no cross-slab DP checkpoint: diagonal chains do not depend on one
another. The saved row statistics are the only cross-pass checkpoint. Adding a
two-dimensional macro-edge hierarchy here would retain more state without
reducing exact candidate work.

For selected capacity `S`, live FP32 scratch is one score slab, an optional
utility slab, and the row triple:

```text
score       B * H * round_up(T,32) * round_up(S,4)
utility     same, only when Q or K gradients are requested
row stats   3 * B * H * T
```

The generic Q/K path uses both slabs. FP16 `Dv=64` fused statistics and reverse
compute utility while scores are resident, so only the score slab is live;
value-only also needs one slab. The normal path aligns `S` to 32 and may use a
four-slot granularity only when an extreme shape cannot fit 32 slots under the
budget. Because `S <= 8192` is independent of sequence length, workspace is
`O(BHT)`. Output gradients are also linear and are not counted as scratch.

The final production candidate must satisfy both:

```text
workspace = O(BHT) plus input/output gradients
candidate work = exact O(BHT^2)
```

The mode is now the production fixed-shape unbounded VJP on CUDA. Macro-edge,
ready-queue, row-stream, and folded schedules remain research-only.

### Initial fixed-512 slab result (historical)

The first fixed-512 milestone produced the following table. It remains useful
for tracing the implementation, but its capacity decision and absolute
latencies are superseded by the clean production recheck below.

```text
T       slabbed VJP   replay G256   folded L32   row stream   slabbed peak
256         0.22 ms       0.30 ms      0.84 ms       9.14 ms       2.20 MiB
512         0.31 ms       0.83 ms      0.96 ms      14.93 ms       8.40 MiB
1024        1.00 ms       2.99 ms      2.91 ms      30.79 ms      16.80 MiB
2048        3.61 ms      12.06 ms     11.32 ms      67.76 ms      33.59 MiB
4096       14.54 ms      46.74 ms     43.95 ms     163.59 ms      67.19 MiB
8192       57.72 ms     185.68 ms    176.30 ms     485.97 ms     134.38 MiB
```

The slabbed peak includes returned FP32 gradients. Declared scratch is `2.01,
8.02, 16.05, 32.09, 64.19, 128.38 MiB`; it remains linear in `T`. Maximum
slabbed-versus-maintained replay gradient error through `T=8192` is below
`7.5e-7`. At `T=4096`, the optimized path is `3.21x` faster than maintained
replay and `3.02x` faster than folded replay; at `T=8192` the ratios are
`3.22x` and `3.05x`.

At that milestone, three low-level changes accounted for the retained gain.
Padding each
`16 x 16` utility route in shared memory to a stride of 17 reduced Nsight
Compute shared-memory excessive wavefronts from about 64% to 7% and reduced a
representative utility kernel from `961 us` to `526 us`. Raising the internal
slab capacity from 256 to 512 reduced replay launches and improved 4K/8K by
about 8%; capacity 1024 bought only another 2.4%/0.9% while doubling scratch
again, so it was rejected. Saving the reciprocal row normalizer removes the
repeated backward division without adding state.

Rejected implementation variants include 32/64-wide feature chunks, an
8-warp score CTA, subwarp Q/K owners, and a two-route dV owner. Each was
numerically correct but slower on the long-sequence sm75 controls. Boundary
replay remains intentionally regular because removing it would add code and
state for negligible work.

The value-owner metadata cache was retained by direct ablation. Removing it
changed V-only latency at `T=512/1024/2048/4096` from
`0.181/0.399/1.296/5.130 ms` to `0.352/0.755/2.563/10.301 ms`. A 64-thread
owner was marginally faster below 2K but about 2% slower at 2K and 4K, so the
fixed 128-thread owner remains the long-sequence choice.

### Clean production recheck (2026-09-05)

Earlier runs were not used to tune the promoted dispatcher after unrelated
load was discovered on GPU1. The retained choices were remeasured on an idle
RTX 2080 Ti with operations interleaved in rotating order over seven rounds.
The shape was FP16, `B=1,D=8,Dv=64`, QKV gradients, with `H/Hv` varied as
shown. `baseline_s256` is the exact generic slab plan at capacity 256.

| T | H/Hv | auto slab | production ms | baseline s256 ms | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1/1 | 2,047 | 0.857 | 1.450 | 1.69x |
| 2,048 | 4/2 | 2,047 | 2.951 | 4.120 | 1.40x |
| 2,048 | 8/4 | 2,047 | 4.699 | 7.599 | 1.62x |
| 4,096 | 1/1 | 4,095 | 2.906 | 5.103 | 1.76x |
| 4,096 | 4/2 | 4,095 | 8.500 | 15.555 | 1.83x |
| 4,096 | 8/4 | 4,095 | 16.405 | 29.460 | 1.80x |
| 8,192 | 1/1 | 8,191 | 8.325 | 19.450 | 2.34x |
| 8,192 | 4/2 | 8,191 | 32.078 | 61.185 | 1.91x |
| 8,192 | 8/4 | 4,096 | 65.133 | 116.889 | 1.79x |

Calling the same implementation through an explicit control slab differed
from production by at most 0.7% in eight cases and 1.0% in the remaining
case. This confirms that private dispatch itself adds no material overhead.
The reproducible H=4 production/fixed-slab run, including all seven samples
and gradient differences, is stored in
`validation/unbounded_production_clean_sm75.json`.

The clean `T=8192` capacity sweep explains why 512 is no longer retained:

| slab | H=1 ms | H=4 ms | H=8 ms |
| ---: | ---: | ---: | ---: |
| 256 | 75.622 | 91.173 | 105.474 |
| 512 | 40.404 | 51.846 | 74.324 |
| 1,024 | 22.765 | 37.896 | 72.101 |
| 2,048 | 14.227 | 36.317 | 68.440 |
| 4,096 | 10.599 | 33.147 | 65.387 |
| 8,192 | 8.366 | 32.392 | 64.369 |

Full capacity is valuable for low head count, while the gain flattens as
`B*H` grows. The fixed 1 GiB planner therefore gives H=8 a 4096 slab rather
than spending another roughly 1 GiB for about 1.6% at this shape.

The retained execution plan has four internal flags: fused statistics,
fused reverse, tensor-core dV, and tiled symbol accumulation. Fusion starts at
`T>=2048` only when there are at least eight series, or at `T>=4096` when
`series*T>=8192`. Tensor dV requires FP16 `Dv=64` and enough sequence/series
work. Tiled Q/K starts at `D>=4`; its clean H=4, T=4096 speedups for
`D=1/2/4/8/16/32` were approximately `-0.7/1.4/4.8/12.4/9.5/12.2%`, so D1
and D2 retain the scalar owner.

Two scheduling conclusions also survived the clean recheck. Increasing the
folded macro tile from 32 to 64 or 128 regressed T=2K..8K by 25-45%.
Hardware work distribution beat paired, fixed-paired, and occupancy-bounded
persistent diagonal queues across random and periodic H=1/H=4 controls; the
closest persistent result was still 2-5% slower. Neither mechanism is in
production.

Two final replay ideas were rejected after kernel-level measurement. At
T=8192,H=4, the standalone backward score pre-scan is 1.34 ms of a 32.09 ms
VJP (4.2%); retaining the last forward slab would therefore have a sub-5%
ceiling while coupling forward and reverse workspace lifetimes. It remains a
regular recompute. A second A/B replaced the fused-statistics kernel's
single-warp row recurrence with eight warps scanning 32 diagonals into a
padded `[diagonal][33]` shared tile. Production-relevant T=4K/8K cases ranged
from a 1.7% win to a 0.5% regression and were neutral at 8K. The extra variant
was removed. This result does not reject section 6.1: complementary packing
targets the intrinsically short `1..32..1` segments inside square wavefront
tiles, while slab diagonals already contain long regular 32-row chunks.

## 10. Research Macro Error Handling

The macro-wavefront research wrapper validates:

- CUDA, contiguity, dtype, rank, shape, and common device;
- `1 <= D <= 32`;
- finite positive `scale` and `mismatch_scale`;
- `0 <= dropout_p < 1`;
- `tile_size == 0` or a supported positive multiple of 32;
- cooperative-launch support before selecting persistent mode.

Unsupported tile sizes fail clearly. Automatic dispatch may choose the exact
multi-launch implementation when cooperative launch is unavailable or when a
tiny problem cannot amortize a persistent grid. Neither fallback changes
results.

## 11. Validation Matrix

Correctness must cover:

```text
T:       1,2,31,32,33,63,64,65,95,96,97,127,128,129
L:       32,64,96,128 and auto
D:       1,8,32
pattern: random, all-match, alternating, single mismatch, periodic
dtype:   FP32, FP16, BF16 where supported
heads:   H=1 and grouped H/Hv
dropout: 0 and fixed nonzero seed
grads:   all seven nonempty Q/K/V masks
```

Required comparisons are:

1. debug raw scores versus the explicit unlimited PyTorch recurrence;
2. multi-launch versus cooperative persistent forward;
3. row statistics versus the dense oracle;
4. log-gate VJP versus autograd and the diagonal recurrence oracle;
5. full Q/K/V VJP versus the production oracle configured with `W=T`;
6. exact hard output independence from every surrogate schedule control;
7. Compute Sanitizer memcheck, racecheck, and synccheck;
8. occupancy, registers, spills, peak workspace, and end-to-end latency.

The post-promotion combined command for production unbounded, slabbed replay,
persistent-wavefront compatibility, and build contracts reports `736 passed`
on sm75. It covers the first multi-slab boundary, all seven gradient masks,
dropout, grouped value heads, D1/D8/D32, FP32/FP16/BF16, structured patterns,
and fullgraph `torch.compile`. Direct production-versus-independent replay
checks through T=4096 differ by at most `6.4e-5` for FP16 dV and below
`2.2e-5` for FP16 Q/K; FP32/BF16 controls are below `6e-7`/`3.3e-7`.
Compute Sanitizer reports zero memcheck errors, racecheck hazards, and
synccheck errors for the macro schedules, fixed-512 slab control, and the
production fully fused unbounded path. The complete repository suite reports
`3424 passed, 589 skipped`; skips are device/build-dependent parameterizations.
The production extension was rebuilt explicitly for sm75 on physical GPU1
before the post-promotion tests.

Tests intentionally cross every 32-cell microtile and every `L`-cell macro
boundary. Passing only square or all-match cases is insufficient.

## 12. Promotion Criteria

The research path may replace no maintained code until all of the following
are true:

- exact output and VJP parity across the validation matrix;
- no candidate omission and no finite suffix cap;
- exact `O(BHT)` workspace mode;
- no synchronization hazard under Compute Sanitizer;
- stable gain in complete backward, not only score generation;
- performance reported against current production finite VJP, unlimited
  diagonal replay, historical fixed suffix attention, and FlashAttention;
- no new public controls beyond the existing estimator parameters.

Failure at any stage is recorded as a result. The old implementation remains
the reference rather than being silently modified to fit a benchmark.
