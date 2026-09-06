# RosaSoft CUDA Optimization Record

This document records the 2026-07-30 through 2026-09-06 CUDA optimization
passes.
It distinguishes production changes from rejected experiments so old branches
and benchmark artifacts do not become accidental design requirements.

## 1. Measurement Contract

The target was physical GPU 1, an idle NVIDIA GeForce RTX 2080 Ti
(`sm_75`), isolated with `CUDA_VISIBLE_DEVICES=1`. The build used PyTorch
2.11.0, CUDA 12.8, `-O3`, `--use_fast_math`, and resource reporting for
`sm_75` and `sm_86`.

Decisions used alternating-order A/B runs after warmup. A change was retained
only when it passed the PyTorch reference, all affected gradient masks, and a
shape matrix rather than one favorable point. Every experiment kept the full
dense causal candidate set. The final retained set was remeasured on an idle
card with five alternating-order rounds, 30 warmups, and 200 timed iterations
per sample; the tables below report medians of those rounds.

The 2026-09-05 unbounded replay decisions were independently rechecked after
unrelated GPU1 load was discovered. Only seven-round rotating-order idle-card
measurements in `PERSISTENT_MACRO_TILE_VJP.md` govern those decisions; older
absolute latency tables remain historical evidence for the finite operator.

## 2. Decision Matrix

| Step | Mechanism | Decision | Reason |
| ---: | --- | --- | --- |
| 1 | Packed value-only static dispatch and value tiling | Keep | Removes dead Q/K work and coalesces `(route,Dv)` updates. |
| 2 | Factor softsign Jacobians into one finalize kernel | Keep | Applies each input Jacobian once and lowers hot-kernel register pressure. |
| 3 | Tile cooperative route utilities and value probabilities | Keep, gated | Wins long `Dv >= 64` mixed-gradient shapes without quadratic state. |
| 4 | Packed sign bits in head-major `[H,N]` | Keep | Coalesces route-lane reads and lowers packed generic register pressure. |
| 5 | Packed Q-only exact score tail cache | Keep, gated | A 1024-score tail wins long Q-only segments; older scores remain exact recomputes. |
| 6 | Dense work- and occupancy-aware cache/recompute plan | Keep | Avoids premature recompute, then prevents large score caches from adding launch waves. |
| 7 | Dense block-local K VJP aggregation | Keep, gated | Collapses overlapping global atomics for long `D <= 8` rows. |
| 8 | Global 64/128/256-thread CTA choice | Keep 128 | 64 and 256 have conflicting dense/packed wins; a ladder needs brittle heuristics. |
| 9 | Cancellation-resistant local diagonal adjoint | Keep as oracle | Removes full-diagonal cancellation and passes long exact-match FP32 stress. |
| 10 | Multi-stage factorized diagonal CUDA VJP | Archived | Parity passed, but cross-GPU mean latency was `1.30x` production and scratch was quadratic; removed from the frozen build. |
| 11 | Exact unlimited hard diagonal DP | Promoted | Bit-exact, `O(BHT)` state, quadratic worst-case work; selected at T>=512 after repeated inputs improved by over `160x` at T=4096. |
| 12 | Offline cross-GPU promotion gate | Keep | Rejects candidates with any material regression, parity failure, or excessive workspace. |
| 13 | Exact tiled-streaming dense VJP | Keep for fixed `T >= 4096` | Preserves every candidate without quadratic state and cuts long SM75 QKV latency by about `2.8x..2.9x`. |
| 14 | Local-gate adjoint aggregation | Keep | Contracts each shared gate with Q/K bits once; gives about `2x` over the first streaming kernel. |
| 15 | Shared-workspace lifetime reuse and 32-row `D=32` | Keep | Lowers shared storage to `40.50 KiB` and removes the production 16-row branch. |
| 16 | Exact rolling diagonal suffix recurrence | Reject | Exact, but serial diagonal dependence and low active-thread count regress common shapes. |
| 17 | First-sweep online value carrier | Reject | Small wide-value wins do not clear the mean gate and narrow values regress. |
| 18 | Coarse gradient-mask template classes | Reject | A small Q/K-only win costs about 81% binary growth and no useful QKV gain. |
| 19 | Finite-window row-log hoist and mismatch-gate LUT | Reject in that kernel | Compiler already removes the bounded-loop row invariant; LUT gains are sub-threshold and shape-dependent there. The unbounded grouped path is covered separately below. |
| 20 | Sparse-tail warp suffix ownership | Keep as research primitive | Wins when only 1-2 queries survive and no recurrence state exists; full-VJP integration is neutral and raises registers. |
| 21 | `32 x 32` block-diagonal tiles | Reject | Higher residency loses to doubled halo, synchronization, reverse, and atomic boundary work. |
| 22 | Sequential TF32 high/low fragments | Reject | NVCC already reuses the registers; aggregate change is noise with a 2.1% worst regression. |
| 23 | Named-barrier warp-specialized double buffer | Keep for research | Exact first-sweep overlap gives a 3.9% integrated mean win on sm_86, but does not clear the 5% production gate. |
| 24 | Exact unbounded diagonal-slab replay | Promoted as separate operator | Visits every causal candidate, has no suffix cap, and keeps live workspace `O(BHT)` with a private bounded slab. |
| 25 | FP16 fused statistics and reverse utility | Keep, gated | Removes the materialized utility slab and improves clean H=4/H=8 long-sequence controls by roughly 39-44% over the selected unfused plan. |
| 26 | FP16 tensor dV plus tiled Q/K owners | Keep, gated | Tensor dV wins `Dv=64` at sufficient work; tiled symbols win reliably from D=4 and remain scalar at D=1/2. |
| 27 | Shape-aware slab up to 8192 under 1 GiB | Keep | Large slabs remove repeated launches; capacity gains flatten with head count, so the fixed budget selects the tradeoff without a public knob. |
| 28 | 64/128 macro tiles and persistent/paired diagonal queues | Reject | Clean sm75 recheck leaves 32 fastest; hardware scheduling remains 2-17% ahead of tested explicit schedules. |
| 29 | CTA-local complementary diagonal packing | Implemented in macro prototype; isolated A/B pending | `packed_diagonal_lane()` maps 63 local segments to 32 warp itineraries with segmented carry reset. It should not complicate slab replay absent measured tail waste. |
| 30 | Multi-warp fused-statistics score scan | Reject | Padded diagonal-major shared storage is exact, but clean production-relevant results range from 1.7% faster to 0.5% slower and become neutral at 8K. |
| 31 | Retain the last forward score slab | Reject | The removable score pre-scan is only 4.2% of H=4,T=8K VJP time; cross-pass lifetime coupling does not clear the complexity gate. |
| 32 | Occupancy-bounded grouped checkpoint reverse | Keep, gated | Complete 32-diagonal ownership needs no inter-CTA progress protocol, stores only `O(GT)` boundaries, and cuts 4K/8K live memory by 10.7-21.3x. Static long/short pairing, gate LUT, row-prior hoist, max-then-exp reduction, warp affine scans, and WMMA utility/dV are retained. |
| 33 | Move utility/WMMA phase storage from shared memory to L2 | Reject | Shared storage fell by 8 KiB, but registers rose from 85 to 101. Forcing three CTAs introduced spills, materially regressed 4K, and did not give a stable long-sequence win. |

## 3. Rejected Local Changes

### 3.1 Finite-Window Mismatch LUT

The tested QKV times for `T=128/256/512/1024` were
`0.426/0.857/2.998/11.566 ms`, versus
`0.402/0.862/2.956/11.412 ms` without the LUT. The result changes sign across
shapes and does not justify a shared table.

This result does not apply to the later unbounded grouped kernel, where the
same gate is otherwise recomputed across stats, checkpoint, and replay passes.
That path retains one 33-entry table per CTA; see
`GROUPED_CHECKPOINT_VJP.md`.

### 3.2 Warp-Per-Route Suffix Scan

The current kernel assigns adjacent lanes to adjacent routes. Every route
executes the same bounded suffix loop; mismatches change arithmetic values,
not loop control. A standalone `W=32, T=1024` microbenchmark assigned
`1/2/4/8/16/32` lanes to one route. Relative to one lane per route, the
multi-lane variants were approximately
`10.7x/14.4x/3.3x/4.4x/5.6x` slower.

This does not reject prefix scan as a primitive. It rejects applying it along
the wrong axis. A diagonal recurrence has a true scan dependency and is
covered in section 6.

### 3.3 Static `W=32`

Static unrolling improved the suffix core by only a few percent, raised
observed registers into the `158..168` range for affected instances, and grew
the extension from roughly 2.3 MiB to 3.5 MiB. Q-only had isolated gains, but
K, QKV, and value paths regressed. Arbitrary positive windows remain supported.

### 3.4 Packed Sequence Lookup

All 128 threads currently perform the same offset binary search. Computing it
once per block requires a barrier and independent shared lifetime; the first
prototype exposed a read/overwrite race, and the corrected form showed severe
scheduling sensitivity. Computing once per warp removed that risk but changed
training time by roughly `-1%..+7%` across segment lengths. The simpler
thread-local search remains.

## 4. Retained Paths

### 4.1 Dense And Packed K Aggregation

Route lanes write many K contributions to overlapping positions. The retained
path accumulates a
`[D, blockDim + W - 1]` FP32 tile in shared memory and flushes it once.

For dense rows it activates only when `D <= 8`, the complete shared layout
fits 48 KiB, and the sequence passes a gradient-mask-specific threshold:

```text
K or KV:  T >= 512
QK or QKV: T >= 1024
```

The higher mixed-gradient threshold matters. At `Dv=32..64`, enabling the
tile for QK/QKV at `T=512..768` regressed by roughly `6%..17%`; at `T=1024`
the same path won. Five-round same-process forced A/B on the idle 2080 Ti gave:

| Shape | `Dv=8` | `Dv=32` | `Dv=64` |
| --- | ---: | ---: | ---: |
| K, `T=512` | `2.37x` | `1.62x` | `1.79x` |
| KV, `T=512` | `2.31x` | `1.45x` | `1.65x` |
| QK, `T=1024` | `1.29x` | `1.09x` | `1.03x` |
| QKV, `T=1024` | `1.30x` | `1.09x` | `1.03x` |

The aggregate and direct paths are separate compile-time variants; leaving a
runtime template branch in the direct path cost `4%..15%`. That split grows
the extension from 6,757,592 to 10,130,880 bytes, about 50%, but the final
SM75/SM86 build remains free of stack and local-memory spills. Raw forced A/B
records are stored in `validation/dense_key_aggregation_ablation.json`.

It activates only when:

```text
K gradient enabled
D <= 8
average segment length >= 256
local segment length >= 256
complete shared layout <= 48 KiB
```

Host selection requires average segment length at least 256; inside an aggregate
launch, a local segment shorter than 256 skips the tile accumulation but still
shares that launch's template and dynamic shared allocation. The idle-card
recheck on uniform segment lengths gave:

| Segment length | K-only speedup | QKV speedup |
| ---: | ---: | ---: |
| 128 | `1.01x` | `0.99x` |
| 256 | `2.10x` | `1.41x` |
| 512 | `1.88x` | `1.37x` |

The 128-token negative control is effectively unchanged, which supports the
256-token threshold. Dense K retains its head-major global accumulator behind
the shared tile, so public layout and finalization are unchanged.

### 4.2 Q-Only Cooperative Utility

Q/K route credit needs:

```text
utility(route) = dot(grad_output[row], sign(value[route]))
```

The generic path scans `Dv` independently in each route lane and recomputes
the utility in the second route pass. For dense Q-only backward with
`Dv >= 32`, the retained path computes each utility once and caches one FP32
scalar per route:

```text
Dv = 32:  8-lane groups
Dv >= 64: 16-lane groups
```

Idle-card `T=512/1024` medians changed as follows:

| `Dv` | Before | After | Speedup |
| ---: | ---: | ---: | ---: |
| 32 | `1.680 / 5.867 ms` | `1.244 / 3.988 ms` | `1.35x / 1.47x` |
| 64 | `2.143 / 7.595 ms` | `1.298 / 4.199 ms` | `1.65x / 1.81x` |
| 128 | `3.182 / 11.511 ms` | `1.260 / 4.353 ms` | `2.53x / 2.64x` |
| 256 | `5.332 / 19.804 ms` | `1.360 / 5.044 ms` | `3.92x / 3.93x` |

`Dv=8` and mixed-gradient paths do not select this cache.

### 4.3 Value-Only Route/Dimension Tiling

Value-only backward needs probabilities but no route utility or suffix
adjoint. The retained path caches one tile of dropped probabilities, flattens
`(route, value_dimension)`, and lets adjacent threads update adjacent value
elements.

At `T=512/1024`, its idle-card speedups were:

| `Dv` | Speedup |
| ---: | ---: |
| 32 | `1.80x / 4.07x` |
| 64 | `3.17x / 4.53x` |
| 128 | `4.60x / 4.92x` |
| 256 | `4.60x / 5.00x` |

The path starts at `Dv=32`; `Dv=8` remained within `3.4%` at `T=512` and
within `0.1%` at `T=1024`.

### 4.4 Factored Symbol Jacobians

The route kernels now accumulate raw FP32 symbol-space adjoints. One finalize
kernel applies `1 / (1 + abs(x))^2` once per enabled Q/K/value input. Dense K
finalization also performs the existing `[B,H,D,T] -> [B,T,H,D]` transpose.
This removed repeated logit reads and Jacobian arithmetic from every
route/suffix/bit contribution. The affected SM75 kernels remained spill-free;
representative dense and packed training paths improved by roughly
`1.7x..2.1x`.

### 4.5 Packed Layout and Exact Tail Cache

Private packed-varlen Q/K sign bits changed from route-unfriendly `[N,H]` to
head-major `[H,N]`; the public `[N,H,D]` API and gradient layout did not
change. All-match hard forward at `T=1024` improved by about
`1.12x/1.43x/3.27x` for `H=1/4/16`.

Packed Q-only backward may additionally retain the most recent 1024 scores in
shared memory when average segment length is at least 256. Scores before that
tail are recomputed exactly in pass two. Capacity 512 left avoidable reuse on
the table, while enabling the cache below average length 256 regressed many
short segments. The retained gate improved tested Q-only segments by about
`8%..11%`; non-Q masks instantiate no cache branch or cache storage.

### 4.6 Dense Occupancy Planner

The complete dense cache layout grows with `T` and can reduce resident blocks
long before the 48 KiB per-block limit is reached. The host now asks CUDA for
the active blocks per SM of the exact cache and recompute template instances,
including dtype, static gradient mask, cooperative mode, and dynamic shared
bytes. Occupancy alone was too aggressive: at `T <= 2048`, avoiding the
second score scan still won even when cache used fewer resident blocks.

The retained selector therefore has two stages:

```text
consider recompute only when T >= 4096
for V-only, also require T >= 64 W
select recompute only when it reduces total grid waves
```

The second stage uses the actual dense row count and device SM count, so a
small grid retains cache even if the recompute kernel has higher theoretical
occupancy. Equal grid waves prefer reuse. The `4096` and `64` gates are
SM75-calibrated private thresholds and should be remeasured before targeting a
substantially different architecture.

One representative SM75 QKV matrix (`D=8`, `Dv=64`, `W=32`) was:

| `T` | Forced cache | Forced recompute | Selected |
| ---: | ---: | ---: | ---: |
| 2048 | `5.92 ms` | `6.51 ms` | cache, `6.06 ms` |
| 4096 | `30.31 ms` | `24.11 ms` | recompute, `24.12 ms` |
| 8192 | `286.77 ms` | `88.53 ms` | recompute, `88.31 ms` |

The recompute path also received the same compile-time value-only
specialization and probability tiling as the cache path; otherwise a forced
comparison confounds score reuse with unrelated dead code.

The route-utility audit also found a real shared-memory lifetime bug in the
first cooperative prototype. The retained temporary tile is warp-owned and
uses paired warp synchronization around consumption and reuse. The persistent
dense Q-only cache remains block-produced and uses a block barrier before
cross-warp reads. Dense full-cache, packed Q-only, and packed QKV tests report
zero racecheck hazards and zero synccheck errors.

### 4.7 CTA-Size Negative Control

Complete 64-, 128-, and 256-thread builds each passed 125 CUDA/varlen tests,
had 79 CUDA kernels, used at most 128 registers, and had zero stack/local
spill. The throughput ranking was not stable:

- 256 threads improved several dense Q/K cases by `7%..45%`;
- the same build regressed packed QK/QKV by roughly `16%..40%`;
- 64 threads improved selected packed QV/QKV cases by `3%..20%`, but made
  dense Q-only up to about `73%` slower.

A runtime ladder therefore needs layout, mask, length, `Dv`, and suffix-window
heuristics. The fixed 128-thread CTA remains the simpler robust choice.

### 4.8 Negative Controls

The retained cooperative path starts at `Dv=64` and sequence length 256.
Dense `Dv=32`, packed QK `Dv=64`, and packed segments with average length
below 256 remain on their simpler paths. Global CTA-size and full packed-cache
experiments were rejected because their ranking changed across gradient masks
and layouts; no historical pre-cooperative QKV number is used as evidence for
the final kernel.

## 5. Why Dense Support Is Unchanged

No retained path changes a score, probability, dropout decision, candidate,
or gradient formula. They:

- reuse a scalar route utility;
- reorder a value-dimension reduction;
- aggregate overlapping FP32 atomics inside one block;
- move one-time softsign Jacobians into finalization;
- change private sign-bit layout;
- cache exact scores or recompute them according to resource cost.

Candidate top-k, bounded lookup, thresholding, and sparse discovery gradients
remain prohibited by `AGENTS.md`.

## 6. Exact Diagonal Recurrence

For one causal diagonal, let `x_n` be its local match gate and let `W` be the
finite suffix window. The score

```text
s_n = x_n + x_n x_(n-1) + ...  (at most W products)
```

obeys:

```text
p_n = product(x_(n-W) ... x_n), or 0 when n < W
s_n = x_n (1 + s_(n-1)) - p_n
```

Once `p_n` is known, each element is an affine transform
`s -> x_n s + (x_n - p_n)`. Affine transforms compose associatively, so one
warp can scan 32 diagonal elements and pass one carry to the next group.
`p_n` should be reconstructed from a sliding integer mismatch-count sum, not
by subtracting long FP32 log-prefix sums.

The original algebraic reverse used external route-score VJP `a_n` and:

```text
b_n = a_n + x_(n+1) b_(n+1)
c_n = b_n p_n
h_m = b_m (s_m + p_m) - sum(c_n, n=m..m+W)
```

This is exact over real arithmetic, but it subtracts quantities that can grow
with the complete diagonal to recover a result supported on only `W` routes.
Long exact matches therefore lose FP32 precision.

The retained numerical oracle expands only real local contributions. For one
route output `n` and gate `m`, where `m <= n < m + W`:

```text
d s_n / d log(x_m)
  = sum over starts r=max(0,n-W+1)..m of product(x_r ... x_n)

h_m
  = sum over n=m..min(N-1,m+W-1)
      a_n * d s_n / d log(x_m)
```

All products and each inner prefix sum are non-negative. The implementation
uses compensated accumulation only across the at-most-`W` signed external
adjoints. It does not form or subtract full-diagonal correction terms. The
PyTorch oracle intentionally uses `O(NW)` scratch so correctness is explicit;
a production warp implementation would keep a bounded age/ring state and
scan starts in 32-element groups.

For the production symbol proxy,

```text
log(x) = -mismatch_scale/2
         + mismatch_scale/(2D) * sum(q_bit * k_bit)
```

so each local gate is owned once by its diagonal and contributes
`h * mismatch_scale/(2D) * k` to Q and the symmetric term to K. The research
prototype now performs these complete scatter-adds and verifies the final
unused key position receives exactly zero gradient.

Validation results for the affine score scan and small exact VJPs remain:

| Matrix | Cases | Max score error | Max VJP error |
| --- | ---: | ---: | ---: |
| FP64, `N=1..65`, `W=1..128` | 100 | `6.66e-16` | `8.88e-16` |
| FP32, `N=257..4096`, `W=1..512` | 48 | `4.77e-7` | `7.15e-7` |

The 91-test research suite additionally covers full causal matrix score and
log-gate mapping, complete FP32/FP64 Q/K symbol VJPs, and exact,
one-mismatch, alternating, and random-Hamming gate patterns. Exact-match FP32
adjoints reached about `7.0e-4` maximum absolute error under unscaled random
route VJPs at `T=257,W=128`; with a real softmax-shaped route-score adjoint the
maximum fell below `3.1e-5`.

The old algebraic reverse did not extrapolate. The deterministic long
exact-match stress now compares both FP32 forms with a direct FP64 formula:

| `N` | `W` | Algebraic max error | Local max error | Local global-relative error |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 512 | `6.59e-3` | `1.08e-3` | `7.67e-8` |
| 4096 | 4096 | `3.63e-2` | `1.36e-2` | `8.62e-8` |

The absolute values grow because exact-match gradients themselves reach about
`1.4e4` and `1.6e5`; the local form stays near one FP32 ulp globally and has
much lower error near small outputs. This resolves the numerical formulation
blocker. It does not by itself establish a faster production schedule.

## 7. Multi-Stage CUDA Prototype

The former multi-stage diagonal CUDA prototype had three stages:

1. a row-owned pass stores max, denominator, and expected utility in
   `O(BHT)` FP32 state;
2. diagonal route blocks reconstruct every raw suffix score and route adjoint,
   update value gradients, and accumulate one scalar `dL/dlog(x)` per local
   Q/K pair;
3. a final pair-owned pass maps each scalar local-gate adjoint to all Q/K bits
   once, followed by the production softsign finalize.

This prototype preserves every causal candidate, exact score/dropout
semantics, grouped value heads, all seven gradient masks, and FP32/FP16/BF16.
Across 44 focused tests its differences from production are only reduction
order. At `T=33`, maximum absolute gradient error was below `6e-7`; the
cross-GPU calibration through `T=1024` observed at most `2.67e-5`.

It was not promoted. It materialized `O(BHT^2)` FP32 local-gate scratch, and on
the RTX 2080 Ti random QKV at `T=128/256/512` was `1.25x/1.27x/1.47x` the
production latency. Across the common 2080 Ti/3070 matrix its mean ratio was
`1.30x` and worst ratio `1.59x`. Its raw operation, benchmark, and default
tests were removed when `rosa-soft-dense-reference-v1` was frozen. The later
single-kernel tiled-streaming schedule removed the quadratic gate matrix and
extra launches without reviving this architecture.

## 8. Exact Hard Diagonal DP

The maintained hard diagonal path assigns one warp to one complete Q/K
diagonal. For each 32-position chunk it computes the nearest preceding
mismatch with an inclusive prefix maximum; the exact unlimited suffix length
is the distance to that mismatch. A 64-bit atomic maximum encodes
`(length, successor_route)`, preserving longest-first and latest-on-tie
semantics without a hard suffix window.

The path compares every causal Q/K endpoint pair once. It is not a bounded
candidate lookup, probabilistic hash, or sparse-gradient mechanism. It needs
one `O(BHT)` winner array and has a deterministic `O(BHT^2)` work bound,
including repetitive inputs where the direct early-exit route scan becomes
cubic. Dense and packed-varlen implementations are selected from T=512; the
direct scan remains the low-overhead short-input fallback.

On the RTX 3070 at `B=1,H=4,D=8,Hv=2,Dv=64,T=4096`, FP16 production DP takes
about `0.268/0.345/0.673 ms` for random/periodic-64/all-equal symbols. The
former direct scan measured `0.188/64.88/108.28 ms`. All tested outputs are
bit-exact with the compact SAM. Random input pays a small absolute overhead in
exchange for removing the pattern-dependent worst case. Compute Sanitizer
memcheck, racecheck, and synccheck pass for dense and empty-segment varlen.

## 9. Cross-GPU Promotion Gate

The archived `validation/execution_plan_calibration.json` records the exact
device, capability, shape, pattern, parity error, latency ratio, and extra
workspace. A future candidate is promoted only when all recorded cases satisfy:

```text
worst latency ratio <= 1.05
mean latency ratio <= 0.95
extra workspace <= 64 MiB
hard parity error == 0
factorized VJP max error <= 3e-4
```

The common `T=128/256/512/1024`, random/all-match, QKV matrix produced:

| Candidate | Cases | Mean ratio | Worst ratio | Max error | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Hard diagonal index | 16 | `1.366` | `1.817` | `0` | research only |
| Factorized diagonal VJP | 16 | `1.305` | `1.592` | `2.67e-5` | research only |

These results retained the row-owned VJP at that checkpoint. They remain the
negative-control evidence for the later tiled-streaming promotion.

## 10. First Tiled-Streaming VJP Promotion

The first promoted fixed-length kernel tiled 16 or 32 adjacent query rows, 32 route
positions, 32 suffix offsets, and 32 or 64 value features. Its first complete
route sweep computes online `(maximum, normalizer, expected utility)` row
statistics. Its second complete sweep recomputes exact scores and fuses
dropout, utility, value accumulation, and dense Q/K credit. It stores no
quadratic state and never prunes candidates.

That checkpoint dispatched at `T >= 4096`, with 32 query rows for `D <= 16`
and 16 rows otherwise. On sm_75, those kernels used 64 and 108 registers
respectively, zero stack/spill, and at most `45.37 KiB` shared memory. All
masks, dropout, suffix boundaries, wide values, and `D=32` passed the `3e-4`
parity gate; Compute Sanitizer reported no synchronization or race errors.

Idle RTX 2080 Ti random FP32 QKV at
`B=1,H=4,Hv=2,D=8,Dv=64,W=32` measured:

| `T` | Frozen row VJP | Streaming production | Ratio |
| ---: | ---: | ---: | ---: |
| 4096 | `100.82 ms` | `35.56 ms` | `0.353` |
| 8192 | `391.72 ms` | `134.84 ms` | `0.344` |
| 16384 | `1553.83 ms` | `529.14 ms` | `0.341` |

These numbers are retained as the independent pre-aggregation baseline. The
public hard forward was unchanged, and short fixed-length and packed-varlen
backward retained the previous row schedules.

## 11. Post-Promotion Exact VJP Refinement

### 11.1 Retained local-gate aggregation

The first streaming kernel expanded every route-suffix credit through every
Q/K bit even when many suffixes referred to the same local `(Q,K)` match gate.
The exact derivative is linear: all scalar credits reaching one local gate can
be summed first, then that gate's Q/K sign contractions can be evaluated once.
The retained kernel accumulates by `(diagonal, query_coordinate)` in shared
memory and performs the unique contractions afterward. It still visits every
route and every suffix term.

Clearing and filling the aggregate is not worthwhile for tiny suffix tails.
An exact direct microtile is used when `suffix_count == 1`, or when
`suffix_count <= 4` and `suffix_count * D <= 16`. Partial aggregate chunks
restrict contraction to their reachable gate band; full 32-suffix chunks keep
the fully unrolled loop. This recovered the initial `W=1..4` regressions and
also improved tail chunks at `W=33/65`.

Shared-memory lifetimes were then shortened: the route-probability buffer is
reused for gate credit after value consumption, and the unused dedicated K
workspace was removed. The 32-row layout now uses `40.50 KiB`, independent of
`D`, so production uses it throughout `D=1..32`. On sm_75 it uses 64 registers,
one barrier, and zero stack or register spill. The 16-row benchmark control
uses 96 registers and is no longer selected by production.

Same-process alternating A/B against the independently compiled first
streaming source, on the idle RTX 2080 Ti with FP32
`B=1,H=4,Hv=2,D=8,Dv=64,W=32,QKV`, gave:

| `T` | First streaming | Current | Ratio | Speedup | Max VJP difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `35.44 ms` | `17.56 ms` | `0.496` | `2.02x` | `2.98e-7` |
| 8192 | `134.30 ms` | `67.17 ms` | `0.500` | `2.00x` | `2.38e-7` |
| 16384 | `528.01 ms` | `265.46 ms` | `0.503` | `1.99x` | `4.77e-7` |

For `D=32`, the 32-row layout improved Q, K, and QKV by about `12%..20%`
relative to the 16-row control. Value-only was about `4%..5%` slower, within
the worst-case promotion bound; the complete mask matrix has a clear mean win
and removes one production branch.

### 11.2 Rejected refinements

Each candidate below passed focused parity before timing and was then removed:

- An exact rolling diagonal suffix recurrence reduced arithmetic but exposed
  only 63 serially dependent diagonal workers. For `W <= 32`, QKV regressed
  `15%..32%` and value-only regressed `66%..82%`.
- A first-pass online value carrier avoided repeated `grad_output` reads. It
  gained only `2%..5%` at `Dv=128`, was neutral at `Dv=32/64`, and regressed
  `Dv=8` by `6%..9%`.
- V-only, QK-only, and QK+V template classes gained `2%..4%` on Q/K-only,
  changed QKV by at most `1%`, and grew the CUDA extension from about
  `907 KiB` to `1.64 MiB`.
- Hoisting the row candidate correction had no measurable effect because NVCC
  already commoned it. A shared mismatch-gate LUT produced only
  `0.4%..1.5%` paired median gains and regressed some `D=32` cases.

The final focused 28-test matrix and full suite (`1204 passed, 2 skipped`) pass.
Compute Sanitizer reports zero memcheck errors, zero race hazards or warnings,
and zero synccheck errors over direct and aggregate suffix paths, `D=32`,
dropout, and `W=1/4/33/65`. There is no startup timing, device allowlist,
model-visible switch, or candidate-dependent dispatch. See
[STREAMING_VJP.md](STREAMING_VJP.md) for the current schedule. Raw paired
timings and the six decisions are stored in
`validation/streaming_vjp_gate_aggregation_sm75.json`; the `D=32` tile control
is in `validation/streaming_vjp_optimized_sm75_d32.json`.

## 12. Tensor-Core Research Follow-Up

An sm_86 research extension tested B1/INT8/FP16 local gates, Tensor-Core
utility and value contractions, an exact coordinate-transformed gate-adjoint
GEMM, and a warp-cooperative suffix scan. Local gates, utility, dV, and the
integrated suffix scan did not pass end-to-end promotion gates. The only
retained research candidate maps the aggregated gate adjoint to FP16 hi/lo
HMMA for `D=32,W=31..32`; all other shapes return the frozen streaming
baseline. It adds no global scratch and averaged about 7% faster on GPU0 over
`T=512..8192`, but has not completed the required sm_75 cross-GPU matrix and
is not production dispatch. See [TENSOR_CORE_VJP.md](TENSOR_CORE_VJP.md).

## 13. Block-Diagonal Tensor-Core VJP

The next research prototype changes suffix ownership from independent route
candidates to `64 x 64` query/route blocks. Exact finite-window diagonal
recurrences remove duplicated suffix work, while TF32 WMMA handles utility,
value, and Q/K sign contractions. It still performs two complete causal-route
sweeps, stores no `T^2` state, and leaves the exact hard forward unchanged.

On the idle RTX 3070 at `W=32,Dv=64,T=4096/8192`, all seven gradient masks
improved. QKV speedups were `17%/23%` for `D=8` and `31%/34%` for `D=32`.
FP32/FP16/BF16, dropout, and random/all-match cases passed; Compute Sanitizer
reported zero memory, race, and synchronization errors after adding the
required zero-padded physical gate row.

Nsight Compute measured only `3.1%..3.9%` Tensor-pipe activity and about
`16.7%` active warps at `T=4096,QKV`. The retained gain therefore comes from
both suffix-work reuse and cheaper contractions; the kernel is still limited
by scalar recurrence/score work and one-CTA-per-SM residency, not by Tensor
Core throughput.

A warp-per-candidate suffix scan was rejected. Despite reducing one route's
loop depth, it replaced 32-way route parallelism with one route per warp and
was about `5x..16x` slower in the isolated matrix. See
[BLOCK_DIAGONAL_TENSOR_CORE_VJP.md](BLOCK_DIAGONAL_TENSOR_CORE_VJP.md) for the
recurrence, complete pipeline, shape limits, raw report names, and promotion
gate. This path remains outside production.

The sparse-tail follow-up confirms a narrow exception. When only one or two
queries remain active and no prior diagonal recurrence state exists, a warp
can scan their complete `W=32` histories in parallel and cut isolated score
latency roughly in half. Once the normal block sweep has already produced the
previous score and rolling mismatch count, one thread advances each endpoint
in constant work while the warp path recomputes 32 gates. The full-VJP tail
integration was neutral (`0.9996` mean ratio) and raised registers from 160 to
168, so it was removed; the exact primitive remains for future genuinely
sparse schedules.

Three occupancy/scheduling follow-ups were then tested. A `32 x 32,Dv=64`
physical tile lost all 28 long-sequence mask cases by about `4%..25%` despite
some two-CTA configurations. Explicitly sequencing the TF32 high/low fragments
did not lower registers. The surviving research plan instead gives producer
warps and consumer warps separate jobs in the first route sweep and exchanges
two exact `64 x 64` score tiles through CUDA named barriers. This
`block_tf32_pipeline` plan averaged a `0.961` latency ratio over
`T=4096/8192,D=8/32` and all masks, with a `1.041` worst case. A 36-case
dtype/dropout/pattern matrix averaged `0.965`; all memory, race, and
synchronization sanitizer checks passed. Its 85.1 KiB shared layout and
sub-5% mean gain keep it outside production.

## 14. Unlimited Hard Index: Parallel Build And Restart Batching

The unlimited hard path had two independent long-sequence bottlenecks. One
CTA per series built each stable occurrence list serially, and periodic
candidate trajectories repeatedly restarted at the same route delta. The
promoted implementation addresses both without bounding suffix length or
removing a candidate:

- at `T >= 8192`, contiguous chunks use count, prefix, and scatter kernels to
  reproduce the exact serial occurrence order in parallel;
- a device-side selector nominates at most one frequent restart delta, an
  exact diagonal mismatch-prefix index reconstructs its suffix length, and
  the original occurrence scan remains the fallback;
- descending routes stop only at the proof `route <= best_length`, where no
  remaining route can beat the score or latest-route tie.

On the RTX 3070 with `B=1,H=4,Hv=2,D=8,Dv=64,FP16,T=65536`, random latency
fell from `2.166 ms` to `1.273 ms`; `periodic64` fell from `9.045 ms` to
`1.076 ms`. The final `periodic16..512` sweep measured `0.942..1.094 ms`
against a `1.255 ms` random run. Focused CPU parity passed all 32 cases and
Compute Sanitizer reported zero memory errors, synchronization errors, or
race hazards.

The detailed algorithm, stability and exactness proofs, workspace accounting,
negative controls, and raw evidence are in
[EXACT_HARD_RESTART_AND_OCCURRENCE_INDEX.md](EXACT_HARD_RESTART_AND_OCCURRENCE_INDEX.md).

## 15. Unlimited Grouped-Checkpoint Cleanup

The production unlimited dense VJP received a final scalar/shared-memory
cleanup without changing candidate coverage, suffix extent, hard-forward
semantics, or estimator equations. The retained changes are:

- direct inverse-normalizer row state, eliminating one kernel and hot-path
  division;
- a padded 32x33 score tile to remove warp endpoint bank conflicts;
- `half2` gradient/value movement and 16-byte probability clears;
- one persistent gradient tile with value/probability lifetime overlay;
- packed-query and row-stat hoisting outside diagonal rounds;
- register reuse of each replay gate by the reverse affine scan.

On the idle RTX 2080 Ti with FP16
`B=1,H=4,Hv=2,D=8,Dv=64`, full Q/K/V latency changed from
`9.43/37.97 ms` at 4K/8K to `7.59/30.57 ms`, about a 20% reduction. The final
stats/reverse kernels use 64/86 registers and 24,836 bytes shared memory with
zero spills. Operator workspace remains 24.2 MiB at 4K, 48.4 MiB at 8K, and
64.8 MiB at 16K.

Complete hard-forward-plus-backward latency is 7.85/31.28 ms at 4K/8K. The
same-process xFormers CUTLASS D64 control is 1.70/5.41 ms, leaving a
4.63x/5.78x gap on sm_75. This GPU cannot execute the current PyTorch
FlashAttention backend, so those figures must not be relabeled as FA1.

The later locally compiled official FlashAttention 1.0.9 control measures
0.874/3.280 ms at 4K/8K and 13.167 ms at 16K. Complete RosaSoft is therefore
9.01x/9.50x/9.85x FA1, while exact hard-only forward remains 1.4-1.6x faster
than FA1 forward. This supersedes CUTLASS as the sm75 FlashAttention baseline;
see [FLASH_ATTENTION1_SM75_COMPARISON.md](FLASH_ATTENTION1_SM75_COMPARISON.md).

Several plausible occupancy and traffic optimizations were rejected after
exact parity and timing: compact three-CTA staging, reciprocal-square-root
utility, warp-owner symbol reduction, full score register caching, and global
dV carry. Their common failure was replacing regular on-chip work with extra
unpacking, shuffles, register pressure, or global scratch. The complete suite
passes with `3679 passed, 589 skipped`; Compute Sanitizer reports zero memory,
race, and synchronization findings. Full execution details and raw measurements
are in [GROUPED_CHECKPOINT_VJP.md](GROUPED_CHECKPOINT_VJP.md) and
`validation/grouped_checkpoint_vjp_sm75.json`.

## 16. Dense Backward Reorganization

The next production pass retained the same 32-diagonal recurrence ownership
and reorganized the work after exact reverse replay. The route-credit tile is
now contracted with binary Q/K signs through compensated FP16 WMMA where that
mapping is profitable. Credits are split into FP16 high and residual parts,
the signs are represented exactly, and both products accumulate into FP32.
For K-bearing optimized instances, four warps replay suffix scores while four
warps stage `dO` and V; all eight then converge for utility, reverse, dV, and
symbol credit.

The private production matrix uses Tensor-K at D8 and Tensor-QK at D16/D32
when both gradients are needed. Narrower symbols and unprofitable one-sided
cases retain scalar accumulation. There are no new public knobs, candidates,
losses, windows, or quadratic activations.

On the idle RTX 2080 Ti at `B=1,H=4,Hv=2,D=8,Dv=64,FP16,QKV`, production
latency moved from `7.57/30.32/128.26 ms` to
`7.21/28.81/121.40 ms` at 4K/8K/16K. Reverse alone improved from
`5.22/21.01/85.35 ms` to `4.86/19.45/78.25 ms`. The compensated contractions
pass the explicit scalar-oracle production matrix at `rtol=3e-4` and
`atol=5e-5`.

Three exact reorganizations were measured and rejected:

- 64/96-diagonal stats macro tiles reduced partial-stat storage but made 8K
  stats 19%/30% slower;
- byte mismatch checkpoints reduced shared memory without changing occupancy
  and made reverse 0.4-0.8% slower;
- a four-scan/four-loader stats split was increasingly slower with sequence
  length because scan, rather than input staging, became the critical path.

The post-change official FA1 ratios are 8.59x/8.92x/9.30x at 4K/8K/16K.
This is a measurable reduction but does not alter the architectural result:
exact suffix recurrence remains the dominant scalar/synchronization pipeline.
Raw evidence is in `validation/dense_backward_reorganized_production_sm75.json`,
`validation/reorganized_reverse_width_ablation_sm75.json`,
`validation/macro_stats_ablation_sm75.json`,
`validation/replay_mismatch_compaction_sm75.json`,
`validation/stats_warp_specialization_sm75.json`, and
`validation/flash_attention1_comparison_sm75_reorganized.json`.
