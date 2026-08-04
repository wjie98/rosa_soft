# RosaSoft CUDA Optimization Record

This document records the 2026-07-30 through 2026-08-04 CUDA optimization
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
| 11 | Exact hard diagonal run-length index | Archived | Bit-exact and up to `12x` faster on collapsed all-match codes, but random codes regressed; removed from the frozen build. |
| 12 | Offline cross-GPU promotion gate | Keep | Rejects candidates with any material regression, parity failure, or excessive workspace. |

## 3. Rejected Local Changes

### 3.1 Mismatch LUT

The tested QKV times for `T=128/256/512/1024` were
`0.426/0.857/2.998/11.566 ms`, versus
`0.402/0.862/2.956/11.412 ms` without the LUT. The result changes sign across
shapes and does not justify a shared table.

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
`1.30x` and worst ratio `1.59x`. A future version must replace the dense gate
matrix with block-local diagonal rings and beat the row kernel before the
extra launches are justified. Its raw operation, benchmark, and default tests
were removed when `rosa-soft-dense-reference-v1` was frozen; the measurements
remain here as historical evidence.

## 8. Exact Hard Suffix Index

The former exact hard diagonal prototype assigned one warp to one Q/K
diagonal. For each 32-position group it computed the nearest preceding
mismatch with an inclusive prefix maximum; the exact suffix length is the
distance to that mismatch, capped by `W`. A 64-bit atomic maximum encodes
`(length, route)` so longest-first and latest-on-tie semantics are unchanged.

The prototype compared every causal Q/K endpoint pair once. It was not a
bounded candidate lookup, probabilistic hash, or sparse-gradient mechanism.
Tests and the calibration matrix covered random, all-match, periodic,
singleton, partial-warp, FP32, FP16, BF16, `W=1..300`, and `T=1..1024`;
output and both packed symbol tensors are bit-exact with the production scan.

Performance depended on hard-code entropy. On the 2080 Ti with
`T=4096,W=128`, all-match latency fell from about `6.69 ms` to `0.56 ms`, a
roughly `12x` speedup. Random codes made the indexed path about `1.1x..1.8x`
slower because the production route scan usually exits after its first symbol.
No input-independent `T/D/W` rule distinguished these states, so this path was
removed from the frozen build. The checkpoint codebook diagnostics in
`benchmarks/pretraining_codebook.py` provide the missing collapse/entropy
signal for future model-level dispatch work.

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

The production hard forward and row-owned surrogate VJP therefore remain the
defaults. Existing cache/recompute selection remains occupancy-aware on the
active GPU and uses the portable 48 KiB shared-memory bound; no startup timing,
device-name allowlist, or hidden environment override was added.
