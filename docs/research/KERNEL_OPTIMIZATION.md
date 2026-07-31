# RosaSoft CUDA Optimization Record

This document records the 2026-07-30 SM75 optimization pass. It distinguishes
production changes from rejected experiments so old branches and benchmark
artifacts do not become accidental design requirements.

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
| 7 | Global 64/128/256-thread CTA choice | Keep 128 | 64 and 256 have conflicting dense/packed wins; a ladder needs brittle heuristics. |
| 8 | Exact diagonal score and Q/K adjoint | Research only | Full symbol VJP validates; row staging and a production CUDA parity gate remain. |

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

### 4.1 Long Packed K Aggregation

For packed segments, route lanes write many K contributions to overlapping
positions. The retained path accumulates a
`[D, blockDim + W - 1]` FP32 tile in shared memory and flushes it once.

It activates only when:

```text
K gradient enabled
D <= 8
average segment length >= 256
local segment length >= 256
complete shared layout <= 48 KiB
```

The aggregate and direct paths are separate compile-time kernel variants. Host
selection requires average segment length at least 256; inside an aggregate
launch, a local segment shorter than 256 skips the tile accumulation but still
shares that launch's template and dynamic shared allocation. The idle-card
recheck on uniform segment lengths gave:

| Segment length | K-only speedup | QKV speedup |
| ---: | ---: | ---: |
| 128 | `1.01x` | `0.99x` |
| 256 | `2.10x` | `1.41x` |
| 512 | `1.88x` | `1.37x` |

The 128-token negative control is effectively unchanged, which supports the
256-token threshold. Dense K keeps its existing head-major accumulator.

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

For external route-score VJP `a_n`, define:

```text
b_n = a_n + x_(n+1) b_(n+1)
c_n = b_n p_n
h_m = b_m (s_m + p_m) - sum(c_n, n=m..m+W)
```

Then `h_m` is the VJP with respect to `log(x_m)`. The PyTorch research
implementation emulates 32-element affine scans with sequential group carries
and local-window reductions; it is not a CUDA warp kernel.

For the production symbol proxy,

```text
log(x) = -mismatch_scale/2
         + mismatch_scale/(2D) * sum(q_bit * k_bit)
```

so each local gate is owned once by its diagonal and contributes
`h * mismatch_scale/(2D) * k` to Q and the symmetric term to K. The research
prototype now performs these complete scatter-adds and verifies the final
unused key position receives exactly zero gradient.

Validation results:

| Matrix | Cases | Max score error | Max VJP error |
| --- | ---: | ---: | ---: |
| FP64, `N=1..65`, `W=1..128` | 100 | `6.66e-16` | `8.88e-16` |
| FP32, `N=257..4096`, `W=1..512` | 48 | `4.77e-7` | `7.15e-7` |

The 89-test research suite additionally covers full causal matrix score and
log-gate mapping, complete FP32/FP64 Q/K symbol VJPs, and exact,
one-mismatch, alternating, and random-Hamming gate patterns. Exact-match FP32
adjoints reached about `7.0e-4` maximum absolute error under unscaled random
route VJPs at `T=257,W=128`; with a real softmax-shaped route-score adjoint the
maximum fell below `3.1e-5`.

That short-context result does not extrapolate. A deterministic long
exact-match stress with random external score adjoints produced:

| `N` | `W` | Score error | VJP error |
| ---: | ---: | ---: | ---: |
| 4096 | 512 | `0` | `5.86e-2` |
| 4096 | 4096 | `0` | `2.76` |

The cancellation is in the reverse correction expression, not the affine
score scan. This is now a production blocker: a diagonal CUDA path needs a
numerically different FP32 adjoint formulation or compensated/higher-precision
state before end-to-end parity and performance work are meaningful.

## 7. Why It Is Not Yet a Production Kernel

The recurrence is diagonal, while softmax statistics, value utilities, and
value gradients are row-oriented. The validated component is the final
score-adjoint-to-Q/K stage, not a complete training kernel. A plausible
workspace-bounded production schedule is:

1. a row-owned pass stores only max, denominator, and expected utility in
   `O(BHT)` state;
2. value-only credit remains row/tile owned;
3. a diagonal-owned reverse pass reconstructs dropout, route utility, and
   route-score VJP from row stats, keeps a `W+1` correction ring, and applies
   each local Q/K bit VJP once.

That design still needs a numerically stable FP32 reverse correction, a
concrete CUDA implementation of finite-window score reconstruction, bounded
worker scratch, packed-varlen and grouped-head scheduling, and measured
launch/atomic costs. It must preserve full dense route support, avoid an
`O(BHT^2)` score/adjoint workspace, pass the long exact-match FP32 gate, and
beat the simpler row kernel at large `W` before it can replace production
code.
