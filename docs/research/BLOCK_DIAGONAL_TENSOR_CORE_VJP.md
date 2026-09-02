# Block-Diagonal Tensor-Core VJP

This document records the block-diagonal RosaSoft VJP research and its later
production promotion. The original prototype was external to the frozen
operator. The ordinary TF32 plan is now an internal production schedule behind
the strict SM80+, `T>=4096,W=32,Dv=64` gate; the scalar and named-barrier plans
remain research-only. All variants keep the complete causal candidate set and
avoid an `O(T^2)` persistent workspace. Hard forward is unchanged.

## 1. Question

The streaming production VJP assigns route candidates to lanes and evaluates
each candidate's suffix loop independently. This is simple and exposes high
candidate parallelism, but neighboring `(query, route)` candidates repeatedly
evaluate most of the same local Q/K match gates.

Two alternative ownership rules were tested:

1. one warp owns one route candidate and maps its suffix offsets to lanes;
2. one CTA owns a `64 x 64` query/route block and maps equal-offset pairs to
   diagonals.

The first reduces the serial depth of one candidate. The second reduces the
total amount of repeated suffix work across candidates.

## 2. Exact Block Recurrence

Let `x_n` be the local soft match gate along one Q/K diagonal, and let `W=32`.
The finite suffix score is

```text
S_n = x_n + x_n x_(n-1) + ... + x_n ... x_(n-W+1).
```

It has the exact rolling recurrence

```text
S_n = x_n (1 + S_(n-1)) - P_n,
P_n = x_n x_(n-1) ... x_(n-W).
```

The implementation reconstructs `P_n` from a rolling integer Hamming-mismatch
count, so it does not divide by a possibly tiny gate. A `64 x 64` candidate
tile has only 127 diagonals. Each diagonal worker initializes its halo once and
then advances with constant work per endpoint. Every valid causal candidate is
still evaluated.

For `W=32`, a direct candidate-owned tile evaluates up to `64*64*32 = 131072`
suffix terms per pass. The diagonal recurrence advances at most 4096 tile
endpoints plus the 31-position halos. This is a reduction in duplicated work,
not candidate pruning.

## 3. Exact Two-Pass VJP

One 256-thread CTA owns 64 query rows and streams over 64-route tiles:

1. Generate exact raw suffix scores by diagonal recurrence.
2. Compute route utility `dO @ sign(V)^T`.
3. Update each row's online softmax maximum, normalizer, and expected utility.
4. Repeat the complete route sweep and reconstruct probabilities and raw-score
   adjoints.
5. Accumulate `dV = P^T @ dO`.
6. Reverse the finite suffix recurrence inside the block-local 95-position
   halo, using a 33-slot correction ring per diagonal.
7. Contract the resulting local-gate adjoints with Q/K signs.
8. Run the unchanged softsign-Jacobian finalization.

The two sweeps visit all causal routes. Global state is `O(BTD)` gradients and
row statistics only; no score, probability, or gate tensor of size `T^2` is
stored.

The physical local-gate tile is zero-padded to `96 x 96`. Its logical extent is
`95 x 95`, but WMMA loads full 16-row tiles. An earlier `95 x 96` allocation
made the final padded row alias the correction ring. Numerical tests happened
to stay within tolerance, but Compute Sanitizer exposed the race. The explicit
padding costs 384 bytes and is required for correctness.

## 4. Tensor-Core Work

The `block_tf32` plan uses Ampere TF32 WMMA for the regular contractions:

- route utility: `64 x 64` upstream gradient by binary value signs;
- value VJP: transposed route probabilities by upstream gradient;
- Q VJP: local-gate adjoint by key signs;
- K VJP: transposed local-gate adjoint by query signs.

Suffix popcount, rolling mismatch state, score transforms, exponentials, and
online softmax statistics remain scalar FP32. Tensor Cores accelerate the
matrix-shaped work; they do not accelerate the recurrence itself.

A single TF32 conversion of the gate-adjoint matrix was inaccurate on long
all-match diagonals. Q/K contraction therefore uses

```text
H = tf32(H) + tf32(H - tf32(H))
```

and two MMA operations. Binary sign matrices are exactly representable. This
high/low decomposition reduced the observed all-match Q/K relative error from
about `3.5e-3` to roughly `1e-4`. Utility and value VJP retain one MMA because
their measured error already passed the gate.

The current TF32 kernel uses about 69.1 KiB dynamic shared memory and 160
registers per thread on sm_86. Shared memory permits one CTA per SM, so the CTA
uses 256 threads (eight warps) to provide enough active warps for latency
hiding. SASS inspection confirms `HMMA.1684.F32.TF32` instructions. Unsupported
`Dv != 64`, `W != 32`, or pre-Ampere shapes return the frozen streaming VJP;
the scalar block plan remains an explicit benchmark control only.

This does not mean Tensor Cores are saturated. Nsight Compute 2025.1.1 on the
representative `T=4096,QKV` launch measured only `3.07%` Tensor-pipe activity
for `D=8` and `3.88%` for `D=32`; active warps were about `16.7%`, exactly the
eight-warps-per-SM residency limit. SM throughput was `14.3%..16.4%` and DRAM
throughput `9.2%..11.3%`. The exact scalar recurrence, score transforms, and
softmax functions dominate wall time. The value of WMMA here is reducing the
remaining contraction cost, not turning the whole operator into a GEMM. Raw
metrics are in `validation/block_diagonal_vjp_ncu_sm86.json`.

## 5. Why Warp-Per-Candidate Loses

For `D <= 32`, one lane can indeed load one packed Q/K symbol and compute its
entire Hamming mismatch with one XOR and popcount. A warp can then scan up to
32 suffix gates in about five shuffle stages. This shortens one candidate's
dependency chain, but it spends 32 lanes on one candidate. The baseline spends
the same warp on 32 independent candidates. Dense RosaSoft backward has
quadratically many candidates, so candidate throughput dominates.

On an idle RTX 3070, `B=1,H=4,D=8,T=4096` isolated score generation measured:

| `W` | Thread per candidate | Warp per candidate | Ratio |
| ---: | ---: | ---: | ---: |
| 32 | `2.034 ms` | `12.562 ms` | `6.18x` |
| 64 | `4.006 ms` | `22.268 ms` | `5.56x` |
| 128 | `7.549 ms` | `42.998 ms` | `5.70x` |

Across the complete isolated matrix the warp plan was about `5x..16x` slower.
It remains a negative control. It could make sense for one already-selected
hard route, but not for a dense all-candidate surrogate gradient.

### 5.1 Sparse causal-tail exception

The dense result does not imply that warp suffix ownership can never help.
An additional exact microbenchmark embeds only an `A x A` causal triangle in
a physical `64 x 64` score block. The physical-block control has no reusable
pre-tail recurrence state: every surviving diagonal must initialize its
`W=32` history. In that specific case, assigning all 32 suffix symbols of one
candidate to a warp helps when only one or two candidates per row survive.

| `D` | `A=1` | `A=2` | `A=4` | `A=8` | `A=16` | `A=32` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8, warp/candidate | `0.511` | `0.461` | `0.548` | `1.344` | `3.165` | `7.452` |
| 32, warp/candidate | `0.460` | `0.436` | `0.595` | `1.317` | `2.964` | `7.689` |
| 8, warp/diagonal | `0.429` | `0.493` | `0.411` | `0.578` | `0.621` | `0.897` |
| 32, warp/diagonal | `0.491` | `0.445` | `0.474` | `0.625` | `0.685` | `0.926` |

Entries are latency ratios to processing the sparse triangle through the full
physical block; lower is better. The warp/diagonal method is the transposed
form: every lane still evaluates one complete packed symbol, but a warp prefix
scan composes the affine recurrence and emits up to 32 diagonal endpoints.
This uses lanes much more efficiently than one warp per candidate. A compact
thread-only tail is an important second control: candidate-warp ownership beat
it consistently only at `D=32,A=1/2`, by roughly `12%..14%`. A low active count
alone is therefore insufficient to justify the warp path.

The exception was then tested under a stricter full-tile control. A normal
thread-owned diagonal advances through the first `64-A` route columns, and
only the final `A x A` triangle changes ownership. Here the previous score and
rolling mismatch are already available, so one thread needs only one `O(1)`
recurrence step per endpoint. A candidate warp instead recomputes 32 gates.
Paired measurements found no stable winning threshold: candidate-warp ratios
were about `0.99..1.01` at `A=1/2`, already reached `1.07..1.11` for
`A=4..16`, and exceeded `2.32` at `A=32`. Warp/diagonal ownership reduced the
isolated full-tile score stage by about `2%` at `A=32`, but that fringe is only
one route tile per query CTA.

For completeness, the warp/diagonal fringe was integrated into the full TF32
VJP and measured against the unchanged `block_tf32` plan over
`T=4096/8192`, `D=8/32`, and all seven gradient masks. The overall mean ratio
was `0.9996`; `T=4096,D=8` regressed by `1.42%` on average and the worst mask
regressed by `4.65%`. The added scan also raised register use from 160 to 168
per thread. The integrated branch was therefore removed. The exact isolated
implementations remain as research controls, not dispatch candidates.

Raw reports are
`validation/block_suffix_tail_ownership_saturated_sm86.json`,
`validation/block_suffix_hybrid_tail_paired_sm86.json`, and
`validation/block_diagonal_vjp_warp_tail_paired_sm86.json`.

## 6. Why Block Granularity Wins

The diagonal block plan preserves candidate-level occupancy while sharing
suffix history across neighboring candidates. On the same RTX 3070 score-only
case:

| `W` | Thread per candidate | Best diagonal tile | Ratio |
| ---: | ---: | ---: | ---: |
| 32 | `2.034 ms` | `1.335 ms` (`32`) | `0.657x` |
| 64 | `4.006 ms` | `1.448 ms` (`64`) | `0.361x` |
| 128 | `7.549 ms` | `1.631 ms` (`64`) | `0.216x` |

For `W=8/16`, initialization, halo, and synchronization are not amortized and
the thread-per-candidate plan remains faster. Tile 32 is narrowly best for the
isolated `W=32` score kernel, but tile 64 is retained for the integrated plan
because it aligns all utility, value, and Q/K contractions with a `64 x 64`
Tensor-Core macro tile.

### 6.1 Smaller physical tiles

The integrated source was parameterized so `32 x 32` tiles could be compiled
with 128, 192, or 256 threads without changing the estimator. This did raise
resident parallelism for selected builds. In particular,
`__launch_bounds__(256, 2)` reduced the 256-thread build to 128 registers and
allowed two CTAs, or 16 warps, to reside on one sm_86 SM. Passing
`--maxrregcount=128` alone did not do this: NVCC ignored the requested cap
under the original one-block launch bound. Occupancy claims must therefore be
checked from the compiled resource report, not inferred from the flag.

The extra residency did not compensate for twice as many query/route tile
boundaries. Against the `64 x 64` TF32 plan at `Dv=64`, all 28 combinations of
`T=4096/8192`, `D=8/32`, and seven gradient masks were slower by about
`4%..25%`. More halo setup, gate reverse passes, barriers, and global atomic
flushes dominate the saved per-CTA storage. A rectangular `32 x 64` tile was
not promoted to implementation: the current high/low contraction scratch
would exceed 50 KiB per CTA, retain one-CTA residency on sm_86, and still
double the query-side boundary work. It remains a lower-priority experiment,
not an assumed improvement.

The sequential high/low TF32 contraction was also tested. NVCC already reused
the fragment registers, so the explicit schedule stayed at 160 registers and
changed aggregate latency by less than one percent, with a `2.1%` worst
regression. That branch was removed.

### 6.2 Warp-specialized first-sweep pipeline

A more useful application of warp specialization keeps the `64 x 64` tile and
splits the eight resident warps by role during the first complete route sweep:

1. producer warps 0--3 generate the next exact suffix-score tile;
2. consumer warps 4--7 consume the current tile, run TF32 route utility, and
   update online softmax statistics;
3. two `64 x 64` shared score buffers alternate as ping and pong;
4. CUDA named barriers use asynchronous `bar.arrive` for ownership transfer
   and `bar.sync` only when a group must wait.

This is a double buffer, not candidate pruning or approximate prefetch. Every
causal score is generated by the same diagonal recurrence, and the second
route sweep, value VJP, gate reverse, and Q/K contractions are unchanged. The
standalone first-sweep benchmark is bitwise exact and improved all 12 tested
`D=8/32`, random/all-match cases by `15.6%..34.8%`.

The integrated `block_tf32_pipeline` plan uses about 85.1 KiB dynamic shared
memory, 158 registers per thread, and a 16-byte stack frame on sm_86, versus
69.1 KiB and 160 registers for `block_tf32`. Both remain limited to one CTA per
SM. Its paired results against `block_tf32` are:

| Matrix | Cases | Mean ratio | Worst ratio |
| --- | ---: | ---: | ---: |
| `T=4096/8192,D=8/32`, all masks | 28 | `0.961` | `1.041` |
| `T=512..4096,D=8/32`, masks 3/4/7 | 24 | `0.958` | `1.031` |
| dtype/dropout/pattern robustness | 36 | `0.965` | `1.037` |

All 14 `T=8192` cases won, by about `4.8%` on average. The robustness matrix
crosses FP32, FP16, BF16, dropout `0/0.2`, random/all-match symbols, and masks
QK/V/QKV; 29 of 36 cases won. These are stable research gains, but the mean
does not yet clear the repository's `>=5%` per-device promotion gate, and the
85 KiB layout has no sm_75 path. The plan is retained as an explicit research
control and is not production dispatch.

Raw reports are
`validation/block_pipeline_named_barrier_sm86.json`,
`validation/block_diagonal_vjp_pipeline_sm86.json`,
`validation/block_diagonal_vjp_pipeline_lengths_sm86.json`, and
`validation/block_diagonal_vjp_pipeline_robustness_sm86.json`.

## 7. Integrated Results

The canonical report is
`validation/block_diagonal_vjp_tf32_hilo_all_masks_sm86.json`. It uses
`B=1,H=4,Hv=2,Dv=64,W=32`, random FP32 symbols, all seven gradient masks, and
same-process alternating medians on an idle RTX 3070.

| Shape | Worst mask ratio | QKV ratio | QKV speedup |
| --- | ---: | ---: | ---: |
| `T=4096,D=8` | `0.883` | `0.829` | `17.1%` |
| `T=8192,D=8` | `0.859` | `0.772` | `22.8%` |
| `T=4096,D=32` | `0.795` | `0.693` | `30.7%` |
| `T=8192,D=32` | `0.756` | `0.664` | `33.6%` |

No mask in this long-sequence matrix regressed. Maximum random-input absolute
errors were about `1.8e-4` for Q/K and `8.9e-4` for V. The robustness report
`validation/block_diagonal_vjp_tf32_robustness_sm86.json` crosses FP32, FP16,
BF16, dropout `0/0.2`, and random/all-match symbols at `T=4096,D=8,QKV`; every
case was faster, with latency ratios `0.770..0.902`.

Focused CUDA tests cover all masks, `D=1/8/32`, boundary lengths around 32 and
64, random/all-match symbols, FP32/FP16/BF16, and dropout. The exact scalar
block plan matches the streaming oracle within reduction-order tolerance; the
TF32 plan uses a `3e-3` robustness bound. The research autograd wrapper returns
the exact production hard forward and uses the selected VJP only in backward.
Compute Sanitizer reports zero memcheck errors, zero race hazards, and zero
synccheck errors after physical gate padding.

The pipeline plan is covered by the same bounded-error and autograd tests.
CUDA 12.8 memcheck, racecheck, and synccheck each report zero errors for its
named-barrier ping/pong schedule. The complete focused totals are 271
block-VJP tests, 185 suffix-ownership tests, 12 standalone-pipeline tests, and
17 configurable-tile tests.

The isolated suffix-ownership smoke also reports zero race and synchronization
hazards. CUDA 12.8 memcheck emits a host-side `cuKernelGetFunction` invalid
handle while still executing this JIT module correctly; rerunning with
`--report-api-errors no` reports zero device-memory errors. This suppression is
limited to that isolated microbenchmark and is not used for the integrated VJP
checks.

## 8. Decision and Next Gate

The useful design is block-diagonal suffix recurrence plus Tensor-Core matrix
contractions. Warp-per-candidate suffix scans are rejected for dense backward;
their narrow exception is a tiny active tail with no reusable recurrence
state. Smaller physical tiles and the named-barrier double buffer were also
rejected for production: the former regressed every long case, while the
latter added 16 KiB shared memory for less than the required mean gain.

The ordinary TF32 plan was promoted on 2026-08-21. Production contains one
`64 x 64` kernel with no plan selector and falls back to tiled streaming for
every unsupported architecture or shape. Its source was reduced from the
research copy to 1198 lines by deleting scalar, pipeline, and generic-template
branches. At `T=4096`, 144 production-boundary outputs spanning all masks,
FP16/BF16/FP32, dropout `0/0.2`, and random/all-match patterns had worst
relative gradient error `5.01e-4`. A long single-bit hard-route fitting test
reached zero loss in step 7 under both production TF32 and exact streaming.

The end-to-end FP16 production times for `B=1,H=4,Hv=2,D=8,Dv=64,W=32` are
`0.84/2.30/7.04/20.20/72.48/280.89 ms` at
`T=512/1K/2K/4K/8K/16K`. These are respectively about
`2.9x/3.8x/4.7x/1.2x/1.3x/1.3x` faster than the pre-promotion operator and
faster than fixed SUFA at every measured length. The canonical integrated
report is `validation/attention_speed_production_v2_sm86.json`; dispatch
boundary reports are `validation/production_dispatch_v2_sm86_fp16.json` and
`validation/production_dispatch_v2_sm75_fp16.json`.

Promotion requires:

1. an sm_75 baseline comparison with an appropriate non-TF32 fallback and an
   Ada/Hopper matrix;
2. measured tile/occupancy choices for `Dv=32/128` without a public schedule
   ladder;
3. long-sequence fitting parity, not only pointwise VJP parity;
4. a production integration review proving that compile size and short-shape
   dispatch do not regress;
5. the existing all-mask `<=1.05` worst-case and `>=5%` mean-speedup gate on
   every promoted device family.
