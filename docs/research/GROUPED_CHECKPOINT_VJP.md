# Grouped Checkpoint VJP

## Status

Implemented on 2026-09-06 as an internal production specialization of
`rosa_soft_unbounded`. The implementation changes execution order and live
storage only. It preserves the frozen dense estimator:

- exact unlimited hard forward;
- softsign Q/K/value STE;
- `exp(-3 * mismatch_rate)` local gate by default;
- exact unlimited suffix DP and all causal candidates;
- square-root suffix utility, null score, `-log(N)`, and dense softmax;
- dense binary-value carrier and PyTorch-compatible attention dropout.

Unsupported shapes and small workloads continue to use the exact slab-replay
fallback. Packed varlen also retains that fallback.

## Recurrence

Use diagonal coordinates `delta = query_position - key_position`:

```text
S[q, delta] = g[q, delta] * (1 + S[q - 1, delta])
g[q, delta] = exp(-mismatch_scale * popcount(Q[q] xor K[q-delta]) / D)
```

For direct score adjoint `a[q,delta]`, reverse mode is another affine scan on
the same diagonal:

```text
A[q,delta] = a[q,delta] + g[q+1,delta] * A[q+1,delta]
d log(g[q,delta]) = S[q,delta] * A[q,delta]
```

Thus backward is not a fundamentally different graph traversal. It is the
same one-dimensional state recurrence in reverse. The engineering problem is
that `a[q,delta]` needs the final row softmax statistics, which couple all
diagonals at a query row. Row statistics must therefore finish before exact
reverse replay begins.

## Production Schedule

The public API exposes no scheduling parameters. The SM70+ FP16/Dv64 path uses:

```text
P = 32  warp scan width
C = 32  complete diagonals per logical task
L = 32  query rows per checkpoint/replay interval
G        maximum resident CTA grid reported by CUDA occupancy
```

On RTX 2080 Ti the scalar reverse kernel uses 256 threads, 86
registers/thread, and 24,836 bytes of shared memory. The optimized D8
Tensor-K instance uses 101 registers/thread and 29,060 bytes; the D16+ joint
Tensor-QK instance uses 100 registers/thread and 29,060 bytes. All three are
spill-free and permit two CTAs per SM, so `G=136` on its 68 SMs. The stats
kernel remains at 64 registers/thread and 24,836 bytes.

### Pass A: Exact Row Statistics

Pass A computes only the final tuple for every `(batch, head, query)` row:

```text
(row_max, inverse_row_normalizer, expected_dropout_utility)
```

It processes a bounded slab of complete diagonal tiles at a time:

1. eight warps scan 32 diagonals in four rounds;
2. each lane advances one full-symbol recurrence endpoint;
3. FP16 `Dv=64` utility products use WMMA with FP32 accumulation;
4. each 32-candidate tile computes max first, then one exponential per
   candidate and two sums;
5. partial tile statistics are merged into the exact online row tuple.

The automatic slab is capped at 4096 diagonals and selected under a 64 MiB
partial-statistics budget. At very large `B*H*T`, at least one 32-diagonal tile
is retained, so live storage is a bounded budget plus an `O(B*H*T)` floor.
Slab width is storage granularity, never a suffix horizon.

### Pass B: Checkpoint, Replay, and Reverse

A fixed occupancy-sized grid persists for the whole reverse pass. Each
physical CTA repeatedly takes a statically assigned pair of logical tasks;
each task owns 32 complete diagonals from origin to sequence end.

For one task the CTA:

1. skips row blocks that precede the first active cell;
2. scans its diagonals forward and saves only the incoming score state every
   32 rows in CTA-private global scratch;
3. traverses those intervals in reverse;
4. reloads one boundary and reconstructs the 32x32 score tile in shared
   memory;
5. obtains exact probabilities from Pass A row statistics;
6. runs the reverse affine scan and forms `d log(g)`;
7. contracts the route-credit tile with Q/K signs, using compensated Tensor
   Core contractions for profitable symbol widths and scalar FP32 reduction
   otherwise;
8. computes dV with WMMA, splitting probabilities into FP16 high/low parts so
   their sum retains the FP32 probability more accurately;
9. reuses the same checkpoint scratch for the next task.

For the optimized K-bearing instances, four producer warps replay the 32
suffix diagonals while four loader warps concurrently stage `dO` and binary
V. All eight warps then converge for utility, reverse scan, dV, and symbol
credit. This is intra-CTA role specialization; it adds no queue, progress
atomics, or inter-CTA dependency.

Checkpoint storage is

```text
G * ceil(T / 32) * 32 * sizeof(float) = O(G*T).
```

No score, probability, gate, or VJP matrix of size `T^2` is persistent. Work
remains `Theta(B*H*T^2)` because every candidate is still evaluated.

## Load Balance

Diagonal tasks have different lengths. A scheduling unit explicitly pairs
tile `d` with tile `last-d`, rather than relying on a folded ordinal that can
preserve parity and leave some CTAs with nearly twice the mean work. Even and
odd physical CTAs execute the two sides in opposite order, which slightly
dephases Q/K/V atomic updates.

This is a static persistent schedule: there is no global ready queue, no
progress spin loop, and no inter-CTA DP dependency. Complete-diagonal
ownership makes those synchronization mechanisms unnecessary in Pass B.

Packing complementary partial diagonals inside one warp remains deferred. It
can only recover the first and last partial 32-row segment of each complete
diagonal, an `O(T)` edge loss against `O(T^2)` useful work, so its long-sequence
benefit does not currently justify the additional reset logic.

## Compute-Density Changes

The production implementation includes all low-complexity exact changes that
survived validation:

1. A 33-entry mismatch-gate LUT changes the candidate hot path from an SFU
   exponential to `popcount + shared load`.
2. `log(row)` is precomputed once per row instead of inside the candidate
   loop.
3. Candidate softmax reduction uses max followed by one exponential per
   active candidate, rather than repeated pairwise exponential merges.
4. Stats score generation is warp-per-diagonal affine scan; no single warp
   serially fills the complete 32x32 tile.
5. Utility and dV use FP16 WMMA with FP32 accumulation.
6. Half and float shared-memory regions are phase-overlaid so their lifetimes
   do not add.
7. Q/K/V are reduced within a tile before global atomic accumulation.
8. Row statistics retain the inverse softmax normalizer directly. This removes
   a restore kernel and changes the reverse hot path from division to
   multiplication without changing the represented probability.
9. The score tile uses a 33-float stride. The padding removes the 32-way bank
   conflict produced when warp lanes write one endpoint for each query row.
10. FP16 gradient loads and binary-value conversion use aligned `half2`
    operations; the probability buffer is cleared with aligned 16-byte stores.
11. The output-gradient tile remains live across utility and dV, while value
    and probability storage overlay one another. This removes a second global
    gradient load and two unnecessary CTA barriers.
12. Row-invariant softmax state and packed query symbols are loaded outside the
    four diagonal rounds. The replayed mismatch gate remains in a register and
    is reused by reverse mode, avoiding a second Q/K load, XOR, popcount, and
    gate lookup.
13. The D8 K-credit contraction and D16/D32 Q/K contractions reshape the exact
    route-credit tile into WMMA products. Each FP32 credit is represented as
    FP16 high plus FP16 residual, while binary signs remain exact in FP16 and
    accumulation stays FP32.
14. Four suffix-producer warps overlap replay with four `dO`/V loader warps on
    the profitable K-bearing paths. The scalar path retains eight scan warps.

The following exact alternatives passed focused parity but lost the timing or
complexity gate and were removed:

- Compact value/probability staging plus `__launch_bounds__(256, 3)` reached
  three CTAs/SM, but extra unpacking and staging made 4K materially slower and
  did not improve 16K.
- Replacing `sqrtf` plus division with reciprocal square root increased
  register pressure and was about 1% slower on sm_75.
- Warp-owner Q reduction reduced symbol reads but serialized bit credit through
  shuffles; the 8K reverse pass regressed from about 21.35 ms to 25.17 ms.
- Caching all replay scores in registers raised reverse use to 96 registers and
  was 0.4-0.7% slower. Retaining only four mismatch gates is the useful point.
- Carrying dV partial sums through global CTA scratch halved some atomic adds,
  but added 8 KiB per CTA and regressed 8K/16K.
- Moving the 8 KiB utility/WMMA phase buffer to CTA-private L2 reduced shared
  memory, but register use rose and occupancy did not improve without spills.
- Expanding one stats task from 32 to 64/96 diagonals reduced partial-stat
  workspace but was 19%/30% slower at 8K because CTA parallelism fell and
  micro-tile barriers remained.
- Replacing the 4,224-byte replay-gate tile with byte mismatch codes reduced
  optimized shared memory to 25,892 bytes, but occupancy was still
  register-limited and the second LUT lookup made 4K/8K/16K about 0.4-0.8%
  slower.
- Splitting stats into four scan warps plus four loader warps lengthened its
  critical scan path. It was 1.9%/4.4%/5.0% slower at 4K/8K/16K and was
  removed.

The simpler shared-memory layout remains production. These results also show
that occupancy alone is not the objective: converting deterministic on-chip
work into unpacking, shuffles, or global scratch can lose despite a higher
resident-CTA count.

## Dispatch

Grouped checkpoint is selected only when all of these are true:

- fixed-length CUDA input;
- FP16 Q/K/value and `Dv=64`;
- Q or K gradient requested;
- `T >= 2048`;
- either `B*H >= 2` and `B*H*T >= 8192`, or `T >= 32768`.

All other cases use the exact unbounded replay implementation. This avoids a
known under-filled grouped-kernel loss for one series at 4K/8K and for two
series at 2K. The threshold is private and does not change estimator output.

Within the grouped reverse path, dispatch is also private:

- D1/D2/D4 use scalar Q/K credit;
- D8 uses Tensor-K when K gradients are requested and scalar Q otherwise;
- D16/D32 use joint Tensor-QK for joint gradients, Tensor-K for K-only, and
  Tensor-Q for Q-only.

This matrix is based on direct width/mask ablation. Padding D1/D2/D4 to a
16-wide Tensor Core tile loses more staging work than it saves.

## Validation

Configuration: idle RTX 2080 Ti, CUDA 12.8, FP16,
`B=1,H=4,Hv=2,D=8,Dv=64`, full Q/K/V VJP. Each entry is the median of five
timing rounds; every round averages ten repeats after five warmups.

| T | production auto | grouped direct | reverse only | peak operator memory | 4096-diagonal slab | slab memory |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 2.18 ms | 2.13 ms | 1.29 ms | 6.1 MiB | 2.90 ms | 129.6 MiB |
| 4K | 7.21 ms | 7.23 ms | 4.86 ms | 24.2 MiB | 8.03 ms | 259.2 MiB |
| 8K | 28.81 ms | 28.78 ms | 19.45 ms | 48.4 MiB | 30.79 ms | 518.4 MiB |
| 16K | 121.40 ms | 121.56 ms | 78.25 ms | 64.8 MiB | 124.69 ms | 1036.8 MiB |

Relative to the frozen pre-reorganization baseline, complete production VJP
improves by 4.7%/5.0%/5.4% at 4K/8K/16K, while reverse alone improves by
6.9%/7.4%/8.3%. At 16K the grouped path is now faster than the 4096-diagonal
slab while using 64.8 MiB instead of 1036.8 MiB. Raw samples are stored in
`validation/dense_backward_reorg_baseline_sm75.json` and
`validation/dense_backward_reorganized_production_sm75.json`.

The same idle-card, alternating-order check measured the complete hard forward
plus unlimited backward at 7.85/31.28 ms for 4K/8K. An output-width-matched
xFormers CUTLASS causal attention control (`H=4,D=64`) measured 1.70/5.41 ms,
so the remaining ratios are 4.63x/5.78x. Turing cannot run the current PyTorch
FlashAttention backend; this is a memory-efficient CUTLASS control, not an FA1
claim. ROSA also uses 8-wide Q/K and two value heads in this comparison, so the
ratio is an execution reference rather than equal-algorithm work efficiency.
Raw alternating samples are in
`validation/unbounded_vs_cutlass_sm75_optimized.json`.

An official FlashAttention 1.0.9 kernel was subsequently compiled for sm75.
It is materially faster than the CUTLASS control:

| T | complete RosaSoft | FA1 QKV-packed | RosaSoft / FA1 | Rosa hard forward | FA1 forward |
|---:|---:|---:|---:|---:|---:|
| 2K | 3.279 ms | 0.569 ms | 5.77x | 0.099 ms | 0.132 ms |
| 4K | 7.561 ms | 0.881 ms | 8.59x | 0.205 ms | 0.322 ms |
| 8K | 29.679 ms | 3.327 ms | 8.92x | 0.718 ms | 1.154 ms |
| 16K | 123.924 ms | 13.327 ms | 9.30x | 2.761 ms | 4.337 ms |

The exact hard forward is 1.3-1.6x faster than FA1 forward on these random
symbols; the dense proxy backward creates the complete training deficit.
Separate-input and QKV-packed FA1 differed by less than 1% after sustained
warmup at 4K/8K, so packing does not explain the gap. FA1 uses `H=4,D=64` for
Q/K/V, whereas ROSA uses `D=8` Q/K and two D64 value heads. The table therefore
matches output shape and causal training mode, not operator equations.

The installation, numerical check, wheel identity, and original control are in
[FLASH_ATTENTION1_SM75_COMPARISON.md](FLASH_ATTENTION1_SM75_COMPARISON.md).
Post-reorganization raw results are in
`validation/flash_attention1_comparison_sm75_reorganized.json`.

Validation gates completed:

- 374 focused production/grouped tests passed; one optional test was skipped;
- the complete repository suite passed with 3679 tests and 589
  hardware/optional skips;
- D1/D8/D16/D32, all gradient masks, dropout, grouped value heads, batches,
  repeated patterns, and repeated execution;
- Compute Sanitizer memcheck, synccheck, and racecheck with zero findings;
- a production-threshold D1/D8/D16/D32 and all-gradient-mask matrix against
  the explicit scalar contraction oracle, with `rtol=3e-4` and `atol=5e-5`.

The official FA1 result supersedes the earlier CUTLASS estimate: current
long-sequence training is about 9-10x FA1, not 5-6x. The grouped design solves
the quadratic activation-capacity problem and removes avoidable scalar work,
but the remaining suffix recurrence, nonlinear utility, shared-memory
barriers, and cross-tile gradient accumulation require architectural rather
than local occupancy tuning. A future Hopper/Blackwell specialization should
change the utility/gradient consumer layout before adding queues or more
checkpoints.
