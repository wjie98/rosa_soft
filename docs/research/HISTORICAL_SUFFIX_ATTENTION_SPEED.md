# Frozen Row VJP versus Historical Fixed-Window SUFA

This record compares the pre-streaming frozen RosaSoft row schedule with the
historical fixed-window suffix-attention surrogate from commit `c776101`.
Measurements were taken on 2026-08-06. The purpose is to separate estimator
cost from old runtime cost; the two implementations do not define the same
gradient.

The long-sequence tables through section 7 are intentionally retained as the
optimization baseline, not as current production timing. The exact
tiled-streaming VJP was subsequently promoted for fixed-length `T >= 4096`.
Its first checkpoint measured `35.56/134.84/529.14 ms` at
`T=4096/8192/16384`. Exact local-gate adjoint aggregation later reduced those
times to `17.56/67.17/265.46 ms`, leaving only a `1.025x..1.044x` latency ratio
to fixed SUFA on this shape. See [STREAMING_VJP.md](STREAMING_VJP.md).

## 1. Compared algorithms

The historical `suffix_attention_proxy` does the following:

1. softsign-STE quantizes Q, K, and V to hard `-1/+1` values;
2. unfolds the previous `W` symbols into a `D * W` feature;
3. applies a geometric decay with default factor `0.5`;
4. calls causal PyTorch SDPA once;
5. mixes in the hard route value through an `endpos`-conditioned gate.

It requires `D * W <= 256`. It represents one fixed, decayed suffix
fingerprint. It does not reproduce current RosaSoft's per-length suffix
competition, normalized Hamming mismatch, score transform, null route,
candidate-count calibration, attention dropout, or masked VJP paths.

Current RosaSoft runs exact hard routing in forward and a custom dense CUDA VJP
in backward. It retains every causal route and every suffix contribution up to
`W`. The production CUDA implementation supports `D in [1, 32]` and arbitrary
positive `W`; the deployment runtime remains the packed `D in [1, 8]` subset.

## 2. Measurement contract

Hardware and software:

- NVIDIA GeForce RTX 2080 Ti, `sm_75`, isolated as physical GPU 1;
- PyTorch `2.11.0+cu128`, CUDA `12.8`;
- row-baseline source commit `c1ccc0f`;
- Intel Core i9-10850K host, four exposed physical cores/eight threads;
- exact historical runtime measurements used `OMP_NUM_THREADS=4`.

The default shape is `B=1, Hq=4, Hv=2, D=8, Dv=64, W=32`, dense QKV
backward, random FP32 inputs, current `scale=1`, `mismatch_scale=3`, and
`dropout_p=0`. Results are medians of alternating-order, warmed A/B rounds.
Short cases used 30 warmups and 100 timed iterations; long cases used fewer,
longer iterations. CUDA event timing is used for the normalized A/B. Peak GPU
memory is incremental allocated memory for one training step.

Two historical columns are intentionally reported:

- **Normalized SUFA** uses the current exact CUDA hard forward for both paths,
  with route `endpos` prepared before timing. It isolates backward estimators.
- **Original full** checks out `c776101` and runs its original CPU-SAM hard
  forward plus the fixed SUFA backward. Timing is synchronized wall time.

The old checkout needed build-only compatibility fixes: an explicit
`<stdexcept>` include, excluding unrelated RWKV CUDA sources from a CPU-only
build, and skipping the optional FLA scan import. No executed SAM or SUFA logic
was changed. Therefore this is the historical algorithm on the current
PyTorch/CUDA stack, not a reconstruction of old driver or library performance.

PyTorch profiler identified the historical SDPA backend as
`_scaled_dot_product_efficient_attention` on this GPU for both FP16 and FP32.

## 3. Sequence scaling

Times are milliseconds per complete forward plus backward. `Current/original`
below one means current RosaSoft is faster; above one means original SUFA is
faster. GPU memory compares the normalized paths and excludes old host-SAM
storage.

| T | Current | Normalized SUFA | Original full | Current/original | Current hard | Current MiB | SUFA MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.379 | 3.133 | 4.894 | 0.08x | 0.031 | 0.12 | 2.24 |
| 128 | 0.385 | 3.127 | 4.994 | 0.08x | 0.032 | 0.24 | 4.47 |
| 256 | 0.694 | 3.182 | 5.069 | 0.14x | 0.033 | 0.48 | 8.95 |
| 512 | 2.078 | 3.153 | 5.331 | 0.39x | 0.032 | 0.95 | 17.89 |
| 1024 | 7.594 | 3.217 | 6.305 | 1.20x | 0.032 | 1.91 | 35.79 |
| 2048 | 30.566 | 6.792 | 11.035 | 2.77x | 0.060 | 3.81 | 71.57 |
| 4096 | 100.126 | 17.155 | 23.055 | 4.34x | 0.152 | 7.62 | 143.14 |
| 8192 | 388.569 | 65.722 | 76.696 | 5.07x | 0.483 | 15.25 | 286.28 |
| 16384 | 1529.401 | 253.778 | 282.288 | 5.42x | 1.721 | 30.50 | 572.56 |

The end-to-end crossover is between 512 and 1024 tokens for this shape. At
short lengths, the old Python composition, nested autograd, CPU SAM, and many
small launches dominate. At long lengths, memory-efficient SDPA is about six
times faster than the current custom dense VJP, before including old hard
runtime cost.

Both training algorithms remain quadratic in `T`. Historical SUFA has much
better long-sequence throughput but about `18.8x` the incremental GPU memory at
the default shape. Current hard forward is below `0.2%` of the 16K training
step, so further hard-forward tuning cannot materially improve training speed.

## 4. Suffix window

This sweep holds `T=1024, D=8, Dv=64` fixed.

| W | Current ms | SUFA ms | Current/SUFA |
| ---: | ---: | ---: | ---: |
| 1 | 1.894 | 3.121 | 0.61x |
| 2 | 2.060 | 3.173 | 0.65x |
| 4 | 2.452 | 3.164 | 0.77x |
| 8 | 3.223 | 3.197 | 1.01x |
| 16 | 4.729 | 3.193 | 1.48x |
| 32 | 7.678 | 3.221 | 2.38x |
| 64 | 12.835 | unsupported | n/a |
| 128 | 26.577 | unsupported | n/a |

At `T=4096`, historical SUFA was already `4.78x` faster at `W=1` and `5.83x`
faster at `W=32`. Current time contains a large route-normalization/value term
plus a roughly linear suffix term. Optimizing only the suffix scan cannot close
the long-sequence gap.

## 5. Symbol and value dimensions

All cells use `T=1024, W=32`; each pair is `current / normalized SUFA` in ms.

| D | Time pair | Current/SUFA |
| ---: | ---: | ---: |
| 1 | 2.931 / 3.803 | 0.77x |
| 2 | 3.632 / 3.332 | 1.09x |
| 4 | 4.973 / 3.135 | 1.59x |
| 8 | 7.649 / 3.394 | 2.25x |

| Dv | Time pair | Current/SUFA |
| ---: | ---: | ---: |
| 8 | 5.568 / 3.171 | 1.76x |
| 32 | 7.432 / 3.296 | 2.25x |
| 64 | 7.629 / 3.361 | 2.27x |
| 128 | 8.475 / 3.421 | 2.48x |
| 256 | 10.268 / 3.742 | 2.74x |

Current suffix work scales visibly with the number of discrete symbol bits.
The efficient-attention backend absorbs these small head dimensions better.
Both paths scale with `Dv`, but current mixed QKV utility/value work grows more.

## 6. Parallel workload and requested gradients

The `T=1024, W=32, D=8, Dv=64` head and batch sweeps were:

| Setting | Current ms | SUFA ms | Current/SUFA |
| --- | ---: | ---: | ---: |
| `B=1, H=1` | 2.827 | 3.122 | 0.91x |
| `B=1, H=2` | 3.959 | 3.204 | 1.24x |
| `B=1, H=4` | 7.558 | 3.308 | 2.28x |
| `B=1, H=8` | 14.770 | 4.212 | 3.51x |
| `B=2, H=4` | 14.835 | 4.235 | 3.50x |
| `B=4, H=4` | 29.240 | 6.990 | 4.18x |

Historical SDPA gains substantially from extra batch/head parallelism. Current
time is close to linear in total query heads for this range.

The old autograd function always computes Q, K, and V gradients. Current
RosaSoft dispatches only requested gradients:

| Requested | Current ms | Historical ms | Current/Historical |
| --- | ---: | ---: | ---: |
| Q | 3.346 | 3.622 | 0.92x |
| K | 3.218 | 3.437 | 0.94x |
| V | 0.756 | 3.490 | 0.22x |
| QK | 6.996 | 3.344 | 2.09x |
| QKV | 7.557 | 3.459 | 2.18x |

At `T=4096`, V-only remained faster (`12.98` versus `17.20 ms`), while Q-only
was slower (`78.08` versus `17.17 ms`). Candidate credit assignment, not value
accumulation, is the remaining long-context bottleneck.

## 7. Dtype and input pattern

FP16 brought no speedup on this SM75 shape. At `T=1024`, current FP32/FP16 was
`7.525/7.564 ms`, and SUFA was `3.369/3.356 ms`. At `T=8192`, the pairs were
`385.521/392.369 ms` and `65.385/65.905 ms`. Current VJP accumulates in FP32;
the selected historical efficient-attention backend also had nearly identical
timing. This result must not be extrapolated to Ampere or Hopper.

The original unlimited CPU SAM updates `endpos` along suffix links. Repetitive
codes expose a historical worst case that random inputs hide:

| Pattern: all match | Current full | Original full | Current hard | Original hard |
| ---: | ---: | ---: | ---: | ---: |
| T=1024 | 7.752 | 7.792 | 0.139 | 3.157 |
| T=4096 | 102.483 | 57.465 | 1.835 | 34.568 |
| T=8192 | 389.275 | 233.132 | 7.163 | 154.712 |

The current bounded hard kernel remains over `20x` faster, but historical SUFA
backward is fast enough that the old complete step still wins at 4K and 8K.

## 8. Validation

An independent fixture was generated by the original
`c776101:rosa_soft/ops_sufa.py` implementation. The current benchmark copy
matched its output and Q/K/V VJP exactly (`max_abs_error=0`) for a multi-head,
grouped-value, gated-route case. FP16 and FP32 smoke matrices covered
`W=1/8/32`; every output and gradient was finite.

## 9. Conclusions and Result

1. RosaSoft is the better short-sequence, small-window, low-bit, and V-only
   implementation. It is also dramatically more memory efficient.
2. Historical fixed SUFA is the faster long-sequence Q/K/QKV estimator because
   it reduces matching to one mature efficient-attention call. That speed does
   not establish gradient equivalence or fitting quality.
3. Hard forward is not the training bottleneck. Hard-route or isolated suffix
   micro-optimization could not recover the former `5x..6x` long-context gap.
4. The resulting production solution is an attention-style streaming VJP. It
   tiles every causal route, computes exact suffix scores and online
   normalization, and fuses route probability, expected utility, Q/K suffix
   credit, and value accumulation without a dense matrix.
5. The first schedule improved the exact estimator by about `2.8x..2.9x` at
   4K-16K. Exact local-gate adjoint aggregation then added another roughly
   `2x`, bringing the complete gain over the frozen row VJP to
   `5.7x..5.9x` and the gap to fixed SUFA down to `2.5%..4.4%` without
   candidate deletion.
6. A fixed-SUFA fallback could be retained only as a deliberately different
   research estimator. It must not silently replace current RosaSoft based on
   these timing results.
