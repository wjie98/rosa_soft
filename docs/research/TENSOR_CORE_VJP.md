# FlashROSA Tensor-Core VJP Study

Status: research-only. The frozen production operator and package build are
unchanged. Measurements below use GPU0, an RTX 3070 (`sm_86`), with CUDA 12.8
and PyTorch 2.8.0.

## Question

The study tests whether the exact dense RosaSoft estimator can use Tensor
Cores without changing its support or storing a quadratic score matrix. The
semantic constraints remain:

- hard forward is unchanged;
- backward visits every causal route;
- score and softmax statistics remain FP32;
- no candidate pruning, sampling, low-rank suffix approximation, or `T^2`
  persistent state is introduced.

The implementation is a separate two-pass research VJP. It recomputes exact
route scores like the production streaming kernel and reuses the same roughly
40.5 KiB dynamic shared-memory allocation.

## 1. Local Match Gate

Four exact local-gate implementations include their layout-conversion cost:

| Method | Arithmetic | Shared scratch | `63x63,D=8` kernel |
|---|---|---:|---:|
| scalar | XOR + POPC + EXP | 0 | 1.23 us |
| FP16 | sign expansion + HMMA + EXP | 2,048 B | 2.80 us |
| INT8 | sign expansion + IMMA + EXP | 1,536 B | 2.58 us |
| B1 | packed XOR-POPC BMMA + EXP | 512 B | 1.65 us |

All methods are bit-exact against scalar for `D=1,2,4,8,16,31,32`, including
partial tiles. SASS inspection confirms real `HMMA`, `IMMA`, and
`BMMA.88128.XOR.POPC` instructions. Scalar still wins because one packed
32-bit XOR/POPC already computes the complete Hamming distance and every path
must still evaluate EXP. Tensor Core layout work cannot be amortized on this
tile.

Decision: keep scalar local gates.

## 2. Utility And Value Contractions

Isolated batched tiles show real Tensor Core throughput. Representative pure
kernel results with 256 independent tiles are:

| Contraction | Scalar | FP16 | TF32 | FP16 hi/lo |
|---|---:|---:|---:|---:|
| utility, `32x64x32` | 32.1 us | 20.6 us | 23.1 us | 24.2 us |
| utility, `32x128x32` | 61.6 us | 36.5 us | 44.0 us | 69.6 us |
| dV, `32x32x64` | 31.5 us | 19.3 us | 21.1 us | 22.2 us |
| dV, `32x32x128` | 60.6 us | 35.8 us | 40.2 us | 42.5 us |

FP16 and TF32 have about `2.9e-4` relative L2 error. FP16 hi/lo reduces this
to roughly `3e-7..8e-7`. Block-scaled FP16 protects range but does not improve
mantissa accuracy and loses much of the speedup.

The isolated result does not survive fusion. Production already assigns 32
warps to 32 rows; the WMMA form uses four warps for four `16x16` output tiles,
then pays conversion and CTA barriers before row-owned consumers resume. In
paired integrated tests over `Dv=32,64,128`, Tensor Core utility averaged
`1.01x` baseline latency and Tensor Core dV averaged `1.02x`. Neither reaches
the required 5% win, and individual shapes regress by more than 5%.

Decision: reject Tensor Core utility and dV for this 32-route schedule.

## 3. Gate-Adjoint Contraction

The local gate adjoint is stored by `(diagonal, query_index)`. A valid GEMM
requires an exact coordinate transform, not the simplified contraction used
by an early microbenchmark:

```text
A[q, k] = local_adjoint[q - k + 31, q]
dQ[q, d] = 0.5 * sum_k A[q, k] * sign(K[k, d])
dK[k, d] = 0.5 * sum_q A[q, k] * sign(Q[q, d])
```

The research kernel constructs each `16x16` A tile directly from the diagonal
layout, without materializing another matrix. The sign operand is exact FP16
`+1/-1`. A uses FP16 hi/lo, so each K tile needs two HMMA operations. Eight
warps cover all eight output tiles when `D=32`.

Unconditional block scaling was robust but erased the speedup. The retained
research path uses a uniform warp overflow vote. Normal tiles take the direct
hi/lo path; only a tile containing `abs(A)>60000` computes a scale and replays
its A conversion. Tests with upstream scales `1e-6` and `1e6` pass parity.

The width threshold is strict:

- `D=8`: scalar is better because WMMA pads eight columns to sixteen;
- `D=16`: measured regressions remain;
- `D=32`: gate-adjoint consistently benefits from all output tiles being
  occupied.

Window length also controls density of the local adjoint. `W<=16` does not
amortize the tiled contraction. The selected research dispatch is therefore
only `D=32 && W in {31,32}`; every other shape calls the frozen baseline.

For `T=512..8192`, `W=32`, `D=32`, `Dv=64`, and all Q/K-containing gradient
masks, the selected path averages `0.927x` baseline latency. QK and QKV cases
run at about `0.88x..0.90x`; K+V is the weakest case at roughly
`0.95x..0.96x`. It remains faster for random/all-match symbols,
dropout 0/0.2, and FP32/FP16/BF16 inputs. No global scratch is added. The
gate-only specialization uses 64 registers and zero local stack.

Decision: retain as a narrowly gated research candidate, not production.

## 4. Warp Suffix Scan

The exact warp scan composes affine recurrences over at most 32 positions and
uses no recursive algorithm. Standalone sequences validate to about `1e-7`
relative L2 error. At `W=32,L=32`, the isolated scan is about `4.2x` faster
than one thread scanning each suffix.

That result also fails after fusion. Production maps one warp to a row and its
32 lanes to route candidates, so all 1,024 route cells progress concurrently.
The diagonal scan remaps warps to 63 diagonals, needs two waves, performs
multiple shuffle scans, and then returns data to row ownership. Paired
`T=4096` tests show:

| W | Integrated warp/direct ratio, D=8 | D=32 |
|---:|---:|---:|
| 8 | 1.61x | 1.51x |
| 16 | 1.31x | 1.31x |
| 31 | 1.14x | 1.13x |
| 32 | 1.17x | 1.13x |

Decision: reject the warp suffix scan for the current 32-row/32-route CTA.
It remains a useful primitive only for a future layout whose natural owner is
a diagonal.

## 5. Fused Result

The first all-Tensor-Core plus warp-scan VJP was numerically correct but
averaged `1.217x` baseline latency over `T=512..8192` and all gradient masks.
Three integration bugs were found and fixed during validation:

1. The frozen shared layout is four-byte packed, so WMMA loads/stores required
   an aligned subregion inside the existing gate scratch.
2. Q/K scratch clearing raced scalar dV readers until an explicit CTA lifetime
   barrier was added.
3. In the rejected combined Tensor-Core plan, dV warps could overwrite the
   shared Tensor-Core output while row warps still consumed utility; a
   compile-time-only barrier now separates those two lifetimes.

The useful result is not a fully Tensor-Core RosaSoft kernel. It is the same
two-pass streaming VJP with only the dense D32 gate-adjoint contraction mapped
to HMMA. Exact local Hamming gates, suffix score construction, online softmax,
utility, dV, and all unsupported shapes remain on the production equations and
scalar execution mapping.

## Reproduction

Key files:

- `benchmarks/tensor_core_vjp.py`: isolated primitives;
- `benchmarks/csrc/rosa_soft_tensor_core_kernels.cu`: scalar/WMMA/IMMA/BMMA
  primitives;
- `benchmarks/flash_tc_vjp.py`: two-pass execution-plan wrapper;
- `benchmarks/csrc/rosa_soft_flash_tc_kernels.cu`: integrated research VJP;
- `benchmarks/tensor_core_vjp_profile.py`: primitive profiler;
- `benchmarks/flash_tc_vjp_profile.py`: paired end-to-end profiler;
- `benchmarks/flash_tc_sanitizer_smoke.py`: focused Compute Sanitizer workload;
- `tests/test_tensor_core_vjp.py` and `tests/test_flash_tc_vjp.py`: parity,
  boundary, dtype, dropout, exact-match, and dynamic-range coverage.

Machine-readable reports are under `validation/tensor_core_primitives_sm86.json`
and `validation/flash_tc_*_sm86.json`. End-to-end timings use interleaved
candidate/baseline measurements to reduce GPU boost-clock drift.

The focused primitive and integrated suites pass `262` tests. The complete
repository passes `1647` tests with `2` skips. On the smoke workload, Compute
Sanitizer reports zero memcheck errors, zero racecheck hazards or warnings,
and zero synccheck errors.

Before production promotion, the D32 gate-only path still requires an sm75
implementation/build and the repository's cross-GPU promotion matrix. The
current evidence is GPU0/sm86 research evidence only.
