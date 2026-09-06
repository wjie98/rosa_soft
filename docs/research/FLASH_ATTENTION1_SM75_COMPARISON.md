# FlashAttention 1 SM75 Comparison

## Purpose

This record establishes a real FlashAttention baseline for the RTX 2080 Ti.
The current PyTorch FlashAttention backend and FlashAttention 2 do not target
Turing, while the official FlashAttention repository directs Turing users to
the 1.x implementation. The previous xFormers CUTLASS control was useful but
was not FlashAttention 1.

## Installation

The environment is Python 3.12.13, PyTorch 2.11.0+cu128 with CXX11 ABI, CUDA
12.8, and compute capability 7.5. A binary-only PyPI request for
`flash-attn==1.0.9` returned no matching distribution, so the official PyPI
source archive was compiled locally.

The source was unchanged except for removing the setup entries for sm80 and
sm90. This limits code generation to the target GPU; no CUDA kernel, launch
parameter, numerical path, or Python operator was edited. The resulting wheel
is:

```text
flash_attn-1.0.9-cp312-cp312-linux_x86_64.whl
size   24,627,167 bytes
sha256 9fa0965ac85976f0ea7d2c66f3bf5ba6a9593a3898cf2088dfa2810dead20385
```

`cuobjdump --list-elf` confirms sm75 cubins for forward and backward head
dimensions 32, 64, and 128. The package is installed in the
`turing-attn-cu128` environment with `einops==0.8.2`; `pip check` reports no
broken requirements. As with old PyTorch CUDA extensions, `torch` must load
before directly importing `flash_attn_cuda`; the official Python interface
already has the correct order.

## Numerical Check

The installed QKV-packed causal operator was checked against an explicit FP32
matmul, causal mask, softmax, and value contraction at `T=127,H=4,D=64`:

| Quantity | Maximum absolute error | Mean absolute error |
|---|---:|---:|
| output | `9.94e-4` | `4.55e-5` |
| Q/K/V gradient | `1.43e-3` | `4.23e-5` |

All outputs and gradients are finite. Additional lengths 17 and 256 stayed
within `2.1e-3` maximum gradient error.

## Measurement Contract

- idle RTX 2080 Ti selected with `CUDA_VISIBLE_DEVICES=1`;
- FP16, causal, dropout disabled, five rotating-order rounds;
- five warmups and seven timed calls per round;
- complete forward plus Q/K/V backward, and a separate forward-only pass;
- projection and input-packing costs excluded for both operators;
- RosaSoft: `B=1,Hq=4,Hv=2,D=8,Dv=64`, exact unlimited hard forward and
  unlimited dense soft backward;
- FA1: standard unpadded QKV-packed `B*T,3,H=4,D=64` operator.

Both outputs have shape `B,T,4,64`, but this is not an equal-equation or
equal-input-width comparison. FA1 computes ordinary dot-product attention;
RosaSoft computes a discrete hard route in forward and an exact dense suffix
proxy VJP in backward.

## Results

| T | RosaSoft train | FA1 train | RosaSoft / FA1 | Rosa hard | FA1 forward |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 3.390 ms | 0.565 ms | 6.00x | 0.102 ms | 0.146 ms |
| 4,096 | 7.874 ms | 0.874 ms | 9.01x | 0.206 ms | 0.315 ms |
| 8,192 | 31.156 ms | 3.280 ms | 9.50x | 0.697 ms | 1.134 ms |
| 16,384 | 129.719 ms | 13.167 ms | 9.85x | 2.742 ms | 4.293 ms |

| T | RosaSoft peak operator memory | FA1 peak operator memory |
|---:|---:|---:|
| 2,048 | 6.41 MiB | 3.06 MiB |
| 4,096 | 24.83 MiB | 6.13 MiB |
| 8,192 | 49.66 MiB | 12.25 MiB |
| 16,384 | 67.31 MiB | 24.50 MiB |

A sustained-warmup layout control measured packed versus separate-input FA1
at 0.929/0.924 ms for 4K and 3.418/3.404 ms for 8K. The difference is below
1%, so the QKV-packed API is not responsible for the production gap.

The prior xFormers CUTLASS D64 control measured 1.70/5.41 ms at 4K/8K. FA1 is
about 1.9x/1.6x faster, so the earlier 4.6x/5.8x ratio underestimated the true
FlashAttention gap.

## Interpretation

The unlimited hard kernel is already competitive: it is 1.4-1.6x faster than
FA1 forward on random symbols. Nearly all ROSA training time is in the dense
proxy backward. At long lengths both measured training paths scale
quadratically in arithmetic, but FA1 keeps Tensor Core matrix operations and
online softmax pipelines dense, while ROSA carries scalar suffix recurrence,
utility transforms, replay barriers, and Q/K/V credit accumulation.

This result invalidates a near-term claim that local shared-memory or
occupancy tuning can bring the current estimator within 1.5-3x of
FlashAttention. Closing a 9-10x gap requires a more fundamental consumer
layout, architecture-specific asynchronous pipelines, or a model-level policy
that applies dense ROSA training to fewer heads/layers. Candidate pruning or a
finite suffix window remains outside the frozen estimator contract.

Raw samples and configuration are stored in
`validation/flash_attention1_comparison_sm75.json`. Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  python -m benchmarks.flash_attention1_comparison \
  --lengths 2048 4096 8192 16384 \
  --warmup 5 --repeats 7 --rounds 5 \
  --output validation/flash_attention1_comparison_sm75.json
```

## Dense-Backward Reorganization Follow-Up

After introducing compensated Tensor Core Q/K credit contraction and
intra-CTA replay/load specialization, the same installed FA1 package and
measurement contract produced:

| T | RosaSoft train | FA1 train | RosaSoft / FA1 | Rosa hard | FA1 forward |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 3.279 ms | 0.569 ms | 5.77x | 0.099 ms | 0.132 ms |
| 4,096 | 7.561 ms | 0.881 ms | 8.59x | 0.205 ms | 0.322 ms |
| 8,192 | 29.679 ms | 3.327 ms | 8.92x | 0.718 ms | 1.154 ms |
| 16,384 | 123.924 ms | 13.327 ms | 9.30x | 2.761 ms | 4.337 ms |

This supersedes the corresponding 4K/8K/16K rows above for current
production code. It narrows but does not change the architectural conclusion:
the unlimited dense proxy remains about 9x FA1 at long sequence lengths. Raw
samples are in
`validation/flash_attention1_comparison_sm75_reorganized.json` and the
separate 2K control
`validation/flash_attention1_comparison_sm75_reorganized_2k.json`.
