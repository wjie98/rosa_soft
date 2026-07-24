# rosa_soft

Production-oriented PyTorch operators for ROSA research.

This release contains three public paths:

- `rosa_soft_reference`: pure PyTorch hard-forward/soft-backward correctness
  oracle.
- `rosa_soft`: CUDA implementation of the same training estimator.
- `RosaRuntime`: stateful hard ROSA inference runtime with explicit resource
  management and optional CUDA-to-CPU overlap.

## Highlights

- CUDA forward/backward for the RosaSoft soft training proxy.
- Exact hard-forward PyTorch reference on CPU and CUDA.
- Exact hard forward in both training paths; no soft value can leak forward.
- Explicit static `route_temperature` and `mismatch_penalty` controls.
- Counter-indexed mismatch sampling in CUDA: backward saves one seed instead of
  an `O(B H T^2 D)` random tensor.
- Detached reference diagnostics kept outside the training hot path.
- Hard runtime with packed `uint8` Q/K/V and FlashAttention-style `cu_seqlens`.
- Bounded compact suffix automaton with the same finite suffix horizon as
  training and `O(max_suffix_length)` latest-end propagation.
- RWKV7+ROSA overlap example for inference.

## Installation

```bash
pip install --no-build-isolation git+https://github.com/wjie98/rosa_soft.git
```

For local development:

```bash
CUDA_HOME=/path/to/cuda MAX_JOBS=1 python setup.py build_ext --inplace
```

Set `USE_CUDA=0` to build only CPU extension pieces.
Set `ROSA_BUILD_EXTENSION=0` to install only the pure PyTorch reference path:

```bash
ROSA_BUILD_EXTENSION=0 pip install -e .
```

## Quick Start: RosaSoft Training Proxy

The reference operator is the current hard-forward design oracle. It returns
only exact hard values while using independently perturbed soft suffix
allocation in backward:

```python
import torch
from rosa_soft import rosa_soft_reference

B, T, H, D = 2, 64, 8, 4
Hv, Dv = 2, 8
q = torch.randn(B, T, H, D, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(B, T, Hv, Dv, requires_grad=True)

y = rosa_soft_reference(
    q,
    k,
    v,
    max_suffix_length=16,
    route_temperature=1.0,
    mismatch_penalty=3.0,
)
y.square().mean().backward()
```

This path materializes quadratic action tensors and is intended for testing and
small training experiments. See
[docs/ROSA_SOFT_REFERENCE.md](docs/ROSA_SOFT_REFERENCE.md) for its exact
semantics and kernel mapping.

`route_temperature` and `mismatch_penalty` are independent static scalars. They default
to `1.0` and `3.0`; pass different values explicitly for manual experiments.
The operator never derives either value from context length, Q/K width,
`max_suffix_length`, training step, telemetry, or the other scalar.

The CUDA operator implements the same estimator:

```python
import torch
from rosa_soft import rosa_soft

B, T, H, D = 2, 1024, 64, 8
Hv, Dv = 16, 8
q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q)
v = torch.randn(B, T, Hv, Dv, device="cuda", dtype=torch.float16, requires_grad=True)

y = rosa_soft(
    q,
    k,
    v,
    max_suffix_length=128,
    route_temperature=1.0,
    mismatch_penalty=3.0,
)
y.float().sum().backward()
```

Both calls return the exact hard route in forward. In backward they use the
same fixed softsign STE, cubic mismatch perturbation, Jacobian anchor,
multiplicative suffix score, null score, action softmax, and soft distributed
V credit.
The CUDA path reconstructs every mismatch sample from one saved seed and its
`(b, h, q_pos, k_pos, bit)` counter.

## Quick Start: Hard RosaRuntime

```python
import torch
from rosa_soft import RosaRuntime

B, T, H, bits = 2, 16, 8, 4
Hv = 2
q = torch.randn(B, T, H, bits, device="cuda")
k = torch.randn_like(q)
v = torch.randn(B, T, Hv, bits, device="cuda")

with RosaRuntime(
    H,
    Hv,
    qk_bits=bits,
    value_bits=bits,
    max_suffix_length=32,
) as runtime:
    out, endpos = runtime.update(q, k, v, return_packed=False)
```

For overlap, pass an explicit CUDA stream:

```python
stream = torch.cuda.Stream()
work = runtime.update(q, k, v, stream=stream, async_op=True, return_packed=False)
# Run other GPU work here.
out, endpos = work.wait()
```

## RWKV7 + ROSA Example

[examples/rwkv7_rosa_overlap.py](examples/rwkv7_rosa_overlap.py) shows an
RWKV7 block with a ROSA branch parallel to `tmix`:

1. compute ROSA Q/K/V on GPU;
2. start `RosaRuntime` with an explicit stream, staging packed bits to CPU;
3. compute RWKV7 `tmix` on GPU;
4. wait for ROSA output, then run the ROSA output projection;
5. continue with the channel mix.

The ROSA branch uses official-style RWKV initialization for token-shift
parameters and Q/K/V projection ranges.

## API Summary

### `rosa_soft_reference`

| Parameter | Default | Description |
| --- | ---: | --- |
| `query_logits`, `key_logits` | required | `[B, T, H, D]` floating tensors on CPU or CUDA, `D in [1, 32]`. |
| `payload_logits` | required | `[B, T, H_v, D_v]`, with `H % H_v == 0`. |
| `max_suffix_length` | `32` | Exact hard and proxy suffix horizon. |
| `route_temperature` | `1.0` | Static backward allocation route_temperature. Forward is unaffected. |
| `mismatch_penalty` | `3.0` | Static mismatch leakage and local-Jacobian control. |

Testing and inspection hooks are intentionally outside the public training
surface:

```python
from rosa_soft.diagnostics import summarize_rosa_soft
from rosa_soft.testing import inspect_rosa_soft

_, inspection = inspect_rosa_soft(q, k, v)
summary = summarize_rosa_soft(inspection)
```

### `rosa_soft`

| Parameter | Default | Recommendation |
| --- | ---: | --- |
| `query_logits` | required | `[B, T, H, D]` CUDA tensor, `D in [1, 32]`. |
| `key_logits` | required | `[B, T, H, D]` CUDA tensor. |
| `payload_logits` | required | `[B, T, H_v, D_v]` CUDA tensor, `H % H_v == 0`. |
| `max_suffix_length` | `32` | Main semantic knob. Use the smallest suffix horizon that captures the task. |
| `route_temperature` | `1.0` | Static action-allocation route_temperature; the kernel uses `1 / route_temperature` as its logit scale. |
| `mismatch_penalty` | `3.0` | Static local mismatch penalty, independent of route_temperature and window size. |

The current training kernel enumerates every causal action. Its persistent
random state is linear in `B H T`, but compute remains quadratic in sequence
length. This dense coverage is intentional: RosaSoft uses dense backward
credit to train a sparse hard route. The default training path must not prune
actions with top-k, ANN/LSH, hard-winner neighborhoods, thresholds, or sampled
candidates, because undiscovered routes would lose their learning signal.
Long-context inference should use `RosaRuntime`; training optimization must
instead preserve all candidates through online reductions, dense tiling,
exact diagonal recurrences, and recomputation/caching tradeoffs.

### `RosaRuntime`

| Parameter | Description |
| --- | --- |
| `num_heads` | Number of Q/K heads. |
| `num_value_heads` | Number of value heads. Defaults to `num_heads`. |
| `qk_bits` | Packed hard runtime Q/K width, `1..8`. |
| `value_bits` | Packed hard runtime payload width, `1..8`. |
| `max_suffix_length` | Exact hard suffix horizon, identical to the training forward meaning. |

`RosaRuntime` currently uses one packed byte per head. Its 8-bit deployment
limit is explicit and narrower than the training operator's 32-bit Q/K
contract. `update_packed` reads only the declared low `qk_bits`/`value_bits`;
unused upper bits do not affect matching or returned payloads.

## Repository Layout

```text
rosa_soft/
  soft.py                   CUDA autograd wrapper for RosaSoft
  soft_contract.py          Shared shape and scalar contract
  soft_reference.py         Pure PyTorch semantic oracle
  testing.py                Deterministic and inspection hooks
  diagnostics.py            Read-only inspection summaries
  runtime.py                Python RosaRuntime wrapper and async staging
  rwkv7.py                  Low-level RWKV7 CUDA wrappers
  csrc/
    rosa_soft.cpp           C++ checks and dispatch
    rosa_runtime.cpp        C++ hard runtime custom class
    cuda/
      rosa_soft_kernels.cu
examples/
  fit_soft_reference.py     Tiny hard-forward fitting benchmark
  rwkv7_export.py           Exported RWKV7 reference module
  rwkv7_rosa_overlap.py     RWKV7+ROSA overlap example
docs/
  CONCEPT.md                User-facing technical report and design history
  ROSA_SOFT_DESIGN.md       RosaSoft operator report
  ROSA_SOFT_REFERENCE.md    Hard-forward reference invariants and kernel map
tests/
  test_soft_cuda.py         CUDA/reference forward and VJP parity
  test_soft_reference.py    Hard and surrogate invariant coverage
  test_diagnostics.py       Detached diagnostic correctness
  test_runtime.py           Bounded Runtime correctness and lifecycle
  rosa_soft_benchmark.py    Training time and peak-memory probe
  rosa_runtime_benchmark.py Runtime scaling probe
```

## Developer Documentation

Read [docs/CONCEPT.md](docs/CONCEPT.md) for the full user-facing report:

- why ROSA needs a training proxy;
- the path from bitflip perturbation to soft-DP, suffix attention, and
  RosaSoft;
- the current RosaSoft formulas and parameter guidance;
- the local validation snapshots.

Read [docs/ROSA_SOFT_DESIGN.md](docs/ROSA_SOFT_DESIGN.md) for a shorter
operator-focused reference.

## Validation Snapshot

The current RosaSoft mainline was selected from hard-forward next-token
experiments: the forward path uses hard ROSA, while the backward path changes
between proxy-gradient methods.

| Check | Result |
| --- | --- |
| Production test suite | `116 passed` on SM75 and SM86, including three-plan VJP parity, dense-support route discovery, exhaustive small-state and bounded-runtime oracle parity, lifecycle tests, packed-width and empty-input invariants, nonfinite-logit VJP agreement, exact recurrence adjoints, `D=32`, grouped value heads, non-contiguous tensors, singleton sequences, FP16/BF16, and causal rows larger than one CUDA block. Historical research experiments are isolated from this count. |
| Counter-exact parity | With the same reconstructed mismatch samples, CUDA and PyTorch forward are bit-exact; FP32 Q/K/V gradient maximum absolute error is `5.96e-8` on the recorded probe. |
| Single-sample next-token fit | At 1000 steps, the single-branch reference reached hard CE `4.78e-4`; CUDA reached `3.83e-4` on model seed 0 and `3.39e-4` on seed 2, all at `100%` accuracy, versus historical bitflip CE `1.61e-4`. Four perturbation seeds on fixed model seed 0 all reached `100%`. |
| Backward planner | CUDA profiler probes instantiated `ScoreCached`, `KeyReduced`, and `Generic` kernels. On RTX 3070, representative FP32 steps took `1.17`, `0.46`, and `1.31 ms`; these shapes exercise planner boundaries rather than comparable workloads. |
| Runtime scaling | For the repeated-symbol worst case with `W=32`, one CPU thread took `3.79/5.62/8.27/13.83/24.81 ms` at `T=8K/16K/32K/64K/128K`, consistent with linear growth after fixed overhead. |
| Training memory | For FP32 `B=1,H=4,D=8,H_v=2,D_v=8,W=32`, CUDA peak operator memory rose from about `0.03 MiB` at `T=64` to `0.23 MiB` at `T=512`; the materialized reference used `50.55 MiB` at `T=256`. |
| Training time | On an RTX 3070 for the same shape, the final long-run CUDA probe took `1.528/6.002/20.426 ms` at `T=256/512/1024`; every causal backward action is still evaluated. |

## Validation

```bash
python -m py_compile setup.py rosa_soft/*.py examples/*.py tests/*.py
python -m pytest -q
python tests/rosa_soft_benchmark.py \
  --operator cuda --sequence-lengths 64 128 256 \
  --max-suffix-length 32 --route-temperature 1.0 \
  --mismatch-penalty 3.0
python tests/rosa_runtime_benchmark.py \
  --B 4 --T 256 --H 8 --Hv 2 --bits 4 --max-suffix-length 32
```

For CUDA build validation:

```bash
CUDA_HOME=/path/to/cuda MAX_JOBS=1 python setup.py build_ext --inplace
```

## Acknowledgements

This project is built upon and inspired by the research of **Peng Bo
(BlinkDL)** in the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)
project. We extend our sincere appreciation for the innovative work that has
significantly influenced this project.
