# rosa_soft

Production-oriented PyTorch operators for ROSA research.

This release contains three public paths:

- `rosa_soft_reference`: pure PyTorch hard-forward/soft-backward correctness
  oracle.
- `rosa_soft`: CUDA implementation of the same training estimator.
- `RosaRuntime`: stateful hard ROSA inference runtime with explicit resource
  management and optional CUDA-to-CPU overlap.

## Highlights

- CUDA forward/backward for the RosaSoft training estimator.
- Exact hard-forward PyTorch reference on CPU and CUDA.
- Exact hard forward in both training paths; no soft value can leak forward.
- Explicit static `route_temperature` and `mismatch_penalty` controls.
- Counter-indexed mismatch sampling in CUDA: backward saves one seed instead of
  an `O(B H T^2 D)` random tensor.
- Detached reference diagnostics kept outside the training hot path.
- Hard runtime with packed `uint8` Q/K/payload symbols and
  FlashAttention-style `cu_seqlens`.
- Bounded compact suffix automaton with the same finite suffix horizon as
  training and `O(max_suffix_length)` latest-end propagation.
- Optional RWKV7 kernels and a ROSA overlap example for inference.

## Installation

```bash
pip install --no-build-isolation git+https://github.com/wjie98/rosa_soft.git
```

The default native build auto-detects CUDA and excludes RWKV7:

```bash
CUDA_HOME=/path/to/cuda MAX_JOBS=1 python setup.py build_ext --inplace
```

Build selection is explicit:

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `ROSA_BUILD_EXTENSION` | `0`, `1` | `1` | Disable all native code for a reference-only install. |
| `USE_CUDA` | `auto`, `0`, `1` | `auto` | Auto-detect, forbid, or require CUDA compilation. |
| `ROSA_BUILD_RWKV7` | `0`, `1` | `0` | Include optional RWKV7 kernels; requires CUDA. |

For a pure PyTorch install:

```bash
ROSA_BUILD_EXTENSION=0 pip install -e .
```

For the CPU hard runtime only, use `USE_CUDA=0`. To include the legacy RWKV7
kernels, use `USE_CUDA=1 ROSA_BUILD_RWKV7=1`. Native builds must use the
already installed PyTorch ABI, so pip builds require `--no-build-isolation`.

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

This path materializes quadratic route tensors and is intended for testing and
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
same fixed softsign VJP, cubic mismatch perturbation, hard-Hamming local VJP,
multiplicative suffix score, null score, route softmax, and distributed soft
payload credit.
The CUDA path reconstructs every mismatch sample from one saved seed and its
`(b, h, q_pos, k_pos, bit)` counter.

## Quick Start: Hard RosaRuntime

```python
import torch
from rosa_soft import RosaRuntime

B, T, H, bits = 2, 16, 8, 4
payload_heads = 2
query = torch.randn(B, T, H, bits, device="cuda")
key = torch.randn_like(query)
payload = torch.randn(B, T, payload_heads, bits, device="cuda")

with RosaRuntime(
    H,
    num_payload_heads=payload_heads,
    qk_bits=bits,
    payload_bits=bits,
    max_suffix_length=32,
) as runtime:
    output, end_positions = runtime.update(
        query,
        key,
        payload,
        return_packed=False,
    )
```

For overlap, pass an explicit CUDA stream while the runtime is open:

```python
stream = torch.cuda.Stream()
with RosaRuntime(H, payload_heads, bits, bits, 32) as runtime:
    work = runtime.update(
        query,
        key,
        payload,
        stream=stream,
        async_op=True,
        return_packed=False,
    )
    # Run other GPU work here.
    output, end_positions = work.wait()
```

## RWKV7 + ROSA Example

[examples/rwkv7_rosa_overlap.py](examples/rwkv7_rosa_overlap.py) shows an
RWKV7 block with a ROSA branch parallel to `tmix`:

1. compute ROSA Q/K/payload logits on GPU;
2. start `RosaRuntime` with an explicit stream, staging packed bits to CPU;
3. compute RWKV7 `tmix` on GPU;
4. wait for ROSA output, then run the ROSA output projection;
5. continue with the channel mix.

The ROSA branch uses official-style RWKV initialization for token-shift
parameters and Q/K/payload projection ranges.

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
| `route_temperature` | `1.0` | Static route-allocation temperature; the kernel uses `1 / route_temperature` as its logit scale. |
| `mismatch_penalty` | `3.0` | Static local mismatch penalty, independent of route_temperature and window size. |

The current training kernel enumerates every causal route. Its persistent
random state is linear in `B H T`, but compute remains quadratic in sequence
length. This dense coverage is intentional: RosaSoft uses dense backward
credit to train a sparse hard route. The default training path must not prune
routes with top-k, ANN/LSH, hard-winner neighborhoods, thresholds, or sampled
candidates, because undiscovered routes would lose their learning signal.
Long-context inference should use `RosaRuntime`; training optimization must
instead preserve all candidates through online reductions, dense tiling,
exact diagonal recurrences, and recomputation/caching tradeoffs.

Lowering `route_temperature` only sharpens the current proxy ordering. With a
finite mismatch penalty, that ordering can differ from the exact hard
latest-longest route, so a low-temperature limit is not a proof of hard-route
equivalence.

### `RosaRuntime`

| Parameter | Description |
| --- | --- |
| `num_heads` | Number of Q/K heads. |
| `num_payload_heads` | Number of payload heads. Defaults to `num_heads`. |
| `qk_bits` | Packed hard runtime Q/K width, `1..8`. |
| `payload_bits` | Packed hard runtime payload width, `1..8`. |
| `max_suffix_length` | Exact hard suffix horizon, identical to the training forward meaning. |

`RosaRuntime` currently uses one packed byte per head. Its 8-bit deployment
limit is explicit and narrower than the training operator's 32-bit Q/K
contract. `update_packed` reads only the declared low `qk_bits`/`payload_bits`;
unused upper bits do not affect matching or returned payloads.

The first update fixes the number of sequence slots until `reset()`.
`sequence_ids` can opt into explicit slot-order checks and must be supplied on
the first update after construction or reset. An update failure poisons the
runtime; close it and construct a new instance. Native automaton checkpointing
has no stable format, so `state_dict()` and `load_state_dict()` explicitly
raise `NotImplementedError`.

## Repository Layout

```text
rosa_soft/
  soft.py                   CUDA autograd wrapper for RosaSoft
  soft_contract.py          Shared shape and scalar contract
  soft_reference.py         Pure PyTorch semantic oracle
  testing.py                Development-only deterministic inspection hooks
  diagnostics.py            Development-only read-only summaries
  runtime.py                Python RosaRuntime wrapper and async staging
  rwkv7.py                  Optional low-level RWKV7 CUDA wrappers
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
  research/                 Historical estimator experiments
benchmarks/
  rosa_soft.py              Training time and peak-memory probe
  rosa_runtime.py           Runtime scaling probe
  legacy/                   Optional historical RWKV7 probes
tests/
  test_soft_cuda.py         CUDA/reference forward and VJP parity
  test_soft_reference.py    Hard and surrogate invariant coverage
  test_diagnostics.py       Detached diagnostic correctness
  test_runtime.py           Bounded Runtime correctness and lifecycle
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

## Validation

```bash
python -m py_compile setup.py rosa_soft/*.py examples/*.py tests/*.py
python -m pytest -q
python benchmarks/rosa_soft.py \
  --operator cuda --sequence-lengths 64 128 256 \
  --max-suffix-length 32 --route-temperature 1.0 \
  --mismatch-penalty 3.0
python benchmarks/rosa_runtime.py \
  --B 4 --T 256 --H 8 --Hv 2 --bits 4 --max-suffix-length 32
```

The latest locally reproduced environment and results are recorded in
[`validation/latest.json`](validation/latest.json). Historical estimator
selection evidence remains under [`docs/research/`](docs/research/); it is not
part of the production test count.

For CUDA build validation:

```bash
CUDA_HOME=/path/to/cuda MAX_JOBS=1 python setup.py build_ext --inplace
```

## Acknowledgements

This project is built upon and inspired by the research of **Peng Bo
(BlinkDL)** in the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)
project. We extend our sincere appreciation for the innovative work that has
significantly influenced this project.
