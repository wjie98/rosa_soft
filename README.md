# rosa_soft

Production-oriented PyTorch operators for ROSA research.

This release contains two public paths:

- `rosa_anchor_ops`: CUDA training proxy for a hard-ROSA-style suffix route.
- `RosaRuntime`: stateful hard ROSA inference runtime with explicit resource
  management and optional CUDA-to-CPU overlap.

## Highlights

- CUDA forward/backward for the RosaAnchor soft training proxy.
- Telemetry for selector confidence, null mass, entropy, and early-stop stats.
- Scale controller for periodic target-top-probability calibration.
- Hard runtime with packed `uint8` Q/K/V and FlashAttention-style `cu_seqlens`.
- Compact suffix-automaton backend, with a map backend kept for correctness
  and benchmark comparison.
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

## Quick Start: RosaAnchor Training Proxy

```python
import torch
from rosa_soft import rosa_anchor_ops

B, T, H, D = 2, 1024, 64, 8
Hv, Dv = 16, 8
q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q)
v = torch.randn(B, T, Hv, Dv, device="cuda", dtype=torch.float16, requires_grad=True)

y, telemetry = rosa_anchor_ops(
    q,
    k,
    v,
    window_size=128,
    scale=None,
    return_telemetry=True,
    logit_epsilon=0.0,
    qk_damper_strength=0.0,
)
y.float().sum().backward()
print(telemetry.as_float_dict())
```

## Scale Control During Training

Use `RosaAnchorScaleController` to periodically adjust scale toward a target top
probability. The default training target is `0.8`.

```python
from rosa_soft import RosaAnchorScaleConfig, RosaAnchorScaleController

controller = RosaAnchorScaleController(
    RosaAnchorScaleConfig(
        seq_len=ctx_len,
        qk_bits=qk_bits,
        window_size=window_size,
        target_top_prob=0.8,
        update_interval=100,
    )
)

probe = controller.should_probe(step)
result = rosa_anchor_ops(q, k, v, window_size=window_size, **controller.kwargs(step))
if probe:
    y, telemetry = result
    controller.observe(telemetry)
else:
    y = result
controller.advance()
```

See [examples/train_rosa_anchor_scale.py](examples/train_rosa_anchor_scale.py)
for a minimal next-token training loop skeleton.

## Quick Start: Hard RosaRuntime

```python
import torch
from rosa_soft import RosaRuntime

B, T, H, bits = 2, 16, 8, 4
Hv = 2
q = torch.randn(B, T, H, bits, device="cuda")
k = torch.randn_like(q)
v = torch.randn(B, T, Hv, bits, device="cuda")

with RosaRuntime(H, Hv, qk_bits=bits, value_bits=bits) as runtime:
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

### `rosa_anchor_ops`

| Parameter | Description |
| --- | --- |
| `query` | `[B, T, H, D]` CUDA tensor, `D in [1, 32]`. |
| `key` | `[B, T, H, D]` CUDA tensor. |
| `value` | `[B, T, H_v, D_v]` CUDA tensor, `H % H_v == 0`. |
| `window_size` | Suffix window. Backward supports up to 512. |
| `scale` | Selector scale. `None` uses the built-in shape estimate. |
| `return_telemetry` | Return `AttentionTelemetry` for calibration. |
| `logit_epsilon` | Early-stop threshold. `0` is exact within the window. |
| `qk_damper_strength` | Optional Q/K saturation guardrail in `[0, 1]`. |

### `RosaRuntime`

| Parameter | Description |
| --- | --- |
| `num_heads` | Number of Q/K heads. |
| `num_value_heads` | Number of value heads. Defaults to `num_heads`. |
| `qk_bits` | Packed hard runtime Q/K width, `1..8`. |
| `value_bits` | Packed hard runtime V width, `1..8`. |
| `backend` | `"compact"` default, or `"map"` for comparison. |

## Repository Layout

```text
rosa_soft/
  anchor.py                 Python autograd wrapper for RosaAnchor
  runtime.py                Python RosaRuntime wrapper and async staging
  training.py               Scale controller helpers
  rwkv7.py                  Low-level RWKV7 CUDA wrappers
  csrc/
    rosa_anchor.cpp         C++ checks and dispatch
    rosa_runtime.cpp        C++ hard runtime custom class
    cuda/
      rosa_anchor_kernels.cu
examples/
  rwkv7_export.py           Exported RWKV7 reference module
  rwkv7_rosa_overlap.py     RWKV7+ROSA overlap example
  train_rosa_anchor_scale.py
docs/
  ROSA_ANCHOR_DESIGN.md     Developer report and formulas
tests/
  rosa_runtime_cpp.py       Runtime correctness tests
  rosa_runtime_benchmark.py Compact-vs-map benchmark
```

## Developer Documentation

Read [docs/ROSA_ANCHOR_DESIGN.md](docs/ROSA_ANCHOR_DESIGN.md) for:

- per-step RosaAnchor formulas;
- design motivation for sink/null, recency tie-break, and weighted mismatch;
- backward surrogate details;
- telemetry definitions;
- parameter tuning and scale-control guidance.

## Validation

```bash
python -m py_compile setup.py rosa_soft/*.py examples/*.py tests/*.py
CUDA_VISIBLE_DEVICES=1 python tests/rosa_runtime_cpp.py
python tests/rosa_runtime_benchmark.py --B 4 --T 256 --H 8 --Hv 2 --bits 4
```

For CUDA build validation:

```bash
CUDA_HOME=/path/to/cuda MAX_JOBS=1 python setup.py build_ext --inplace
```
