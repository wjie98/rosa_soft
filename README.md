# rosa_soft

`rosa_soft` contains the current ROSA training proxy and a stateful ROSA
suffix-automaton runtime for PyTorch.

The public package intentionally exposes only the production path:

- `rosa_anchor_ops`: CUDA autograd operator used as the current ROSA soft
  training proxy.
- `RosaRuntime`: CPU suffix-automaton runtime with packed `uint8` state,
  standard `cu_seqlens` varlen input, optional CUDA staging, and explicit
  resource management.

Historical proxy experiments, legacy snapshots, and internal ablation assets
are not part of this release branch.

## Installation

```bash
pip install --no-build-isolation git+https://github.com/wjie98/rosa_soft.git
```

Set `USE_CUDA=0` to build only CPU extension pieces.

For low-memory build environments, keep extension parallelism conservative:

```bash
MAX_JOBS=1 pip install --no-build-isolation .
```

## RosaAnchor Training Proxy

```python
import torch
from rosa_soft import rosa_anchor_ops

query = torch.randn(2, 1024, 64, 8, device="cuda", dtype=torch.float16)
key = torch.randn_like(query)
value = torch.randn(2, 1024, 16, 8, device="cuda", dtype=torch.float16)

output = rosa_anchor_ops(
    query,
    key,
    value,
    window_size=128,
    scale=None,
    logit_epsilon=1e-3,
    qk_damper_strength=0.2,
)
```

### `rosa_anchor_ops`

| Parameter | Type | Description |
| --- | --- | --- |
| `query` | `Tensor` | `[B, T, H, D]` Q bit logits. |
| `key` | `Tensor` | `[B, T, H, D]` K bit logits. |
| `value` | `Tensor` | `[B, T, H_v, D_v]` V bit logits. `H % H_v == 0`. |
| `window_size` | `int` | Maximum suffix window. CUDA backward supports up to 512. |
| `scale` | `float | None` | Selector logit scale. `None` uses the built-in shape estimate. |
| `return_telemetry` | `bool` | Return lightweight selector statistics for scale tracking. |
| `logit_epsilon` | `float` | Early-stop threshold. `0` runs the full suffix scan. |
| `qk_damper_strength` | `float` | Optional Q/K gradient direction damper in `[0, 1]`. |

## RosaRuntime Inference Runtime

`RosaRuntime` is stateful. Use `close()` or a context manager when deterministic
resource release matters; it does not rely on Python `__del__`.

```python
import torch
from rosa_soft import RosaRuntime

B, T, H, D = 2, 16, 8, 4
Hv = 2
q = torch.randn(B, T, H, D, device="cuda")
k = torch.randn_like(q)
v = torch.randn(B, T, Hv, D, device="cuda")

with RosaRuntime(num_heads=H, num_value_heads=Hv, qk_bits=D, value_bits=D) as rt:
    out, endpos = rt.update(q, k, v, return_packed=False)
```

Packed dense input is also supported:

```python
q = torch.randint(0, 16, (B, T, H), device="cuda", dtype=torch.uint8)
k = torch.randint(0, 16, (B, T, H), device="cuda", dtype=torch.uint8)
v = torch.randint(0, 16, (B, T, Hv), device="cuda", dtype=torch.uint8)

rt = RosaRuntime(H, Hv, qk_bits=4, value_bits=4)
out_packed, endpos = rt.update_packed(q, k, v)
rt.close()
```

Varlen input uses FlashAttention-style cumulative sequence lengths:

```python
cu_seqlens = torch.tensor([0, 5, 13, 21], dtype=torch.int32)
q = torch.randint(0, 16, (21, H), dtype=torch.uint8)
k = torch.randint(0, 16, (21, H), dtype=torch.uint8)
v = torch.randint(0, 16, (21, Hv), dtype=torch.uint8)

out, endpos = rt.update_packed(q, k, v, cu_seqlens=cu_seqlens)
```

For CUDA overlap, pass an explicit stream:

```python
stream = torch.cuda.Stream()
work = rt.update(q, k, v, stream=stream, async_op=True)
out, endpos = work.wait()
```

If no stream is supplied, the call is blocking and uses the caller's current
device semantics. The runtime does not create hidden CUDA streams.

### `RosaRuntime`

| Parameter | Type | Description |
| --- | --- | --- |
| `num_heads` | `int` | Number of Q/K heads. |
| `num_value_heads` | `int | None` | Number of V heads. Defaults to `num_heads`; `num_heads % num_value_heads == 0`. |
| `qk_bits` | `int` | Q/K bit width, `1..8`. |
| `value_bits` | `int` | V bit width, `1..8`. |
| `backend` | `str` | `"compact"` default, or `"map"` for semantic/benchmark comparison. |

`stats()` returns `(states, edges, values)` for runtime memory diagnostics.

## Validation

```bash
python test/rosa_sam_cpp.py
python test/rosa_runtime_benchmark.py --B 4 --T 256 --H 8 --Hv 2 --bits 4
```
