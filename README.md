# rosa_soft

English | [简体中文](README.zh-CN.md)

A PyTorch extension for ROSA, a discrete suffix-based retrieval operator.
It provides exact CUDA matching with dense soft or bitflip training gradients,
and a C++ suffix automaton for stateful CPU inference.

## Operators

| API | Purpose | Input layout |
| --- | --- | --- |
| `rosa_soft` | Hard CUDA forward, dense soft surrogate backward | Dense or packed |
| `rosa_bitflip` | Hard CUDA forward, complete bit-edit output differences | Dense |
| `rosa_hard` | Stateless CPU SAM inference and validation; returns values and routes | Dense or packed |
| `RosaSam` | Stateful CPU SAM routing; caller owns the V history | Dense or packed chunks |

Both training operators use the same exact, causal, unlimited-suffix forward.
For query position `i`, select the longest matching Q/K suffix with K end
`j < i`; ties select the latest `j`. Return binary `V[j+1]`, or zero if
there is no match. Positive inputs encode `+1`; zero and negative inputs
encode `-1`. Neither training forward exposes soft values.

## Installation

Requires Python 3.10+, PyTorch 2.11.x, and a C++17 compiler. CUDA training
also requires a CUDA toolkit compatible with the installed PyTorch build.
From a source checkout, install PyTorch first, then build:

```bash
pip install --no-build-isolation .
```

CUDA is detected automatically. Set `USE_CUDA=1` to require a CUDA build or
`USE_CUDA=0` for CPU-only inference and validation. Set `CUDA_HOME` if the
toolkit is outside the default search path. Extensions build at installation,
not on import. See [API and deployment](docs/API.md) for execution constraints.

## Training

```python
import torch
from rosa_soft import rosa_soft, rosa_bitflip

q = torch.randn(1, 2048, 4, 8, device="cuda",
                dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(1, 2048, 2, 64, device="cuda",
                dtype=torch.float16, requires_grad=True)

y = rosa_soft(q, k, v)  # [1, 2048, 4, 64]
# Alternative estimator, with the same hard forward:
# y = rosa_bitflip(q, k, v, rows=64, chunks=2)
y.float().square().mean().backward()
```

Dense Q/K use `[B,T,H,D]`, with `1 <= D <= 32`. V uses `[B,T,Hv,Dv]`,
where `H % Hv == 0`. CUDA training supports FP16, BF16 and FP32, FP32
gradient accumulation, first-order autograd, and `torch.compile`.

`rosa_soft` is the default estimator. Its static options are `scale=1.0`,
`dropout_p=0.0` and `mismatch_scale=3.0`; they affect only backward.
`rosa_bitflip` evaluates complete activation-bit output changes at fixed dY,
then maps credit to continuous inputs using softsign. Explicit `tied="qk"`
and `tied="qkv"` modes perform simultaneous edits of shared activations.
Neither estimator is an unbiased derivative of an arbitrary nonlinear task
loss, and neither guarantees better training than the other.

For independent QKV 4-bit models, use `[B,T,C/4,4]`. The
[Rosa4Bit adapter](examples/rosa_4bit.py) retains a trainable output amplitude
outside V quantization. See the [integration guide](docs/API.md#qkv-4-bit-models)
and [residual-block training example](examples/train_bitflip.py).

## Memory and Scaling

CUDA training performs quadratic work in sequence length at fixed widths.
Unlimited suffix matching does not imply linear-time GPU training.

**Bitflip workspace is O(T), not O(T log T), for fixed batch size, head counts,
bit/value widths, and `rows`.** Let `S = B*H/chunks` and `R = min(rows,T)`:

- Independent edits use `O(S*R*T)` band storage.
- Joint QK/QKV edits use `O(S*R*T*D)` band storage.
- Full inputs, saved routes and final gradients remain linear-sized and are
  not divided by `chunks`.

`rows` (default 256) limits live query rows, not suffix length. `chunks`
(default 1) is the number of equal head groups processed sequentially in
backward; it must divide both H and Hv. Both reduce workspace when adjusted
appropriately, with a throughput tradeoff. Linear space can still be large:
see the [allocation ledger and sizing examples](docs/MEMORY.md).

## Inference

```python
import torch
from rosa_soft import rosa_hard, RosaSam

q = torch.randn(1, 16, 4, 8)
k = torch.randn_like(q)
v = torch.randn(1, 16, 2, 64)
y, ends = rosa_hard(q, k, v)  # No-match ends are -1.

sam = RosaSam(num_heads=4, symbol_bits=8)
ends = sam.update(q, k)
# Subsequent updates continue each sequence's history until sam.reset().
```

CPU SAM avoids the all-pairs CUDA DP and supports incremental retrieval, but
its state grows with history; this implementation does not guarantee constant
worst-case time per token. CPU calls are synchronous, including staging GPU
inputs. Calling `model.eval()` does not automatically switch operators.

## Documentation

- [中文版 README](README.zh-CN.md): matching overview and quick start in Chinese.
- [API and deployment](docs/API.md): parameters, packed input, binding and model integration.
- [Memory and complexity](docs/MEMORY.md): allocations, sizing and measurement scope.
- [Methods and related work](docs/METHODS.md): forward semantics and gradient definitions.
- [Production design](docs/DESIGN.md): equations, kernel organization and source map.

Run the semantic and training integration tests from a source checkout:

```bash
pip install --no-build-isolation ".[test]"
python -m pytest -q
```

Tests use independent DP/math and literal bit-edit oracles. They cover
causality, ties, unlimited suffixes, layouts, dtypes, gradient masks,
compilation and checkpointed training. Current bitflip execution is tested
on SM75 and SM86; native BF16 model/Inductor tests require SM80 or newer.

## Acknowledgements

ROSA was introduced by **Peng Bo (BlinkDL)**. We thank him and the
[RWKV-LM project](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)
for the original design and implementation.
