# rosa_soft

Hard-forward ROSA operators for PyTorch.

`rosa-soft-dense-reference-v1` freezes the current exact-hard/dense-gradient
training implementation. New estimator families should use a separate module
or operator instead of changing this baseline. The frozen contract and
reproduction commands are recorded in
[`docs/DENSE_REFERENCE_FREEZE.md`](docs/DENSE_REFERENCE_FREEZE.md).

The package has three supported implementation families:

- `rosa_soft_reference` and `rosa_soft_varlen_reference`: pure PyTorch
  correctness oracles.
- `rosa_soft` and `rosa_soft_varlen`: CUDA training operators with the same
  hard forward and dense surrogate VJP.
- `RosaRuntime`: stateful packed CPU runtime for exact hard inference.

## Documentation

| Start here | Contents |
| --- | --- |
| [`docs/PRODUCTION_GUIDE.md`](docs/PRODUCTION_GUIDE.md) | Public API, build matrix, data flow, source ownership, Runtime lifecycle, validation, and release checklist. |
| [`docs/CONCEPT.md`](docs/CONCEPT.md) | Exact hard-forward and dense surrogate-backward equations. |
| [`docs/ROSA_SOFT_DESIGN.md`](docs/ROSA_SOFT_DESIGN.md) | Python, C++, CUDA, autograd, and execution-plan design. |
| [`docs/ROSA_SOFT_REFERENCE.md`](docs/ROSA_SOFT_REFERENCE.md) | PyTorch oracle and inspection conventions. |
| [`docs/DENSE_REFERENCE_FREEZE.md`](docs/DENSE_REFERENCE_FREEZE.md) | Frozen tag, reproduction snapshot, and change policy. |

The complete index is [`docs/README.md`](docs/README.md). Research documents
record evidence and rejected alternatives; they do not extend the production
API.

## Semantic Contract

RosaSoft trains a discrete suffix route without exposing a soft value path:

- forward returns the exact latest-longest hard ROSA match;
- backward evaluates every valid causal route;
- query/key receive a dense suffix VJP with a fixed concave route-score
  transform;
- values receive probability-weighted dense credit;
- null output is exactly zero;
- ties select the latest matching route.

The public operator has four keyword-only controls:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `max_suffix_length` | `32` | Hard and surrogate suffix horizon. |
| `scale` | `1.0` | Multiplicative backward attention-logit scale. |
| `dropout_p` | `0.0` | Post-softmax inverted attention dropout in backward. |
| `mismatch_scale` | `3.0` | Static mismatch leakage and local-VJP scale. |

These controls are static. `scale` and `mismatch_scale` are not scheduled or
derived from sequence length, symbol width, horizon, step, or diagnostics.
Dropout never participates in hard forward.

The current pre-release API does not retain aliases for older control names:
`scale = 1 / route_temperature`, `mismatch_scale` replaces
`mismatch_penalty` without changing its numeric convention, and the training
tensor formerly called `payload` is now `value`. `RosaRuntime` keeps
`payload` because it names the packed inference storage protocol.

The default `mismatch_scale=3.0` is the conservative choice for the default
`dropout_p=0` recipe. Synthetic fitting did not find one value that dominates
across dropout settings, so values such as `9.0` must be passed explicitly
when evaluating a different recipe.

`max_suffix_length` must be an integer at least one and is clipped to the
available sequence length. `scale` and `mismatch_scale` must be finite and
positive. `dropout_p` must be finite and lie in `[0, 1 - 2^-24]`. CUDA also
requires these scalars and their checked products to fit normal FP32
arithmetic.

## Installation

Python 3.10 or newer and an installed PyTorch are required.

```bash
pip install --no-build-isolation git+https://github.com/wjie98/rosa_soft.git
```

Native builds compile against the PyTorch ABI in the current environment:

```bash
CUDA_HOME=/path/to/cuda MAX_JOBS=1 \
  python setup.py build_ext --inplace
```

Only two build variables are supported:

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `ROSA_BUILD_EXTENSION` | `0`, `1` | `1` | Build or omit all native code. |
| `USE_CUDA` | `auto`, `0`, `1` | `auto` | Detect, forbid, or require CUDA. |

Reference-only and CPU-runtime installs are explicit:

```bash
ROSA_BUILD_EXTENSION=0 pip install --no-build-isolation -e .
USE_CUDA=0 pip install --no-build-isolation -e .
```

Build state is reported by one immutable object:

```python
import rosa_soft

print(rosa_soft.BUILD_CAPABILITIES)
# BuildCapabilities(
#   variant="reference" | "cpu-runtime" | "cuda",
#   compiled_extension=...,
#   rosa_runtime=...,
#   rosa_soft_cuda=...,
# )
```

A missing `_C` module selects reference mode. An existing but broken or
ABI-incompatible extension fails during import instead of silently falling
back.

## Training

```python
import torch
from rosa_soft import rosa_soft

B, T, H, D = 2, 1024, 64, 8
value_heads, value_dim = 16, 8

query = torch.randn(
    B, T, H, D, device="cuda", dtype=torch.float32, requires_grad=True
)
key = torch.randn_like(query, requires_grad=True)
value = torch.randn(
    B,
    T,
    value_heads,
    value_dim,
    device="cuda",
    dtype=torch.float32,
    requires_grad=True,
)

output = rosa_soft(
    query,
    key,
    value,
    max_suffix_length=128,
    scale=1.0,
    dropout_p=0.1,
    mismatch_scale=3.0,
)
output.float().square().mean().backward()
```

Q/K symbol width is `1..32`. Inputs must share device and dtype; CUDA supports
FP32, FP16, and BF16. Value heads may be grouped when
`H % value_heads == 0`.

Both CUDA calls are validated with
`torch.compile(..., backend="aot_eager", fullgraph=True)`. Inductor and CUDA
Graphs are not yet part of the validated contract. CUDA gradients use global
atomic accumulation, so the VJP participates in PyTorch deterministic
algorithm checks and is numerically, not bitwise, reproducible. Backward
checks `needs_input_grad` and skips disabled query, key, or value work. With
`dropout_p > 0`, one scalar RNG seed is saved and each route mask is
reconstructed from its indices; no quadratic random tensor is retained.

FP32 inputs preserve the widest numerically nonzero dense route support.
With FP16 projections, use `torch.amp.GradScaler`; scaling recovers additional
small route gradients, but values below the FP16 range can still disappear.
Keep the RosaSoft query/key/value projections in FP32 when full long-tail support
is required. BF16 has a wider exponent range when the target GPU supports it.

### Packed Variable-Length Training

Use packed tensors when a batch contains unequal sequence lengths:

```python
import torch
from rosa_soft import rosa_soft_varlen

lengths = [1024, 768, 1536]
cu_seqlens = torch.tensor(
    [0, 1024, 1792, 3328],
    device="cuda",
    dtype=torch.int32,
)
N, H, D = sum(lengths), 64, 8

query = torch.randn(N, H, D, device="cuda", requires_grad=True)
key = torch.randn_like(query, requires_grad=True)
value = torch.randn(N, 16, 8, device="cuda", requires_grad=True)

output = rosa_soft_varlen(
    query,
    key,
    value,
    cu_seqlens,
    max_suffix_length=128,
)
output.square().mean().backward()
```

Packed query/key shapes are `[N,H,D]`; value is `[N,H_v,D_v]`.
`cu_seqlens` is a CUDA `int32` tensor, starts at zero, ends at `N`, and is
nondecreasing. Empty segments are allowed. Routes, suffixes, candidate
normalization, and null behavior are all segment-local; no forward value or
gradient crosses a segment boundary.

## Reference

```python
import torch
from rosa_soft import rosa_soft_reference

query = torch.randn(1, 32, 4, 4, requires_grad=True)
key = torch.randn_like(query, requires_grad=True)
value = torch.randn(1, 32, 2, 8, requires_grad=True)

output = rosa_soft_reference(query, key, value)
output.sum().backward()
```

The reference accepts FP16, BF16, FP32, and FP64 on CPU or CUDA. Its packed
variant accepts the same `cu_seqlens` contract. Both materialize quadratic
route tensors per sequence and are intended for correctness checks and small
experiments.

Development-only inspection stays outside the training API:

```python
from rosa_soft.diagnostics import summarize_rosa_soft
from rosa_soft.testing import inspect_rosa_soft

_, inspection = inspect_rosa_soft(query, key, value)
diagnostics = summarize_rosa_soft(inspection)
```

## Hard Runtime

`RosaRuntime` is the packed `1..8`-bit deployment subset. This narrower
contract is intentional and does not reduce the training operator's `1..32`
symbol width.

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
    output, matched_key_end_positions = runtime.update(
        query,
        key,
        payload,
        return_packed=False,
    )
```

An explicit CUDA stream overlaps staging and CPU matching:

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
    output, matched_key_end_positions = work.wait()
```

Async submission snapshots inputs and uses a bounded pending queue. The first
update fixes the number and order of sequence slots until `reset()`;
`sequence_ids` enables explicit reorder checks. A failed update poisons the
instance. Close it and create a new runtime.

The suffix horizon bounds matching work, not total retained state.
Automata and payload history currently grow approximately linearly with
processed context. Native checkpoint serialization is unsupported.

## Complexity

The current CUDA training path preserves dense candidate support:

```text
hard forward:
  O(B H T D + B H T^2 min(W,T) + B H T D_p)
full-QKV surrogate backward:
  O(B H T^2 (min(W,T) D + D_p))
operator-owned auxiliary CUDA state:
  O(B H T)

packed hard forward:
  O(N H D + H sum_s L_s^2 min(W,L_s) + N H (log S + D_p) + S)
packed full-QKV surrogate backward:
  O(H sum_s L_s^2 (min(W,L_s) D + D_p) + N H log S + S)
packed operator-owned auxiliary state:
  O(N H + S)
```

Auxiliary state excludes caller-owned inputs/output and the query/key/value
references retained by autograd. CUDA reconstructs each deterministic Hamming
gate and, when enabled, each counter-based dropout decision on demand. It does
not store an `O(B H T^2 D)` comparison tensor, an `O(B H T^2)` score matrix,
or an `O(B H T^2)` dropout mask. Candidate pruning, top-k, ANN/LSH, sampled
negatives, and suffix-index-selected gradients are not valid optimizations of
the default estimator.

## Repository

```text
rosa_soft/
  soft.py                 CUDA autograd wrapper
  soft_contract.py        shared input and scalar contract
  soft_reference.py       PyTorch semantic oracle
  runtime.py              packed Runtime and async lifecycle
  testing.py              deterministic inspection hooks
  diagnostics.py          detached summaries
  csrc/
    export.cpp            dispatcher schemas
    rosa_soft.cpp         CUDA checks and dispatch
    rosa_runtime.cpp      hard suffix-automaton runtime
    cuda/rosa_soft_kernels.cu
tests/                    semantic, lifecycle, compile, and parity gates
benchmarks/               manual training and Runtime probes
docs/                     current design and historical estimator evidence
contrib/rwkv7_legacy/     unbuilt, unsupported source-history archive
```

Start with the [production guide](docs/PRODUCTION_GUIDE.md) and the
[documentation index](docs/README.md). Current semantics are specified in
[CONCEPT.md](docs/CONCEPT.md), [ROSA_SOFT_DESIGN.md](docs/ROSA_SOFT_DESIGN.md),
and [ROSA_SOFT_REFERENCE.md](docs/ROSA_SOFT_REFERENCE.md). Historical design
decisions and estimator evidence remain under [`docs/research/`](docs/research/).

## Validation

```bash
python -m py_compile setup.py rosa_soft/*.py examples/*.py tests/*.py
python -m pytest -q

python examples/fit_soft_reference.py \
  --operator cuda --device cuda --steps 1000 \
  --success-loss-threshold 0.01

python examples/contextual_rnn_recall_gate.py \
  --operator cuda --device cuda --seeds 0 1 2 3 \
  --dropout-p 0.1 --steps 1000

python benchmarks/trained_fit_alignment.py \
  --operator cuda --device cuda --model-seeds 0 2 10 27 29 \
  --target-mode strict-longest-latest --steps 1000

python benchmarks/rosa_soft.py \
  --operator cuda --sequence-lengths 64 128 256 \
  --max-suffix-length 32 --scale 1 --dropout-p 0.1 \
  --mismatch-scale 3

python benchmarks/rosa_soft.py \
  --operator cuda --layout varlen --segment-length 64 \
  --sequence-lengths 128 256 512 --gradients qkv

python benchmarks/rosa_runtime.py \
  --B 4 --T 4096 --H 8 --Hv 2 --bits 4 \
  --max-suffix-length 32
```

Reproduced environment, fit, test, and benchmark results belong in
[`validation/latest.json`](validation/latest.json), not in semantic code.

## Acknowledgements

ROSA development was influenced by Peng Bo's work on recurrent language
models in [RWKV-LM](https://github.com/BlinkDL/RWKV-LM).
