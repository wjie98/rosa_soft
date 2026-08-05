# RosaSoft Production Guide

This guide describes the maintained production surface of `rosa_soft` version
`0.1.0`. It is an implementation and release map for engineers integrating or
changing the package. Mathematical derivations live in [Concept](CONCEPT.md),
and kernel details live in [Design](ROSA_SOFT_DESIGN.md).

The tag `rosa-soft-dense-reference-v1` freezes the current training semantics:
exact hard ROSA forward and dense surrogate backward. Documentation,
reproducibility, and semantics-preserving kernel fixes may continue on `main`.
New estimator families must use a separate module or operator.

## 1. Scope And Sources Of Truth

Production behavior is defined in this order:

1. Public symbols exported by `rosa_soft/__init__.py`.
2. Input and scalar validation in `rosa_soft/soft_contract.py`.
3. Exact equations in `rosa_soft/soft_reference.py` and
   [Concept](CONCEPT.md).
4. CUDA parity with that reference, as enforced by the production tests.
5. Runtime hard-routing behavior in `rosa_soft/runtime.py` and
   `rosa_soft/csrc/rosa_runtime.cpp`.

The following are deliberately not public contract extensions:

- raw `torch.ops.rosa_soft.*` schemas;
- helpers whose names start with `_`;
- `rosa_soft.testing` inspection tensors;
- `rosa_soft.diagnostics` reporting fields;
- anything under `benchmarks/`, `docs/research/`, or `contrib/`.

## 2. Public Package Surface

The package exports exactly these names:

```python
from rosa_soft import (
    __version__,
    BUILD_CAPABILITIES,
    RosaRuntime,
    rosa_soft,
    rosa_soft_reference,
    rosa_soft_varlen,
    rosa_soft_varlen_reference,
)
```

`BUILD_CAPABILITIES` is an immutable `BuildCapabilities` value with:

```text
variant: "reference" | "cpu-runtime" | "cuda"
compiled_extension: bool
rosa_runtime: bool
rosa_soft_cuda: bool
```

Availability by build variant is:

| Surface | reference | cpu-runtime | cuda |
| --- | :---: | :---: | :---: |
| `rosa_soft_reference` | yes | yes | yes |
| `rosa_soft_varlen_reference` | yes | yes | yes |
| `RosaRuntime` | placeholder error | yes | yes |
| `rosa_soft` | placeholder error | placeholder error | yes |
| `rosa_soft_varlen` | placeholder error | placeholder error | yes |

A missing extension selects reference mode. If `_C` exists but has incomplete
runtime or CUDA registration, import fails. The package never silently hides a
stale or ABI-incompatible native build.

## 3. Build And Installation

Requirements:

- Python `>=3.10`;
- PyTorch `>=2.11,<2.12` installed before a native build;
- a C++ compiler with OpenMP for `RosaRuntime`;
- a CUDA toolkit visible through `CUDA_HOME` for CUDA training kernels.

Build against the active PyTorch ABI:

```bash
pip install --no-build-isolation -e .
```

Only two environment variables control the build:

| Variable | Accepted values | Default | Result |
| --- | --- | --- | --- |
| `ROSA_BUILD_EXTENSION` | `0`, `1` | `1` | Omit or build `_C`. |
| `USE_CUDA` | `auto`, `0`, `1` | `auto` | Detect, forbid, or require CUDA sources. |

Canonical commands:

```bash
# Pure PyTorch references, no native extension
ROSA_BUILD_EXTENSION=0 pip install --no-build-isolation -e .

# PyTorch references plus CPU RosaRuntime
USE_CUDA=0 pip install --no-build-isolation -e .

# Runtime plus CUDA training operators
USE_CUDA=1 CUDA_HOME=/path/to/cuda \
  pip install --no-build-isolation -e .
```

`USE_CUDA=1` is strict and fails when the toolkit is unavailable. `auto` adds
CUDA sources only when PyTorch's extension API finds `CUDA_HOME`.

## 4. Training APIs

Dense API:

```python
rosa_soft(
    query,                 # [B, T, H, D]
    key,                   # [B, T, H, D]
    value,                 # [B, T, H_v, D_v]
    *,
    max_suffix_length=32,
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)                       # -> [B, T, H, D_v]
```

Packed variable-length API:

```python
rosa_soft_varlen(
    query,                 # [N, H, D]
    key,                   # [N, H, D]
    value,                 # [N, H_v, D_v]
    cu_seqlens,            # CUDA int32 [S + 1]
    *,
    max_suffix_length=32,
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)                       # -> [N, H, D_v]
```

Common constraints:

- Q, K, and value share device and floating dtype.
- CUDA supports FP32, FP16, and BF16.
- `1 <= D <= 32`.
- `H >= 1`, `H_v >= 1`, and `H % H_v == 0`.
- `D_v >= 1`.
- Dense `B` and `T` are positive.
- Packed `N` is positive; empty segments inside `cu_seqlens` are valid.
- Non-contiguous public inputs are accepted and normalized internally where
  required by the implementation.

`cu_seqlens` starts at zero, ends at `N`, and is nondecreasing. Segment
boundaries isolate hard routes, suffix products, null competition, dropout
indices, and all gradients.

The pure PyTorch reference functions have the same signatures and output
shapes. They additionally support FP64 and CPU execution, but materialize
quadratic route state and are not intended for full training workloads.

## 5. Exact Forward Contract

Every input component is converted to a binary symbol:

```text
sign(x) = +1 when x > 0, otherwise -1
```

At query row `t`, non-null route `a` is valid for `1 <= a <= t`. It compares
the query suffix ending at `t` with the key suffix ending at `a - 1`, then
returns signed value position `a`.

Forward behavior is fixed:

1. Compute exact all-bit symbol equality.
2. Measure the exact matching suffix length, bounded by
   `max_suffix_length` and sequence boundaries.
3. Select the largest suffix length.
4. Resolve equal lengths by the largest route index, the latest route.
5. Return exact signed value for a non-null route.
6. Return exact zero when every non-null suffix length is zero.

No surrogate score, probability, dropout mask, or continuous value leaks into
forward. Changing `scale`, `dropout_p`, or `mismatch_scale` cannot change the
hard result.

## 6. Dense Backward Contract

Backward is a custom VJP, not the derivative of the discrete winner. It keeps
hard symbols numerically and assigns them a softsign derivative:

```text
dz/dx := 1 / (1 + |x|)^2
```

For every valid causal route it computes:

```text
normalized mismatch:
  m = sum_i (1 - z(q_i) z(k_i)) / (2D)

local gate:
  g = exp(-mismatch_scale * m)

raw suffix evidence:
  S = sum_l product_{r < l} g_r

route utility:
  U(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
```

The null route has fixed score `0.5`. Every non-null logit receives the
candidate-count correction `-log(N_t)`, then all valid routes participate in
one dense softmax. Values receive probability-weighted credit from the same
distribution used for Q/K credit.

Dense route support is semantic. Top-k candidates, ANN/LSH, sampled negatives,
hard-winner neighborhoods, score thresholds, and suffix-index pruning are not
valid optimizations of this operator.

## 7. Static Controls And RNG

| Control | Validation | Role |
| --- | --- | --- |
| `max_suffix_length` | integer `>=1`; clipped to available length | Hard and surrogate horizon. |
| `scale` | finite, positive, normal FP32 on CUDA | Multiplies route logits. |
| `dropout_p` | finite in `[0, 1 - 2^-24]` | Probability of dropping a post-softmax backward weight. |
| `mismatch_scale` | finite, positive, normal FP32 on CUDA | Controls mismatch leakage in local gates. |

These values are explicit per call. The operator does not derive or schedule
them from `D`, `T`, horizon, active suffix length, diagnostics, or training
step.

At `dropout_p=0`, the call does not advance PyTorch RNG state. With dropout
enabled and a backward graph required, Python saves one scalar seed. CUDA and
the reference reconstruct route masks from stable route indices. No quadratic
random tensor is saved. Dropout uses inverted scaling and affects Q, K, and
value VJPs, never hard forward.

## 8. Python To Kernel Data Flow

Dense training call:

```text
rosa_soft()
  -> shared Python validation
  -> hard_forward CUDA op
       output [B,T,H,D_v]
       packed Q signs [B,H,T] int32
       packed K signs [B,H,T] int32
  -> save Q/K/value, packed signs, controls, optional scalar seed
  -> surrogate_vjp_masked CUDA op during backward
       gradient mask: Q=1, K=2, value=4
  -> cast requested FP32 accumulators back to input dtype
```

Packed training follows the same sequence with token-major public inputs and
private sign buffers shaped `[H,N]`. If no input requires gradients, Python
calls hard forward directly and does not build custom autograd state.

The custom functions are once differentiable. Higher-order gradients are not
supported. Disabled Q, K, or value gradients select specialized work through
the gradient mask instead of computing and discarding full VJPs.

## 9. Internal Extension Schemas

The CUDA build registers four internal dispatcher schemas:

| Schema | Owner | Purpose |
| --- | --- | --- |
| `rosa_soft::hard_forward` | CUDA | Dense exact hard output and packed Q/K signs. |
| `rosa_soft::hard_forward_varlen` | CUDA | Segment-local packed exact hard output. |
| `rosa_soft::surrogate_vjp_masked` | CUDA | Dense requested-gradient VJP. |
| `rosa_soft::surrogate_vjp_varlen_masked` | CUDA | Packed requested-gradient VJP. |

These schemas are implementation details. Call the Python functions so input
validation, RNG ownership, autograd state, dtype restoration, and capability
errors remain coherent.

The CPU extension also registers `torch.classes.rosa_soft.RosaRuntime`. A CUDA
build contains both the CPU runtime and CUDA operators.

## 10. RosaRuntime Deployment API

`RosaRuntime` is the exact hard, stateful, packed deployment subset. Its Q/K
and payload widths are intentionally limited to `1..8` bits.

Constructor:

```python
RosaRuntime(
    num_heads,
    num_payload_heads=None,
    qk_bits=8,
    payload_bits=8,
    max_suffix_length=32,
)
```

Unpacked update quantizes floating logits and returns unpacked signed payloads
by default:

```python
output, matched_key_end_positions = runtime.update(
    query_logits,
    key_logits,
    payload_logits,
    cu_seqlens=None,
    *,
    stream=None,
    async_op=False,
    return_packed=False,
    sequence_ids=None,
)
```

`update_packed` accepts `torch.uint8` Q/K/payload symbols and defaults to a
packed uint8 output. Inputs may be dense `[B,T,H]` or varlen `[N,H]` with CPU
or CUDA `cu_seqlens`. Bits above the declared width are ignored by native
matching.

The second output uses global key end positions and `-1` for null. When
`return_packed=False`, null payloads are exact zeros rather than unpacked
negative bits.

Runtime lifecycle:

- states are `OPEN`, `CLOSING`, `CLOSED`, and `FAILED`;
- use a context manager or call `close()`;
- `async_op=True` returns work with idempotent, thread-safe `wait()`;
- at most two operations may be pending, providing bounded backpressure;
- an explicit CUDA stream overlaps staging transfers and CPU matching;
- the first update fixes slot count and order until `reset()`;
- optional unique `sequence_ids` make slot reorder errors explicit;
- a native update failure poisons the instance;
- `state_dict()` and `load_state_dict()` intentionally raise because no stable
  native checkpoint schema exists.

`stats()` returns state, edge, and deduplicated payload-symbol counts.
`memory_stats()` adds automata, sequence, and stable logical-byte counts. The
suffix horizon bounds matching work, not retained history; automata and payload
state currently grow approximately linearly with processed context.

## 11. Source Ownership

| Path | Responsibility | May define semantics? |
| --- | --- | :---: |
| `rosa_soft/__init__.py` | Public exports, build capability detection, unavailable-feature errors. | surface only |
| `rosa_soft/soft_contract.py` | Shared shapes, devices, dtypes, controls, FP32 bounds, dropout seed. | yes |
| `rosa_soft/soft_reference.py` | Inspectable equation-level hard forward and dense VJP oracle. | yes |
| `rosa_soft/soft.py` | CUDA wrapper, fake registrations, custom autograd, gradient masks. | orchestration only |
| `rosa_soft/csrc/export.cpp` | Internal dispatcher schemas and extension module initialization. | no |
| `rosa_soft/csrc/rosa_soft.cpp` | Native validation and CUDA dispatch. | must match contract |
| `rosa_soft/csrc/cuda/rosa_soft_kernels.cu` | Packed signs, hard route scan, online softmax, Q/K/value VJPs, private execution plans. | must match reference |
| `rosa_soft/runtime.py` | Runtime packing, staging, async ownership, lifecycle, output restoration. | runtime surface |
| `rosa_soft/csrc/rosa_runtime.cpp` | Exact suffix automata and payload history. | runtime hard semantics |
| `rosa_soft/testing.py` | Materialized deterministic inspection state. | no |
| `rosa_soft/diagnostics.py` | Detached route, symbol, and gradient summaries. | no |
| `setup.py` | Build variants, source list, ABI and package metadata. | build only |

Production code must not import from `benchmarks/`, `docs/research/`, or
`contrib/`. The build contains no legacy RWKV source or research-only operator.

## 12. Numerical And Integration Boundaries

- Hard output parity is exact for the same binary symbols.
- CUDA VJPs accumulate in FP32 and return the original input dtype.
- Global CUDA atomics make gradients numerically reproducible, not bitwise
  reproducible across launches.
- FP16 can underflow small dense-route gradients after the FP32 accumulator is
  cast back. Use GradScaler or keep projections in FP32 when tail support is
  important. BF16 preserves a wider exponent range.
- `torch.compile(..., backend="aot_eager", fullgraph=True)`, autocast,
  GradScaler, and RNG-preserving checkpoint use are validated.
- Inductor, CUDA Graph capture, higher-order gradients, and Runtime checkpoint
  serialization are not part of the current contract.
- Dense backward remains quadratic in route count. CUDA avoids quadratic
  persistent score, comparison, and dropout tensors by online computation and
  recomputation.

## 13. Validation Ownership

| Test area | Primary files |
| --- | --- |
| Shared reference equations and hard ties | `tests/test_soft_reference.py` |
| Dense CUDA parity and gradient masks | `tests/test_soft_cuda.py` |
| Packed segment isolation and parity | `tests/test_soft_varlen.py` |
| Compile, AMP, GradScaler, checkpoint | `tests/test_soft_integration.py` |
| Build variants, schemas, public placeholders | `tests/test_build_contract.py` |
| Runtime routing and lifecycle | `tests/test_runtime.py` |
| Inspection and detached summaries | `tests/test_diagnostics.py` |

Minimum local checks:

```bash
python -m py_compile setup.py rosa_soft/*.py
python -m pytest -q
```

Reference-only build check:

```bash
ROSA_BUILD_EXTENSION=0 \
  pip install --no-build-isolation -e .
python -m pytest -q tests/test_build_contract.py tests/test_soft_reference.py
```

CPU runtime build check:

```bash
USE_CUDA=0 pip install --no-build-isolation -e .
python -m pytest -q tests/test_build_contract.py tests/test_runtime.py
```

CUDA promotion also requires all dtype, grouped-head, non-contiguous,
gradient-mask, dropout-seed, varlen, compile, fitting, latency, register, and
spill gates documented in [Design](ROSA_SOFT_DESIGN.md) and `AGENTS.md`.

## 14. Change And Release Checklist

Before editing:

1. Classify the change as documentation, semantic contract, execution-only,
   runtime-only, or research-only.
2. Reject changes that silently alter dense candidate support, hard tie/null
   behavior, RNG indexing, defaults, or packed segment boundaries.
3. Put a new estimator in a separate module/operator instead of adding a mode
   switch to the frozen functions.

Before merging or releasing:

1. Confirm `git status` contains only intended files.
2. Run Python compilation and the full available test suite.
3. Build each affected variant: reference, CPU runtime, and CUDA.
4. Verify the four-schema CUDA surface and package capability matrix.
5. Compare reference and CUDA hard output and requested Q/K/value VJPs.
6. Run packed segment-boundary and dropout reproducibility gates.
7. Record environment and measured results under `validation/`.
8. Update production documentation only when the supported contract changes.
9. Keep research evidence under `docs/research/` and out of the package graph.

## 15. Known Production Limits

- No realistic full-model pretraining run has established scaling behavior.
- Dense backward is `O(T^2)` in valid routes and bounded suffix work.
- Runtime history is not checkpointable and retained state grows with context.
- Defaults are empirically conservative, not universally optimal.
- CUDA execution-plan thresholds are private measured choices, not semantic
  parameters.
- The frozen dense VJP is a training reference, not an exact derivative of the
  discrete route selection.

See [Dense Reference Freeze](DENSE_REFERENCE_FREEZE.md) for the reproduced
environment, fit results, runtime spot checks, and freeze policy.
