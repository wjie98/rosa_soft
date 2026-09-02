# RosaSoft Production Guide

This guide describes the maintained `rosa_soft` package. The package has one
job: train exact hard ROSA routes with a dense surrogate gradient and provide
small, independent correctness references. Long-context inference runtimes,
paging, compression, and storage policy are intentionally out of scope.

## 1. Sources Of Truth

Use these sources in order:

1. `rosa_soft/__init__.py` defines the supported Python surface.
2. `rosa_soft/soft_contract.py` defines tensor and scalar validation.
3. `rosa_soft/soft_reference.py` defines the surrogate equations.
4. `tests/test_sam.py` defines hard-route landmarks through an independent DP.
5. CUDA is an optimized implementation of those semantics.

Anything under `benchmarks/`, `docs/research/`, `validation/`, or `contrib/`
is evidence or history, not a public API. Raw `torch.ops` schemas and private
testing/diagnostic helpers are also not public extensions.

## 2. Public Surface

```python
from rosa_soft import (
    __version__,
    BUILD_CAPABILITIES,
    RosaSam,
    rosa_hard_reference,
    rosa_hard_varlen_reference,
    rosa_soft,
    rosa_soft_reference,
    rosa_soft_varlen,
    rosa_soft_varlen_reference,
)
```

`BUILD_CAPABILITIES` is an immutable value with:

```text
variant: "reference" | "cpu" | "cuda"
compiled_extension: bool
rosa_sam: bool
rosa_soft_cuda: bool
```

| Surface | reference | cpu | cuda |
| --- | :---: | :---: | :---: |
| PyTorch soft references | yes | yes | yes |
| `RosaSam` and hard references | unavailable | yes | yes |
| CUDA training operators | unavailable | unavailable | yes |

Import rejects a stale native extension that registers only part of the
required class/operator set. Unavailable public names raise an actionable
runtime error instead of silently changing implementation.

## 3. Build

Install PyTorch first and compile against that exact ABI:

```bash
pip install --no-build-isolation -e .
```

Build controls:

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `ROSA_BUILD_EXTENSION` | `0`, `1` | `1` | Disable or enable native code. |
| `USE_CUDA` | `auto`, `0`, `1` | `auto` | Detect, disable, or require CUDA. |

```bash
ROSA_BUILD_EXTENSION=0 pip install --no-build-isolation -e .
USE_CUDA=0 pip install --no-build-isolation -e .
USE_CUDA=1 pip install --no-build-isolation -e .
```

The CPU extension is ordinary C++ and does not require OpenMP. The source
distribution must include `.h`, `.cuh`, `.cpp`, and `.cu` files below
`rosa_soft/csrc`.

## 4. Training API

Dense tensors:

```python
output = rosa_soft(
    query,                    # [B,T,Hq,D]
    key,                      # [B,T,Hq,D]
    value,                    # [B,T,Hv,Dv]
    max_suffix_length=32,
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)
```

Packed variable-length tensors:

```python
output = rosa_soft_varlen(
    query,                    # [N,Hq,D]
    key,                      # [N,Hq,D]
    value,                    # [N,Hv,Dv]
    cu_seqlens,               # int32 [S+1]
    max_suffix_length=32,
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)
```

Common constraints:

- Q, K, and V share device and floating dtype.
- CUDA accepts FP32, FP16, and BF16.
- Q/K symbol width `D` is in `1..32`.
- `Hq % Hv == 0`; grouped value heads are repeated across Q/K heads.
- Packed offsets start at zero, end at `N`, and are nondecreasing.
- Empty packed segments are valid.
- Non-contiguous public tensors are accepted and normalized internally.

The PyTorch references have matching signatures, support CPU and FP64, and
materialize quadratic route state. They are correctness oracles for small
inputs, not throughput implementations.

## 5. Exact Hard Forward

Every Q/K vector becomes a symbol by `x > 0`. At query row `t`, route `a > 0`
compares the query suffix ending at `t` with the key suffix ending at `a-1`
and returns signed value position `a`.

The hard algorithm is:

1. Consider null and all causal key ends `< t`.
2. Find the globally longest exact symbol suffix.
3. On a tie, choose the greatest key end.
4. Gather the successor value and quantize it to `+1/-1`.
5. Return zero if no symbol matched.

Hard suffix length is unlimited. `max_suffix_length` controls backward only.
No surrogate control, dropout decision, or continuous magnitude can change
the hard result.

The CUDA hard scan is suitable for training batches and validation. Its
quadratic work is not presented as the solution for million-token inference.

## 6. Dense Surrogate Backward

For a local Q/K pair:

```text
m = popcount(q xor k) / D
g = exp(-mismatch_scale * m)
```

Each candidate accumulates consecutive gate products, up to
`max_suffix_length`, into suffix evidence `S`. Its utility is:

```text
U(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
```

Null has score `0.5`. Non-null candidates receive
`-log(candidate_count)`, and all logits are multiplied by `scale` before
softmax. Standard inverted post-softmax dropout is applied only to the VJP
carrier. Q/K use a softsign derivative at their hard signs.

Every valid route participates. The production CUDA path recomputes route
state with online softmax statistics instead of storing a quadratic matrix.
It saves Q, K, V, packed signs, static scalar controls, and at most one dropout
seed. `ctx.needs_input_grad` suppresses work only for unrequested input
gradients, never for candidate routes needed by another gradient.

## 7. Compact SAM Reference

`RosaSam` validates hard routing independently from the dense CUDA scan:

```python
sam = RosaSam(num_heads=4, symbol_bits=8)
matched_key_end = sam.update(query, key)
```

Accepted input forms:

- logits `[B,T,H,D]`;
- packed logits `[N,H,D]`, optionally with `cu_seqlens`;
- int32 symbols `[B,T,H]` through `update_packed`;
- int32 symbols `[N,H]` through `update_packed`, optionally with offsets.

Returned ends are local to each sequence. `-1` means null. The automata retain
Q/K state across chunks; sequence count is fixed after the first update until
`reset()`. Updates mutate state synchronously; concurrent callers must use
separate instances.

The implementation has three layers:

| File | Responsibility |
| --- | --- |
| `csrc/rosa_sam_core.h` | Plain C++ state, edges, match, append, clone. |
| `csrc/rosa_sam.cpp` | Batch/head parallelism and PyTorch custom class. |
| `sam.py` | Sign packing, device staging, shape handling, value gather. |

The C++ state contains only `max_length`, `suffix_link`, `latest_end`, and a
sparse edge-list head. Matching is performed before appending the same-row K.
Appending updates `latest_end` directly along the suffix-link chain. This is
compact and transparent, with adaptive typical behavior and quadratic worst
case on adversarial repetition. It is a validation backend, not an inference
framework.

`rosa_hard_reference` and `rosa_hard_varlen_reference` use the returned key end
to gather the signed successor value. `RosaSam` itself never stores values.

## 8. Semantic Oracle

SAM expected results must not come from another automaton. The test oracle is
the direct dynamic program:

```text
L[i,j] = Q[i] == K[j] ? L[i-1,j-1] + 1 : 0
```

For row `i`, it selects among `j < i` by maximum `L[i,j]`, breaking ties by
the greatest `j`. Computing `j == i` is still necessary because that DP state
can contribute to later rows; it is excluded only from current selection.

The focused suite covers exhaustive binary Q/K strings, random widths through
D32, clone-producing and periodic strings, ties, chunk boundaries, reset,
high-bit masking, grouped successor values, packed segments, and empty
segments.

## 9. Source Ownership

```text
rosa_soft/
  __init__.py
  soft_contract.py
  soft_reference.py
  soft.py
  sam.py
  initialization.py
  csrc/
    export.cpp
    rosa_sam_core.h
    rosa_sam.cpp
    rosa_soft.cpp
    cuda/rosa_soft_kernels.cu
    cuda/rosa_soft_vjp_common.cuh
    cuda/rosa_soft_streaming_kernels.cu
    cuda/rosa_soft_block_diagonal_kernels.cu
```

`setup.py` is the authoritative build graph. Historical Runtime and hard-index
sources are preserved under `contrib/runtime_legacy/`; they must not be
included, imported, or collected by default tests.

## 10. Validation And Release

Minimum CPU/SAM gate:

```bash
USE_CUDA=0 pip install --no-build-isolation -e .
python -m pytest -q tests/test_build_contract.py tests/test_sam.py
```

CUDA gate:

```bash
USE_CUDA=1 pip install --no-build-isolation -e .
python -m pytest -q \
  tests/test_soft_reference.py \
  tests/test_soft_cuda.py \
  tests/test_soft_varlen.py \
  tests/test_streaming_vjp.py \
  tests/test_block_diagonal_vjp.py
```

Run the full suite before release. Kernel changes also require register/spill,
latency, and peak-memory measurements on the intended GPU. Build an sdist and
inspect it to ensure native headers are present and archived Runtime sources
are absent.

The tag `rosa-soft-dense-reference-v1` remains the frozen comparison baseline.
New estimators use new names and stay outside the default operator until their
semantics and validation contract are explicitly adopted.
