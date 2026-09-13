# Memory and Complexity

[English overview](../README.md) | [中文概览](../README.zh-CN.md) |
[API](API.md) | [Methods](METHODS.md) | [Design](DESIGN.md)

## Scope

Current GPU bitflip storage is **O(T), not O(T log T)**, for fixed batch,
head counts, symbol/value widths and `rows`. It uses reusable row bands, not
a suffix array, sparse table, logarithmic hierarchy or all-edited-world tensor.
Exact filtering reduces work without storing a growing history of repairs.

Distinguish three quantities:

1. Saved forward tensors, whose lifetimes extend to backward.
2. Live backward scratch, allocated and reused while one head group runs.
3. Whole-model peak memory, including activations, gradients, optimizer state,
   temporary layout conversions and allocator effects.

Linear scratch does not imply a small constant, linear runtime, or a bound on
whole-model peak. In the formulas below, `T` is one dense sequence's length:

```text
C = chunks
S = B*H/C          Q/K streams in the current head group
Sv = B*Hv/C        value streams in the current head group
R = min(rows,T)    live query rows, with 1 <= rows <= 256
```

C must divide both H and Hv. Full inputs, saved packed Q/K and routes, output,
dY and final gradients do not shrink by C. Their storage is linear in T;
their distinct sizes and shared/aliased storage must be counted only once.

## Independent Bitflip

Allocation source: [`credit` in bitflip.cu](../rosa_soft/csrc/cuda/bitflip.cu).
The following arrays coexist and are reused for each successive query band.
Let `m = 1` byte for D<=8, otherwise `m = 4` bytes.

| Buffer | T < 65536, bytes | T >= 65536, bytes |
| --- | --- | --- |
| `meta` | `8*S*R*T` | `8*S*R*T` |
| `ends` | Alias of packed Q; no allocation | `4*S*R*T` |
| `counts` | `4*S*R` | `4*S*R` |
| `summary` | `16*S*R*(T+1)` | `32*S*R*(T+1)` |
| `work` | `8*S*R*(T+1)` | `16*S*R*(T+1)` |
| `events` | `8*S*R*T` | `4*S*R*T` |
| `excluded` | `2*m*S*R*(T+1)` | `2*m*S*R*(T+1)` |

The sum of these band allocations, before allocator rounding, is:

```text
T < 65536:   S*R * ((40 + 2*m)*T + 28 + 2*m)
T >= 65536:  S*R * ((64 + 2*m)*T + 52 + 2*m)
```

Thus the leading coefficients are 42/48 bytes per band cell for D<=8/D>8
on the narrow path, or 66/72 bytes on the wide path. The 65536 threshold
widens integer fields; it is not a suffix limit or a log(T) index dimension.

Other live arrays include two DP state buffers (`24*S*T` bytes), FP32 Q/K
credit (`8*S*D*T`), common credit (`8*S*T`), a reordered dY, and packed binary
V (`4*Sv*T*ceil(Dv/32)`). Native I/O and head slicing may introduce additional
linear-sized copies. The two Q/K credit buffers are allocated together when
either Q or K needs a gradient; V-only backward skips the bit-edit work.

In asymptotic notation, current-group scratch is bounded by
`O(S*R*T + S*T*(D+Dv) + Sv*T*Dv)`. Since R is capped at 256, it is O(T) at
fixed other dimensions. Bit width affects time and linear buffers even when
the narrow band coefficient stays unchanged.

## Joint QK and QKV

Allocation source: [`joint_credit` in joint.cu](../rosa_soft/csrc/cuda/joint.cu).
These modes perform one simultaneous shared-bit edit; their storage differs
from independent edits.

| Buffer | T < 65536, bytes | T >= 65536, bytes |
| --- | --- | --- |
| `meta` | `8*S*R*T` | `8*S*R*T` |
| `summary` | `16*S*R*T` | `32*S*R*T` |
| `repair` | `8*S*R*T*D` | `8*S*R*T*D` |
| `mark` and `list` | `8*S*R*T` | `8*S*R*T` |

The band total is `(32 + 8*D)*S*R*T` bytes below 65536 tokens, or
`(48 + 8*D)*S*R*T` otherwise. Additional arrays include state (`8*S*T`),
FP32 shared credit (`4*S*D*T`), converted dY and packed V. T<=1 returns
before allocating the band. Joint scratch is
`O(S*R*T*D + S*T*(D+Dv) + Sv*T*Dv)`, still O(T) for fixed widths and rows.

For example, below 65536 tokens the band coefficient is 64 bytes at D=4,
96 bytes at D=8, and 288 bytes at D=32. Do not use the independent 42-byte
estimate to size joint modes. Joint index limits are documented in [API](API.md).

## Sizing Example

For independent edits with B=1, H=Hv=192, D=Dv=4, T=4096 and rows=256:

| chunks | Active heads | Independent band | Joint band at D=4 |
| --- | --- | --- | --- |
| 1 | 192 | 8065.4 MiB | 12288 MiB |
| 3 | 64 | 2688.5 MiB | 4096 MiB |
| 6 | 32 | 1344.2 MiB | 2048 MiB |

These are **calculated buffer sizes**, not measured CUDA peaks or hardware
performance claims. They exclude all other arrays and model allocations.
MiB means 2**20 bytes. FP16 does not halve these integer band buffers.

At fixed rows and chunks, doubling T approximately doubles the band size
within one integer-width regime. When T<rows, R=T and the small-input band
grows quadratically until the row cap is reached. The width switch at 65536
introduces a step. Neither behavior changes the bounded-row O(T) storage
claim, but both matter when interpreting measured curves.

Reduce rows for finer memory control, or increase chunks to process fewer
heads simultaneously. Both add execution overhead; there is no universal
best setting. Head chunks preserve the entire history and final gradients.

## Other Paths

Hard CUDA forward visits O(BH*T^2) pairs and retains O(BH*T) packed symbols
and winning priorities, plus input/output-sized buffers. It does not retain
the whole DP matrix.

The generic soft backward stores a slab of width K: `O(BH*T*K)`, plus linear
statistics and gradient arrays. K is chosen internally, capped at 8192 and
reduced toward a score/utility budget of 1 GiB. Padding, minimum allocation
granularity and other buffers mean this is not a total-memory cap. Short
inputs can fit all diagonals in one slab, so their memory curve can look
quadratic before the cap. The FP16/Dv64 specialization instead replays
checkpoints with storage `4*G*ceil(T/32)*32` bytes, where G is the internally
selected, occupancy-capped CTA count. Its statistics pass also uses bounded
partial buffers. See [`plan`](../rosa_soft/csrc/cuda/soft.cu) and
[`launch`](../rosa_soft/csrc/cuda/soft_fp16.cu).

Packed soft backward executes segments separately; peak scratch follows the
largest live segment, while full packed inputs/gradients remain resident.
The CPU SAM grows with accumulated history, not just the latest input chunk.
It stores states/transitions, not all end positions or V history. Its current
latest-end propagation does not guarantee linear total time on every input.

## Measuring Peak Memory

For an operator-only measurement, allocate inputs and dY first, warm up the
chosen forward/backward path, then synchronize and record allocated bytes.
Reset peak statistics, run one complete forward/backward, synchronize, and
report the peak minus the baseline. Keep output/gradient lifetimes consistent
and avoid retaining previous autograd graphs between measurements.

Report B/T/H/Hv/D/Dv, dtype, gradient mask, tied/rows/chunks, execution mode
(eager, compiled or CUDA graph), and whether forward is included. Distinguish
`max_memory_allocated` from allocator-reserved memory and device-wide usage.
Measure end-to-end model peaks separately; activation checkpointing and head
chunking reduce different allocations.
