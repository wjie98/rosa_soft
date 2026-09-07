# Production Design

## Public Data Flow

```text
q, k, v
   |
   +-- training -----------------------------------------------+
   |    pack sign bits -> exact CUDA diagonal DP -> hard y     |
   |                            |                               |
   |                            +-> save packed q/k             |
   |                                                            |
   |    dy -> dense unlimited suffix VJP -> dq, dk, dv <--------+
   |
   +-- inference / validation
        pack sign bits -> CPU suffix automaton -> matched end
                                               -> binary V[end+1]
```

Dense tensors are `[B,T,H,D]`. Packed tensors are `[N,H,D]` with int32 CUDA
`cu_seqlens`. Q and K have the same head count and width `D <= 32`; V may use
fewer heads when `H % Hv == 0`.

## Native Surface

Only two dispatcher schemas are registered:

```text
rosa_soft::forward(q, k, v, cu) -> (y, packed_q, packed_k)
rosa_soft::backward(q, k, v, dy, packed_q, packed_k, seed, cu,
                    scale, dropout_p, mismatch_scale, mask)
                    -> (dq, dk, dv)
```

An empty CUDA int32 `cu` identifies dense input. The Python autograd wrapper
saves Q/K/V, two `O(BHT)` packed symbol tensors, an optional scalar seed, and
offsets. It returns gradients only for requested Q/K/V inputs.

## Hard CUDA DP

`cuda/hard.cu` assigns one warp to each Q/K matrix diagonal. Every lane compares
one complete packed symbol, then a warp prefix maximum locates the previous
mismatch. This yields all equal-run lengths without recursion or lane
divergence over symbol bits. An atomic 64-bit priority stores `(length,
key_end+1)`, so unsigned `atomicMax` implements longest-first and latest-first
tie breaking. A final kernel gathers binary successor V. The scan covers every
diagonal and has no `W` parameter.

## Dense Soft VJP

For zero-based query position `i`, route `a` returns `V[a]` and matches
against the key ending at `a - 1`. With binary Q/K signs, the carrier is:

```text
m[i,a] = mean_d (1 - q[i,d] k[a-1,d]) / 2
g[i,a] = exp(-mismatch_scale * m[i,a])
S[i,a] = g[i,a] * (1 + S[i-1,a-1])
U(S)   = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
z[i,a] = scale * U(S[i,a]) - log(i)    for 1 <= a <= i
z[i,0] = scale * 0.5
p[i,:] = softmax(z[i,:])
carrier[i] = sum_{a=1..i} dropout(p[i,a]) * sign(V[a])
```

Out-of-domain suffix states are zero. Route zero is a null route with zero
value, not `V[0]`; row zero has only this route and never evaluates `log(0)`.
All signs use `1 / (1 + abs(x))^2` as their backward derivative. Dropout is
applied after softmax and only to the carrier. The actual forward uses only
the exact hard route.

`cuda/soft.cu` is the generic FP16/BF16/FP32 implementation. It scans exact
diagonal suffix recurrence in automatically sized tiles, maintains online
softmax statistics, replays score tiles for Q/K/V credit, and stores no full
candidate matrix. Workspace size is selected internally and capped; it is not
a public tuning parameter.

`cuda/soft_fp16.cu` is the long-sequence FP16/Dv64 specialization. It fuses
score/statistics/reverse work, uses checkpoint replay, and uses tensor cores
where the dense V contraction is suitable. Dispatch is internal and exact
with respect to the same surrogate.

Statistics scan four consecutive steps per thread and combine the local
affine maps using an ordered eight-lane CUB scan. Checkpoint generation uses
the corresponding ordered reduction. Affine composition is associative but
not commutative; a reordered reduction would change the suffix recurrence.
The statistics input loader issues two independent half2 loads before their
conversion and shared-memory stores. Invalid value routes remain zero.
Input and utility strides are padded for Tensor Core loads and utility reads.
Probability and credit share a compact 64x32 layout: `(route, row)` maps to
`route * 32 + (row ^ ((route & 6) << 2))`. This permutes aligned eight-half
segments for diagonal stores and both normal and transposed `ldmatrix` loads.
Two empty corners of the utility rectangle are not computed; every actual
causal candidate is still included. Probability and credit contractions retain
high and residual FP16 components with FP32 accumulation.

On SM75 and newer, the FP16 backward uses a small `Mma` helper with documented
`ldmatrix` and `mma.m16n8k8` register layouts. Q/K/V contractions retain their
results in FP32 registers. Four warp shuffles arrange each 16x16 result for
coalesced global atomic writes; no shared-memory output tile is needed.
Adjacent reverse row blocks overlap by 32 value rows. Eight FP32 registers per
thread retain that dV overlap until the next block, and only final writes are
reordered. The accumulator resets for every diagonal task and flushes at its
first row block. Different tasks and GQA heads still accumulate atomically;
their FP32 summation order is not deterministic, so repeated gradients need
not be bitwise identical even though hard outputs are exact.
The helper uses CUDA instructions directly, without a CUTLASS build dependency;
older devices use the existing generic path.

Packed backward copies only the small offset vector to host, then applies the
same dense implementation to each nonempty segment and writes gradients back
to packed storage. This keeps one mathematical implementation and limits work
to the sum of squared segment lengths. Metadata synchronization and
per-segment launches are the variable-length execution tradeoff.

## CPU SAM

`sam.h` contains the automaton and `sam.cpp` contains the PyTorch custom-class
binding. Each sequence/head owns one stateful automaton. Query matching occurs
before the key at the same position is appended, preserving causality. Every
state tracks its latest end position, and clone transitions are copied rather
than shared so later mutation cannot corrupt another state.

`RosaSam.update` returns local matched key ends; no match is `-1`. It can process
successive chunks as long as the number of sequences does not change. The
convenience `rosa_hard` call is stateless and gathers only values present in
the supplied chunk.

## Source Map

```text
rosa_soft/__init__.py          public exports
rosa_soft/soft.py              CUDA/autograd wrapper
rosa_soft/sam.py               CPU SAM wrapper and hard gather
rosa_soft/csrc/export.cpp      two dispatcher schemas
rosa_soft/csrc/rosa_soft.cpp   validation and dense/packed dispatch
rosa_soft/csrc/sam.{h,cpp}     exact suffix automaton
rosa_soft/csrc/cuda/hard.cu    exact unlimited hard DP
rosa_soft/csrc/cuda/common.cuh shared dense-VJP primitives
rosa_soft/csrc/cuda/soft.cu    generic dense VJP
rosa_soft/csrc/cuda/soft_fp16.cu FP16/Dv64 specialization
```
