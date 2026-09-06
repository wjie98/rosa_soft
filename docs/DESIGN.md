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

`cuda/soft.cu` is the generic FP16/BF16/FP32 implementation. It scans exact
diagonal suffix recurrence in automatically sized tiles, maintains online
softmax statistics, replays score tiles for Q/K/V credit, and stores no full
candidate matrix. Workspace size is selected internally and capped; it is not
a public tuning parameter.

`cuda/soft_fp16.cu` is the long-sequence FP16/Dv64 specialization. It fuses
score/statistics/reverse work, uses checkpoint replay, and uses tensor cores
where the dense V contraction is suitable. Dispatch is internal and exact
with respect to the same surrogate.

Packed backward copies only the small offset vector to host, then applies the
same dense implementation to each nonempty segment and writes gradients back
to packed storage. This keeps one mathematical implementation and avoids the
old finite-window row kernel's cubic behavior. The metadata synchronization
and per-segment launches are the known varlen tradeoff.

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
