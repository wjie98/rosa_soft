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
   |    dy -> selected backward -> dq, dk, dv <-----------------+
   |          rosa_soft: dense unlimited suffix carrier
   |          rosa_bitflip: independent hard bit edits
   |
   +-- inference / validation
        pack sign bits -> CPU suffix automaton -> matched end
                                               -> binary V[end+1]
```

Dense tensors are `[B,T,H,D]`. Packed tensors are `[N,H,D]` with int32 CUDA
`cu_seqlens`. Q and K have the same head count and width `D <= 32`; V may use
fewer heads when `H % Hv == 0`.

Bitflip currently accepts dense tensors only, including T=0; it must not
be used to concatenate independently bounded documents.

## Native Surface

The existing soft path registers two dispatcher schemas:

```text
rosa_soft::forward(q, k, v, cu) -> (y, packed_q, packed_k)
rosa_soft::backward(q, k, v, dy, packed_q, packed_k, seed, cu,
                    scale, dropout_p, mismatch_scale, mask)
                    -> (dq, dk, dv)
```

An empty CUDA int32 `cu` identifies dense input. The Python autograd wrapper
saves Q/K/V, two `O(BHT)` packed symbol tensors, an optional scalar seed, and
offsets. It returns gradients only for requested Q/K/V inputs.

Bitflip adds two separate schemas without changing the soft ABI:

```text
rosa_soft::bitflip_forward(q, k, v, rows) -> (y, packed_q, packed_k, route)
rosa_soft::bitflip_backward(q, k, v, dy, packed_q, packed_k, route,
                            d, rows, mask) -> (dq, dk, dv)
```

Its autograd setup retains the winning priority and only the raw Q/K
activations that need gradients. Packed Q/K are retained whenever either
needs a gradient. V-only backward skips bit-edit DP. Both schemas have
FakeTensor implementations and a registered first-order autograd rule.

## Hard CUDA DP

Dense forward uses the shared `cuda/hard.cuh` integer DP in both extensions.
Each warp scans four adjacent Q/K diagonals, reuses query loads and combines
their latest-longest priorities before the atomic write. The two extensions
keep separate type/layout wrappers and separate floating-point build flags.

Packed `cuda/hard.cu` assigns one warp to each Q/K matrix diagonal. Every lane compares
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

### Soft Host Organization

The soft host interface uses `Input` for tensor handles and `Args` for the
three static estimator parameters. These are host-only types; CUDA kernels
still receive explicit pointers and scalar dimensions. `plan` in `soft.cu`
owns shape/device selection, including the generic fallback. The FP16 path
derives its loader split from the compile-time tensor-gradient mask instead
of exposing a second independent template choice.

Both soft paths share prior initialization and the final softsign derivative.
The FP16 statistics and checkpoint passes use the same O(T) prior allocation.
Generic statistics and reverse replay also share score/credit launch helpers;
this does not fuse additional kernels or change their arithmetic order.

## Independent-Bit Backward

For each Q/K activation bit e, let Y[e] be the complete hard output after
flipping only that bit, and Y the original output. At fixed upstream dY:

```text
credit[e] = sum_i dot(dY[i], Y[e,i] - Y[i])
dX[e]     = -sign(X[e]) * credit[e] / (2 * (1 + abs(X[e]))^2)
```

Here `sign(x)` is +1 for x>0 and -1 otherwise, including zero.

V backward scatters dY along the original hard route, then applies
`1/(1+abs(V))^2`. Thus bitflip and soft share forward semantics, not
backward semantics. Independent activation edits are not simultaneous
edits of shared projection parameters. The final nonlinear loss is not
re-evaluated for every bit.

The CUDA implementation has three stages:

1. Pack Q/K and find exact latest-longest hard priorities. Each warp scans
   four adjacent matrix diagonals, sharing query loads and combining their
   priorities in registers; ballots identify the previous unequal symbols.
   The packed priority is `length:32 | successor_position:32`; zero is null.
2. Replay a band of query rows. Diagonal state records the exact suffix
   length, the length available after repairing the last mismatch, and
   that mismatch's bit. Only mathematically irrelevant records are omitted,
   never a potentially nonzero output edit. Below T=65536, one 64-bit record
   holds both lengths, the mismatch bit, and the key end. Longer sequences
   use wider length fields and a separate end-position array.
3. Reduce each row into replacement summaries, combine repairs by bit and
   edited position, then accumulate complete original-to-replacement
   credit. Q/K credit is converted back to the input layout and dtype.

Destructive edits need more than a row's second-best candidate: a K edit
can invalidate several overlapping candidates. Four prefix-maximum fields
encode the required Q clipping and K left/clipped/right fallback ranges.
CUB performs the ordered row scan; exact repair priorities are aggregated
with integer atomics. Short and long rows use complementary warp/CTA
predicates, so exactly one launch owns each row.

Bounded shared caches hold summaries and local winners when they fit; larger
ranges use the same global representation. A first-touch K owner list avoids
revisiting unused positions, with an exact full-range fallback on overflow.
Record, event, and winner buffers reuse storage only after their last reader.
Large row CTAs prefetch one record ahead during summary and event processing.

If the original winner is also the latest terminal-symbol match, destructive
Q edits need only the current position and K fallback needs only the left
prefix. A uniform exact predicate selects this scalar prefix scan; all other
rows retain the four-field summary. Repair competition is unchanged. This
reuses existing workspace and adds no public parameter or kernel launch.

Credit subtracts binary values before multiplication by dY, avoiding a
large baseline and repair correction that could cancel after rounding.
For Dv<=512, a four-bit coefficient table shares those products within a row.
When all table groups fill complete packed words, a separate instantiation
omits the per-group bounds checks. The condition is `ceil(Dv/4) % 8 == 0`;
partial final groups keep their original channel mask. No padding, extra
accumulation terms, or workspace is added, and other shapes use the same code.
For wider values, large row CTAs cache the first 480 dimensions and accumulate
the remaining changed bits directly. Small row CTAs use only direct credit.
The fixed cache bound limits shared-memory use; it is not a value truncation.
These are exact representations of the same output difference.

The live band is `min(rows,T)`. Integer generation tags let repair storage
be reused without clearing it for every bit; it is reset before tag order
wraps. No full edited-world tensor or debug mode exists in the native API.
Space is `O(BH*rows*T + BHTD + BHTDv)`; time is quadratic in T at fixed
widths. Dense repeated codes generate more repair/credit work than
independent random codes, even though both have the same asymptotic bound.
Smaller `rows` reduces the workspace but adds band launches and leaves fewer
row CTAs available per launch. It is a memory/throughput tradeoff, not a change
to the estimator or its matching range.

The package builds `_bitflip` separately from `_C` to preserve bitflip's
non-fast-math floating arithmetic while leaving the soft/SAM build intact.
Neither extension JIT-compiles during import. CPU-only builds retain SAM
and expose clear CUDA-required errors for both training APIs.

### Bitflip Function Boundaries

| Function | Responsibility |
|---|---|
| Python `rosa_bitflip` | Validate the layout/rows argument and call the dispatcher |
| `_setup_context`, `_backward` | Save requested activations and register first-order autograd |
| Native `forward` | Validate/pack inputs, compute hard priorities, gather binary V |
| CUDA `dp` | Carry exact/repair suffix state across row bands and emit packed records |
| CUDA `row` | Build replacement summaries, combine repairs and accumulate output-difference credit |
| `baseline` | Read the fallback priority after a destructive edit, before considering repairs |
| Row-local `credit` | Compute `dot(dY, replacement_value - original_value)` |
| Native `credit` | Allocate/reuse the live band and launch DP/row kernels |
| Native `backward` | Validate saved inputs, obtain credit, apply the activation derivative and scatter dV |

`Record` and `RowSummary` are non-owning views of packed arrays, not persistent
model state. `Summary` contains the four fallback fields; `Maximum`, `Prefix`
and `ScalarPrefix` are small CUB reduction/scan functors. `priority` and `stamp`
encode winner ordering and generation tags. Packing, gather/scatter, validation
and FakeTensor functions implement the surrounding tensor interface.

Private `Credit` owns Q/K buffers in `[BH,D,T]` and a common buffer in
`[BH,2,T]`. Their layout is fixed, so no internal stride adapter or dynamic
tensor-list protocol is needed. Validation happens before invoking this
internal producer; public tensor layouts and returned gradients are unchanged.

`D` is Q/K symbol width; `Dv` is value width. `rows` is the live query-band size,
not a suffix window or CUDA block size. Internal `N` selects 32/256-thread row
blocks, `M` the excluded-bit mask type, `P` the priority width, and `C` the credit
representation. These are implementation choices, not additional public knobs.
The operators own no projection weights and do not retain a cache across calls.
The caller constructs Q/K/V and explicitly chooses the soft or bitflip estimator;
switching `model.train()`/`eval()` does not automatically select CPU SAM.

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

`Sam::step` calls `match` before `extend`; `go`/`set` handle transitions and
`copy_edges` makes independent clone edges. `State` stores length, suffix link,
latest end and the first edge; `Edge` stores a symbol, target and next edge.
The wrapper manages one automaton per sequence/head, not the external V cache.
This compact implementation walks linked edges and propagates latest ends along
suffix links; it does not guarantee linear total runtime on adversarial inputs.
Storage grows geometrically when a chunk exceeds current capacity. This avoids
copying the state arrays at every single-token update while still reserving
room for a large prefill in one request. Empty chunks preserve matching history
and the sequence-count contract.

## Source Map

```text
rosa_soft/__init__.py          public exports
rosa_soft/soft.py              CUDA/autograd wrapper
rosa_soft/bitflip.py           independent-bit API and autograd registration
rosa_soft/sam.py               CPU SAM wrapper and hard gather
rosa_soft/csrc/export.cpp      two dispatcher schemas
rosa_soft/csrc/rosa_soft.cpp   validation and dense/packed dispatch
rosa_soft/csrc/soft.h          private soft host arguments and declarations
rosa_soft/csrc/dispatch.h      shared FP16/BF16/FP32 type dispatch
rosa_soft/csrc/sam.{h,cpp}     exact suffix automaton
rosa_soft/csrc/cuda/hard.cu    exact unlimited hard DP
rosa_soft/csrc/cuda/hard.cuh   shared dense integer DP
rosa_soft/csrc/cuda/common.cuh shared dense-VJP primitives
rosa_soft/csrc/cuda/soft.cu    generic dense VJP
rosa_soft/csrc/cuda/soft_fp16.cu FP16/Dv64 specialization
rosa_soft/csrc/bitflip.cpp     bitflip dispatcher schemas and native entry
rosa_soft/csrc/cuda/bitflip.cuh bitflip packed fields and private declarations
rosa_soft/csrc/cuda/bitflip_io.cu bitflip hard forward and gradient conversion
rosa_soft/csrc/cuda/bitflip.cu  exact independent-bit DP and row credit
```
