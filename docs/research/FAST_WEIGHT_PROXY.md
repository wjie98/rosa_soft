# Parameter-Free Fast-Weight Proxy

## Status

This is a research-only PyTorch estimator. It does not modify the frozen
`rosa_soft` package, CUDA schemas, hard routing, or dense production VJP.

The experiments ask whether one ROSA head can use either a short suffix
fingerprint or a compact current-symbol feature as an online memory key. The
proxy has:

- exact hard ROSA forward;
- no trainable proxy parameters;
- no auxiliary or imitation loss;
- hard `+1/-1` numerical features with the production softsign VJP;
- an ephemeral per-sequence fast-weight state;
- a custom backward that never exposes the proxy value to the residual stream.

It is a biased VJP. Neither a recurrent model nor a deterministic sketch can
be an unbiased derivative of the deterministic hard route.

## Exact local feature

For one hard bit, let

```text
c = exp(-mismatch_scale / D)
A = (1 + c) / 2
B = (1 - c) / 2
psi(x) = [sqrt(A), sqrt(B) x]
```

Then `<psi(q), psi(k)>` is `1` for a match and `c` for a mismatch. For a
`D`-bit symbol,

```text
phi(x) = tensor_product_d psi(x_d)
```

has dimension `2**D` and exactly satisfies

```text
<phi(q), phi(k)> = exp(-mismatch_scale * Hamming(q, k) / D).
```

For suffix length `L`, the tensor product of the `L` local features has an
inner product equal to the product of the `L` local ROSA match gates. Joining
all lengths therefore reproduces the production raw suffix evidence before
utility, null calibration, and route softmax.

## Online memory

Candidate `a` writes hard `V[a]` under the K fingerprint ending at `a-1`.
Query row `t` reads with the Q fingerprint ending at `t`.
The fast-weight proxy has no candidate softmax and therefore does not use the
production attention `scale`/temperature.

The two online updates are normalized linear attention and a normalized delta
rule. With `M = transpose(S)`, the latter is:

```text
k = normalize(K_fingerprint[a - 1])
prediction = transpose(M) @ k
M = M + outer(k, hard_V[a] - prediction)
y_proxy[t] = transpose(M) @ normalize(Q_fingerprint[t])
```

For a repeated unit key, the update makes `transpose(M) @ k` equal the newest
value exactly. A later contextual experiment found that this overwrite
property is harmful when several associations should coexist. The additive
state

```text
M = M + outer(k, hard_V[a])
z = z + k
y_proxy[t] = (transpose(M) @ q) / dot(z, q)
```

was substantially more reliable despite being less similar to hard latest
routing.

## Ablation ladder

`benchmarks/fast_weight_proxy.py` implements:

| Proxy | Fingerprint | Memory |
|---|---|---|
| `linear_attention` | current symbol | additive normalized read |
| `delta_rule` | current symbol | normalized delta rule |
| `state_linear_attention` | complete degrees `0..1` | additive normalized read |
| `state_quadratic_attention` | complete degrees `0..2` | additive normalized read |
| `state_cubic_attention` | complete degrees `0..3` | additive normalized read |
| `state_*_delta` | the same degree ladder | normalized delta rule |
| `single_suffix_delta` | one exact fixed-length suffix | delta rule |
| `single_suffix_sketch_delta` | one fixed-length TensorSketch | delta rule |
| `exact_suffix_delta` | exact direct sum of lengths `1..L` | delta rule |
| `sketch_suffix_delta` | sketched direct sum of lengths `1..L` | delta rule |

The exact suffix variants are validation oracles. Their largest level grows as
`2**(D*L)`. TensorSketch uses fixed hash/sign maps and FFT convolution; these
maps are constants reconstructed in backward, not model parameters.

The state feature ladder truncates only the current `D`-bit symbol, never the
suffix. For `D=8`, degrees one, two, three, and full have dimensions `9`, `37`,
`93`, and `256`. Every complete degree is retained, so there is no random hash
or feature-selection seed.

## GPU 0 results

All measurements below used the RTX 3070, identical token/model seeds, hard
forward, optimizer, and 500 fitting steps unless stated otherwise. Success is
`best_loss < 1e-3`. Gradient cosine is Q/K cosine against complete bitflip on
16 small random cases.

### Complete D=4, hard W=3 ladder

| Estimator | Success | Median best loss | Q/K cosine | ms/step |
|---|---:|---:|---:|---:|
| production | 2/4 | 0.065657 | 0.518 | 3.56 |
| complete bitflip | 2/4 | 0.252625 | 1.000 | 10.54 |
| local linear attention | 1/4 | 0.223012 | 0.392 | 25.46 |
| local delta rule | 1/4 | 0.149261 | 0.520 | 23.12 |
| exact fixed L=3 suffix | 3/4 | 0.000745 | 0.165 | 24.07 |
| exact lengths 1..3 | 2/4 | 0.044947 | 0.417 | 24.07 |
| sketched lengths 1..3, r=64 | 2/4 | 0.127244 | 0.333 | 30.52 |
| sketched fixed L=3, r=256 | 3/4 | 0.000645 | -0.006 | 29.20 |

Every estimator matched the hard forward in all 16 gradient cases. The
fixed-length result supports the one-head/one-short-pattern hypothesis. Mixing
all suffix lengths in one normalized memory introduced interference and did
not improve fitting.

The fixed exact fingerprint has 4096 features. The useful `r=256` sketch
reduces its delta state by 16 times, from 16,384 to 1,024 floats per head for
`Dv=4`.

### Width and hash sensitivity

For the fixed `D=4,L=3` sketch with hash seed zero:

| Width | Success | Median best loss |
|---:|---:|---:|
| 64 | 2/4 | 0.122843 |
| 128 | 2/4 | 0.001841 |
| 256 | 3/4 | 0.000645 |
| 512 | 2/4 | 0.127954 |

The curve is not monotone because changing the width changes the fixed hash
collisions. At `r=256`, four hash seeds produced `3/4`, `2/4`, `2/4`, and
`3/4`, or `10/16` total. Model seed 2 failed under every hash and also failed
the exact fixed-length proxy. TensorSketch is therefore useful but not yet a
seed-insensitive final representation.

### Hard W=8 with proxy L=3

The scalable test kept the proxy horizon at three while increasing the hard
ROSA horizon to eight and trained for 1000 steps:

| Estimator | Success | Median best loss | ms/step |
|---|---:|---:|---:|
| production | 2/4 | 0.070646 | 3.46 |
| complete bitflip | 2/4 | 0.252264 | 12.39 |
| fixed suffix sketch, L=3, r=256 | 3/4 | 0.000250 | 29.32 |

This is positive evidence that proxy fingerprint length need not equal the
hard retrieval horizon. It is not evidence for arbitrary-context scaling;
the fitting task is still a tiny repeated-motif probe with a trainable readout.

### Context-conditioned state fitting

A follow-up tested the original fast-weight interpretation directly: `S` is a
per-sequence compressed `K/V` state, and `S q` supplies credit to contextual
Q/K/V projections. Hard ROSA still supplies every numerical forward value.

The shortcut-free reset-RNN gate gives Q/K contextual residuals while proving
that the post-reset query residual alone cannot identify the answer. Eight
model/data seeds used `D=8`, `Dv=4`, two heads, 400 steps, 32 train pairs, and
16 held-out pairs on GPU 0:

| Estimator | Strict passes | Mean held-out exact | Worst exact | Median first train-exact |
|---|---:|---:|---:|---:|
| production | 6/8 | 0.9873 | 0.9375 | 98.0 |
| degree-2 linear state, 37 features | **7/8** | **0.9971** | **0.9766** | 89.5 |
| degree-3 linear state, 93 features | 6/8 | 0.9902 | 0.9688 | 78.5 |
| full linear state, 256 features | 7/8 | 0.9951 | 0.9766 | 87.5 |

The degree-one state passed only `1/4` in the first seed block. More capacity
is therefore necessary, but degree three and the full basis did not improve on
the complete quadratic level. For `Dv=4`, the quadratic state stores 148
matrix floats plus 37 normalizer floats per head, versus 1,024 plus 256 for the
full map.

The update rule mattered more than maximum feature dimension. On the difficult
contextual seed 1, all delta-rule dimensions failed after 1,000 steps; the full
normalized linear state passed. This supports an associative `S = sum(KV)`
credit state rather than a latest-overwrite proxy.

The token-only fitting model has no contextual input to Q/K. On that control,
degree one/two/three linear states passed `1/4`, while the full state passed
`2/4`. Expansion cannot manufacture contextual patterns by itself.

On 16 random `D=8` VJP cases, the quadratic state's Q/K cosine to exact
bitflip was `0.047`, versus production's `0.203`. Its contextual gain therefore
does not come from approximating bitflip more closely. Exact bitflip can still
be used in the batch-one fitting probe; its current research implementation
rejects the paired batched contextual gate.

### Controlled association, head, and depth scaling

The follow-up matrix changed one axis at a time around `A=4`, two Q/K heads,
and one reset-GRU layer. Four seeds used 400 steps, 16 train pairs, eight
held-out pairs, `D=8`, `Dv=4`, and one shared value head. `Depth` below is the
reset-GRU encoder depth; the model still contains one ROSA route layer.

| Associations | Q/K heads | Depth | Production exact | Quadratic exact | Production payload route | Quadratic payload route | Production / quadratic ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | 0.719 | **0.969** | 0.719 | **0.984** | 9.6 / 20.0 |
| 4 | 1 | 1 | **0.789** | 0.754 | **0.820** | 0.738 | 13.8 / 30.1 |
| 4 | 2 | 1 | 0.797 | **0.953** | 0.902 | **0.969** | 13.9 / 30.2 |
| 4 | 2 | 2 | 0.711 | **0.965** | 0.832 | **1.000** | 21.6 / 37.3 |
| 4 | 2 | 4 | 0.656 | **0.867** | 0.840 | **0.898** | 36.1 / 53.0 |
| 4 | 4 | 1 | 0.918 | **0.949** | 0.977 | **1.000** | 16.1 / 33.8 |
| 8 | 2 | 1 | 0.789 | **0.797** | **0.883** | 0.859 | 23.6 / 52.3 |
| 16 | 2 | 1 | 0.619 | **0.672** | 0.696 | **0.770** | 40.5 / 95.9 |

Every shortcut check passed, but four seeds expose substantial tails. At
`A=8`, for example, quadratic exact accuracies were `1.000`, `0.758`, `1.000`,
and `0.430`. The matrix supports three bounded conclusions:

1. the quadratic field is unreliable with only one Q/K head;
2. two heads give a large benefit on the small controlled task, while four
   heads make production competitive by adding route redundancy;
3. neither estimator solves `A=16` in 400 steps, and a fourth GRU layer hurts
   both, so this is not evidence of a monotone scaling law.

The reported timing includes a Python token loop and an unfused PyTorch proxy.
It is useful only for paired comparisons within a row, not as a ROSA kernel
throughput result.

### Uniform next-token training

`state_attention_pretraining.py` removes the directed recall loss. An episode
contains storage `(cue, payload)` pairs followed by query triples
`(cue-specific reset, cue, payload)`. The reset makes paired complementary
assignments have bit-identical query residuals. Training uses one uniform
next-token cross entropy over every sequence position and samples fresh random
mappings on every optimizer step. There is no recall weighting, auxiliary
loss, proxy parameter, or finite mapping set to memorize.

For `A=4` and 16 payload codes, storage payloads are sampled without
replacement. The irreducible full-sequence loss is therefore

```text
(log(16) + log(15) + log(14) + log(13)) / 19 = 0.56235
```

while query recall can still reach one. Four GPU-0 seeds used 2,000 steps, 32
pairs per fresh training batch, 16 held-out pairs, depth two, `D=8`, and
`Dv=4`:

| Q/K heads | Estimator | Validation loss | Recall mean / worst | Episode exact | Payload route | ms/step |
|---:|---|---:|---:|---:|---:|---:|
| 2 | production | 0.951 | 0.406 / 0.063 | 0.055 | 0.525 | 30.9 |
| 2 | quadratic | **0.738** | **0.723 / 0.188** | **0.500** | **0.770** | 54.5 |
| 4 | production | **0.632** | **0.908 / 0.852** | **0.680** | **0.949** | 29.4 |
| 4 | quadratic | 0.662 | 0.896 / 0.797 | 0.633 | 0.777 | 52.6 |

Zero-route and current-value controls remained between `0.053` and `0.078`
mean recall, so the successful predictions require hard history. With four
heads, exact routing to the designated storage payload is sufficient but not
necessary: the output projection can compose several historical hard values.

The two-head quadratic estimator has a clear average advantage but a long
initialization tail. Its failed seed 2 improved only from `0.188` at 2,000
steps to `0.484` at 5,000. Raising both estimators to four heads removed the
catastrophic tail; production then slightly outperformed quadratic. This is a
negative result for treating the proxy as a universal production replacement,
and positive evidence that route redundancy is more important than adding
more surrogate machinery.

### Controlled initialization and V capacity

The next experiment tested whether the two-head tail starts before the first
optimizer step. It adds no proxy parameter, auxiliary loss, parameter tying,
or training schedule. The benchmark-only initializer constructs an
orthogonal basis per head. For four of eight Q/K bits,

```text
Wq = W
Wk = rho W + sqrt(1 - rho**2) U,  rho = 0.97
```

where `W` and `U` are orthogonal rows. The other four Q/K rows are private and
mutually orthogonal. All Q/K weights remain independent trainable parameters
after initialization. The row gain targets a median absolute projected logit
of `0.6`; this is an initialization constant, not a runtime temperature.

The same experiment decouples the four-bit payload vocabulary from hard V
width. Orthogonal V rows use the same `0.6` target. `Dv=8` therefore changes
the actual hard V symbol and output projection; it is not a hidden proxy-only
feature. Module-specific seeds keep all common modules identical across the
controlled configurations. Ordinary `state_attention_pretraining.py` runs do
not enable that reset unless a custom initializer is requested.

Before training, partial sharing reduced the positive cue Q/K Hamming distance
from `4.35` to `2.24`, but only `12.5%` of designated payload routes were hard
selected. The gain is therefore not an initialized hard solution. It gives
the quadratic state field a roughly distance-independent similarity bias from
which the exact hard route can learn.

Four model seeds crossed with two independent fresh-data streams used 1,000
steps and one fixed 64-pair validation set on GPU 0:

| Configuration | Recall mean / worst | Episode exact | Payload route | Mean order delta |
|---|---:|---:|---:|---:|
| E0: default Q/K, default V4 | 0.549 / 0.148 | 0.145 | 0.736 | 0.317 |
| E7: independent orthogonal Q/K, default V4 | 0.600 / 0.123 | 0.166 | 0.713 | 0.189 |
| E1: partial-shared Q/K, default V4 | 0.728 / 0.309 | 0.345 | 0.922 | 0.196 |
| E6: partial-shared Q/K, orthogonal V4 | 0.858 / 0.750 | 0.532 | 0.986 | 0.072 |
| E3: partial-shared Q/K, orthogonal V8 | **0.969 / 0.828** | **0.886** | 0.959 | **0.053** |

E7 shows that orthogonality and the larger Q/K logit scale do not remove the
tail without Q/K correlation. E6 shows that a well-conditioned V projection
is a large, zero-parameter-count improvement even at `Dv=4`. E3 then gives an
additional `0.111` mean recall from actual V widening. Its model has 16,128
parameters versus 15,744 for V4, an increase of 384 or 2.4%; the quadratic
fast-weight matrix grows from `37x4` to `37x8` entries per head. The measured
step-time difference was below one percent on this short Python/GRU probe, but
that result does not predict a fused long-sequence kernel's V-dimension cost.

Zero-route and current-value controls remained near the `1/16` chance level.
The exact-paired-episode gate was still strict: E3 passed only 2/8 runs at
1,000 steps, and its slowest run had 0.828 token recall. This is a substantial
tail reduction, not proof that bad basins are gone.

The same eight initial models and data streams were then trained with the
production VJP. This isolates initialization from the estimator:

| Configuration | Production mean / worst | Quadratic mean / worst | Mean paired delta | Quadratic wins |
|---|---:|---:|---:|---:|
| E0 | 0.248 / 0.064 | 0.549 / 0.148 | +0.302 | 7/8 |
| E6 | 0.292 / 0.047 | 0.858 / 0.750 | +0.566 | 8/8 |
| E3 | 0.832 / 0.699 | 0.969 / 0.828 | +0.137 | 6/8 |

Partial-shared Q/K plus orthogonal V4 is therefore not a generally solved
hard route that any VJP can exploit: production barely improved on E0, while
the quadratic state field used the same initialization in all eight runs.
V8 helped both estimators and reduced their gap.

The two slow E3 data-order runs were retrained from scratch for 2,000 steps.
Quadratic recall reached `0.998` and `1.000`; production reached `0.977` and
`0.980`. Median first train-exact step was `1055.5` for quadratic and `1344`
for production. These two cases support a convergence-speed interpretation,
but they do not exclude unseen bad seeds.

A 500-step width/output screen gave no reason to use V16: E3 V8 reached
`0.922 / 0.781` mean/worst recall, while V16 reached `0.908 / 0.658`.
Initializing the output as a scaled V transpose was actively unstable at
`0.765 / 0.082`. Keep paired output only as a recorded negative control; do
not promote it into the candidate path.

### V representation and credit assignment

`value_codec_proxy.py` and `state_attention_value_ablation.py` test whether V
needs finer numerical values than the Q/K match symbols. They are research
paths only. Every variant uses the same exact hard suffix route, controlled
Q/K initialization, next-token objective, and quadratic Q/K state VJP. The V
representations are:

- `binary`: the existing hard `-1/+1` value with softsign VJP;
- `uniform`: `2**k` fixed levels on `[-1, 1]` with a clipped identity STE;
- `exponential`: one sign bit and `k-1` exponent bits, with fixed magnitudes
  spanning `2**exponent_min` through one and a clipped identity STE;
- `rms`: continuous V normalized to unit RMS per token/head;
- `float`: unconstrained continuous V.

The quantized codecs have no learned or per-token scale. In particular,
`exp4e8` changes the fixed exponent interval from `[-4, 0]` to `[-8, 0]`; it
does not add parameters. `k` is the total scalar bit count, including sign.

Four model seeds with one data seed used 500 steps. The first screen fixed
`Dv=4`; binary V8 is the capacity control:

| Representation | Recall mean / worst | Payload route |
|---|---:|---:|
| binary V4 | **0.680 / 0.545** | 0.940 |
| uniform2 V4 | 0.602 / 0.541 | 0.866 |
| uniform4 V4 | 0.471 / 0.070 | 0.744 |
| exp2 V4 | 0.570 / 0.420 | 0.854 |
| exp3 V4 | 0.665 / 0.418 | 0.930 |
| exp4 V4 | 0.579 / 0.494 | 0.997 |
| exp4e8 V4 | 0.520 / 0.111 | 0.812 |
| RMS V4 | 0.295 / 0.199 | 0.996 |
| float V4 | 0.127 / 0.070 | 0.732 |
| binary V8 | **0.922 / 0.781** | 0.934 |

More scalar levels did not replace state width. Uniform2 V4 and binary V8
both store eight bits per value, but binary V8 was much better. The quadratic
state has eight independent value directions in the latter and only four in
the former. Codebook cardinality alone is therefore not the useful capacity
measure for this proxy.

A matched V8 screen reached the same conclusion:

| Representation | Recall mean / worst | Payload route |
|---|---:|---:|
| binary V8 | **0.922 / 0.781** | 0.934 |
| uniform2 V8 | 0.814 / 0.732 | 0.938 |
| exp2 V8 | 0.811 / 0.707 | 0.917 |
| uniform4 V8 | 0.540 / 0.070 | 0.688 |
| exp3 V8 | 0.619 / 0.078 | 0.750 |
| exp4 V8 | 0.616 / 0.434 | 0.797 |
| RMS V8 | 0.183 / 0.066 | 0.982 |
| float V8 | 0.125 / 0.074 | 0.750 |

The continuous failures are not just a shortage of route capacity. RMS V8
kept unit value RMS and reached near-perfect routes, yet failed to decode the
payload. Raw float also exploited an unsafe amplitude degree of freedom: at
500 steps, V4 representation RMS ranged from `11.4` to `20.3` and maximum
absolute values from `34.6` to `69.2`. Float V8 showed the same behavior. Hard
Q/K routing alone does not prevent value-amplitude hacking when a biased dense
V carrier supplies the backward signal.

The second screen kept the same hard forward and Q/K VJP but replaced dense V
credit with the direct derivative of the currently selected hard value. This
gradient is exact conditional on the selected route for float/RMS, and a
selected-value STE for quantized codecs. It cannot teach unselected V
candidates. Representative proxy-to-selected changes were:

| Representation | Dense proxy mean / worst | Selected mean / worst |
|---|---:|---:|
| binary V4 | **0.680 / 0.545** | 0.492 / 0.090 |
| binary V8 | **0.922 / 0.781** | 0.881 / 0.531 |
| uniform2 V4 | 0.602 / **0.541** | **0.698** / 0.086 |
| exp3 V4 | **0.665 / 0.418** | 0.515 / 0.070 |
| RMS V4 | 0.295 / **0.199** | **0.427** / 0.080 |
| float V4 | 0.127 / **0.070** | **0.387** / 0.051 |

Selected credit reduced float amplitude growth, but produced a strongly
bimodal training outcome. For example, uniform2 V4 had per-seed recalls
`[1.000, 0.820, 0.887, 0.086]`. The dense V carrier is therefore biased but
useful: it preconditions values at candidates that the current hard route has
not selected. Removing that credit makes Q/K/V co-adaptation less stable.

Finally, eight model/data seed pairs used 1,000 steps for the two competitive
two-bit V8 codecs. The matching binary E3 result is reused from the controlled
initialization experiment; its 500-step result is exactly reproduced by the
codec path.

| Representation | Recall mean / worst | Payload route | Strict passes |
|---|---:|---:|---:|
| binary V8 | **0.969 / 0.828** | **0.959** | 2/8 |
| uniform2 V8 | 0.894 / 0.715 | 0.967 | 2/8 |
| exp2 V8 | 0.866 / 0.736 | 0.867 | 1/8 |

Uniform2 had two runs with perfect routes but only `0.805` and `0.715` token
recall, directly exposing a value-code/decode failure. Low exp2 runs more often
lost the route as well, so its value trajectory also disturbed Q/K learning.
The exponent code is viable, but it did not beat uniform quantization or the
simpler binary value. Increasing the exponent span to `[-8, 0]` was worse.

Keep binary V8 plus dense proxy credit as the current research default. Do not
add a value codec or selected-gradient option to the production ABI. Revisit
an exponent code only on a task whose V semantics genuinely span multiplicative
scales, decay rates, or jump magnitudes; the current payload is a discrete
content code, for which extra state directions and binary saturation are more
useful than scalar dynamic range.

### Exact bitflip versus current estimators

The final comparison gives all three estimators the optimized E3
initialization: partial-shared orthogonal Q/K and orthogonal binary V8. For
each model/data seed pair, production, exact bitflip, and quadratic state
attention start from deep copies of the same model and see the same fresh data
stream. The only changed behavior is the backward estimator. Before training,
the designated payload route accuracy is still only `0.125`; the initializer
conditions the projections but does not install the solution.

The exact bitflip implementation now supports batches by splitting all
activation-coordinate counterfactuals into memory-bounded chunks. Every Q, K,
and V bit is still flipped exactly once, and every affected hard output is
recomputed. A batched VJP is bit-exact against concatenated batch-one VJPs.
Chunking changes neither the estimator nor the hard forward.

Eight model/data seed pairs used 1,000 steps, 32 complementary training pairs
per fresh batch, one fixed 64-pair validation set, two Q/K heads, one V head,
depth two, `D=8`, `Dv=8`, and `W=1` on GPU 0. The table includes both exact
bitflip scales studied at the target horizon. It also separates deterministic
reference RosaSoft from two repetitions of the compiled CUDA estimator:

| Estimator / implementation | Bitflip VJP scale | Validation loss | Recall mean / worst | Episode exact | Payload route | Strict passes | Median first train-exact | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production RosaSoft, reference | - | 0.713 | 0.839 / 0.750 | 0.431 | 0.878 | 0/8 | none | 33.6 |
| production RosaSoft, CUDA repeats | - | 0.718-0.737 | 0.835-0.854 / 0.725-0.760 | 0.381-0.444 | 0.879-0.883 | 0/8 | 877-981 | **29.2-29.8** |
| complete bitflip, raw | 1.0 | 0.779 | 0.940 / 0.773 | 0.808 | 0.937 | 1/8 | **193.5** | 68.2 |
| complete bitflip, calibrated | 0.003 | 0.758 | 0.773 / 0.516 | 0.385 | 0.757 | 1/8 | **161.5** | 67.2 |
| quadratic state | - | **0.617** | **0.969 / 0.828** | **0.886** | **0.959** | **2/8** | 443.5 | 52.1-53.0 |

All shortcut checks passed and all numerical forwards were the same hard
route. The CUDA RosaSoft VJP uses atomic reductions. With dropout disabled,
identical initialization, data, and explicit RNG seeds still produced the two
CUDA ranges above, while the reference run and quadratic runs reproduced
exactly. This is trajectory sensitivity to reduction order, not an
initialization difference. It does not change the estimator ranking.

The raw finite-difference unit is not naturally comparable to the other
backward fields. Before global clipping, its mean Q/K parameter-gradient norm
was `6.260`, versus `0.00345` for CUDA RosaSoft, `0.00357` for reference
RosaSoft, and `0.01150` for quadratic state. Exact bitflip therefore gained a
static `gradient_scale` which multiplies Q/K/V VJPs only; it leaves the hard
forward and every counterfactual direction unchanged.

A 300-step screen on the four difficult `data_seed=1` runs showed a sharp
short-horizon optimum:

| Bitflip scale | Validation loss | Recall mean / worst | Episode exact | Payload route | Mean Q/K gradient norm |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.934 | 0.794 / 0.451 | 0.498 | 0.798 | 2.876 |
| 0.1 | 0.723 | 0.874 / 0.500 | 0.744 | 0.859 | 0.271 |
| 0.03 | 0.691 | 0.888 / 0.576 | 0.732 | 0.874 | 0.0678 |
| 0.01 | 0.658 | 0.924 / 0.766 | 0.703 | 0.917 | 0.0205 |
| 0.003 | **0.630** | **0.987 / 0.963** | **0.949** | **0.991** | 0.00627 |
| 0.001 | 0.726 | 0.914 / 0.744 | 0.676 | 0.913 | 0.00223 |

The `0.003` point generalized at 300 steps to held-out model seeds 4-7
(`0.993/0.984` mean/worst recall), but it did not remain stable through 1,000
steps. Five formal runs ended at `0.770` or below (`0.770`, `0.762`, `0.631`,
`0.525`, and `0.516`). The raw scale had a better 1,000-step mean, yet its
2,000-step recheck also bifurcated: one difficult run recovered to `0.998`,
while another collapsed to `0.480`. A fixed bitflip scale selected at short
horizon is consequently not a general stabilization rule. Exact
single-coordinate counterfactuals learn
routes quickly but retain a long optimization tail.

The full-sequence loss adds another distinction. Raw bitflip recalled better
than production but had worse loss, so it traded ordinary next-token modeling
for more aggressive route learning. The quadratic field improved both recall
and loss and came closest to the `0.56235` data-distribution floor.

On 16 matched random `D=8,Dv=8,W=1` VJP cases, hard forward equality was
`16/16` for every estimator:

| Estimator | Q/K cosine to bitflip | Q/K sign agreement | Q/K norm ratio | V cosine | Combined cosine |
|---|---:|---:|---:|---:|---:|
| production | 0.155 | 0.903 | 0.087 | 0.306 | 0.051 |
| complete bitflip | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| quadratic state | 0.055 | 0.688 | 0.508 | 0.280 | 0.039 |

The quadratic estimator wins while being less aligned with bitflip than
production. Bitflip alignment is consequently not a useful selection metric
for this training problem; the useful signal is a different, smoother credit
field rather than an approximation to the coordinate counterfactual.

Current implementation performance must be separated from algorithmic
potential. For the formal batch-64, T=20 run, representative pure Q/K/V VJP
measurements were `0.43 ms / +0.37 MiB` for production,
`39.9 ms / +33.1 MiB` for complete bitflip, and
`23.4 ms / +6.94 MiB` for the quadratic PyTorch prototype. Memory is the
incremental allocator peak above the live input/cache baseline. At batch one,
quadratic took `23.1/43.4/84.8 ms` for
`T=20/40/80`, because its token scan is still Python/PyTorch; production stayed
near `0.43 ms` with its fused CUDA implementation. These timings do not
justify deploying the quadratic path before a fused scan kernel exists.

The engineering decision is therefore split:

- production RosaSoft remains the only deployment-ready estimator and is by
  far the fastest;
- binary V8 quadratic state remains the best current research training method
  on this gate, with the best mean, tail, route, and full-sequence loss;
- complete bitflip is a valuable exact oracle and can learn quickly, but its
  coordinate cost, horizon-sensitive static scale, and unstable long-run tail
  make it unsuitable as the default pretraining VJP. SAM/filtered indexes can
  reduce exact-bitflip computation; they do not change this
  estimator-quality result.

### Long-suffix and context extrapolation gate

The `W=1`, `T=20` comparison above does not establish long-suffix credit or
length extrapolation. `long_suffix_extrapolation.py` adds a deliberately
narrow mechanism gate for those claims. It uses a global K sequence of length
`T`; every position remains a candidate, but only 32 requested final query
rows are materialized during training. The exact hard route is therefore the
same latest-longest ROSA route while training costs `O(P*T*W)`, not `T^2`.

Each aligned W-position block repeats one target trajectory. The target block
is exact. Every other aligned block flips one content bit at the oldest
position, so after solving it is a strict `W-1` distractor. Fixed binary phase
bits prevent shifted cross-block windows from pretending to be aligned
blocks, but those windows still participate in the route. Before training,
one shared read-role scalar flips the corresponding oldest query bit; many
later distractors are then exact matches and win the hard route. There are no
trainable values or readout. The main condition uses balanced binary V8 and
explicitly changes any distractor V code that equals the target, removing the
value-collision shortcut.

The four paths have bit-identical hard outputs:

- `production` is exactly the final-row VJP of `rosa_soft_reference`, including
  softsign-STE symbols, exponential Hamming gates, normalized square-root
  suffix utility, null route, and `-log(N)` non-null calibration;
- `state_quadratic_attention` is exactly the final-row degree-2 additive
  fast-weight carrier already used above;
- `temporal_quadratic_attention` first constructs a fixed finite-window
  suffix state and then applies the same class of additive fast-weight read;
- `exact_relevant_bitflip` flips the gate's one trainable activation bit and
  computes its exact output counterfactual. It is an estimator oracle, not a
  claim that full-model bitflip has this cost.

The temporal path is intentionally parameter-free. For each hard Q/K symbol
`x_t` it uses an orthonormal Hadamard projection `B` and fixed pairwise
rotations `P`, then computes

```text
u_t = normalize(sum_{r=0}^{min(W-1,t)} P^r B sign(x_{t-r}))
phi(u_t) . phi(u_s) = (u_t . u_s)^2
```

The complete homogeneous degree-two map has `R(R+1)/2` coordinates. The
causal write stores `phi(k_t) outer V[t+1]` plus the corresponding normalizer;
the read contracts both with `phi(q_t)`. At the selected `R=64`, this is 2,080
features and `2080*(Dv+1)` persistent floats per head. The PyTorch prototype
materializes features, but the mathematical state is independent of sequence
length. Q, K, and V are hard-sign values in the numerical carrier with the
same softsign VJP as production, so sign-preserving logit rescaling cannot
leak information into the hard forward.

Small full-sequence tests embed candidate `K[a]` and payload `V[a+1]` back into
the existing operators and match both final-row VJPs exactly. An independent
scalar suffix oracle matches the dense scan, and packed chunked scans match the
dense result for both faulty and solved symbols.

GPU 0 results used `D=8`, binary `Dv=8`, 32 probes, eight data seeds, 50 Adam
steps at `T=8192`, and frozen evaluation at `8192/65536/1048576` candidates:

| W | Aligned distractors at 8K / 1M | Production | Relevant bitflip | Current-symbol quadratic | Temporal quadratic |
|---:|---:|---:|---:|---:|---:|
| 1 | 8,191 / 1,048,575 | 8/8 | 8/8 | 8/8 | 8/8 |
| 2 | 4,095 / 524,287 | 5/8 | 8/8 | 0/8 | 6/8 |
| 4 | 2,047 / 262,143 | 4/8 | 8/8 | 0/8 | 6/8 |
| 8 | 1,023 / 131,071 | 8/8 | 8/8 | 0/8 | 7/8 |
| 32 | 255 / 32,767 | 8/8 | 8/8 | 0/8 | 7/8 |

Every run that repaired the bit at 8K remained exact at 64K and 1M, including
all appended later distractors. Failed training runs remained failed. This
separates two conclusions: exact hard suffix routing extrapolates naturally
once the code is correct in this collision-free construction, but the
backward estimator still has to create that code.

The quadratic result is structural, not a small-gradient threshold. For
`W>1`, its carrier reads only `q[-1]`; the independently controlled oldest bit
has no path to the loss and its gradient is exactly zero. A shared recurrent
trunk could let a final-symbol gradient modify earlier symbols indirectly, so
this does not prove that quadratic state is useless in a model. It does prove
that the carrier itself supplies no suffix credit and that the earlier W=1
table cannot be generalized to W>1.

The temporal path fixes that missing dependency: the final query reaches
every and only the last `W` query symbols. It improves the old quadratic path
from zero successes at every `W>1` to `6/8, 6/8, 7/8, 7/8`. Its failures are
not a horizon failure. Every successful 8K code remained exact at 64K and 1M,
while failed seeds started with the wrong scalar gradient sign. Increasing R
from 16 to 64 helped the long-window gate, but R=128 was not uniformly better;
finite low-rank candidate/value interference remains.

Production reaches every suffix offset, but its candidate-weighted V signal is
not uniformly reliable. At W=2 and W=4, three and four of eight balanced-V
seeds respectively started with the wrong shared credit sign. In an
adversarial control where every distractor V is the same negative target,
production passed only W=32 (`0/4, 0/4, 0/4, 0/4, 4/4` for
`W=1/2/4/8/32`). Temporal quadratic and bitflip both passed all five
conditions (`4/4` each). `-log(N)` calibrates total
non-null mass against null; it cannot prevent coherent or noisy distractor V
credit from overwhelming one target inside the non-null set. Longer suffix
utility can improve separation, explaining why W=32 is easier than some
shorter windows here, but this should not be mistaken for a monotone general
law.

### Temporal decay ablation

An ordinary scalar decay is not a free improvement. Applying `alpha^r` only
inside the suffix fingerprint gave the following balanced-V successes; the
unbounded causal KV memory was never decayed:

| alpha | W=2 | W=4 | W=8 | W=32 |
|---:|---:|---:|---:|---:|
| 1.00 | 6/8 | 6/8 | 7/8 | 7/8 |
| 0.99 | 6/8 | 6/8 | 7/8 | 6/8 |
| 0.95 | 6/8 | 6/8 | 7/8 | 6/8 |
| 0.90 | 6/8 | 5/8 | 7/8 | 6/8 |
| 0.75 | 5/8 | 5/8 | 8/8 | 5/8 |
| 1.05 | 6/8 | 6/8 | 7/8 | 6/8 |
| 1.25 | 6/8 | 5/8 | 7/8 | 7/8 |

The oldest decisive bit receives a direct factor `alpha^(W-1)`, so a fixed
`alpha<1` predictably destroys long-horizon credit. The opposite direction
only changes which offsets dominate and also fails to improve reliability.
On 16 generic `T=6,D=2,W=3` VJP cases, however, `alpha=0.5` improved mean Q/K
cosine to bitflip from `0.143` to `0.189`. This is genuine evidence that short
and long tasks prefer different effective horizons, not evidence for one
better scalar.

A parameter-free multiscale control assigned state pairs half-lives
`1/2/4/8/16/32/infinity`. It raised generic Q/K cosine to `0.182` and changed
the long-suffix successes to `6/8, 5/8, 8/8, 8/8`: useful at W=8/32, worse at
W=4. Permanent-channel fractions of 25%, 50%, and 75%, plus a binary
`0.5/1.0` split, produced no dominating allocation. The multiscale variant
also remained `0/4` on the 500-step repeated-motif fit. It is therefore an
ablation, not part of the minimal operator.

A truly data-dependent decay needs information that this parameter-free
operator does not possess. A gate computed from continuous Q/K magnitude
reintroduces a proxy-only amplitude channel that the model can hack. A
dedicated hard gate bit changes the symbol ABI and spends code capacity. A
gate conditioned on a particular Q/K candidate match is semantically sound,
but is route-dependent and restores the dense pairwise scan. Until one of
those contracts is chosen and tested in a shared-trunk language task, the
minimal temporal prototype keeps equal finite-window weights (`alpha=1`) and
does not expose a decay parameter.

### Matched generic fit and VJP control

The targeted suffix gate is not a general estimator win. A matched 500-step
`T=16,D=2,W=3` repeated-motif run gave final-threshold successes of `1/4` for
production, `1/4` for complete bitflip, and `0/4` for temporal quadratic.
Median best losses were `0.135`, `0.447`, and `0.420`; mean prototype step
times on RTX 3070 were `3.54`, `11.23`, and `31.05 ms`. On 16 random exact
bitflip VJP cases, mean Q/K cosine was `0.561` for production and `0.143` for
temporal quadratic. All hard forwards were exactly equal.

The temporal result therefore establishes one narrower fact: a two-pass
subquadratic state can deliver direct gradients to old suffix symbols and can
solve most controlled W>1 gates without trainable proxy parameters. It does
not yet justify replacing the dense production VJP. A fused implementation
would cost `O(T*W*R + T*R^2*Dv)` arithmetic and
`O(R^2*(Dv+1))` persistent state per head, avoiding T-squared storage and
candidate deletion but still carrying a large R-squared constant. These are
carrier bounds only: the current correctness prototype deliberately calls the
dense exact-hard reference in forward. A scaling implementation would have to
pair the carrier with the separate exact hard ROSA runtime/index.

This remains a controlled estimator gate, not an LM scaling result. It fixes
the previous single-position and short-context omissions, but it does not test
a learned shared symbolizer, naturally occurring suffix collisions, multiple
ROSA layers, or next-token quality. The next gate must train a shared causal
trunk at 8K and evaluate held-out language-like trajectories at longer lengths;
the current result is the lower-level mechanism prerequisite for that test.

### Factorized aligned suffix-kernel experiment

The temporal quadratic control above compresses a history before comparing Q
and K. That mixes offsets and does not implement ROSA's aligned suffix
algebra. A second route tested the reverse construction directly: build an
independent suffix feature for Q and K, then use a causal fast-weight scan to
aggregate all candidates in linear sequence memory.

#### Exact identity and exact local VJP

For one hard D-bit symbol, production uses

```text
m(q, k) = exp(-mismatch_scale * Hamming(q, k) / D).
```

The complete Walsh feature `phi(x)` has dimension `2**D` and satisfies
`phi(q) dot phi(k) = m(q, k)` at every binary symbol. Define one aligned level
per suffix length:

```text
F_t[1] = phi(x_t)
F_t[l] = phi(x_t) tensor F_{t-1}[l-1]
```

Then

```text
F_i[l] dot F_j[l]
  = product(r=0..l-1, m(Q[i-r], K[j-r]))

sum(l=1..W, F_i[l] dot F_j[l]) = production raw suffix evidence.
```

This is an exact dual of the pairwise suffix DP. The pairwise form keeps a
small state for every `(i,j)` and costs quadratic T. The independent form
keeps one state per token but its exact feature rank grows as
`sum(l=1..W, 2**(D*l))`.

The ordinary derivative of the finite Walsh polynomial is not production's
exponential-Hamming derivative, even though their binary forward values are
equal. The prototype therefore uses an analytic custom VJP. If `R` denotes a
Walsh subset, `w_R` its squared feature coefficient, and
`alpha=mismatch_scale/(2D)`, the feature-space Jacobian is chosen as

```text
psi[i,R](q) = alpha * w_(R xor {i}) * q_(R xor {i}) / sqrt(w_R).
```

For every binary K symbol this gives

```text
psi[i](q) dot phi(k) = alpha * k_i * m(q,k),
```

which is exactly the local production gate derivative before the existing
softsign-STE multiplier. Because all later tensor, sum, and fast-weight
operations are linear in each local feature VJP, the complete exact raw
suffix score and its Q/K VJP match the dense production raw-score oracle.
The tests check both, rather than accepting forward-only equality.

#### Read and route controls

`benchmarks/suffix_kernel_proxy.py` retains exact hard ROSA forward and exposes
three route kernels:

| route kernel | candidate kernel | purpose |
|---|---|---|
| `raw` | `S` | exact raw suffix evidence |
| `quadratic` | `S**2` | sharpen after all suffix levels, including cross-level terms |
| `level_quadratic` | `sum_l S_l**2` | nonnegative per-level control without W-squared level pairs |

Each can drive additive normalized linear attention or a normalized delta
rule. Additive writes `K_feature tensor V` and a matching normalizer; delta
writes the residual against the current memory. No route is deleted and no
proxy parameter or auxiliary loss is introduced.

The exact small-shape candidate diagnostic used `D=3,W=3,T=192`, eight seeds,
and balanced binary V. All three additive kernels moved the deliberately
wrong oldest bit in the correct direction in 8/8 cases:

| estimator | target coefficient | effective routes | null mass | correct direction |
|---|---:|---:|---:|---:|
| production | 0.00629 | 13.15 | 0.265 | 8/8 |
| exact raw | 0.01084 | 82.63 | 0 | 8/8 |
| exact global quadratic | 0.00965 | 65.69 | 0 | 8/8 |
| exact level quadratic | 0.01082 | 68.10 | 0 | 8/8 |

The result explains a real semantic difference. Production's utility,
softmax, and null route make a much more concentrated field. Linear attention
distributes useful but weaker credit over many more candidates. Squaring the
suffix evidence sharpens that field, but does not reproduce production's null
calibration.

#### Exact fitting result

On the matched `T=16,D=2,W=3` repeated-motif fit, the 500-step comparison was:

| estimator | successes | median best loss | median final accuracy | step ms |
|---|---:|---:|---:|---:|
| production | 1/4 | 0.221 | 0.864 | 3.51 |
| complete bitflip | 1/4 | 0.447 | 0.727 | 11.31 |
| exact raw additive | 1/4 | 0.00370 | 1.000 | 27.53 |
| exact global quadratic additive | 1/4 | 0.00169 | 1.000 | 28.30 |
| exact level quadratic additive | 1/4 | 0.128 | 0.909 | 30.63 |
| exact raw delta | 0/4 | 0.130 | 0.909 | 24.38 |
| exact global quadratic delta | 0/4 | 0.260 | 0.727 | 25.72 |
| exact level quadratic delta | 0/4 | 0.302 | 0.818 | 28.10 |

At 1,000 steps, exact raw reached 2/4 with median best loss `7.78e-4`;
global quadratic reached 3/4 with `6.00e-4`. Both failed the same model seed,
so exact factorization improves the optimization field but does not remove all
initialization or value-credit failures. Delta is a negative control, not a
retained candidate.

The gain is also not explained by closer agreement with bitflip. Across 16
small complete-bitflip VJP cases, mean Q/K cosine was `0.561` for production,
`0.395` for exact raw additive, `0.375` for exact global quadratic, and
`0.351` for exact level quadratic. The exact suffix state is useful as a
smoother field, not as a better pointwise bitflip approximation.

#### Fixed-width approximation

Exact tensor rank cannot scale. The fixed-width implementation uses one
independent CountSketch map per suffix offset and multiplies their FFT spectra
cumulatively. All W levels therefore need W factor transforms, rather than
the earlier W-squared reconstruction. Multiple branches concatenate with
`1/sqrt(C)` scaling, which averages their induced kernels.

Two variance and state controls were tested:

1. global quadratic first CountSketches the direct sum of all levels back to
   R, then applies the exact homogeneous quadratic feature; its persistent
   feature dimension is `C*R*(R+1)/2`, independent of W;
2. hybrid raw keeps the complete `2**D` local level exactly and sketches only
   levels 2 through W. Its feature dimension is
   `2**D + C*(W-1)*R`, and W=1 has no hash dependence.

For constant 8-bit K, the 8K benchmark does not expand every token to 256
features. It builds the 256 possible local features once, sketches that table
per offset, gathers by packed symbol code, and streams each suffix level
directly into the fast-weight memory. The prototype never stores a T-squared
candidate matrix for this path.

The current small random `D=2,W=3` sketch screen shows why generic cosine is
not a sufficient promotion gate. At `R64,C4`, hybrid raw had score cosine
`0.993`, Q/K VJP cosine `0.986`, and top-route agreement `0.938`, but still
assigned negative scores to `3.86%` of candidates. Per-level quadratic had
score cosine `0.991` and VJP cosine `0.979` with no negative weights. The
bounded global quadratic was weaker: at `R64,C4`, score/VJP cosine was
`0.963/0.961`.

The engineering costs are:

| route | persistent feature state | carrier arithmetic before V |
|---|---:|---:|
| hybrid raw | `2**D + C*(W-1)*R` | `O(T*C*W*R log R)` |
| bounded global quadratic | `C*R*(R+1)/2` | suffix scan plus `O(T*C*R**2*Dv)` |
| level quadratic | `C*W*R*(R+1)/2` | `O(T*C*W*R**2*Dv)` |

The level-quadratic `R128,C2,W32` control solved 8/8 coherent-negative cases
at T=1024, but its W-scaled R-squared V update is too expensive for the 8K
formal matrix. It remains a diagnostic, not an implementation target.

#### 8K training and hard length extrapolation

The formal balanced-binary gate uses 32 probes, eight seeds, D=8, V8, and
trains at 8K. Existing production, bitflip, and temporal numbers are included
for direct context. The exact suffix rows below use a diagnostic O(P*T*W)
candidate computation; hybrid raw uses `R64,C4` and a pre-aggregated linear
state.

| estimator | W=1 | W=2 | W=4 | W=8 | W=32 |
|---|---:|---:|---:|---:|---:|
| production | 8/8 | 5/8 | 4/8 | 8/8 | 8/8 |
| relevant-bit bitflip | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| temporal quadratic | 8/8 | 6/8 | 6/8 | 7/8 | 7/8 |
| exact suffix raw | 8/8 | 5/8 | 4/8 | 8/8 | 8/8 |
| hybrid raw R64,C4 | 8/8 | 5/8 | 3/8 | 8/8 | 5/8 |

Exact raw proves that independent aligned suffix states can express the W=32
credit field. It also proves that W=2/4 failures are already present in the
raw additive kernel under balanced V, so more sketch width cannot repair
them. Hybrid raw's additional W=32 failures are approximation error. Raising
it to `R128,C4` changed W=2/4/32 to `5/8,4/8,4/8`; using `R64,C16` gave the
correct initial W=32 direction in only 4/8 cases. Branch count therefore does
not converge at a practical budget as the suffix product gets long.

The coherent-negative control uses four seeds at the same 8K shape. Exact raw
and hybrid raw solved every W in 4/4 runs. Production solved only W=32 because
its null route can be closer to the target than the coherent wrong values at
short W; bitflip and temporal quadratic solved all W. This condition proves
that the factorized state can carry old-bit credit when V signals agree, but
does not rescue its balanced-value reliability.

Every run that repaired the hard bit at 8K retained exact latest-longest hard
routing at 64K and 1M. This is a hard ROSA length-extrapolation result after
training, not a claim that the proxy itself was trained or evaluated over one
million candidates. Failed 8K runs remained failed at longer lengths, with
the constructed exact distractor count growing proportionally to context.

On the 8K balanced matrix, hybrid raw context construction ranged from about
`36 ms` at W=1 to `785 ms` at W=32. Its measured PyTorch training step rose
from about `15.7 ms` to `145 ms`; the direct diagnostic candidate oracle was
about `57 ms` at W=32. A fused kernel would reduce Python and FFT-launch
overhead, but cannot remove the measured estimator variance or the exact rank
lower bound.

#### Result

The reverse factorization is feasible and mathematically cleaner than the
earlier temporal-state heuristic. It establishes that no explicit dynamic
decay is needed to represent aligned suffix evidence: pair dependence emerges
when independent tensor levels are dotted. It also exposes the fundamental
tradeoff cleanly:

* exact independent state has the desired score and VJP but exponential rank;
* fixed-width random state has linear T but long-product variance large enough
  to flip balanced-value gradients;
* global quadratic improves exact small-fit convergence, but bounded route
  compression loses too much W=32 information;
* per-level quadratic removes negative weights, but its W-scaled R-squared V
  state and arithmetic are not a practical replacement;
* additive read is retained as the only useful control; delta is rejected.

This route is therefore frozen as a research oracle and a negative scaling
result. It does not replace production RosaSoft or the temporal control. A
future linear-state attempt must demonstrate a structured low-rank map whose
error remains bounded with W and candidate count, or admit rank growth. More
random CountSketch branches, another decay scalar, or a faster kernel do not
address the observed failure.

## Decision

Keep the suffix ladder as a research reference, but demote the hash-sensitive
suffix sketch. Retain `state_quadratic_attention` as a research candidate, not
as a production replacement:

1. it has no suffix feature, hash, seed, FFT, trainable proxy parameter, or
   auxiliary loss;
2. its 37-dimensional current-symbol feature is 6.9 times smaller than the
   full `D=8` feature;
3. it passed the original shortcut-free contextual gate more reliably than
   production and improved the difficult two-head next-token setting;
4. it is a biased credit-assignment field, not an approximation to bitflip;
5. its earlier advantage disappeared at four heads, while the controlled
   two-head E3 gate still has a 1,000-step tail; the new direct long-suffix gate
   gives its oldest decisive bit exactly zero gradient for every `W>1`, so the
   current evidence does not justify a production kernel;
6. binary V8 remained more accurate and seed-stable than uniform, exponential,
   RMS-normalized, or float V at matched width, and dense V credit was more
   stable than selected-only credit;
7. under the same optimized initialization, it also exceeded both raw and
   short-horizon-calibrated complete bitflip in 1,000-step recall tail, route
   accuracy, episode exactness, and full-sequence loss, although the current
   PyTorch implementation remains slower than the production CUDA operator.

Use E6, partial-shared Q/K plus orthogonal V4, as the minimum controlled
research initialization. Use E3 only when the extra hard V capacity is an
explicit architecture choice. Keep both opt-in; neither belongs in the
production operator ABI or default model initialization. Do not retain the
paired-output initializer outside its negative-control experiment.

Further gates should vary `D`, `Dv`, sequence length, and multiple stacked ROSA
layers on genuinely streaming corpora. If later evidence restores a stable
advantage, a kernel should generate constant/linear/pairwise interactions
directly and fuse the numerator state, normalizer state, read, and reverse
scan. It must not materialize the complete `2**D` basis or per-token state
history.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/fast_weight_proxy_ablation.py \
  --device cuda --estimators production bitflip linear_attention \
    delta_rule single_suffix_delta exact_suffix_delta sketch_suffix_delta \
  --model-seeds 0 1 2 3 --steps 500 \
  --qk-bits 4 --value-bits 4 --max-suffix-length 3 \
  --fingerprint-length 3 --sketch-dim 64 \
  --gradient-qk-bits 4 --gradient-value-bits 4 \
  --gradient-max-suffix-length 3 --gradient-fingerprint-length 3 \
  --json-out validation/fast_weight_proxy_d4w3_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/fast_weight_proxy_ablation.py \
  --device cuda --estimators production bitflip \
    single_suffix_sketch_delta \
  --model-seeds 0 1 2 3 --steps 1000 \
  --qk-bits 4 --value-bits 4 --max-suffix-length 8 \
  --fingerprint-length 3 --sketch-dim 256 \
  --gradient-sequence-length 8 --gradient-qk-bits 4 \
  --gradient-value-bits 4 --gradient-max-suffix-length 8 \
  --gradient-fingerprint-length 3 \
  --json-out validation/fast_weight_proxy_d4w8_l3_r256_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/contextual_estimator_recall.py \
  --operator cuda --device cuda \
  --estimators production state_quadratic_attention \
    state_cubic_attention state_full_linear_attention \
  --seeds 0 1 2 3 4 5 6 7 --steps 400 \
  --train-pairs 32 --validation-pairs 16 --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_scaling.py \
  --operator cuda --device cuda \
  --estimators production state_quadratic_attention \
  --seeds 0 1 2 3 --steps 400 \
  --association-values 2 4 8 16 --head-values 1 2 4 \
  --depth-values 1 2 4 --train-pairs 16 --validation-pairs 8 \
  --json-out validation/state_attention_scaling_sm86.json --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_pretraining.py \
  --operator cuda --device cuda \
  --estimators production state_quadratic_attention \
  --seeds 0 1 2 3 --steps 2000 --train-pairs 32 \
  --validation-pairs 16 --associations 4 --context-depth 2 --heads 2 \
  --json-out validation/state_attention_pretraining_sm86.json --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_pretraining.py \
  --operator cuda --device cuda \
  --estimators production state_quadratic_attention \
  --seeds 0 1 2 3 --steps 2000 --train-pairs 32 \
  --validation-pairs 16 --associations 4 --context-depth 2 --heads 4 \
  --json-out validation/state_attention_pretraining_h4_sm86.json --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_init_ablation.py \
  --operator cuda --device cuda \
  --estimators state_quadratic_attention \
  --model-seeds 0 1 2 3 --data-seeds 0 1 \
  --validation-seed 200000 --steps 1000 \
  --train-pairs 32 --validation-pairs 64 \
  --experiments E0_default_dv4 E1_shared_dv4 E3_shared_dv8 \
  --json-out validation/state_attention_init_cross_order_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_init_ablation.py \
  --operator cuda --device cuda \
  --estimators state_quadratic_attention \
  --model-seeds 0 1 2 3 --data-seeds 0 1 \
  --validation-seed 200000 --steps 1000 \
  --train-pairs 32 --validation-pairs 64 \
  --experiments E6_shared_dv4_orthogonal E7_independent_qk_dv4 \
  --json-out validation/state_attention_init_controls_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_init_ablation.py \
  --operator cuda --device cuda --estimators production \
  --model-seeds 0 1 2 3 --data-seeds 0 1 \
  --validation-seed 200000 --steps 1000 \
  --train-pairs 32 --validation-pairs 64 \
  --experiments E0_default_dv4 E6_shared_dv4_orthogonal E3_shared_dv8 \
  --json-out validation/state_attention_init_production_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_value_ablation.py \
  --operator cuda --device cuda \
  --model-seeds 0 1 2 3 --data-seeds 0 \
  --validation-seed 200000 --steps 500 \
  --train-pairs 32 --validation-pairs 64 \
  --experiments binary_v8_proxy uniform2_v8_proxy exp2_v8_proxy \
    exp3_v8_proxy rms_v8_proxy float_v8_proxy \
  --json-out validation/state_attention_value_codec_v8_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_pretraining.py \
  --operator cuda --device cuda \
  --estimators production bitflip state_quadratic_attention \
  --model-seeds 0 1 2 3 --data-seeds 0 1 \
  --validation-seed 200000 --steps 1000 \
  --train-pairs 32 --validation-pairs 64 \
  --associations 4 --payload-bits 4 --hidden-size 32 \
  --context-depth 2 --heads 2 --qk-bits 8 \
  --value-heads 1 --value-bits 8 \
  --qk-init partial_shared_orthogonal --value-init orthogonal \
  --controlled-module-reset \
  --json-out validation/state_attention_estimator_comparison_v8_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_pretraining.py \
  --operator cuda --device cuda \
  --estimators production bitflip state_quadratic_attention \
  --model-seeds 0 1 2 3 --data-seeds 0 1 \
  --validation-seed 200000 --steps 1000 \
  --train-pairs 32 --validation-pairs 64 \
  --associations 4 --payload-bits 4 --hidden-size 32 \
  --context-depth 2 --heads 2 --qk-bits 8 \
  --value-heads 1 --value-bits 8 \
  --qk-init partial_shared_orthogonal --value-init orthogonal \
  --controlled-module-reset --bitflip-gradient-scale 0.003 \
  --json-out \
    validation/state_attention_estimator_comparison_v8_calibrated_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_pretraining.py \
  --operator reference --device cuda --estimators production \
  --model-seeds 0 1 2 3 --data-seeds 0 1 \
  --validation-seed 200000 --steps 1000 \
  --train-pairs 32 --validation-pairs 64 \
  --associations 4 --payload-bits 4 --hidden-size 32 \
  --context-depth 2 --heads 2 --qk-bits 8 \
  --value-heads 1 --value-bits 8 \
  --qk-init partial_shared_orthogonal --value-init orthogonal \
  --controlled-module-reset \
  --json-out validation/state_attention_production_reference_v8_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_estimator_vjp.py \
  --device cuda --cases 1:20 1:40 1:80 64:20 \
  --json-out validation/state_attention_estimator_vjp_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/fast_weight_proxy_ablation.py \
  --device cuda --gradient-only \
  --estimators production bitflip state_quadratic_attention \
    state_cubic_attention linear_attention \
  --gradient-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --gradient-qk-bits 8 --gradient-value-bits 4 \
  --gradient-max-suffix-length 3 --gradient-fingerprint-length 3

CUDA_VISIBLE_DEVICES=0 python benchmarks/long_suffix_extrapolation.py \
  --device cuda --windows 1 2 4 8 32 --seeds 0 1 2 3 4 5 6 7 \
  --probes 32 --bits 8 --value-bits 8 --value-mode balanced_binary \
  --train-context-length 8192 \
  --eval-context-lengths 8192 65536 1048576 \
  --steps 50 --chunk-size 65536 --summary-only \
  --json-out validation/long_suffix_extrapolation_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/suffix_kernel_ablation.py \
  --device cuda --skip-diagnostics --skip-sketch-scan --skip-gradient \
  --estimators exact_raw_additive exact_quadratic_additive \
  --model-seeds 0 1 2 3 --steps 1000 --summary-only \
  --json-out validation/suffix_kernel_exact_fit_1k_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/suffix_kernel_ablation.py \
  --device cuda --skip-diagnostics --skip-fit --skip-gradient \
  --sketch-route-kernels raw quadratic level_quadratic \
  --sketch-dims 16 32 64 --sketch-counts 1 2 4 \
  --sketch-data-seeds 0 1 --sketch-hash-seeds 0 1 2 3 \
  --summary-only \
  --json-out validation/suffix_kernel_current_sketch_screen_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/long_suffix_extrapolation.py \
  --device cuda --estimators suffix_sketch_raw_attention \
  --windows 1 2 4 8 32 --seeds 0 1 2 3 4 5 6 7 \
  --probes 32 --bits 8 --value-bits 8 --value-mode balanced_binary \
  --train-context-length 8192 \
  --eval-context-lengths 8192 65536 1048576 \
  --steps 50 --suffix-sketch-dim 64 --suffix-sketch-count 4 \
  --chunk-size 65536 --summary-only \
  --json-out validation/suffix_kernel_hybrid_raw_balanced_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/long_suffix_extrapolation.py \
  --device cuda \
  --estimators exact_suffix_raw_attention suffix_sketch_raw_attention \
  --windows 1 2 4 8 32 --seeds 0 1 2 3 \
  --probes 32 --bits 8 --value-bits 8 --value-mode coherent_negative \
  --train-context-length 8192 \
  --eval-context-lengths 8192 65536 1048576 \
  --steps 50 --suffix-sketch-dim 64 --suffix-sketch-count 4 \
  --chunk-size 65536 --summary-only \
  --json-out validation/suffix_kernel_coherent_sm86.json
```
