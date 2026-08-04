# Stateful Internal-Language Gates

## Status

This document records four research implementations, one checkpoint
diagnostic, and their August 2026 RTX 2080 Ti results. None changes the public
RosaSoft API or its default estimator/runtime path. The experiments answer
narrower questions:

1. Can a minimal deterministic state generate ROSA symbols that are not
   synchronized one-for-one with residual inputs?
2. Does state make a finite multi-symbol grammar representable, and does the
   production VJP train it as the required trajectory grows?
3. Can one exact hard recalled value causally form the next query?
4. Can hard-state conflicts drive deterministic alphabet growth without a
   new weighted loss?
5. Can checkpoint Q/K trajectories separate read/write drift, insufficient
   code capacity, and long-suffix fragility?

The answer is structurally yes to the first four. The optimization answer is much
less favorable: two-symbol trajectories train reliably, four-symbol
trajectories are seed-sensitive, eight-symbol trajectories fail, and
two-hop chains commonly lose an initially correct codebook during continued
training.

Machine-readable primary runs are in:

- `validation/latent_grammar_capacity.json`;
- `validation/latent_grammar_l2.json`;
- `validation/latent_grammar_l4.json`;
- `validation/latent_grammar_l8.json`;
- `validation/multihop_recall_gate.json`;
- `validation/symbol_growth_trained.json`;
- `validation/symbol_growth_frozen.json`;
- `validation/pretraining_codebook.json`;
- `validation/pretraining_codebook_long.json`.

## Experimental Contract

Every gate keeps exact hard ROSA forward and the production dense soft VJP.
There is no soft information in a forward output. The grammar and growth
gates inject hard payload values directly and have no trainable value
projection, residual bypass, or readout. Their MSE is exactly zero only when
all requested hard payloads are returned.

The two-hop gate pairs examples with identical input positions and different
pre-query values. Source-to-intermediate maps vary by sample. The second
query receives no source cue or recurrent state; only the first hard routed
value can distinguish the required second address.

These are codebook fitting gates, not held-out language modeling. They isolate
route causality and optimization but do not establish semantic generalization
or realistic pretraining behavior.

## 1. Minimal Stateful Symbolizer

`benchmarks/internal_language.py` implements one transition:

```text
p_t = tanh(W_x LN(x_t) + W_s s_(t-1) + W_f f_t)
s_t = (1 - alpha) s_(t-1) + alpha p_t
q_t = W_q LN(s_t)
k_t = W_k LN(s_t)
```

`alpha` is one fixed `update_rate`. There is no learned gate, scheduler,
noise source, reconstruction objective, or auxiliary loss. Feedback `f_t` is
optional and absent outside the two-hop use case. A reset clears state before
the current input is consumed.

Q and K use a shared latent state but different linear heads. Copying Q into
K at initialization is an optional prior, not weight tying. On the strict
length-2 grammar, aligned and independent initialization both finished at
4/4 exact runs. Median first-exact step was 2 for aligned heads and 60.5 for
independent heads. Alignment is therefore a useful initialization prior but
not a representational requirement on the smallest gate.

Unit tests cover determinism, state persistence, exact reset behavior, the
stateless control, separate Q/K parameter storage, feedback gradients, and
gradients through the recurrent transition.

## 2. Strict Latent Grammar

### Construction

A memory phrase has `L` symbol positions followed by its payload route. Only
the first input identifies the cue; every later input is the same blank
token. State is reset at each phrase boundary. The query repeats the cue and
blank pattern after memory.

Each ROSA symbol has:

- one learned content bit;
- one fixed phase bit that distinguishes a phrase endpoint from every
  internal route.

The content bit is exposed only at a fixed outline:

| Phrase length | Active content positions | Candidates |
| ---: | --- | ---: |
| 2 | `0,1` | 3 |
| 4 | `0,2,3` | 5 |
| 8 | `0,4,5,6,7` | 17 |

The first active position carries the highest binary code bit. Any suffix
shorter than `L` loses that position and has capacities 2, 4, and 16,
respectively, one fewer than the candidate count. A complete suffix has
capacities 4, 8, and 32. This makes the exact required window 2, 4, or 8,
not merely the next tested power of two.

An explicit binary codebook routes at 100% for every full window. At `W=L-1`
its measured route accuracies are 33.3%, 40.0%, and 47.1%. Parameterized tests
enforce full-window success and shortened-window failure for all three tasks.

A stateless projector sees the cue only at position zero and identical blanks
afterward, so it can distinguish at most two candidates even with the full
window. A stateful projector can carry the cue and emit a longer trajectory.

### Training Results

The primary matrix uses production CUDA, aligned but separate Q/K heads,
`scale=1`, `dropout_p=0`, `mismatch_scale=3`, AdamW at `1e-3`, 500 steps, and
four seeds. Each task queries every candidate through complementary value
pairs.

| Model | L | Initial route acc. | Final route acc. | Final exact runs | Ever exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| stateful | 2 | 0.750 | 1.000 | 4/4 | 4/4 |
| stateless | 2 | 0.583 | 0.250 | 0/4 | 0/4 |
| stateful | 4 | 0.500 | 0.650 | 2/4 | 2/4 |
| stateless | 4 | 0.400 | 0.300 | 0/4 | 0/4 |
| stateful | 8 | 0.118 | 0.000 | 0/4 | 0/4 |

The L=2 stateful code is both learnable and stable. The two non-exact aligned
initializations first reached exact at steps 24 and 4 and then remained exact.
The independent-head ablation also reached 4/4, at first-exact steps 23, 27,
94, and 124.

L=4 is a transition regime. Two seeds finished exact; one entered the exact
basin only at step 476. The failures ended at 40% and 20%. L=8 is not close:
all seeds fell from 11.8% initial route accuracy to zero, and best loss stayed
near 1.0. Mean instrumented step time also rose from 15.4 ms at L=2 to 35.8
ms at L=4 and 189.9 ms at L=8.

The state adds the required representational capacity, but it also couples
several hard decisions through time. The production VJP supplies dense local
credit; it does not provide a low-variance trajectory-level edit. Longer
state Jacobians and coordinated sign changes therefore make training less,
not more, reliable in this implementation.

The fixed phase and outline are oracle grammar scaffolds. These experiments
do not show that a model can discover phrase boundaries or outline timing
from raw residuals.

## 3. Hard-Value Multi-Hop Recall

### Construction

Each sample stores two tables:

```text
source address       -> hard intermediate-address Value
intermediate address -> hard answer Value
```

The source-to-intermediate permutation and answer table vary per sample.
After the first query, the model executes:

```text
f = ROSA(Q(source), memory K, memory V)       # exact hard Value
q2 = symbolizer(f, reset_state)
y = ROSA(q2, memory K, memory V)              # exact hard answer
```

The second symbolizer state is reset, and no source feature is supplied at
that step. The final loss reaches the first Q/K projection through the custom
hard-forward/soft-backward VJP. A unit test verifies that detaching `f`
changes the shared projection gradient.

### Results

The primary matrix uses six source and intermediate addresses, eight address
bits, eight Q/K bits, 24 complementary pairs, four seeds, 500 steps, and
AdamW at `1e-3`.

Before training, aligned Q/K initialization already gives mean first-hop
Value accuracy 91.7% and final answer accuracy 85.9%. Zero-feedback accuracy
is 0%; shuffled feedback is 7.3%. The high initial result is therefore a real
hard two-hop chain, not a source/readout shortcut.

After training with routed feedback:

- final gate success is 1/4;
- mean final answer accuracy is 28.6%;
- mean first-hop Value accuracy is 33.3%;
- zero-feedback accuracy remains 0%;
- shuffled-feedback accuracy is 4.7%.

Three seeds were exact at least once. Their exact-step counts were 1, 8, and
501. The first two subsequently lost the first-hop codebook; only the seed
that started exact remained exact. Training with zero feedback passed 0/4
and ended at 1.0% mean answer accuracy. Detached-feedback training passed
1/4 and ended at 29.2%, providing no empirical improvement over routed
feedback.

Thus the differentiable causal path exists, but this experiment does not show
that end-to-end gradient improves the chain. A single first-hop bit change
changes the entire second query and route, so codebook instability is
multiplied across hops.

## 4. Conflict-Driven Dormant Bits

### Statistics and Split

`benchmarks/symbol_growth.py` preallocates eight Q/K projection rows and
exposes only an active prefix. It groups hard key codes and computes:

- number and size of hard states;
- states/samples with more than one continuation label;
- pairwise conflicting labels;
- empirical `H(continuation | hard state)`;
- query/key self-alignment as a separate metric.

For every conflicting hard state, sorted continuation classes are divided
into two balanced sides. A least-squares direction predicts those sides from
the latent features. That direction initializes exactly one dormant Q bit and
the corresponding K bit; both receive the same finite margin. Existing rows
are not rewritten by the split.

This is an external supervised structural edit because it consumes
continuation labels. It is not a surrogate gradient and not evidence of
unsupervised symbol emergence.

### Results

The gate has 32 one-hot concepts, starts with two active bits, permits eight,
uses W=1, and checks conflicts every 50 steps for 300 steps.

| Strategy | Initial route | Final route | Final Q/K align | Final conflict pairs | Exact runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed 8, train at `5e-3` | 0.891 | 0.570 | 0.727 | 5.25 | 1/4 |
| grow, margin 1, train at `5e-3` | 0.000 | 0.727 | 0.773 | 1.00 | 0/4 |
| grow, margin 4, train at `5e-3` | 0.000 | 0.844 | 0.898 | 1.00 | 0/4 |
| grow, freeze gradients | 0.000 | 1.000 | 1.000 | 0.00 | 4/4 |

The frozen control activates four bits and stops at six active bits in every
seed. Conflict pairs fall from 113-124 to zero, proving that deterministic
state splitting can grow a sufficient alphabet. It does not prove stability
after realistic LM gradients resume: this MSE becomes zero once routing is
exact, so there is no remaining unrelated objective to perturb the symbols.

With simultaneous production-VJP updates, key conflicts can disappear while
routes remain wrong. In the first smoke run, key conflicts reached zero but
route accuracy and Q/K self-alignment were both 71.9%. This directly shows
that alphabet capacity and read/write alignment are different failure modes.
A larger split margin improves stability but suppresses the softsign VJP and
still does not make any run exact.

## 5. Pretraining Checkpoint Codebook Diagnostics

`benchmarks/pretraining_codebook.py` accepts matched Q/K head logits with
shape `[occurrence, trajectory position, bit]` and one continuation label per
occurrence. It does not prescribe a model, auxiliary objective, or training
schedule. A pretraining run can collect repeated semantic occurrences from
its residual stream and evaluate its existing heads directly.

For every suffix horizon it reports:

- paired Q/K bit, symbol, and full-trajectory agreement;
- exact longest/latest route and compatible-continuation accuracy;
- target versus strongest-distractor suffix lengths;
- hard key-state occupancy and conflicting continuation labels;
- empirical `H(continuation | hard key trajectory)`;
- bit entropy and dead-bit fraction separately from Q/K alignment.

The controlled `N=64,L=8,D=4` matrix distinguishes the intended failure
modes. Aligned trajectories route at 100% with zero conditional entropy. Ten
percent independent Q corruption lowers full-horizon routing to about 43%; a
shared role-specific drift lowers it to about 52%. Keeping only one active
bit per position preserves Q/K alignment but introduces about 0.253 bits of
continuation conditional entropy and lowers routing to about 88%.

The longer `N=256,L=32,D=2` matrix shows why neither bit width nor trajectory
length is sufficient alone:

| Mode | Horizon 1 route | Horizon 8 route | Horizon 32 route | Horizon 32 Q/K trajectory agreement |
| --- | ---: | ---: | ---: | ---: |
| aligned | `0.016` | `0.998` | `1.000` | `1.000` |
| 10% independent corruption | `0.012` | `0.404` | `0.404` | `0.002` |
| role drift | `0.012` | `0.508` | `0.507` | `0.000` |
| one active bit/position | `0.008` | `0.628` | `1.000` | `1.000` |

Long trajectories can supply combinatorial address capacity even with one
active bit per position: the collapsed condition reaches zero conflict entropy
by horizon 32. But independent 10% bit errors make exact 32-position Q/K
agreement nearly vanish. ROSA still routes around 40% because longest-suffix
competition can use a shorter intact tail. Therefore pretraining should track
the full horizon curve, not only final trajectory uniqueness or mean bit
entropy.

## Decisions

1. Keep the stateful symbolizer and all three gates in research code. Do not
   add a state, phase, outline, growth control, or split margin to the public
   RosaSoft operator.
2. Treat an internal language as a sequence grammar with explicit boundaries
   and sparse update slots, not an iid fingerprint emitted at every token.
3. Do not train an under-capacity codebook continuously. First collect hard
   conflicts, activate capacity, verify conflict entropy and Q/K alignment,
   and only then resume broader updates.
4. Preserve old hard bits during a split. The simplest next prototype should
   freeze the shared trunk and active rows, train or initialize only the new
   row, then unfreeze under a hard-code rollback check.
5. Parameterize read/write roles as a shared address base plus small separate
   residuals. Fully independent heads are learnable at L=2, but Q/K drift is
   the dominant longer-chain failure.
6. Gate multi-hop feedback on one-hop codebook stability. Adding more hops
   before one-hop conflict and alignment metrics are stable only amplifies
   route discontinuities.
7. Retain one loss. Conflict statistics should schedule structural edits, not
   become another weighted reconstruction/entropy/margin loss.

The next meaningful experiment is the frozen-growth protocol inside the
shortcut-free contextual LM gate, with held-out cue compositions and an
unrelated LM loss still active after routing becomes exact. That is the
smallest test that can distinguish a genuinely grown internal language from
a supervised one-hot codebook edit.
