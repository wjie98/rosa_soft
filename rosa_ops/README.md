# ROSA QKV Operator Implementation

This directory contains the core, low-level implementations of the softened ROSA QKV operator.

> **🚀 Latest Update**: The newest core operators have been migrated to the local `rosa_cpp` extension.
>
> **Installation & Usage**:
> 1. Install the extension: `pip install --no-build-isolation .`
> 2. Import the operator: `from rosa_cpp import rosa_bits_ops`


The numbered subdirectories (`2025XXXX/`) contain historical snapshots of the implementation, preserved for research and comparison purposes.

## Implementation History

Below is a timeline of major changes and refinements made to the operator logic.

- **20251026: Initial Version**
  - **Concept**: The initial approach was to directly "soften" the classic dynamic programming algorithm for finding the Longest Common Substring.
  - **Hard Mode Formula**: The standard DP recurrence for this task is:
    ```
    # If query at `i` matches key at `j`, extend the match from the diagonal. Otherwise, reset to 0.
    dp[i][j] = (dp[i-1][j-1] + 1) if q[i] == k[j] else 0
    ```
    Here, `dp[i][j]` stores the length of the common suffix.
  - **Soft Mode Formula**: To make this differentiable, the boolean check (`==`) was replaced by a continuous similarity score `a[i][j]` (e.g., from a sigmoid or scaled dot product), and the `if/else` was replaced by arithmetic operations:
    ```
    dp[i][j] = dp[i-1][j-1] * a[i][j] + a[i][j]
    ```
    In this formula, `a[i][j]` acts as a "gate". If the match is strong (`a[i][j] ≈ 1`), the formula approximates `dp[i-1][j-1] + 1`, accumulating the match length. If the match is weak (`a[i][j] ≈ 0`), it resets `dp[i][j]` to nearly zero.
  - **Limitation**: This formulation has a strong **diagonal dependency** (`dp[i][j]` depends on `dp[i-1][j-1]`). While parallel scan algorithms (like wavefront parallelism) can be implemented on GPUs for this pattern, the degree of parallelism is fundamentally limited by the number of anti-diagonals. This dependency structure complicates the design of highly efficient, hardware-saturating fused kernels, and early versions required storing intermediate results in global memory between kernel launches, creating a performance bottleneck.

- **20251030: New Softening Formula**
  - **Concept**: To improve parallelization potential, a new softening formula was adopted. This approach is based on primitives that are more amenable to massively parallel computation.
  - **Soft Mode Formula**: The new logic is expressed as:
    ```
    # Let `a` be a row/column of the similarity matrix
    t = cumsum(a)
    dp = t - cummax(t * (1 - a))
    ```
  - **How It Works**: This is a clever trick to perform a "gated cumulative sum" in parallel.
    1.  `a` represents the probability of a "match" (value close to 1). Consequently, `(1 - a)` represents the probability of a "reset" (value close to 1).
    2.  `t = cumsum(a)` is the running total of match scores.
    3.  `t * (1 - a)` will be a large value (close to `t`) at reset points and a small value (close to 0) at match points.
    4.  `cummax(t * (1 - a))` finds the running maximum of these reset-point values. This effectively finds the value of the cumulative sum *at the last significant reset point*.
    5.  `t - cummax(...)` subtracts the cumulative sum at the last reset from the current cumulative sum, yielding the **sum since the last reset**.
  - **Advantage**: This formulation transforms the computation from a diagonally-dependent scan into a row/column-wise operation composed of standard parallel primitives (`cumsum`, `cummax`). While not necessarily faster in a naive implementation, this structure is **significantly more flexible and rearrangeable**. This change is crucial as it provides much greater optimization space for future fused kernel implementations that can better exploit the GPU's memory hierarchy and parallelism.

- **20251102: C++ Gradient Estimation**
  - A C++ version of a global perturbation differential estimation method for ROSA gradients was added.
  - This serves as a reference implementation for comparing the performance and correctness of future gradient approximation techniques.
  - **Note**: The QKV implementation in this version was not yet aligned with Peng Bo's reference logic.

- **20251103: Major Refactoring & Alignment**
  - The core QKV implementation was significantly refactored to align more closely with the reference logic from Peng Bo's version. This introduced several key behavioral changes:
    - **Matching Logic**: The query `q[t]` now exclusively matches against historical keys `k[i]` where `i < t`. Upon finding the longest match ending at `i`, the operator returns the *subsequent* value `v[i+1]`.
    - **Default Value**: If no match is found, the operator returns a default value of `0`, a change from the previous behavior of returning the current value `v[t]`.
    - **Discretization Modes**: The implementation now formally supports two methods for tokenization: `bits_mode=True` (bitwise representation) and `bits_mode=False` (one-hot/`argmax` representation).
    - **Performance Note**: This alignment introduced more computational steps, making `torch.compile` highly recommended for achieving maximum efficiency.
  - **Note on a previous modification**: An earlier version set unmatched items to their own values (`v[t]`). This was a practical convenience that naturally incorporated positional information but has been superseded by the alignment with the canonical ROSA logic.

- **20251105: Masking Support**
  - Added the `attn_mask` parameter to the operator, allowing specific tokens in the sequence to be ignored during the matching process.

- **20251107: Python Suffix Automaton & API Refinement**
  - Added `RapidOnlineSuffixAutomaton`, a pure Python implementation of the suffix automaton logic for reference and testing. For a potentially faster C++ implementation, please refer to the code in the `20251102/` directory.
  - The input parameter format for the `rosa_qkv_ops` function was adjusted for greater clarity and flexibility, particularly in how embedding vectors are handled.

- **20251116: Refactored with STE to Fix Model Hacking Issue**
  - The previous `sigmoid` + temperature `tau` scheme had a vulnerability where the model could learn to output small-magnitude logits to bypass the discretization pressure from annealing. The new version adopts the Straight-Through Estimator (STE), which enforces a strictly discrete forward pass while using a surrogate gradient for the backward pass, fundamentally solving this issue.

- **20251117: Refactored the core learning mechanism to a more holistic Straight-Through Estimator (STE) framework**
  - This update introduces an **Interpolation Annealing** strategy, controlled by a schedulable `alpha` coefficient, to smoothly transition from a fully continuous soft proxy to the discrete hard operation during training. This approach ensures the gradient is always derived from the smooth proxy, significantly improving training stability and addressing smoothness issues in the loss landscape while converging to a deterministic forward pass.

- **20251118: CPU Offloading for Hard Forward Pass**
  - The discrete "hard" forward pass can now be offloaded to the CPU for execution via a custom C++ operator, controlled by the `host_ops=True` flag. This allows for better hardware utilization by running the hard pass on the CPU in parallel with the soft proxy computation on the GPU. This can lead to a significant performance improvement, especially when the GPU is the primary bottleneck.

- **20251121: Fine-tuned Proxy Function**
  - The SUFA proxy now supports disabling positional information injection and uses a standard causal mask to activate underlying optimizations.

- **20251204: Query Shaping (ROSA-GQS)**
  - Replaced dense soft proxies with **ROSA-Guided Query Shaping**, compatible with Flash Attention.
  - Introduced **Look-ahead Masking** to optimize the gradient horizon and prune hallucinated gradients from noise.
  - Implemented **Continuous Entropy-Aware Boosting**, which dynamically amplifies queries with high temporal variance during long matches, promoting robust long-context lock-in while suppressing degenerate solutions.

- **20251209: Value Detach & Decay (VDD) Operator**
  - **Concept**: Solves the "co-adaptation" trap (V-Dominance) where the model learns a blurry mean Value to satisfy a soft attention proxy, causing Q/K optimization to stagnate and failing to explore long-range dependencies.
  - **Mechanism**: The `rosa_vdd_ops` operator introduces a specific gradient flow strategy:
    - **Soft Branch (Search)**: Uses Suffix Attention (SUFA) with **Detached Values**. This strictly forces gradients to optimize $Q$ and $K$ to align geometrically with the correct historical keys, rather than modifying $V$ to fit a suboptimal alignment.
    - **Hard Branch (Content)**: The $V$ tensor is updated *exclusively* by the precise indices selected by the discrete Hard ROSA pass, ensuring representation sharpness.
  - **Refinements**:
    - **Geometric Decay**: Reintroduced a strict decay factor ($\lambda \approx 0.45$) to mathematically align continuous Flash Attention dot products with the hierarchical "Longest Common Suffix" objective.
    - **Hypercube Optimization**: Shifted from Spherical optimization (`F.normalize`) to Hypercube optimization (Tanh/Clamp). This aligns the soft proxy's geometry with the discrete Hamming space, utilizing "bang-bang" gradients to encourage saturation at hypercube vertices for more effective bit-flipping.

- **20251219: Adaptive Suffix Window & Parameter Pruning**
  - **Suffix Window Optimization**: Empirical analysis revealed that a suffix history of 3-4 tokens is sufficient for effective context locking. We have refactored the interface to use a specific `suffix_window` parameter instead of implicitly expanding to the full `head_dim`. This prevents gradient noise arising from overly long, irrelevant context windows.
  - **Adaptive Decay**: The decay factor is now calculated adaptively by default based on the suffix window size, significantly reducing the tuning overhead. Manual control is preserved via the `suffix_factor` parameter for specialized use cases.
  - **API Simplification**: A major cleanup of the interface was performed. We removed a large volume of experimental and ineffective parameters, exposing only the optimal combination to ensure stability and ease of use.