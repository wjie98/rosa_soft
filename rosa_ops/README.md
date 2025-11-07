# ROSA QKV Operator Implementation

This directory contains the core, low-level implementations of the softened ROSA QKV operator.

**The latest, most stable, and recommended version of the operator is located in `rosa.py`**.

The numbered subdirectories (`20XXXXXX/`) contain historical snapshots of the implementation, preserved for research and comparison purposes.

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
  - **Limitation**: This formulation has a strong **diagonal dependency** (`dp[i][j]` depends on `dp[i-1][j-1]`). On a GPU, this requires a "scan" operation along the diagonals of the attention matrix, which is inherently sequential and slow. This version required storing intermediate results in global memory between kernel launches, creating a significant performance bottleneck.

- **20251030: New Softening Formula**
  - **Concept**: To break the sequential dependency and enable massive parallelism, a new, more relaxed softening formula was adopted. This approach is based on highly parallelizable "prefix scan" primitives.
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
  - **Advantage**: This formulation transforms the computation from a strictly associative recursion into a more flexible, rearrangeable form. `cumsum` and `cummax` are standard parallel primitives with highly efficient GPU implementations. This change was crucial as it provides significant optimization space for future fused kernel implementations, moving away from the slow diagonal scan.

- **20251102: C++ Gradient Estimation**
  - A C++ version of a global perturbation differential estimation method for ROSA gradients was added.
  - This serves as a reference implementation for comparing the performance and correctness of future gradient approximation techniques.

- **20251103: Major Refactoring & Alignment**
  - The core QKV implementation was significantly refactored to align more closely with the reference logic from Peng Bo's version. This introduced several key behavioral changes:
    - **Matching Logic**: The query `q[t]` now exclusively matches against historical keys `k[i]` where `i < t`. Upon finding the longest match ending at `i`, the operator returns the *subsequent* value `v[i+1]`.
    - **Default Value**: If no match is found, the operator returns a default value of `0`, a change from the previous behavior of returning the current value `v[t]`.
    - **Discretization Modes**: The implementation now formally supports two methods for tokenization: `bits_mode=True` (bitwise representation) and `bits_mode=False` (one-hot/`argmax` representation).
    - **Performance Note**: This alignment introduced more computational steps, making `torch.compile` highly recommended for achieving maximum efficiency.
  - **Note on a previous modification**: An earlier version set unmatched items to their own values (`v[t]`). This was a practical convenience that naturally incorporated positional information but has been superseded by the alignment with the canonical ROSA logic.

- **20251105: Masking Support**
  - Added the `attn_mask` parameter to the operator, allowing specific tokens in the sequence to be ignored during the matching process.