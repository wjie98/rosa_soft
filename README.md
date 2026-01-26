# Softened ROSA Operators and Explorations for Next-Generation LLMs

[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/your-username/rosa_soft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a robust, end-to-end trainable implementation of the **ROSA (Rapid Online Suffix Automaton)** operator. We make the discrete, non-differentiable ROSA mechanism compatible with gradient-based optimization by employing a **Straight-Through Estimator (STE)** framework.

This approach combines the best of both worlds:
1.  **A true ROSA forward pass**, which executes the discrete, parameter-free logic for maximum efficiency and faithfulness to the original concept.
2.  **A smooth, stable backward pass**, which uses a **Suffix Attention (SUFA)** mechanism as a **proxy for gradients**, enabling stable and effective training.

## Background: What is RWKV-8 ROSA?

ROSA is a groundbreaking concept proposed by Peng Bo (BlinkDL) as part of the RWKV-8 architecture. It aims to replace the standard attention mechanism with what is described as a "neurosymbolic infinite-range lossless information propagator."

The core idea is to predict the next token in a sequence based on the longest exact match found in the history. For a given sequence `x`, the output `y_i` is determined by finding the longest suffix of `x` ending at `i-1` that matches a previous substring. If such a match is found at index `j`, the output is the token that followed that match, `x_{j+1}`.

This mechanism has several powerful properties:
- **Parameter-Free**: The core logic has no trainable weights.
- **No Dot Product / Softmax**: It operates on discrete tokens, eliminating the quadratic complexity of standard attention.
- **No Float KV Cache**: It only needs to store the history of discrete tokens.
- **Efficient Inference**: The underlying Suffix Automaton can be processed very quickly on a CPU, in parallel with GPU-based layers.

This project is inspired by and builds upon the original research found at the [RWKV-LM Repository](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8).

## The Core Challenge: Making ROSA Differentiable

The primary challenge of the original ROSA is its inherently discrete and non-differentiable nature. The token matching is based on exact equality, a step function that provides zero gradient information almost everywhere. This makes it impossible to train a model that produces ROSA-compatible inputs using standard backpropagation.

Our work is dedicated to solving this problem, allowing ROSA to be integrated as a trainable layer within any modern deep learning architecture.
## Our Solution: A Straight-Through Estimator (STE) Framework

Our solution elegantly decouples the forward and backward passes to achieve both correctness and trainability. The entire logic is encapsulated in the `rosa_vdd_ops` function.

### 1. Suffix Attention (SUFA) as the Gradient Proxy

The backward pass relies on **Suffix Attention (SUFA)**. unlike standard attention which calculates similarity based on global semantics, SUFA calculates the dot-product similarity between the **suffixes** of Q and K within a geometric decay window.

**Why SUFA?**
*   **Goal Alignment**: Both ROSA and SUFA strive to match queries with relevant keys based on suffix similarity. The gradient signal from SUFA—"make similar suffixes have higher dot products"—is exactly the right incentive for the model to learn representations that will form discrete matches in the ROSA forward pass.
*   **Stable Gradients**: Standard attention provides a well-behaved, dense gradient signal via the softmax function, smoothing the loss landscape.
*   **Efficiency**: SUFA is implemented using `scaled_dot_product_attention`, leveraging the **Flash Attention** ecosystem for speed.

### 2. Value Detach

A critical innovation in our training recipe is **detaching the Value tensor** in the Soft Proxy branch.
*   **The Problem**: If `V` is optimized via Soft Attention, it tends to learn the weighted mean of multiple keys to minimize loss, becoming "blurry." This reduces the incentive for `Q` and `K` to find the single *correct* match.
*   **The Solution**: We detach `V` in the soft branch. The soft branch is used *only* to train `Q` and `K` to find the correct location. The `V` is updated *only* by the Hard ROSA branch (via explicit injection). This forces `Q/K` to align geometrically to find the sharp, correct `V`.

### 3. Geometric Decay

To bridge the gap between continuous dot products and discrete suffix matching, we apply a **Geometric Decay** to the query and key projections. The decay factor is calculated dynamically based on the suffix window size, allowing the model to establish an optimal effective context horizon.

This mechanism enforces a **strict temporal hierarchy**: it assigns exponentially higher weights to recent tokens while suppressing distant history. By ensuring that the immediate token match contributes most significantly to the similarity score, this weighting scheme constructs a unique "state fingerprint" that mathematically aligns the Flash Attention objective with the "Longest Common Suffix" logic of ROSA, allowing historical context to serve as disambiguation without dominating the primary signal.

## Installation & Usage

### 1. Installing the C++ Kernel (`rosa_cpp`)

The high-performance discrete Suffix Automaton logic is implemented in C++. You must compile and install this extension to use the operator.

```bash
cd rosa_cpp
pip install --no-build-isolation .
```

### 2. Importing the Operator

```python
from rosa_cpp import rosa_bits_ops
```

## API Reference: `rosa_bits_ops`

The core logic is encapsulated in the `rosa_bits_ops` function:

```python
def rosa_bits_ops(
        query: Tensor, key: Tensor, value: Tensor,
        suffix_window: int = 8,
        suffix_factor: Optional[float] = 0.5,
        attention_mask: Optional[Tensor] = None,
        attention_tau: float = 1.0,
        schmitt_trigger: float = 0.0,
):
    """
    Performs the Rapid Online Suffix Automaton (ROSA) attention-like operation.

    This function computes a differentiable, attention-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be tensors of logits that will be binarized). The operation is designed
    to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): (B, H, T, D) Logits for Query bits (pre-tanh).
        key (Tensor): (B, H, T, D) Logits for Key bits (pre-tanh).
        value (Tensor): (B, H, T, D_v) Logits for Value bits.
        suffix_window (int): Size of the lookback window for fingerprinting.
        suffix_factor (Optional[float]): Decay factor for the window.
        schmitt_trigger (float): Threshold for Schmitt trigger to prevent noise.

    Returns:
        Tensor: The result of the Hard SAM lookup.
    """
```

## Project Structure

- **`modules/`**: Contains PyTorch model definitions, including base classes and specific architecture integrations (e.g., Qwen3).
- **`rosa_cpp/`**: The home of the latest, high-performance C++ kernels.
- **`rosa_ops/`**: An archive of historical implementation snapshots, preserving the evolution of the operator logic.

## Status & Roadmap

This project is currently in an active, experimental phase. Our goal is to provide a stable, performant, and easy-to-integrate ROSA layer for the broader research community.

## Acknowledgments

This work is heavily inspired by the original research and innovations of Peng Bo (BlinkDL) in the RWKV project.