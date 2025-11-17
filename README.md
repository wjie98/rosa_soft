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

Our solution elegantly decouples the forward and backward passes to achieve both correctness and trainability. The entire logic is encapsulated in the `rosa_bits_ops` function.

- **The Hard Forward Pass**: The forward pass executes the *true, discrete* ROSA logic. This ensures that during training, the model is optimized for the exact computation that will be used at inference time. For maximum performance, this pass can even be offloaded to a highly optimized C++ kernel on the CPU (`host_ops=True`), running in parallel with other GPU operations.

- **The Soft Proxy for Gradients**: The backward pass is the key to trainability. Gradients are derived *exclusively* from a continuous, differentiable "soft proxy" function. While the forward pass sees the discrete world, the backward pass provides a smooth landscape for the optimizer to navigate.

### Why Use Suffix Attention (SUFA) as the Proxy?

While our initial research led us to implement a mathematically faithful "soft ROSA," we discovered that an alternative approach often yields better results. For completeness, **we provide both proxy methods:**
1.  **A direct simulation of ROSA (`proxy='rosa'`)**, which uses a softened dynamic programming algorithm to create a differentiable version of the original suffix matching logic.
2.  **Suffix Attention (`proxy='sufa'`)**, which we have found to be a far superior proxy for generating stable and effective gradients in practice.

**SUFA's Basic Principle**: Unlike standard attention which computes similarity between a single query (Q) vector and all key (K) vectors, SUFA's core idea is to better approximate ROSA's suffix matching mechanism. It achieves this by calculating the dot-product similarity between the suffixes of Q and K within a defined range. This "suffix-to-suffix" comparison provides a gradient signal that more directly encourages the model to learn representations where similar sequences have similar endings, which is precisely the behavior required by the discrete ROSA forward pass.

The reasons for preferring SUFA are threefold:

1.  **Goal Alignment**: Both ROSA and SUFA strive to match queries with relevant keys based on suffix similarity. The gradient signal from SUFA—"make similar suffixes have higher dot products"—is exactly the right incentive for the model to learn representations that will form discrete matches in the ROSA forward pass.

2.  **Stable & Smooth Gradients**: Standard attention provides a well-behaved, dense gradient signal via the softmax function. SUFA inherits this stability, leading to a much smoother loss landscape and more reliable training compared to the complex and potentially sparse gradients from the softened dynamic programming approach.

3.  **Computational Efficiency**: The SUFA proxy can be implemented using `scaled_dot_product_attention`, leveraging the highly optimized **Flash Attention ecosystem**. This makes the training backward pass significantly faster and more memory-efficient.

By using Suffix Attention as the proxy, we get the training stability and speed of a mature architecture while optimizing for the unique, parameter-free inference capabilities of ROSA.


## How It Works: The `rosa_bits_ops` Function

Our implementation is centered around a single, flexible function that manages the entire process:
```python
def rosa_bits_ops(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
        alpha: Union[Tensor, float] = 1.0, proxy: Literal["rosa", "sufa"] = "sufa",
        bits_tau: Union[Tensor, float] = 1.0, attn_tau: Union[Tensor, float] = 1.0,
        host_ops: bool = False, training: bool = False, sufa_head_dim: int = 128,
):
    """
    Performs the Rapid Online Suffix Automaton (ROSA) attention-like operation.

    This function computes a differentiable, attention-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be tensors of logits that will be binarized). The operation is designed
    to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): The query tensor of shape (B, H, T, D_qk).
        key (Tensor): The key tensor of shape (B, H_kv, T, D_qk).
        value (Tensor): The value tensor of shape (B, H_kv, T, D_v).
        attn_mask (Optional[Tensor]): An optional boolean attention mask.
        alpha (Union[Tensor, float]): The interpolation coefficient for the STE.
            - `alpha = 0.0`: The output is purely from the soft, differentiable proxy.
            - `alpha = 1.0`: The forward pass output is purely from the hard, discrete op.
            This should be annealed from 0 to 1 during training.
        proxy (Literal["rosa", "sufa"]): The type of soft proxy to use for gradient
            computation. 'rosa' uses a custom dynamic programming approach to
            soft-simulate the ROSA matching algorithm. 'sufa' uses Suffix Attention
            (a standard scaled dot-product attention) as the proxy.
        bits_tau (Union[Tensor, float]): Temperature for converting query/key/value logits
            into continuous bit representations (e.g., via tanh or sigmoid). A smaller
            tau makes the bits "harder".
        attn_tau (Union[Tensor, float]): Temperature for the softmax in the soft attention
            calculation. A smaller tau makes the attention distribution sharper.
        host_ops (bool): If True, the 'hard' part of the operation is offloaded to the CPU
            for potentially faster execution. This requires a custom C++ extension.
        training (bool): If True, enables the training path which uses a soft proxy for
            gradients via a Straight-Through Estimator (STE).
        sufa_head_dim (int): The head dimension to use for the 'sufa' proxy's
            scaled dot-product attention.

    Returns:
        Tensor: The resulting attention output tensor, with the forward pass
                determined by the interpolation of hard and soft ops, and the
                backward pass determined solely by the soft op.
    """
```

## Project Structure

- **`minirosa/`**: Contains the code for integrating ROSA into existing LLM architectures. It provides a **monkey patch** designed to seamlessly replace standard attention layers in **Hugging Face `transformers` models**.
- **`rosa_ops/`**: Contains the core, low-level implementations of the trainable ROSA operator. See the `rosa_ops/README.md` for a detailed implementation history.
- **`train_distillation.py`**: Demonstrates how to apply the monkey patch from `minirosa/` to add a ROSA layer to a **Qwen3-based architecture** and fine-tune it.

## Status & Roadmap

This project is currently in an active, experimental phase. Our goal is to provide a stable, performant, and easy-to-integrate ROSA layer for the broader research community.

## Acknowledgments

This work is heavily inspired by the original research and innovations of Peng Bo (BlinkDL) in the RWKV project.