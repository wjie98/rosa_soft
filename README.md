# Softened ROSA Operators and Explorations for Next-Generation LLMs

[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/your-username/rosa_soft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository explores the implementation and evolution of **ROSA (Rapid Online Suffix Automaton)**, a novel attention-free mechanism for sequence processing. Our work proceeds along two parallel tracks:

1.  **Faithful Softening**: Implementing a differentiable, "softened" version of the original ROSA algorithm, enabling its integration into end-to-end trainable neural networks.
2.  **Attention-like Simplification**: Proposing a new, practical variant of ROSA based on sliding windows and Hamming distance, which leverages the existing Flash Attention ecosystem for efficient training.

## Background: What is RWKV-8 ROSA?

ROSA is a groundbreaking concept proposed by Peng Bo (BlinkDL) as part of the RWKV-8 architecture. It aims to replace the standard attention mechanism with what is described as a "neurosymbolic infinite-range lossless information propagator."

The core idea is to predict the next token in a sequence based on the longest exact match found in the history. For a given sequence `x`, the output `y_i` is determined by finding the longest suffix of `x` ending at `i-1` that matches a previous substring. If such a match is found at index `j`, the output is the token that followed that match, `x_{j+1}`.

This mechanism has several powerful properties:
- **Parameter-Free**: The core logic has no trainable weights.
- **No Dot Product / Softmax**: It operates on discrete tokens, eliminating the quadratic complexity of standard attention.
- **No Float KV Cache**: It only needs to store the history of discrete tokens.
- **Efficient Inference**: The underlying Suffix Automaton can be processed very quickly on a CPU, in parallel with GPU-based layers.

This project is inspired by and builds upon the original research found at the [RWKV-LM Repository](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8).

## Project Philosophy & Our Approaches

Our exploration aims to make ROSA and its principles practical for training and integration into modern LLMs.

### Track 1: Faithful Softening of ROSA

The primary challenge of the original ROSA is its non-differentiable, discrete nature. Our initial work focuses on creating a "soft" version using dynamic programming that is mathematically equivalent in its hard form but allows for gradient flow during training.

This approach uses specialized `cumsum` and `cummax` primitives to create a parallelizable scan, making the operation differentiable and suitable for end-to-end training.

> **Code Location**: The core, low-level operator implementations for this track, along with their development history, have been moved to the `rosa_ops/` subdirectory.

### Track 2: A Practical, Attention-like Simplification

We propose a novel perspective: **view ROSA as an extreme form of quantized attention**.

- Standard attention measures similarity via dot products between continuous query/key vectors.
- ROSA measures similarity via exact historical suffix matching between discrete tokens.

We can relax the strict conditions of ROSA to create a more practical, hardware-friendly mechanism that retains its core spirit. Our proposed simplification involves two key changes:
1.  **Infinite History → Finite Sliding Window**: Instead of matching against the entire history, we only consider a fixed-size sliding window.
2.  **Exact Match → Fuzzy Match (Hamming Distance)**: Instead of requiring a perfect suffix match, we use Hamming distance as a similarity metric between quantized query and key vectors.

This approach yields significant benefits:
- **Training Efficiency**: The softened version of this simplified kernel is highly compatible with the **Flash Attention ecosystem**, allowing for highly optimized training on GPUs.
- **Inference Efficiency**: During inference, the operation becomes a fast Hamming distance search over a sliding window, which can be aggressively optimized on CPUs or specialized hardware.

This new approach offers a compelling bridge between the efficiency of ROSA and the mature tooling of conventional attention mechanisms.

## Project Structure

- **`/`**: The main project directory.
- **`minirosa/`**: Contains a new, self-contained mini-model (`model_minirosa.py`) for quickly experimenting with and validating the different ROSA implementations within a standard Transformer architecture.
- **`rosa_ops/`**: Contains the core, low-level implementations of the softened ROSA QKV operator. See the `rosa_ops/README.md` for a detailed implementation history.

## Status & Roadmap

This project is currently in an active, experimental phase.

## Acknowledgments

This work is heavily inspired by the original research and innovations of Peng Bo (BlinkDL) in the RWKV project.