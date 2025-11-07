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

### Track 2: SuffixAttention (SUFA) - A Practical, Attention-like Simplification

We propose a novel perspective: **view ROSA as an extreme form of quantized attention**.

- Standard attention measures similarity via dot products between continuous query/key vectors.
- ROSA measures similarity via exact historical suffix matching between discrete tokens.

Based on this perspective, we introduce **SuffixAttention (SUFA)**, a new mechanism that fundamentally alters the matching process to achieve greater hardware efficiency. SUFA makes two key simplifications:
1.  **Sequential Match → Direct Similarity**: It completely replaces the complex, sequential **suffix matching** algorithm with a direct, point-wise **Hamming distance calculation** between individual quantized query and key vectors.
2.  **Infinite History → Finite Range**: To ensure computational feasibility, the search for the best key (the one with the minimum Hamming distance) is constrained to a **finite range** of the recent history, effectively capping the maximum look-back distance.

This approach yields significant benefits:
- **Training Efficiency**: The softened version of the SUFA kernel—a highly regular, parallel search—is perfectly suited for optimization within the **Flash Attention ecosystem**.
- **Inference Efficiency**: During inference, the operation becomes a fast Hamming distance search, which can be aggressively optimized on CPUs or specialized hardware using bit-manipulation instructions.

SUFA offers a compelling bridge between the parameter-free efficiency of ROSA and the mature, highly-optimized tooling of conventional attention mechanisms.

## Project Structure

- **`minirosa/`**: Contains self-contained mini-models for experimenting with our different approaches, both integrated into a Qwen3-based architecture:
  - `model_minirosa.py`: An example of integrating the **Faithful Softening** ROSA (Track 1) into a dense Transformer model.
  - `model_minisufa.py`: An example of integrating **SuffixAttention (SUFA)** (Track 2) into a dense Transformer model.
- **`rosa_ops/`**: Contains the core, low-level implementations of the softened ROSA QKV operator (Track 1). See the `rosa_ops/README.md` for a detailed implementation history.

## Status & Roadmap

This project is currently in an active, experimental phase.

## Acknowledgments

This work is heavily inspired by the original research and innovations of Peng Bo (BlinkDL) in the RWKV project.