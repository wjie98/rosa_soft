# Softened ROSA Operators

[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/your-username/rosa_soft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**rosa_soft** provides a robust, end-to-end trainable implementation of the **ROSA (Rapid Online Suffix Automaton)** operator for next-generation LLMs.

> **Note:** This project makes the discrete, non-differentiable ROSA mechanism compatible with gradient-based optimization using a **Straight-Through Estimator (STE)** framework.
> For a deep dive into the theoretical background, the Suffix Attention (SUFA) proxy, and our design philosophy, please read **[The ROSA Concept & Method](./docs/CONCEPT.md)**.

## 📂 Project Structure

* **`rosa_soft/`**: The core Python package containing the C++ kernel for the ROSA operator.
* **`rwkv_cuda/`**: Standalone, high-performance CUDA operators for standard RWKV models.
* **`modules/`**: PyTorch implementations of ROSA layers and reference models (e.g., Qwen3, RWKV7).
* **`legacy/`**: Historical snapshots of the operator evolution.

## 🚀 Installation

Prerequisites: **PyTorch** and **CUDA** toolkit.

### 1. Install from Source (Recommended)

You can install the core `rosa_soft` operator directly from GitHub:

```bash
$ pip install --no-build-isolation git+https://github.com/wjie98/rosa_soft.git
```

### 2. (Optional) Install RWKV CUDA Operators

If you are using the provided RWKV-based reference models in `modules/` or need standalone RWKV ops:

```bash
$ git clone https://github.com/wjie98/rosa_soft.git && cd rosa_soft
$ pip install --no-build-isolation rwkv_cuda
```

## 🛠 Usage

### Using the ROSA Operator

Two operators are exported:
- **`rosa_bits_ops`**: Uses standard SDPA attention as the gradient proxy.
- **`rosa_scan_ops`** (experimental): Uses linear attention as the gradient proxy (currently under fine-tuning).

```python
import torch
from rosa_soft import rosa_scan_ops

# B: Batch, H: Heads, T: Sequence Length, D: Bits
# Inputs are usually logits (before tanh or sign)
query = torch.randn(2, 64, 1024, 8).cuda()
key   = torch.randn(2, 64, 1024, 8).cuda()
value = torch.randn(2, 64, 1024, 8).cuda()

# For standard SDPA proxy:
output: Tensor = rosa_bits_ops(
    query, key, value,
    suffix_window=8,        # Lookback window for fingerprinting
    suffix_factor=0.5,      # Decay factor for the window
    schmitt_trigger=0.1     # Threshold to prevent noise toggling
)

# For linear attention proxy (experimental):
output: Tensor = rosa_scan_ops(
    query, key, value,
    suffix_window=8,        # Lookback window for fingerprinting
    suffix_factor=0.5,      # Decay factor for the window
    schmitt_trigger=0.1     # Threshold to prevent noise toggling
)

```

### API Reference: `rosa_bits_ops` & `rosa_scan_ops`

| Parameter | Type | Description |
| --- | --- | --- |
| `query` | `Tensor` | (B, H, T, D) Logits for Query bits. |
| `key` | `Tensor` | (B, H, T, D) Logits for Key bits. |
| `value` | `Tensor` | (B, H, T, D_v) Logits for Value bits. |
| `suffix_window` | `int` | Size of the lookback window for geometric decay fingerprinting (SUFA proxy). |
| `suffix_factor` | `float` | Decay factor for the window. |
| `schmitt_trigger` | `float` | Hysteresis threshold to stabilize bit flipping against noise. |

## 🧩 Reference Modules

The `modules/` directory contains experimental integrations of the ROSA layer into modern architectures.

* **`rwkv7rosa.py`**: A hybrid architecture combining RWKV-7 token mixing with the ROSA operator.
* **`qwen3.py`**: Experiments with Qwen-based architectures.

> **Warning:** The architectures in `modules/` are subject to change as we refine the best practices for placing the ROSA layer.

## 📅 Roadmap & History

* **2026-02**: Adopted **Suffix Linear Attention** as the gradient proxy. This formulation achieves **O(T)** training time and space complexity and demonstrates superior stability compared to the standard dot-product attention version.
* **2026-01**: Implemented **Gated Value Detach**. This mechanism balances the QK matching signal with the V semantic signal, ensuring smoother convergence compared to the previous hard detach method.
* **2025-12**: Initial implementation of the STE framework and SUFA proxy.

## Acknowledgments

This project is built upon and inspired by the groundbreaking research of **Peng Bo (BlinkDL)** in the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8) project. We extend our sincere appreciation for the innovative work that has significantly influenced our approach.
