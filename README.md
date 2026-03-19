# Softened ROSA Operators

[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/your-username/rosa_soft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**rosa_soft** provides a robust, end-to-end trainable implementation of the **ROSA (Rapid Online Suffix Automaton)** operator for next-generation LLMs.

> **Note:** This project makes the discrete, non-differentiable ROSA mechanism compatible with gradient-based optimization using a **Straight-Through Estimator (STE)** framework.
> For a deep dive into the theoretical background, the Suffix Attention (SUFA) proxy, and our design philosophy, please read **[The ROSA Concept & Method](./docs/CONCEPT.md)**.

## 📂 Project Structure

* **`rosa_soft/`**: The core Python package containing the C++ kernel for the ROSA operator.
* **`legacy/`**: Historical snapshots of the operator evolution.
* **`demo/`**: PyTorch implementations of ROSA layers and reference models (e.g., RWKV7).

## 🚀 Installation

Prerequisites: **PyTorch** and **CUDA** toolkit.

You can install the core `rosa_soft` operator directly from GitHub:

```bash
$ pip install --no-build-isolation git+https://github.com/wjie98/rosa_soft.git
```

## 🛠 Usage

### Using the ROSA Operator

Two operators are exported:
- **`rosa_soft_ops`**: Uses softened dynamic programming as the gradient proxy.
- **`rosa_sufa_ops`**: Uses suffix attention as the gradient proxy.
- **`rosa_scan_ops`** (experimental): Uses linear attention as the gradient proxy (currently under fine-tuning).

```python
import torch
from rosa_soft import *

# B: Batch, T: Sequence Length, H: Heads, D: Bits
# Inputs are usually logits (before tanh or sign)
query = torch.randn(2, 1024, 64, 8).cuda()
key   = torch.randn(2, 1024, 64, 8).cuda()
value = torch.randn(2, 1024, 16, 32).cuda()

# For softened dynamic programming proxy
output: Tensor = rosa_soft_ops(
    query, key, value,
    schmitt_trigger=0.1     # Threshold to prevent noise toggling
)

# For suffix attention proxy:
output: Tensor = rosa_sufa_ops(
    query, key, value,
    suffix_window=8,        # Lookback window for fingerprinting
    suffix_factor=0.5,      # Decay factor for the window
    schmitt_trigger=0.1     # Threshold to prevent noise toggling
)

# For suffix linear attention proxy (experimental):
output: Tensor = rosa_scan_ops(
    query, key, value,
    suffix_window=8,        # Lookback window for fingerprinting
    suffix_factor=0.5,      # Decay factor for the window
    schmitt_trigger=0.1     # Threshold to prevent noise toggling
)

```

### API Reference: `rosa_soft_ops` & `rosa_sufa_ops` & `rosa_scan_ops`

| Parameter | Type | Description |
| --- | --- | --- |
| `query` | `Tensor` | (B, T, H, D) Logits for Query bits. |
| `key` | `Tensor` | (B, T, H, D) Logits for Key bits. |
| `value` | `Tensor` | (B, T, H_v, D_v) Logits for Value bits. |
| `suffix_window` | `int` | Size of the lookback window for geometric decay fingerprinting (SUFA proxy). |
| `suffix_factor` | `float` | Decay factor for the window. |
| `schmitt_trigger` | `float` | Hysteresis threshold to stabilize bit flipping against noise. |

## 🧩 Reference Demo Modules

The `demo/` directory contains experimental integrations of the ROSA layer into modern architectures.

* **`rwkv7.py`**: PyTorch implementation of the RWKV-7 model.
* **`rwkv7rosa.py`**: A hybrid architecture combining RWKV-7 with the ROSA token mixing layer.

> **Warning:** The architectures in `demo/` are subject to change as we refine the best practices for placing the ROSA layer.

## 📅 Roadmap & History

* **2026-02**: Adopted **Suffix Linear Attention** as the gradient proxy. This formulation achieves **O(T)** training time and space complexity and demonstrates superior stability compared to the standard dot-product attention version.
* **2026-01**: Implemented **Gated Value Detach**. This mechanism balances the QK matching signal with the V semantic signal, ensuring smoother convergence compared to the previous hard detach method.
* **2025-12**: Initial implementation of the STE framework and SUFA proxy.

## Acknowledgments

This project is built upon and inspired by the groundbreaking research of **Peng Bo (BlinkDL)** in the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8) project. We extend our sincere appreciation for the innovative work that has significantly influenced our approach.
