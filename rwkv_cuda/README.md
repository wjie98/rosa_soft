# RWKV CUDA Operators

This directory contains high-performance CUDA operator implementations for RWKV models.

## ⚠️ Attribution

The core CUDA kernel code in this module is primarily ported and adapted from the following open-source projects by **Peng Bo (BlinkDL)**:

1.  **RWKV-CUDA**
    * Repository: [https://github.com/BlinkDL/RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA)
    * Description: Provides high-performance training operators.

2.  **Albatross**
    * Repository: [https://github.com/BlinkDL/Albatross](https://github.com/BlinkDL/Albatross)
    * Description: Provides high-performance FP16 inference operators.

We sincerely thank BlinkDL for his selfless contributions to the community. This project wraps these operators to facilitate standalone installation and usage.

## Installation

If you need to use these operators independently (e.g., running pure RWKV models without installing the `rosa_soft` core package), you can install them directly from this directory:

```bash
$ pip install --no-build-isolation .
```