# rosa_soft
Softened ROSA QKV Operators for Training Next-Generation LLM Models

## Overview
This repository implements softened ROSA (Rapid Online Suffix Automation) QKV operators that enable efficient training of next-generation large language models with enhanced sequence processing capabilities.

## Core Idea
We replace the traditional ROSA automaton algorithm with a dynamic programming approach, generating attention matrices similar to conventional attention mechanisms. Through specialized softening techniques, these attention matrices become differentiable during training, enabling true end-to-end training of models incorporating ROSA operations.

## Technical Approach

### Dynamic Programming Formulation

For **hard mode**:
```
dp[i][j] = dp[i][j] ? dp[i-1][j-1] + 1 : 0
```

For **soft mode**:
```
# Previous version
dp[i][j] = dp[i-1][j-1] * a[i][j] + a[i][j]

# Current version (more parallelizable)
t = cumsum(a)
dp = t - cummax(t * (1 - a))
```

### Implementation Notes
- **20251030**: Adopted a new, more relaxed softening formula based on cumsum and cummax primitives. This transforms the scan computation from a strictly associative recursion into a more flexible, rearrangeable form. This change is crucial as it provides significant optimization space for future fused kernel implementations.
- **20251026**: Initial version requires interleaved scanning in DP mode, with intermediate results currently stored in global memory
- We modified the original ROSA implementation for practical convenience. In our implementation, unmatched items are set to their own values (latest values) rather than zero, which naturally incorporates positional information

## Key Features
- Differentiable ROSA operations for gradient-based training
- Efficient DP-based implementation
- Compatible with modern transformer architectures
- Support for both hard and soft matching modes

## Background
ROSA (Rapid Online Suffix Automation) is a novel sequence retrieval mechanism proposed by pengbo as part of the next-generation long-sequence processing architecture. This work builds upon the research from [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8).

## Applications
- Long-context language modeling
- Efficient sequence matching in transformers
- Enhanced retrieval-augmented generation
- Next-generation LLM architectures

## Status
Initial development version - active research and improvements ongoing.
