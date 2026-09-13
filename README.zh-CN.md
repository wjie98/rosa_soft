# rosa_soft

[English](README.md) | 简体中文

用于 ROSA 离散后缀检索的 PyTorch 扩展。提供精确 CUDA 匹配、dense soft 与
bitflip 两种训练梯度，以及用于有状态 CPU 推理的 C++ 后缀自动机。

## 算子

| API | 用途 | 输入布局 |
| --- | --- | --- |
| `rosa_soft` | CUDA hard 前向，稠密 soft 代理反向 | Dense 或 packed |
| `rosa_bitflip` | CUDA hard 前向，完整 bit 翻转输出差分 | Dense |
| `rosa_hard` | 无状态 CPU SAM 推理与验证，返回值和匹配位置 | Dense 或 packed |
| `RosaSam` | 有状态 CPU SAM 路由，V 历史由调用方管理 | Dense 或 packed 分块 |

两个训练算子使用相同的精确、因果、无限后缀前向。对查询位置 `i`，选择 K 结束位置
`j < i` 的最长 Q/K 后缀匹配；长度相同则选最近的 `j`。输出二值 `V[j+1]`，
没有匹配则输出零。正数编码为 `+1`，零和负数编码为 `-1`。训练前向不输出软表征。

## 安装

需要 Python 3.10+、PyTorch 2.11.x 和 C++17 编译器。CUDA 训练还需要与已安装
PyTorch 构建兼容的 CUDA toolkit。先安装 PyTorch，再从源码目录构建：

```bash
pip install --no-build-isolation .
```

默认自动检测 CUDA。设置 `USE_CUDA=1` 可要求构建 CUDA 扩展；设置 `USE_CUDA=0`
则仅构建 CPU 推理与验证功能。toolkit 不在默认搜索路径时设置 `CUDA_HOME`。
扩展在安装阶段编译，导入时不编译。执行限制见 [API 与部署](docs/API.md)。

## 训练

```python
import torch
from rosa_soft import rosa_soft, rosa_bitflip

q = torch.randn(1, 2048, 4, 8, device="cuda",
                dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(1, 2048, 2, 64, device="cuda",
                dtype=torch.float16, requires_grad=True)

y = rosa_soft(q, k, v)  # [1, 2048, 4, 64]
# 使用另一种梯度估计器，hard 前向不变：
# y = rosa_bitflip(q, k, v, rows=64, chunks=2)
y.float().square().mean().backward()
```

Dense Q/K 使用 `[B,T,H,D]`，其中 `1 <= D <= 32`。V 使用 `[B,T,Hv,Dv]`，
要求 `H % Hv == 0`。CUDA 训练支持 FP16、BF16、FP32 输入，使用 FP32 累积梯度，
支持一阶 autograd 和 `torch.compile`。

默认估计器为 `rosa_soft`。静态参数 `scale=1.0`、`dropout_p=0.0`、
`mismatch_scale=3.0` 只影响反向。`rosa_bitflip` 在固定 dY 下计算完整的激活 bit
翻转输出变化，再用 softsign 将信用映射回连续输入。显式 `tied="qk"` 和
`tied="qkv"` 模式会同时翻转共享激活的对应 bit。两者都不是任意非线性任务损失的
无偏导数，也不保证其中一种在所有训练任务上更好。

独立 QKV 4-bit 模型使用 `[B,T,C/4,4]`。
[Rosa4Bit 适配器](examples/rosa_4bit.py) 将可训练输出幅度放在 V 量化之外。
用法见 [模型集成指南](docs/API.md#qkv-4-bit-models) 和
[残差块训练示例](examples/train_bitflip.py)。

## 显存与扩展性

固定宽度下，CUDA 训练沿序列长度执行二次规模的计算。
无限后缀匹配不代表 GPU 训练时间为线性。

**固定 batch、head 数、bit/value 宽度和 `rows` 后，bitflip 工作空间为 O(T)，
不是 O(T log T)。** 令 `S = B*H/chunks`、`R = min(rows,T)`：

- 独立翻转的 band 存储为 `O(S*R*T)`。
- 联合 QK/QKV 翻转的 band 存储为 `O(S*R*T*D)`。
- 完整输入、保存的路由和最终梯度仍占线性空间，不随 `chunks` 等比例缩小。

`rows` 默认 256，限制同时处理的 query 行数，不限制后缀长度。`chunks` 默认 1，
表示反向依次处理的等分 head 组数，必须同时整除 H 和 Hv。适当减小 `rows` 或增大
`chunks` 可以降低工作空间，但存在吞吐量取舍。线性空间的绝对开销仍可能很大，
具体分配和算例见 [显存与复杂度](docs/MEMORY.md)。

## 推理

```python
import torch
from rosa_soft import rosa_hard, RosaSam

q = torch.randn(1, 16, 4, 8)
k = torch.randn_like(q)
v = torch.randn(1, 16, 2, 64)
y, ends = rosa_hard(q, k, v)  # 没有匹配时 ends 为 -1。

sam = RosaSam(num_heads=4, symbol_bits=8)
ends = sam.update(q, k)
# 后续 update 延续各序列的历史，直到调用 sam.reset()。
```

CPU SAM 避免全候选 CUDA DP，支持增量检索，但状态随历史增长；当前实现不保证
每个 token 的最坏耗时为常数。CPU 调用是同步的，GPU 输入也需要同步经主机处理。
调用 `model.eval()` 不会自动切换算子。

## 文档

- [English README](README.md)：英文概览和快速开始。
- [API 与部署](docs/API.md)：参数、packed 输入、绑定模式和模型集成。
- [显存与复杂度](docs/MEMORY.md)：分配账本、容量估计和测量口径。
- [方法与相关工作](docs/METHODS.md)：前向语义与梯度定义。
- [生产设计](docs/DESIGN.md)：公式、kernel 组织与源码导览。

从源码目录运行语义测试和训练集成测试：

```bash
pip install --no-build-isolation ".[test]"
python -m pytest -q
```

测试使用独立 DP、数学定义和直接 bit 翻转 oracle，覆盖因果性、平局处理、无限后缀、
布局、dtype、梯度掩码、编译和 checkpoint 训练。目前 bitflip 已在 SM75、SM86 上
验证；原生 BF16 模型与 Inductor 测试需要 SM80 或更新架构。

## 致谢

ROSA 由 **Peng Bo（BlinkDL）** 提出。感谢他和
[RWKV-LM 项目](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)
提供的原始设计与实现。
