# Executable Block Catalogue · 45 verified numerical blocks

本页是 SeetheTruth Learning Studio 的**实际执行合同**。现有 45 种积木均已具有对应的小矩阵数值实现。但**今后纳入新积木不再以数值矩阵实现为强制条件**：最低标准是可检查的输入输出语义、数学公式、形状与状态变换、前提条件和错误测试；仅有定性实现的积木必须明确标注，不能伪装成可数值执行。当前共 45 种（基础 11 + 扩展 34），12 个引导课程以及空白创作模式。

## 分层积木及形状语义

| 范畴 | 可执行积木 | 数学含义与关键限制 |
| --- | --- | --- |
| 输入与表征 | Input, Token IDs, Embedding | Input: 小型连续向量；Token IDs: 整数 [T,1]；Embedding: E[ID] 查表，有范围校验与固定教学参数 |
| 线性及标量算术 | Linear, MatMul, Scale, Add scalar, Square | Linear 采用确定性微型权重/偏置；MatMul 检查内维；Scale/Bias 只用固定标量 |
| 张量变换 | Transpose, Slice columns, Concat columns, Repeat rows | 转置、切片、通道拼接和行重复均实际生成新数值；二维轴上限被严格检查 |
| Reduction | Sum(last axis), Mean(last axis), Variance(last axis) | 各 token 独立按隐藏特征列做聚合；方差采用 population / biased estimator |
| 激活与非线性 | ReLU, SiLU, GELU(tanh), Sigmoid, Tanh, Softplus, Softmax | 按定义逐元素或逐行；Softmax 采用 max-subtraction 数值稳定计算；GELU **明确是 tanh 近似版** |
| 归一化 | RMSNorm, LayerNorm, L2 Normalize | RMSNorm 不减均值；LayerNorm 使用总体方差且固定 γ=1、β=0；L2 使用 max(||x||,eps) |
| 分支与门控 | Add, Multiply, Subtract, Divide, SwiGLU, GEGLU | 按元素计算并要求张量形状完全匹配；SwiGLU/GEGLU 输入是独立的 gate/up 投影，内部不虚构参数 |
| 位置及注意力 | Sinusoidal PE, RoPE(2D), Mask, Masked Softmax, SDPA, GQA | 位置加法与正交旋转可直接比较；Masked Softmax 使屏蔽位置严格为 0；GQA 保留真实命名轴 |
| 类别预测 | Temperature, ArgMax, Top-k | 温度变换 logits；ArgMax 稳定 tie-break；Top-k 只产出有序数值并提供选中索引，不假装随机采样 |
| 正则化 | Dropout (seeded) | 固定伪随机种子可复算的 inverted Dropout；evaluation 恒等；不谎称每轮独立采样 |
| 学习目标 | MSE, MAE, Cross Entropy, KL(P∥Q) | CrossEntropy 采用稳定 logsumexp 和整数 class ID；KL 要求两边都是有效概率分布；此处仅前向损失，没有反传 |

注：列表中形状统一是 T×D 的二维微型矩阵，部分 SDPA/GQA 积木进入之前验证过的多维 IR。输入最多 T=8、D=16，K/V Cache 使用另一个独立有状态教学引擎。用户不会因为只看到教学结果就被告知已运行真实 PyTorch 训练或 LLM 推理。

## 12 个可以实际执行的课程

01 FFN、02 残差、03 单头 Attention、04 MSE Loss、05 GQA/RoPE/Causal、06 Decode 只读 KV 前缀、07 双上投影 SwiGLU、08 RMSNorm vs LayerNorm、09 Token ID→Embedding→位置编码、10 Classifier 与 CrossEntropy、11 构造因果 Mask 后 Softmax、12 Temperature→Softmax/ArgMax/Top-k。另有 Blank 自定义。

每个课程不仅显示概念积木，还要能够按执行次序计算每个节点的矩阵、在 Inspector 中看到公式/数学目的/参数/真实数值与确定性示例的证据边界。

## 与框架和学术共识对齐的数值口径

- GELU 默认 PyTorch GELU 的 `approximate='none'` 是精确 CDF 形式；本课程**只采用** `approximate='tanh'`，两者不允许混称：https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU
- LayerNorm 使用每个 token 特征轴的 biased population variance，缺省含可学习仿射参数；本课程为 γ=1、β=0 的特例：https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- Dropout 在真实 PyTorch 训练下，每次前向独立采样并乘以 1/(1-p)，推理时是恒等映射；固定随机序列只是为了课堂重现实验，不代表真实训练随机性：https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- CrossEntropy 用未归一化 logits 和离散 class target，返回平均 Loss。课程没有 class weight、ignore_index、soft labels 或 label smoothing：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- 注意 PyTorch `kl_div` 默认 reduction 与严格按行求和再平均 KL 的定义不同；本课程显式计算 D_KL(P||Q) 的**batchmean-like** 每 token 平均，不能拿默认 `reduction='mean'` 直接比较：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html

## 测试证据和 UI 审计

`node tests/block_catalog_contracts.js` 覆盖新算子的正常数值 oracle、形状拒绝、浮点与非法标签。`node tests/learning_lab_contracts.js` 保留基础 11 种计算的回归。`node tests/semantic_tensor_ir_contracts.js` 保留 GQA/RoPE 独立合同。新增 `tests/browser_catalog_screenshots.py` 真正打开 Chromium，验证 45 个目录入口、搜索、类别、SwiGLU、CrossEntropy、Masked Softmax 及所有主要旧课程能运行，并生成可查看的 PNG 截图。

截图在 CI 中上传到 `seethetruth-45-block-library-real-chromium` Artifact，作为视觉证据。截图由正在运行的浏览器渲染，绝不是手绘效果图。没有截图不宣称完成了 UI 可见性验证。

## 尚未进入实际可执行目录

真实 Tensor rank 任意值的自动广播、完整 MoE 分组路由/专家负载、FlashAttention 内核、实际位置编码布局与各种特定模型权重、反向自动微分、Optimizer step、RL rollout/PPO/GRPO、实测 GPU 加速：当前版本未实现，不在 45 种计数中。应该在具备源与数值验收后逐项纳入。

## Learning-first 分类与扩展标准

新积木按**学习证据能力**分类，不按是否可在 GPU 上运行分类：

- `CONCEPT`（基础）：可教学的数学功能、输入输出/轴语义、依赖与状态变化，并保留无法仅由 shape 验证的前置条件。不要求完整参数矩阵或数值 kernel。
- `MICRO`（可选增强）：小型确定性数值计算、数值 oracle 与条件分支，用于解释个别元素计算。已有 45 种积木都支持。
- `EXTERNAL`（将来可选）：外接框架/真实模型/性能或训练证据，按来源显示。这不是默认课程依赖。

积木可以只有 `CONCEPT`，例如未来的 rollout、advantage、optim state 或动态 MoE 路由概念积木：在未实现独立数值合同之前，它们必须明确表述为形状或状态的教学模型，不允许推断真实训练结果，也不能计入可数值执行的积木数量。

原理推演优先于大规模矩阵数值模拟。默认工作台见 docs/learning-first-architecture.md。
