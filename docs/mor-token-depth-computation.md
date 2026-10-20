# 44 · MoR 启发：逐 Token 循环预算、状态冻结和真实计算成本边界

## 原始研究与真实设计动机

- [Mixture-of-Recursions（NeurIPS 2025）](https://proceedings.nips.cc/paper_files/paper/2025/hash/8b08bbf8b420faa6eeb4020720582ec7-Abstract-Conference.html) 结合**共享循环权重**与**Token 级自适应计算**：不是每个 Token 都必须接受同样深度；论文还涉及活跃 Token 的 Attention、动态 K/V 缓存管理等关键实现。
- [Ouro / Looped Language Models](https://ouro-llm.github.io/) 与 [Huginn](https://github.com/seal-rg/recurrent-pretraining) 研究的是潜在计算深度与可变预算，不意味着循环计算天然比普通 Transformer **更快**。
- [MoR 开源实现](https://github.com/raymin0223/mixture_of_recursions) 可用于后续审查实际压缩查询 Token、未完成序列的 K/V 缓存、训练的路由损失；本课尚未移植那些服务内核与策略。

为什么这么设计？固定深度 `D` 对所有 Token 都执行 `D` 次，即使有些位置很容易。共享权重缩减参数存储，但**不会自动缩减同样次数的矩阵乘**。如果某些 Token 早已形成有效的表示，可尝试在后续轮次保留原状态，并将计算预算留给困难 Token。想把这件事变成实际吞吐量收益，还必须有*物理的活跃 Token 压缩、Attention/KV 语义管理及 kernel 级性能验证*。

## 已实现的可组合数学流（不是一个黑盒 MoR）

从项目现有的 **2 轮共享参数 Attention + SwiGLU Loop** 生成一张真实的、仍然无环的 DAG：

```text
budget_ids [3,1] = [2,1,0]
    │
    ├─ token_depth_mask(step=1,width=4) -> M0 [3,4]
    │                             ┌────────────────────────┐
    lookup -> full loop 0 -> C0   │ state_mask_blend        │ -> H0
       └──────── original H ──────┤ previous + candidate M0│
                                  └────────────────────────┘
    ├─ token_depth_mask(step=2,width=4) -> M1 [3,4]
    │                             ┌────────────────────────┐
    H0 -> full loop 1 -> C1       │ state_mask_blend        │ -> H1
       └──────── previous H0 ─────┤ previous + candidate M1│
                                  └────────────────────────┘
    H1 -> shared Embeddingᵀ LM Head + cross entropy
```

```text
M_r[t,d] = 1   if budget[t] >= r+1, else 0   (same value across d)
H_r[t,d] = M_r[t,d] * Candidate_r[t,d] + (1-M_r[t,d]) * H_(r-1)[t,d]
```

每一个 `token_depth_mask` 和 `state_mask_blend` 都是普通可增删、可改线的积木，不依赖内置循环或隐藏代码。**输入预算本身是合成整数 ID**，不是神经网络学习的退出决策。它不包括 router、预算惩罚或训练策略。

| Token | 预算 | 第 1 轮 | 第 2 轮 | 最终状态 |
|---|---:|---|---|---|
| t=0 | 2 | 更新 | 更新 | 第二轮候选状态 |
| t=1 | 1 | 更新 | 保留旧状态 | 第一轮采纳状态 |
| t=2 | 0 | 保留输入 | 保留输入 | 输入 Embedding 原状态 |

### 如何证明梯度正确？

```text
dL/dCandidate_r = M_r * (dL/dH_r)
dL/dPrevious_r = (1-M_r) * (dL/dH_r)
dL/dBudget = None  # hard discrete controller
```

候选分支在已退出行上不接收来自该混合节点的直接梯度。**注意**：在其它 Token 的 Attention 内，一个已退出 Token 的保留状态仍可能被作为 Key/Value；这不等同于所有属于该 Token 的计算贡献或训练梯度都为零。

## 为什么这还不是真实稀疏执行？必须区分两个计数

此实验完整计算两轮所有 Token 的 Q/K/V、Attention 和 SwiGLU 候选张量，*再*应用 Mask。对于示例预算 `[2,1,0]`：

- 状态 **可采纳的更新行次**：`3 / 6`，数学有效率 **50%**。
- 实际被跳过的稠密候选 **GEMM 次数**：`0`。
- 实际硬件时间、峰值 KV 显存、吞吐量：**未测量，不能推断**。

要升级到 MoR 样式的真正动态运行，必须证明“只取活跃 Query / 用正确版本的前序隐藏状态计算所有可见 Key 与 Value / 实际执行紧凑 GEMM / 正确 scatter 回所有 Token / KV 缓存按时间及递归深度索引”，然后与现有稠密对照逐元素比对。在未实现这些内核前，不能把状态掩码比例宣传为 FLOPs 加速率。

## 可证伪的最小对照

1. 同样的 8 组 Parameter 与两轮 Transformer，单独覆盖 `budget_ids` 为全 0、`[2,1,0]`、全 2，验证采纳状态与旧状态逐数一致。
2. JavaScript、真实 NumPy、CPU PyTorch 对比每个中间张量；不能只对最后 CE 做一次近似比较。
3. `state_mask_blend` 反向：Mask=0 的候选位置梯度严格为 0；Mask=1 的 previous **直接**梯度为 0；两轮复用参数的梯度与 PyTorch Autograd 数值一致。
4. 对共享 Wq、Wv 和 Embedding 做有限差分；两步 AdamW 与保存/恢复第三步均对照。
5. 必须额外确认 **没有**跳过真实 MatMul，再考虑 Token packing / KV 位置重映射的独立升级。

运行：

```bash
node tests/adaptive_token_depth_contracts.js  # 需要本地 Python, NumPy, CPU PyTorch
```

或在 `learn.html` 中选择 **第 44 课**，查看预算/冻结状态、悬停论文来源、运行反传和两步训练，再下载两套 Python 程序。

## 工程决策矩阵

| 方案 | 为什么采用 | 确实的主要代价 | 应该反对的场景 |
|---|---|---|---|
| 固定循环深度 | 便于同形 GEMM、缓存语义简单、性能可预测 | 简单 Token 也付出最大深度 | Token 难度差异大且服务算力紧张 |
| 逐 Token 语义停止（本课） | 首先验证输出一致性、保留状态与梯度边界 | 稠密算子不减，控制 Mask 还有额外操作 | 目标是立即降低推理延迟 |
| MoR 风格物理 Token 压缩 | 试图同时减少参数存储和活跃 Token 计算 | 路由、Token gather/scatter、K/V 缓存与位置语义复杂 | Batch 很小、路由开销大、跨 Token 依赖不能正确维护 |
| 仅整句自适应深度 | 单请求共享同一轮数，批处理更容易 | 句内 Token 难度异质性利用不足 | 强依赖逐 Token 细粒度计算优化 |

**结论边界：** 本节真实验证的是*硬深度选择的数学语义*，不是 MoR 论文的完整神经架构，也不是可供工业部署的稀疏 Attention 引擎。