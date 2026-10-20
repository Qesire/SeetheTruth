# Huginn / xLLM-Loop：为什么把原始输入反复注入循环 Transformer？

<text-only technical documentation; no pretrained-model results>

## 一、首先重建真实源码

以 **ifm-ai/xllm-loop** 的 [`xllm/models/looped.py`](https://github.com/ifm-ai/xllm-loop/blob/main/xllm/models/looped.py) 为直接依据：

- `LinearInputInjection`（源文件约第 141–159 行）：首先 `torch.cat([state, input_embeds], dim=-1)`，然后经 `ColumnParallelLinear(2d,d,bias=cfg.apply_bias_term)`，输出下一循环轮次的输入。这是**带可训练投影的输入与隐状态融合**，不是简单在 Attention 输出上加一个残差。
- `Huginn._recur`（约第 533–546 行）：在每次共享循环 Block 执行**之前**，先执行 `self.input_injection(state,input_embeds)`。
- 同仓库的 `InputInjection` 类采用完全不同的**对角衰减注入**：`exp(-dt*a)·state+dt·(W·input)`，其中 `dt=softplus(dt_bias)`，`a=exp(A_log)`。这不等价于 LinearInputInjection。
- 官方还有 `OrthogonalInputInjection`：在对角注入基础上处理 retained state 沿 injected update 的投影，并非一条普通线性残差。
- [Part II 训练协议](https://github.com/ifm-ai/xllm-loop/blob/main/release/paper-part2/README.md) 的 Table 3 分别实验了 Huginn-Linear、Parcae-Decay、OrthoInj；它们是**不同机制的竞争解释**，不能统称为“多做几次循环”。

参考研究：[Towards Looped Models Done Right, Part II](https://arxiv.org/abs/2610.06833)，[官方 xLLM-Loop 实现](https://github.com/ifm-ai/xllm-loop)。

## 二、为什么工业界研究这种路径？

循环模型复用参数提高**计算深度/模型参数**的比值，但原始输入的信息只能随隐状态反复传递。注入提供一条额外、可学习的信息流，可能缓解迭代过程中的输入信息淡化；如果迭代收敛到不理想的固定点，还可能改变状态转移函数，使其保留任务所需的信息。

这并不是“注入一定更好”：它增加投影参数和每轮 FLOPs，可能使状态漂移、干扰更深表征，或在已具备良好残差和特征保留时重复低层内容。**是否改善实际语言模型质量必须在等训练 FLOPs、相同 token 数据和有意义的测试深度下做真实训练比较。**

### 可证伪的三种竞争解释

| 对比假说 | 设计 | 验证信号 |
|---|---|---|
| H1 原始信息衰减 | 不注入 vs 每轮 `W[state;input]` | 逐轮表征与初始输入相似度、下游 CE、早/晚轮梯度 |
| H2 迭代固定点问题 | 无注入 vs 对角衰减注入 vs 线性注入 | 隐状态变化 RMS、Jacobian 谱、长循环稳定性 |
| H3 只是参数量增加 | 基础 Loop vs 注入并控制等参数/训练 FLOPs | 质量与延迟/内存的 Pareto 对比，而非训练 loss 直觉 |

## 三、第 45 课到底运行什么

课程：**Huginn-Linear 启发 · 共享循环输入重新注入**。在完全可编辑的 Scratch 风格图中，重复两次执行**同一**因果 Attention + SwiGLU Block，并在**每次入口**通过普通积木完成：

```text
E          = gather(Embedding, prefix)                     [3,4]
h0         = E
z0         = concat_cols(h0, E)                            [3,8]
h0_injected= z0 @ Winject                                  [3,4]
h1         = TransformerBlockθ(h0_injected)
z1         = concat_cols(h1, E)                            [3,8]
h1_injected= z1 @ Winject                                  [3,4]
h2         = TransformerBlockθ(h1_injected)
loss       = depth-weighted CE(logits0, logits1)
```

- **唯一共享注入参数**：`Winject[8,4]`。不是独立两套输入投影。初始化 `Winject=[I; 0.15I]`，两轮共享它的值、梯度和 AdamW 状态；保留原本八组循环参数，再增一组，即共九组。
- **严格函数保持对照**：设 `Winject=[I;0]`，逐轮隐状态、各轮 CE 和最终 loss 必须与**没有注入**的两轮图一致；它是一个低成本但能真正区分“拼接与投影实现错误”的数值对照。
- **模型算量**：该微型任务额外投影各有 `3 × 8 × 4 = 96` MAC，共 **192 MAC**，新增加 **32 个参数元素**。这里只计算这些 MatMul 的理论 MAC；不是 GPU 延迟、实际测得能耗或带宽。
- **完整反向**：两轮经同一 `Winject` 的梯度在它的唯一 Parameter 上相加；`E` 既从原先 Embedding 路径接收梯度，也从反复注入的两轮接收梯度。
- **独立导出**：NumPy Python 执行数值前向；PyTorch CPU 使用真实 Autograd、两步 AdamW、保存 checkpoint、加载后第三步续训。

**重要差异：** 此课程仅对齐 `LinearInputInjection` 的无 bias 数学子集。官方 Huginn 有 Prelude/Recurrent/Coda、特定状态初始化及高性能执行栈；本课程在既有共享两轮 Transformer 上注入，不宣称与 Huginn 模型同构，更不是可导入官方 checkpoint 的端到端复现。

## 四、如何逐步检查与复现实验

在 `learn.html` 打开第 45 课。选中 `Winject`、`r0_injection_concat` 或 `r1_injected_state`，悬停查看官方源码指针与“设计目标/替代方案/失败条件”。运行前向，观察真实中间矩阵；选择 `loss` 执行反向梯度、两步优化，然后导出 NumPy 与 PyTorch 文件。

`tests/loop_input_injection_contracts.js` **实际运行 Python 子进程**，验证 JavaScript/NumPy/PyTorch 的每个中间张量、函数保持、修改共享权重后影响两轮、因果不泄漏、中心有限差分、两步优化以及 checkpoint 恢复。

`tests/browser_catalog_screenshots.py` 在真正的 Chromium 中执行保存和下载，并检查论文悬停解释确实可见。

### 下一步应当独立增加的前沿变体

1. **Diagonal-Decay / OrthoInj**：先复刻实码的 per-channel `softplus(dt_bias)` 与 `exp(A_log)`，然后区分轨迹和梯度机制。
2. **真实 ACT/深度先验**：额外训练退出路由、控制预算，并核对前向稀疏执行是否真的少做 GEMM（现有 MoR 课程明确没有）。
3. **真正 KV Cache 策略**：明确当前循环深度、Token 时间轴、缓存来源和恢复条件，再讨论专家/Token 的动态收缩和实际 kernel 速度。

以上新增功能应先通过源码配置核对与**独立跨语言数值合同**，再升级公开课程与“已验证”标识。
