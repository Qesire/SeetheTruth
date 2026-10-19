# SeetheTruth: Architecture Decisions — 工业界为什么这样设计？

**原则**：模型某个模块的 *是什么*（真实源码、形状、数学运算）与 *为什么*（作者动机、经验规律、性能假设）是两类不同的陈述。后一类必须说明替代方案、代价、失效条件、实验判据。

本文件支撑工作台中的 `loop_design_tradeoffs.js`、`architecture_why.js` 两个可交互、可审计的知识层。所有定量示例必须由当前计算图推导，或在外部原始研究中注明场景；禁止把教学 CPU 小矩阵推断为工业级速度。

## 1 · Loop Transformer：以更多串行计算换更少独立权重

**为什么不把所有层都设置为不同参数？** 因为“网络有效深度”和“模型持有的独立权重”本应是两个维度。令一个真实 Block 为 `F_θ`：

```text
H(1) = F_θ(H(0))
H(2) = F_θ(H(1))
H(3) = F_θ(H(2))
```

这有三项不同的资源后果：

- **物理权重**：一个 `θ`，不因循环数量增加；部署模型参数存储及 AdamW 对这一权重的状态仅需一份。
- **实际执行**：每轮仍要执行 `F_θ` 的全部乘法、归一化、非线性与 Attention；深度增加会增加串行推理成本。**参数更小并不意味着 FLOPs 更小。**
- **训练反传**：完整 BPTT 必须沿时间展开路径求导，累加每轮共享参数梯度；为了回传通常还需要保存或重算不同轮的激活。**不能把减少模型权重说成减少峰值激活内存。**

本项目提供可直接验证的“同深度不共享权重”的结构反事实。按当前的完整教学图：独立可训练矩阵共有 8 张；Embedding 32 元素，7 个共享核心矩阵 136 元素，合计 **168 参数元素**。固定深度 1/2/3 轮都是 168；若只把这 7 个矩阵展开成互相独立的权重，则同深度对照分别需要 **168 / 304 / 440 元素**。这些数来自程序读取真实 `Parameter` shape，**不是某个大型 LLM 的参数估计**。MatMul MAC 总数也从实际已连线图逐个推算，随着递归次数增加。

**业界研究观点**：<Link url="https://arxiv.org/abs/2510.25741">Ouro / Scaling Latent Reasoning</Link> 希望增加潜空间推理的多步计算，而不是每次增加知识存储容量；<Link url="https://arxiv.org/abs/2610.00673">Closing the Loop</Link> 通过控制数据与计算预算分析 Looped LM 相比参数匹配 Dense 模型的增益。可以将其视为*待验证的归纳偏置*，不是所有文本任务下的胜出保证。

**什么时候不适合**：纯事实记忆、严格低延迟以及训练算力不足时，额外串行深度可能难以换来收益。

## 2 · Ouro：为什么监督多个循环深度？

若只有最深处 `CE_R` 接收监督，模型不一定学会在中途预算停止时产生有效概率分布。第 41 门课使用：

```text
L_depth = Σ_r α_r CE_r,  α_r≥0, Σ_r α_r=1
```

这个损失是实际图中的 `Scale + Add`，且每轮输出头真实连接到同一 `Embedding` Parameter 的转置。**严格边界**：固定的多步监督不等于 Ouro 的 learned exit；也不等于自动节约计算，因为全部轮次仍被执行。随机深度训练、末轮监督、等权多轮监督是有意义的最小对照，须同时测每轮 CE、最终 CE 与每步梯度方向。

## 3 · LoopFormer：为什么用深度一致性与停止梯度？

论文 [LoopFormer (ICLR 2026)](https://arxiv.org/abs/2602.11451) 希望模型在不同推理预算下表现稳定。其真实 <Link url="https://github.com/armenjeddi/loopformer/blob/master/train.py">train.py</Link> 中，长轨迹与短轨迹分别经模型推理，并同时优化语言模型损失和 `MSE(long.detach(),short)`。其精确实现还涉及 **time/step-size conditioning** 与**独立采样的短计算轨迹**，对这些要点不能只看多次循环就声称已经复现。

第 42 门课是刻意缩小的*机制消融*：

```text
H_short = first shared-weight block state
H_deep = second shared-weight block state
L = L_depth + λ·MSE(H_short, stop_gradient(H_deep))
```

- **目标**：以较深表征为固定教师，让浅层表征也接近它。
- **为什么 Detach**：不希望一致性项同时把教师拖向学生，导致训练目标共同漂移。**重要实现细节**：当前教学 MSE 的 target 端本身就定义为固定监督目标，图中显式的 Detach 为说明信息流边界而存在；它不会神奇产生原本不存在的额外梯度阻断。
- **代价**：一个额外 MSE 前向与梯度；若深层状态本身较差、与浅层表征职责不同，一致性目标可能干扰预测。
- **实验**：同一初始参数、数据和循环数，分别 `λ=0` / `λ>0`；记录深浅 CE、MSE、验证集 loss、梯度差异。只降低 MSE 本身不能证明模型质量提升。
- **缺失**：没有真正的 LoopFormer Short/Long 两次带不同 `Δt` 的计算轨迹、时间调制层、shortcut-consistency 的全尺度实验。**不可命名为 LoopFormer 官方复现。**

## 4 · 为什么让 Token 在不同深度退出？

固定执行 3 轮虽然算法简单，却浪费“简单 Token”的潜在计算预算。<Link url="https://arxiv.org/abs/2507.10524">Mixture-of-Recursions</Link> 的研究方向是 Token 级深度路由；Ouro 也研究学习式深度分配。工程实现必须额外处理退出概率/负载、Batch 重整、梯度估计、退出后残余权重与每轮 KV Cache。这些并非第 39–42 课已经实现的功能。

**替代方案**：固定序列级预算；按请求难度选择统一循环数；真正逐 Token 退出。

**何时失败**：算术 FLOPs 减少但 GPU 因小 batch、cache 访问与调度不规则，端到端 latency 并未变好。正确实验同时记录逻辑 FLOPs 和真实硬件时间，不混为一谈。

## 5 · Qwen3/GQA：缓存带宽与表达自由度

GQA 中 (H_q>H_{kv})，在相同 `head_dim` 下 KV Cache 每 Token 存储量与 (2H_{kv}d) 成比例，因此 Query 可保持较多关注子空间而减少缓存吞吐；代价是多个 Query 必须共用某些 Key/Value 表示。可以比较 MHA、GQA、MQA，清晰区分 Prefill（算力密集）与 Decode（可能带宽受限）。

Qwen3 的逐头 Q/K Norm 与可训练 `gamma` 是模型实际源码顺序的一部分；它不是“Norm 一定提升任何模型”的理论定律。第 38 课可直接观察共享 Q Norm 增益的两条梯度路径。来源：[Qwen3Attention 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)、[GQA 论文](https://arxiv.org/abs/2305.13245)。

## 6 · MoE / MLA / FlashAttention 的三个**不同**效率目标

| 模块 | 主要想降低什么 | 增加什么成本 | 为什么不能混淆 |
|---|---|---|---|
| 稀疏 MoE | 每 Token 活跃 FFN 乘法占比，同时保留大量总容量 | Router、All-to-All、分组 GEMM、小专家负载均衡 | 稀疏激活不等于 HBM/网络成本减少 |
| MLA | 长上下文 KV 缓存宽度与访问字节 | Latent 投影、RoPE 解耦、吸收路径复杂度 | 不是多个 Query 简单复用 KV Head 的 GQA |
| FlashAttention | Attention 的 HBM IO，避免显式写出完整 (T×T) 权重 | 在线 Softmax 和 GPU Tiling / Kernel 设计 | 数学上精确 Attention，并非换成低秩/线性 Attention |

来源：[DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)、[FlashAttention](https://arxiv.org/abs/2205.14135)。

以上结构的关键验证均应是**计算流重建**：形状、信息源、实际依赖、参数身份、梯度、状态更新，之后再比较真实质量和硬件预算。只有“某个论文声称用了该技术”不足以证明本项目已经复现该模型。

## 7 · 发布判据

- **代码已实现**：所有画布显示的新增图必须真的可计算、可 NumPy/PyTorch 导出，不能预写结果常量。
- **机制已验证**：独立 Python 前向与 JS 每个中间张量一致；反向与有限差分 / PyTorch 一致；两步 AdamW + checkpoint 恢复一致。
- **设计动机**：为每一类机制标注具体来源、目标、替代设计、代价、失效条件及可证伪实验；不把研究者的条件性结果表述为行业绝对共识。
- **尚未实现**：学习式提前退出、Token 级 MoR 真实路由、完整 LoopFormer 时间调制及短轨迹训练、递归深度 KV Cache 服务、实际 GPU/FP8 kernel。以上不得在 UI 中标成已支持的完整实现。
