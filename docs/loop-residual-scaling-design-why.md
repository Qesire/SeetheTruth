# 循环 Transformer 的残差缩放：为什么不是继续堆深度？

**第 43 课**把业界的一个具体设计问题变成真实可运行的计算图：对同一组 Transformer 权重，在固定循环次数 `N=2`、相同 Token、相同每轮监督、相同优化器超参数下，仅切换**每次残差加法前的系数**。

参考研究：[On the Residual Scaling of Looped Transformers: Stability and Transferability (2026)](https://arxiv.org/abs/2606.18524)，[DeepLoop: Depth Scaling for Looped Transformers (2026)](https://arxiv.org/abs/2607.13491)，[Relaxed Recursive Transformers (ICLR 2025)](https://deepmind.google/research/publications/122290/)，[Universal Transformers (ICLR 2019)](https://research.google/pubs/universal-transformers/)。

## 设计问题是什么？

对于普通、**不共享权重**的网络，层 0、1、2 的权重分别获得自己的梯度更新。对于循环网络：

```text
H^(r+1) = H^r + ε F_θ(H^r)
∇θ L = Σr [(∂L/∂H^(r+1)) · (∂F_θ(H^r)/∂θ)] + 跨更早循环的链式路径
```

同一组 `θ` 被第 0、1、……轮重复读取，梯度来自不同访问的累加。不同轮的残差更新可能具有**相关方向**：它们并不是独立噪声，因此按独立层方差估算的 `1/√N` 缩放，未必能充分抑制共享访问所造成的累计变化。这是研究动机，**不是**所有数据与架构都应该使用 `1/N` 的定律。

2026 年 *On the Residual Scaling…* 对共享循环的 residual scaling 进行分析，提出对循环次数采用 `1/N` 因子；对于多物理层结构，还提出分离循环数与独立层数的 `λ/(N√L)` 参数化。另一个研究 *DeepLoop* 针对不同的 **Post-LN DeepNorm** 结构调整 `α,β`，并非把我们的 Pre-Norm 数值 `ε` 直接换成这对系数。

## 实际加入的自由积木

所有循环轮次共用唯一的 `Wq/Wk/Wv/Wo/Wup/Wgate/Wdown` 以及 `embedding`，不会因为展开两个循环而复制权重或优化器状态。

每轮经过：

```text
S_r = H_r + ε · Attention(RMSNorm(H_r); shared Q/K/V/Output)
H_(r+1) = S_r + ε · SwiGLU(RMSNorm(S_r); shared Up/Gate/Down)
```

其中 attention/FFN 的真实 `MatMul → Softmax / SiLU → MatMul` 没有被替换为伪输出；新增的是四个普通的 `scale` 积木：

- `r0_attn_delta`、`r0_ffn_delta`
- `r1_attn_delta`、`r1_ffn_delta`

对应的 `Scale` 后再与残差主路径 `Add`。**重要**：这仅是 Pre-Norm 双残差、单物理 Block、两次循环的机制消融，不是上述论文完整架构或实际预训练模型复刻。

## 为什么有三种规则？

| 规则 | 本课 ε | 研究假设 | 潜在失败模式 |
|---|---:|---|---|
| `unit` | 1 | 保留原始 Block 更新尺度 | 相关残差反复累积，隐藏状态/梯度扩大 |
| `inverse_sqrt` | `1/√2` | 借鉴更一般的独立残差尺度控制 | 共享权重导致相关方向，可能依然偏大 |
| `inverse_loops` | `1/2` | 针对循环次数抑制累计更新 | 过度限制有效的单步更新，出现表示停滞 |

这些并非已被该微实验验证的性能优劣。**可观测事实**只有当前固定的小矩阵前向、梯度及参数状态。泛化质量、收敛速度与硬件性能必须在实际模型、数据和训练配置上测量。

## 你可以在工作台中做什么？

打开 **43 · Loop Residual Scaling**，选中 `r0_attn_delta` 或 `r1_ffn_delta`，右侧 WHY 面板会展示：

1. 与设计对应的原始论文及互相竞争的设计观点；
2. 不同方案的明确数学假设、系统代价、失败条件；
3. **从真实小张量执行**得到的 `Loss`、最终隐藏状态相对输入变化的 RMS、`∥∇Wq∥₂`；
4. 严格控制的参数元素数、`MatMul MAC` 是否相同，防止把性能差异误认为更小的网络。

顶部的 `ε = 1`、`ε = 1 / √N` 和 `ε = 1 / N` 选项修改**当前图中四个实际 Scale 积木**，不会创建新模型、覆盖共享参数或重置 AdamW 动量；普通 Undo/Redo 仍然可用。任何自行修改连线使其不再是所宣称的消融结构，系统都应拒绝标注为“匹配的残差对照”。

## 实际执行的测试及真实性边界

`tests/loop_residual_scaling_contracts.js` 的任务不是判断哪种方案赢，而是证明比较**真的公平且可复现**：

- 三种 ε **相同物理参数身份与值**、相同输入、相同目标、相同 MatMul MAC；
- 所有中间张量逐节点与独立 Python 3.10+ NumPy、CPU PyTorch 执行结果比较；
- JS 解析 VJP、PyTorch Autograd 与中心有限差分的梯度一致；
- AdamW 两步更新与 checkpoint 恢复后继续训练一致；
- 非法循环数、未知规则、Scale 断线、同轮系数不一致必须明确失败；
- 真正 Chromium 测试切换 ε，实际操作反向与训练，下载两种后端的代码。

不模拟 GPU 提速、不引入没有源码证明的“当前所有顶级模型都在采用这种规则”之类结论。

**继续向工业级推进的条件**：构建同一架构的长跨度 `N` 曲线（重建大于 80 节点图时应先扩容并验证内存），对比同 FLOPs、同 Token、同学习率的真实训练曲线，再考虑论文 DeepLoop 的 Post-LN `α/β` 公式、LoopFormer 可变时间/步长条件、Ouro 动态退出与 MoR 的 Token 级深度路由。这些是**不同的算法机制**，需要独立计算合同与数值证据。
