# 第 41 课：LoopLM/Ouro 启发的共享深度与每轮损失

这是在先前 **39/40 课（两轮/三轮最终 CE）** 之上增加的**新训练目标和状态变更语义**，不是重复复制一个 Loop Transformer 名称。各轮是真实矩阵运算，且可用 NumPy 与 PyTorch CPU 独立执行。

## 论文与真实系统依据

- [Ouro / Scaling Latent Reasoning via Looped Language Models（arXiv:2510.25741）](https://arxiv.org/abs/2510.25741)；[官方项目](https://ouro-llm.github.io/)；[官方实现](https://github.com/Ouro-LLM/Ouro)。其核心是共享权重的层组在深度上递归复用，并研究每轮输出、多深度监督及学习式退出分布。
- [Huginn / Scaling up Test-Time Compute with Latent Reasoning（arXiv:2502.05171）](https://arxiv.org/abs/2502.05171)；[官方 recurrent-pretraining](https://github.com/huskydoge/recurrent-pretraining)。这是一个与 Ouro 训练配方**不同**的递归深度架构族。
- [Closing the Loop: Practical Training Recipes for Looped Language Models（2026，arXiv:2610.00673）](https://arxiv.org/abs/2610.00673)。继续探究模型转换、输入混合与 exit loss；本课**尚未复刻其学习式退出或输入混合机制**。
- [The Surprising Effectiveness of Shared Memory in Looped Transformers（2026，arXiv:2610.02383）](https://arxiv.org/abs/2610.02383)。明确指出共享 Transformer 参数**不意味着循环之间复用同一 KV Cache**。本课在 CPU 微图上做完整因果 prefill，不实现跨循环缓存。

## 真正运行的数学流

令 `z=[z₀,z₁,z₂,z₃]`，`H₀=Embedding(z[:3])`，`y=z[1:4]`，`D=4`，`V=8`。

每一轮的参数全是**同一批 ID**：`embedding, Wq, Wk, Wv, Wo, Wup, Wgate, Wdown`，恰有 **8 组**。

```text
H_r → RMSNorm → shared Q/K/V projection
    → (QKᵀ/√4, Causal Mask, MaskedSoftmax) → PV·Wo
    → Residual → RMSNorm → shared Wup/Wgate
    → SwiGLU → shared Wdown → Residual → H_(r+1)
                                      │
                                   × Eᵀ  (同一个 Embedding Parameter)
                                      │
                            logits_r → CrossEntropy_r
```

固定次数 `R ∈ {1,2,3}`，每轮输出标量损失 `L_r`，按教学默认系数 `α_r=(r+1)/Σ_{j=0}^{R-1}(j+1)`：

[
L_{mathrm{depth}}=sum_{r=0}^{R-1} alpha_r,L_r,qquad
L_r=operatorname{CE}(H_{r+1}E^	op,,y).
]

对共享权重 `θ`，反向传播必须经过所有循环和所有监督分支：

[
rac{dL_{mathrm{depth}}}{d	heta}
=sum_r alpha_rrac{dL_r}{d	heta}.
]

由于后面轮次的 `H_r` 也依赖先前轮次，这里的每个 `dL_r/dθ` **包含通过早期隐藏状态展开的完整链式梯度**；绝不是对最后一个 CE 直接乘一个系数。

### 哪些部分已经可执行

| 机制 | 实现 |
|---|---|
| 1/2/3 次深度循环 | 在标准自由积木 DAG 中展开，完整可见，不允许真实图循环 |
| 共享存储参数 | 8 个唯一 Parameter ID；多个轮次和 tied LM Head 重复使用 |
| 每轮损失 | 每个 loop 都有独立的 Logits、CrossEntropy |
| 合法标量组合 | 任意由 CE/MSE/MAE 经非负 Scale 与 Add 构成的损失表达式可进行实际反向与训练；拒绝伪标量目标 |
| CPU 执行 | JavaScript、NumPy 前向和独立 PyTorch Autograd；Tensor 不使用 JS 结果冒充 |
| 参数梯度 | 所有循环与所有监督头的贡献累加到同一 Parameter ID |
| 深度选择 | 工作台切换 `1 → 2 → 3` 或反向，**不会复制或重置已训练权重/AdamW 动量** |
| 变更保护 | 任意自定义改线/变更核心算子后拒绝静默重建，避免丢失使用者作品 |
| 完整训练 | 真实两步 AdamW、checkpoint 加载后续训、每个中间节点跨后端对照 |

### 需要严格区分的三种机制

1. **固定深度监督**：当前真实实现，每轮都计算并得到可训练的深度加权 CE。改变 R 会改变执行计算量，但不会产生新参数。
2. **学习式退出分布**：需要额外的 Gate Network、`p(exit=r)` 概率归一化、期望计算成本/正则与梯度路径。当前未实现；不应在 UI 中标注“已实现早停”。
3. **Token 级动态深度**：MoR 或类似方法可能只让部分 Token 进入后续轮，必须明确 token-活跃集合、因果依赖和 KV Cache 策略；不能用全序列固定 R 冒充。

**推理计费语义：** 参数组数恒定，但逐轮 Attention/FFN 的乘法次数与激活/中间态开销实际增长。完整微图只在 CPU 小矩阵上运行，不生成高性能 fused kernel，也不能当作真实模型速度评测。

## 用户实验与直接可检验的反例

在 `learn.html` 选择 **任务 41**，启动数值模式、运行前向、选择最终 `loss` 并查看反传，然后进行两步 AdamW。记录 8 个权重与动量的哈希。使用「调整 Loop 深度」将 `2→3→1`，再次保存工程并逐元素检查状态完全一致。深度权重会依据新 R 重新归一化并明确提示。

测试入口：`tests/loop_depth_supervision_contracts.js` 和 Chromium `tests/browser_catalog_screenshots.py`；独立运行生成的 NumPy/PyTorch 后端，逐节点比较数值、完整 BPTT、有限差分、两步 AdamW、checkpoint 继续训练和未来 Token 不泄漏。

## 仍有价值的前沿后续

优先研究 Ouro 官方实现中的 learned halting / exit distribution 与 entropy regularization；随后再做 Huginn 的 state/input injection 对照、以及 2026 年 shared-memory loop Transformer 的跨轮缓存身份。每一步都必须根据源码重建准确的缓存张量、梯度路径和实际执行策略，而不是添加没有数值语义的积木。
