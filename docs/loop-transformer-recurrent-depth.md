# Loop Transformer / recurrent depth：真实数学流、源码与可执行积木

## 论文与官方实现核对

- [Universal Transformers (2018)](https://arxiv.org/abs/1807.03819)：同一个状态转移在深度上反复应用。原论文还包含每个位置的自适应计算停止（ACT），这**不是**固定循环的必然组成部分。
- [Huginn: recurrent pretraining（官方代码）](https://github.com/seal-rg/recurrent-pretraining)：重用深度方向的计算模块，通过增加测试时计算量提高潜空间推理能力，同时需要认真处理隐藏状态初始化、输入注入和循环次数。
- [Ouro / Scaling Latent Reasoning via Looped Language Models（官方项目）](https://ouro-llm.github.io/) 与 [官方代码](https://github.com/Ouro-LLM/Ouro)：重复执行**共享参数的 Transformer 层组**，并研究学习式退出分布、熵正则和不同的计算深度。它不是“对输出文本做多轮 CoT”的同义词。
- [Mixture-of-Recursions（官方代码）](https://github.com/raymin0223/mixture_of_recursions)：逐 Token 深度选择和路由、可选 KV 复用；它的离散路由、每层有效 Token 集合和 KV Cache 语义不能由统一固定循环次数代替。

## 课程 39 / 40：可执行的计算流

新增 `loop_transformer_two` 和 `loop_transformer_three`，分别在可自由编辑的有向无环图中**显式展开 2 / 3 次深度循环**。这是解决“用户想看见每个中间乘法”与“运行时执行的是循环”之间的张力：图没有逻辑上无限反馈的边，而是展开为一系列有界、完整可执行的节点。

```text
ids[4,1] ─> prefix[3,1] ─> gather(embed[8,4]) ─> sinusoidal PE ─> H₀[3,4]
         └> labels[3,1]                                       │
                                                              ▼
┌────────────────── 同一个参数化 Block Fθ ───────────────────┐
│ preNorm(Hᵣ)                                                │
│ Q=preNorm Wq     K=preNorm Wk       V=preNorm Wv            │
│ RoPE(Q), RoPE(K), Kᵀ                                      │
│ score=Q·Kᵀ/√4   ── causal visibility[3,3]                  │
│ P=MaskedSoftmax(score); context=P·V                          │
│ A=Hᵣ+context·Wo                                           │
│ F=SiLU(RMSNorm(A)·Wgate) ⊙ (RMSNorm(A)·Wup)                │
│ Hᵣ₊₁=A+F·Wdown                                           │
└─────────────────────────────────────────────────────────────┘
                    │
  H₀ ── Fθ ── H₁ ── Fθ ── H₂  [── Fθ ── H₃]
                           │          │
               RMSNorm(Hfinal)·embedᵀ ─> logits[3,8]
                                            │
                                     CrossEntropy(labels)
```

**有且只有八组 Parameter：** `embed`、`wq`、`wk`、`wv`、`wo`、`wgate`、`wup`、`wdown`。其 ID、值、AdamW 一阶/二阶状态在 2 次与 3 次循环中都保持单份。LM Head 用 `transpose2d(embed)` 绑定同一矩阵。增加循环深度**增加算子执行次数与激活，但不增加 Block 参数组数**。

### 完整反向传播 BPTT

对于固定循环次数 `R`，即便计算图经过 `R` 次相同权重的调用，参数微分必须对所有使用路径汇总：

```math
H_{r+1}=F_\theta(H_r),\quad
\frac{\partial L}{\partial \theta}=
\sum_{r=0}^{R-1}
\left(\frac{\partial L}{\partial H_{r+1}}\right)
\frac{\partial F_\theta(H_r)}{\partial \theta}
\quad\text{(链式传播中的每轮局部贡献)}.
```

`graph_autodiff.js` 的多消费者梯度累加使各轮最终到达**同一参数 ID**。PyTorch 代码生成也只分配一个 `torch.nn.Parameter` 并在不同轮的 `forward()` 中多次读取。非训练型的布尔 causal mask 与整型 token 索引不接受梯度；RoPE 和 SwiGLU 的连续数值路径有严格的 VJP。

**注意：** 简写中的 `∂L/∂Hᵣ` 是包含所有后续轮依赖的总导数，不能把各轮看成彼此独立的浅层 loss。

### 何种意义上实现“真实 Loop Transformer”

| 机制 | 当前已实现 | 没有实现，不能混称 |
|---|---|---|
| 共享参数循环深度 | 真实相同 Parameter 对象被 2/3 次调用 | Ouro 的全部层组、初始化/深度调度及规模 |
| 全序列 causal Attention | 逐轮 QK/RoPE/Mask/Softmax/Value 和 FFN | 快速推理的 per-loop KV Cache、SDPA kernel、paged attention |
| 权重共享训练 | 前向中间值、完整 BPTT、两步 AdamW、保存/恢复 | Truncated BPTT、random-k 训练、gradient checkpointing |
| 深度调整 | 同一模型可比较 2/3 次循环且参数组数一致 | 学习式 Token halting、硬 early-exit 和实际节省 kernel 运行 |
| MoR / Ouro 的其他方向 | 论文级机制说明与计算图对照 | Token 层级稀疏深度路由、entropy-regularized exit、KV 共享、真实权重导入 |

**不能推断**：循环可以免费增加推理质量；固定两三步教学结果不能推出 Ouro 的性能，也不能据未公开结构对当前闭源模型做断言。

## 严格测试

`tests/loop_transformer_codegen_contracts.js` 直接启动真实 Python subprocess：

1. 分别导出 56/75 节点的完整计算图，对每一个节点比较 JS、NumPy、PyTorch 的真实中间 Tensor。
2. 验证 2/3 轮共享参数组集合完全一致、两轮隐藏状态确实不同。
3. 修改未来 token，检查**每一次循环**的过去位置概率和隐藏状态不变；下三角掩码的概率精确归零。
4. JS VJP 与真实 PyTorch Autograd 对全部 8 组参数比较；用中心差分检验 Q/K/FFN 与 Embedding 梯度。
5. 两步 AdamW 对照 JS，再导出 checkpoint、恢复并要求第三步得到相同参数。
6. 手动构造跨迭代真实反馈依赖，必须被 DAG 环路校验拒绝。

`tests/browser_catalog_screenshots.py` 使用真实 Chromium 验收课程加载、逐轮反向、两步训练、论文源码悬停、保存工程以及 NumPy/PyTorch 双代码下载。增加图上限是为了真实展开循环深度：`MAX_NODES=80`、`MAX_EDGES=176`，没有关闭 DAG、参数值或尺寸检查。

## 接下来合理的前沿扩展

先从 Ouro/Huginn 的实际训练流分别增加**输入注入 / 每轮 loop embedding / 多深度监督**，再实现学习式退出的概率守恒和熵正则。与此平行，MoR 的 Token 路由需要额外证明每轮有效 Token 和 KV 状态如何改变，不能只把 router 连接在 loop 外围。

实现前先冻结精确的 upstream commit 和配置，逐模块梳理输入输出、头宽、状态缓存和反向链，再开放“模型等价”选项；之前版本仍只能称作 source-inspired bounded demonstration。
