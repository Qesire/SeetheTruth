# 37 · 可训练因果 Transformer：从自由积木到独立 CPU 代码

这里用**组合现有的底层算子**，而不是新增一个不可拆开的 `Transformer` 黑盒。学习工作台画布中的每个参数、矩阵收缩、下三角 mask、概率归一化和损失都是可单独编辑的计算节点。课程名：`causal_transformer_lm`。

## 计算图与形状

固定合成 Token 序列 `z=[z0,z1,z2,z3]`，`T=3, D=4, V=8`：

| 节点 | 操作 | 输出 |
|---|---|---|
| `prefix` / `labels` | 同源时间切片 `z[:3]` / `z[1:4]` | `[3,1]` / `[3,1]` |
| `lookup` / `pos` | Trainable Embedding gather `E[z_t]` + fixed absolute sinusoidal PE | `[3,4]` |
| `q,k,v` | 分别与 3 个独立可训练矩阵 `Wq,Wk,Wv` 相乘 | 各 `[3,4]` |
| `kt` / `scores` | `Kᵀ`、`QKᵀ` | `[4,3]`、`[3,3]` |
| `scaled` / `mask` | `1/sqrt(4)` 缩放；独立产生下三角可见性 | `[3,3]` |
| `probs` | 只在可见 `j≤i` 位置上 Softmax | `[3,3]` |
| `context` | `P·V`，上下文从过去和当前 Token 汇聚 | `[3,4]` |
| `residual` / `norm` | `X+Context`、RMSNorm | `[3,4]` |
| `logits` / `loss` | `H·W_head`、与 `z[t+1]` 交叉熵 | `[3,8]`、`[1,1]` |

数学上，这个训练目标是：

```text
X_t  = E[z_t] + PE_t
Q_t  = X_t Wq; K_t = X_t Wk; V_t = X_t Wv
S_ij = (Q_i·K_j)/sqrt(D)
M_ij = 1 if j <= i else 0
P_ij = exp(S_ij)/sum_{k<=i} exp(S_ik) for j<=i, else 0
H_t  = RMSNorm(X_t + sum_{j<=t} P_tj V_j)
L    = mean_t CrossEntropy(H_t W_head, z_{t+1})
```

**五组可训练参数**：`embedding_table`、`wq`、`wk`、`wv`、`lm_head`。所有表只在本课程中存储为固定小矩阵，可用工作台参数编辑功能替换；不宣称使用真实预训练模型。

## 如何保证不泄露未来信息

Mask 是 **0/1 可见性条件**，不是一个可导的概率权重。对每个查询行 `i`，Softmax 只归一化 `j≤i` 的项；因此：

```text
P[0,1]=P[0,2]=P[1,2]=0
```

更强的可证伪检验是只修改输入 `z2`：位置 `0,1` 的 `context` 与 `logits` 必须保持不变，而位置 `2` 的结果应变化。请注意，标签位置 `1` 同时会因为 Token 右移而改变，但这只是监督目标改变，不是因果 Attention 泄漏。

## 反向传播与实现边界

- `dV=Pᵀ dContext`；`dP=dContext·Vᵀ`。
- Softmax VJP：对可见 `j`，`dS_ij=P_ij[dP_ij−Σ_k P_ik dP_ik]`；不可见位置的梯度**严格为 0**。
- 对 `QKᵀ/sqrt(D)` 用普通矩阵 VJP，再沿共享输入 `X` 累加 Q/K/V 和残差分支回传的梯度。
- Mask 生成边以及离散 Token ID / 监督标签边刻意停止梯度；对 Embedding 同 ID 的梯度使用 Scatter-add 聚合。
- 单头、单序列、固定大小、无 dropout、无 KV Cache、无 RoPE、无推理采样和多层堆叠；这是可执行的教学级**完整因果自注意力训练链**，不是前沿开源 LLM 的同构复刻。

NumPy 导出全图前向；PyTorch CPU 导出实际 Autograd、AdamW、参数保存及 checkpoint 恢复。这些程序使用的是算子计算而不是硬编码的输出。

## 可重复验收

`tests/causal_transformer_codegen_contracts.js` 分别启动 NumPy 和 PyTorch 进程，对照浏览器 JavaScript 每个中间张量，验证三角 Mask、未来 token 不可见、所有五组参数梯度、有限差分、两步 AdamW，以及 checkpoint 恢复后第三步一致性。

`tests/browser_catalog_screenshots.py` 在真实 Chromium 中打开第 37 课，点击前向、反向、两步优化、保存工程，并导出两种 `.py` 代码。

后续升级应首先考虑可训练 `RoPE + GQA`、参数绑定的 `Embedding↔LM Head` 和共享层堆叠。在这些升级前，不应把单头实现直接标注为某个指定开源模型的实际计算图。
