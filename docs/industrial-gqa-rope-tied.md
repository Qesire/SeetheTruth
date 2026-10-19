# 从 Qwen3 真实源码到 SeetheTruth 第 38 课：可训练 GQA / RoPE / 权重绑定

**实现依据（源码，不仅是论文概念）：**

1. [Hugging Face Transformers 的 Qwen3Attention 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)：
   `q_proj, k_proj, v_proj` 的输出宽度分别按 `num_attention_heads` 与 `num_key_value_heads` 选择；对 Q/K 的每个 head 做 `q_norm/k_norm`，**先归一化、后 RoPE**；在 Attention backend 中复用 K/V Head；输出再经 `o_proj`。
2. [Qwen3 模型配置](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/configuration_qwen3.py)：包含 `num_attention_heads`、`num_key_value_heads`、`head_dim`、`rms_norm_eps`、`tie_word_embeddings` 与滑动注意力配置。**Qwen3Config 的 `tie_word_embeddings` 默认是 false**。因此第 38 课的权重绑定是用于理解共享参数的**可选教学变体**，**不能宣称是所有 Qwen3 检查点的默认实际行为**。
3. [DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)：采用 MLA 和 DeepSeekMoE，KV 压缩缓存与 GQA **不是同一种结构**。切勿用该课程冒充 DeepSeek MLA 实际计算流。

## 第 38 课的全部运算由积木组成

维度取极小但完全可执行的 `T=3, D=4, Hq=2, Hkv=1, head_dim=2, vocab=8`。这只是**工业计算逻辑的缩尺实验**，不是任何 Qwen 模型的原始宽度、头数或权重。

```text
ids[4,1] ─┬─ prefix[3,1] ─ gather(E[8,4]) ─ X[3,4] ─ RMSNorm
          └─ labels[3,1]                                   │
                              ┌──────────────────────────────┤
                              │                              │
                  Q = norm(X) @ Wq[4,4]                  K = norm(X) @ Wk[4,2]
                    ├─ slice(0:2) → Q0norm → RoPE0          │
                    └─ slice(2:4) → Q1norm → RoPE1        Knorm → RoPEk → Kᵀ
                              │                              │
                      scores0=Q0·Kᵀ / √2            scores1=Q1·Kᵀ / √2
                              │                 │
                          Causal Mask[3,3] shared by both softmaxes
                              │                 │
                         P0 = masked-softmax     P1 = masked-softmax
                              │                 │
                         P0·V[3,2]           P1·V[3,2]    V = norm(X) @ Wv[4,2]
                              └── concat(2 heads) ────────────┘
                                      [3,4]
                                      ↓
                                   @ Wo[4,4]
                                      ↓
                             residual with original X
                                      ↓
                                   RMSNorm
                                      ↓
                              @ transpose(E[8,4])   ← exact SAME Parameter
                                      ↓
                                 logits[3,8]
                                      ↓
                            CrossEntropy(labels)
```

### 实际计算流与参数身份

- **QK Norm**：本课程的 `qk_norm` 对每个 `[T,2]` 头按 RMS 做归一化，`γ=1`；Qwen3 原实现有可训练 `weight`，因此当前还不是参数完全等价的复刻。
- **RoPE**：每个相邻通道对做正交二维旋转，偏移为位置 `t`；`rope2d` 的 VJP 精确应用对应正交矩阵的转置（逆旋转）。部署模型的 `rotate_half` 坐标布局及频率配置应通过源码另行验证。这里仅实现相邻偶奇配对布局。
- **GQA**：两个 Q 头各自有 `Q0,K` 和 `Q1,K` 相关性分数；但 K/V 都是同一片参数及同一次计算产物，而不是各自复制出的 KV 参数。这种共享在图结构上可见。
- **权重绑定**：`transpose2d(embed)` 直接作为词表输出投影右操作数，参数始终只有一个 `embed`。两条路径（输入查表与输出词表分类）的梯度在同一 Parameter 上相加；保存/恢复 checkpoint 不会创建第二份“头部参数”。
- **因果性**：`P_h[t,j]=0` 对所有 `j>t`；修改第三个上下文 Token 不得改变前两个位置的 Context 和 Logits。Mask 作为离散可见性，在 VJP 中明确停止梯度。
- **训练实现**：PyTorch CPU `torch.nn.Parameter`、实际 Autograd、状态化 AdamW、独立 checkpoint 及重载续训。NumPy 为**独立前向代码**而非自动微分后端。

### 与真实工业运行系统的距离

| 维度 | 本课程已实现 | 仍未实现 |
|---|---|---|
| Attention 拆解 | Q/K/V、逐头归一化、RoPE、GQA 共享头、Mask、合流、输出投影 | 真实模型的每头可训练 γ、RoPE 频率配置与布局变体、可变头数和泛化 4D 布局 |
| Embedding/Head | 真正共享 Parameter 与两条梯度路径 | 从 Hugging Face checkpoint 导入真实 tensor keys、bias/尺度、混合精度 |
| 推理执行 | CPU 微张量与严格数学身份 | fused SDPA/FlashAttention kernel、Paged KV Cache、连续 batching、prefill/decode kernel/延迟 |
| 模型结构 | 单层教学 GQA，参数化可被用户改动 | 多层 Transformer 堆叠、真实权重初始化、分布式训练/并行 |
| DeepSeek 路径 | 已有 MLA/MoE 的定性课程与模块 | 真实 DeepSeek MLA 参数绑定、KV cache 压缩路径的端到端训练导出 |

## 可证伪验收

`tests/industrial_gqa_tied_codegen_contracts.js` 使用全部 **38 个节点**比较 JavaScript、独立 NumPy、独立 PyTorch 的中间张量，并验证两头共用 KV、RoPE 反向公式、未来 Token 不影响过去、Embedding 与 LM Head 是同一块存储参数、5 组参数梯度与有限差分、两步 AdamW、checkpoint 第三步恢复。

`tests/browser_catalog_screenshots.py` 在真实 Chromium 打开第 38 课，运行→反传→两步优化→工程保存→两个后端的真实 Python 下载。

这也是增加后续前沿模块的流程：**从真实配置与源码提取每个张量合同 → 用可组合图表达实际数学顺序 → 对照两套独立运算 → 证明梯度与状态身份 → 才升级为已支持积木**。
