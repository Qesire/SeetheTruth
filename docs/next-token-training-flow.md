# 第 36 课：从 Token 序列构造真正可训练的 Next-token 计算图

这是一条**可自由拆分、重新连线、能数值执行和导出代码**的教育计算图，而不是静态示意图。依赖：`learning_lab.js`、`block_catalog.js`、`graph_autodiff.js`、`torch_codegen.js`。

## 1. 为什么需要显式时间对齐

给定完整的离散 Token 序列 `ids = [z0, z1, z2, z3]`：

- `input_tokens = slice_rows(ids, start=0, length=3) = [z0, z1, z2]`
- `labels = slice_rows(ids, start=1, length=3) = [z1, z2, z3]`

这样监督位置 `t` 的真实标签是 `z_(t+1)`。两个分支来自**同一个**源，不能用两个独立随机的 Token 序列替代。

## 2. 完整计算流与张量形状

```
token_ids [4,1] ─┬─ slice_rows(start=0,len=3) [3,1] ─┐
                 │                                      gather_rows ── H[3,4] ── MatMul ── logits[3,8] ── CrossEntropy ── L[1,1]
                 │ embedding_table θ_E [8,4] ─────────┘               ↑                              ↑
                 └─ slice_rows(start=1,len=3) [3,1] ──────────────────│──────────────────────────────┘
                                                         lm_head θ_O [4,8]
```

辅助 `log_softmax(logits)` 支路供检查完整的词表分布，并不自动等同分类目标。

用符号写为：

\[
H_t = E[z_t,:],\qquad Z_t=H_t W_O,\qquad
L=-\frac1{T-1}\sum_{t=0}^{T-2}\log\operatorname{softmax}(Z_t)_{z_{t+1}}.
\]

这里 (V=8,D=4,T=4)，是**固定大小、有界的 CPU 教学任务**。

- **前向**：Token ID 是离散下标，Gather 必须读取整个参数表中对应的行；CrossEntropy 是稳定的 LogSumExp，不先 ArgMax。
- **反向**：`∂L/∂Z_t=(softmax(Z_t)-onehot(z_(t+1)))/(T-1)`；`∂L/∂H_t=(∂L/∂Z_t)W_Oᵀ`。
- **共享行梯度**：`∂L/∂E[v,:]=Σ_{t:z_t=v} ∂L/∂H_t`。相同 ID 出现多次，梯度累加而不是覆盖。
- **不对标签求导**：`labels` 的整数索引仅指定监督类别；切片是数学上可求导的恒等选取，但在这里目标边被明确 `stop_gradient`。
- **优化**：每步重新计算 Forward 和 VJP，对 `embedding_table`、`lm_head` 使用有状态的 AdamW。PyTorch 导出真实 Autograd 和 checkpoint，不是假定更新成功。

## 3. 如何实验

在 `learn.html` 中打开 **36 · Next-token → 可训练 Embedding → CE**，切换到数值模式，依次运行、选中 `loss`、查看反向、执行两步 AdamW，再分别导出 NumPy 和 PyTorch CPU。

你还可以自行加入 `pos_encoding`、`LayerNorm`、`SwiGLU` 或 Attention 并重新连线。只要支持的形状、语义和后端运算完整，正常计算图即可执行；尚未实现的反向或代码生成，会明确拒绝并指出节点。

## 4. 这与真正的现代 LLM 相差什么

当前例子是**一次只读取当前 Token 的小型 bigram 分类器**，不是已经完成因果 Transformer。它没有用到历史 `z0...z_t` 的 Attention 上下文、真实 tokenizer、批次/填充 mask、模型训练语料或大规模预训练权重。若要表示条件分布 `p(z_(t+1) | z_0...z_t)`，还需添加严格的因果 Attention、位置/缓存语义以及对应的模型参数和计算图。

`token_ids` 来源仍是固定种子的**合成整数序列**；这不等于 TinyStories 真模型实测激活，也不是一个公开可用的 tokenizer。

## 5. 可执行证据

`tests/token_embedding_ce_codegen_contracts.js` 会运行真实 `python3` 进程，对比 JS、NumPy、PyTorch 的所有中间张量，检查一格时间偏移、重复索引的 Scatter-add 梯度、有限差分、两步 AdamW、checkpoint 续训，以及修改源 Token 后下游真实重算。`tests/browser_catalog_screenshots.py` 则在真实 Chromium 页面完成训练与两个后端的下载。

**参考资料**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)；[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)；[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
