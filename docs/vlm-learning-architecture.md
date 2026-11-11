# VLM 计算课程：视觉信息如何进入语言读出

SeetheTruth 的 VLM 课程让学习者从输入数据开始，沿着实际执行的计算图追踪视觉 Token、文本 Query、注意力权重和最终读出。四个示例使用确定性生成的小型 Float64 张量；它们用于解释与检验**计算机制**，真实图片语义理解与模型性能属于后续模型验证。

四条课程构成递进关系：**如何融合信息 → 如何减少视觉 Token → 如何判断哪些 Token 值得保留**。每一步都由已注册算子执行，复用现有 `LearningLab`、`SeetheTruthUnifiedCore`、知识卡与 PC/Android 课程，而非另建 VLM 引擎。

## 1. 输入对象与形状

图像张量按通道在后的行优先顺序存为 \([H,WC]\)，`vision_patchify` 将 \(P_h\times P_w\) 像素展平成一个 Patch：

\[
R=\operatorname{Patchify}(I)\in\mathbb R^{N_v\times P_hP_wC},
\qquad N_v=\frac{H}{P_h}\frac{W}{P_w}.
\]

`vision_projector` 是原生 `linear`，产生 \(V\in\mathbb R^{N_v\times d}\)。文本分支由 `token_ids` 与 `embedding` 产生 \(T\in\mathbb R^{N_t\times d}\)。线性权重由教学运行时确定性生成，文本 ID 也不是自然语言分词器输出；特征维度相等允许张量连接，却不证明跨模态语义对齐。

默认输入取 \(H=W=4,\ C=1,\ P_h=P_w=2\)，所以四个 \(2\times2\) Patch 对应四个视觉 Token，另有两个文本 Token，隐藏宽度 \(d=4\)。所有计数与下文的分数矩阵均由这些配置推导。

## 2. 两种融合方式，两个实际计算

### 视觉前缀：共同序列中的因果注意力

`extension_vlm_early_fusion` 的节点顺序为：

```text
图像 → Patchify → vision_projector ──┐
                                      ├─ vlm_token_concat → Q/K/V → Masked Softmax → 读出文本行 → Logits
Token IDs → text_embedding ──────────┘
```

拼接结果 \(X=[V;T]\in\mathbb R^{6\times4}\)。该课程**确实包含三个独立的线性节点**：

\[
Q=\operatorname{Linear}_{query}(X),\quad
K=\operatorname{Linear}_{key}(X),\quad
U=\operatorname{Linear}_{value}(X).
\]

随后由 `matmul`、`scale`、`causal_mask`、`masked_softmax`、`weighted_sum` 实现

\[
s_{ij}=\frac{Q_iK_j^\top}{\sqrt d},\quad
p_{ij}=\frac{\mathbf1[j\le i]\exp(s_{ij})}
{\sum_{\ell\le i}\exp(s_{i\ell})},\quad
O_i=\sum_{j\le i}p_{ij}U_j.
\]

`vlm_text_tail(prefixTokens=4)` 取出 \(O\) 的最后两行，再由 `language_head` 得到 \([2,8]\) 的 Logits。因果分数矩阵为 \(6\times6\)，允许读取的位置有 21 个。此 Mask 只规定教学融合序列的一层注意力可见性；真实视觉编码器中的图像 Self-Attention、位置编码与多层解码结构尚未由本课程复刻。

### 独立 Cross-Attention：文本查询视觉记忆

`extension_vlm_cross_attention` 使用另一组边：`text_embedding` 直接接入 `attention.q`，`visual_projector` 同时接入 `attention.k` 和 `attention.v`，之后把结果与文本 Embedding 相加。

**这里没有额外的 Q/K/V 线性节点。** 因此精确的执行公式是：

\[
Q=T,\qquad K=V,\qquad U=V,\qquad
A=\operatorname{softmax}\left(\frac{TV^\top}{\sqrt d}\right),\qquad
T'=T+AV.
\]

注意力矩阵大小为 \(2\times4\)，最后由 `language_head` 得到 \([2,8]\) Logits。与视觉前缀不同，文本 Query 读取独立的视觉 Key/Value；但这只是基本 Cross-Attention 与残差，不含 Flamingo 的门控、BLIP-2 Q-Former 的查询训练或特定模型的预训练权重。

## 3. 压缩视觉前缀：减少了什么

`extension_vlm_compressed_prefix` 在视觉投影之后执行相邻两行均值：

\[
V'_j=\frac{V_{2j}+V_{2j+1}}{2},\quad j=0,1.
\]

视觉 Token 从 4 减为 2，文本 Token 仍为 2；融合序列长度从 6 变成 4。分数矩阵从 36 项缩为 16 项，因果允许位置从 21 变成 10。均值会合并不同的输入细节，所以信息保持与长度降低是两个不同的问题。上述数字是**数学运算规模**，不是推理延迟或视频质量的实测提升。

## 4. 文本条件 Patch 选择：建立误差与选择条件的关系

`extension_vlm_query_patch_selection` 复用原生 `matmul`、`inference_topk`、`attention`、`inference_sparse_attention`、`subtract` 和 `mse`，并由 `vlm_patch_grid_lookup` 将选中 Patch 编号转回空间网格：

```text
文本 T ── QKᵀ/√d ─→ Top-K 索引 ──────→ 稀疏 Attention ─┐
   │                     └→ 原图 Patch 坐标              ├→ 差分 / MSE
   └─────────── 读取全部视觉 Key/Value → Dense Attention ┘
```

每个文本 Query \(t\) 对视觉 Patch \(j\) 的分数为 \(s_{tj}=T_t^\top V_j/\sqrt d\)，Softmax 权重为 \(p_{tj}\)。令 \(S_t\) 为 Top-K 集合，\(m_t=\sum_{j\notin S_t}p_{tj}\) 为丢弃的注意力质量。完整与稀疏读出分别是

\[
o_t=\sum_jp_{tj}V_j,\qquad
\hat o_t=\frac{\sum_{j\in S_t}p_{tj}V_j}{1-m_t}.
\]

对保留与排除的 Value 分别求加权均值 \(\mu_{S_t}\)、\(\mu_{\overline S_t}\)，将完整读出分为两组便有

\[
\boxed{o_t-\hat o_t=m_t(\mu_{\overline S_t}-\mu_{S_t})}.
\]

若 \(\|V_j\|_2\le R\)，可直接推出 \(\|o_t-\hat o_t\|_2\le 2m_tR\)。因而决定误差的不仅是丢失多少注意力质量，还包括被保留与排除的 Value 是否具有不同方向或幅值。

Top-K **不保证**随 \(k\) 增大而单调减少读出误差。例如 \(p=(0.5,0.3,0.2)\)，Value \(=(0,100,-100)\) 时，全量输出为 \(10\)；Top-1 输出为 \(0\)，误差 \(10\)；Top-2 输出为 \(37.5\)，误差增加到 \(27.5\)。此反例和全量 Top-K 恢复 Dense 的等价性均有可运行测试。

`vlm_patch_grid_lookup` 将行优先编号 \(j\) 还原为 \(y=\lfloor j/W_p\rfloor,\ x=j\bmod W_p\)，观测记录保留 Query 行、选择排名、Patch ID 和坐标。它回答“选择了哪一块”，无法单独回答“选中的物体是否正确”。

`readout_error` 的 \([1,1]\) MSE 是 \([2,4]\) 读出向量的均方差，**不是语言交叉熵、Grounding 准确率或视频画质**。目前的 Top-K 实验还先计算全部 QK 分数、再运行稀疏路径，并额外运行 Dense 路径作为对照，属于机制验证而非实测加速。

## 5. 观察、验证和研究延伸

将课程按 **G0 任务 → G1 架构 → G2 模块 → G3 张量 → G4 坐标** 组织，是对学习者观察粒度的安排，复用已有图与观察引擎。当前教学图的可执行和数值追踪主要覆盖 G2–G4 的部分对象；G0 的真实图像问答正确性要在接入权重和任务数据后才能测量。

| 需要判断什么 | 在程序中观察 | 目前证据 |
| --- | --- | --- |
| 视觉/文本来源是否对应 | `patches`、`text_embedding`、`prefix` | 形状、拼接顺序和元素来源 |
| 未来 Token 是否泄漏 | `mask`、`probabilities` | Masked Softmax 的严格零值与反例 |
| Cross-Attention 是否按图执行 | `cross_attention`、`text_residual` | 独立标量 QK/PV 与文本残差 |
| 视觉压缩改变了什么 | `visual_pool`、`text_context` | 有损均值、长度和数值输出 |
| Patch 选择和误差关系 | `top_patches`、`patch_coordinates`、`readout_delta`、`readout_error` | Dense/稀疏读出、误差分解、Top-K 反例 |

`tests/vlm_learning_contracts.js` 运行四条图并核验数值、来源、Mask、错误连接和算子边界；`tests/vlm_cross_backend.js` 把真实 JavaScript 中间张量交给**独立 NumPy 与 PyTorch CPU Float64** 参考程序，对 TopK=2、TopK=4 的分数、索引、读出和 MSE 进行比对。独立固定案例对照并不意味着四个 VLM 扩展积木已经支持任意图的 NumPy/PyTorch 代码导出；不支持的 VLM 图必须明确报错。

下一步有明确依赖：先固定一份轻量真实 VLM 的预处理、视觉编码器、投影与位置 Mask 源码和权重，再以图像—问题对为独立实验单元评测 Token 选择的 Grounding/QA 质量。相同图像里的多 Token 是相关观测，不能重复计作独立样本。之后才值得测设备延迟、视频时序和与 LLM 后训练、视觉 Agent 状态循环的实际组合。Agent 的工具执行和状态变化沿用既有状态/事件接口，不混入普通张量算子。

相关来源用于辨认架构差异：[LLaVA](https://arxiv.org/abs/2304.08485)、[BLIP-2](https://arxiv.org/abs/2301.12597)、[Flamingo](https://arxiv.org/abs/2204.14198) 及 [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)。数值等价与功能覆盖则只以当前仓库的实际节点和验收结果为依据。
