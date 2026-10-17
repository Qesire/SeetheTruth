# Academic Mechanism Atlas — 机制、数学形式与终端效果

SeetheTruth 新增与源代码/运行时观测相互独立的 **LITERATURE / MECHANISM** 知识层。

## 核心语义

一个条目依次包含：模块家族、典型数学公式、局部计算机制、预期终端效果、机制到效果的假设链、适用条件、最小对照测量、常见误区、原始论文来源。

- **数学机制**说明公式对张量执行的变换；通常可从公式严格推导。
- **经验效果**说明论文报告的特定实验结论；不能直接推广到所有模型、设备和数据。
- **工程取舍**说明访存、算力、通信和内存带来的现实成本变化；需要真实 Profile 证实。

## 证据不可混同

| 层级 | 能支持的结论 | 不能支持的结论 |
|---|---|---|
| SOURCE / AST | 某个源码文件存在某个运算和控制分支 | 分支实际执行、实测张量形状、模型精度提高 |
| RUNTIME / TRACE | 在明确输入下某算子/张量事件已被观测 | 数值结果在所有输入上都成立、等同于 CUDA Kernel 统计 |
| LITERATURE / MECHANISM | 通用模块有怎样的数学形式，设计者希望达到什么效果 | 当前仓库一定采用该机制、能取得论文的性能或质量收益 |

对未知的自定义源码模块，知识库必须保持**未匹配**，而非根据名称臆测设计意图。

## 典型模块机制

**Attention**：匹配分数 QKᵀ → 归一化权重 → AV 聚合；目标是使 token 表示利用上下文。参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

**GQA**：减少 KV heads；目标是降低 KV Cache 字节占用和 Decode 访存，实际延迟依赖设备与 kernel。参考：[GQA](https://aclanthology.org/2023.emnlp-main.298/)。

**RoPE**：旋转 Q/K，把相对位置差引入内积，不保证任意长度外推。参考：[RoFormer](https://arxiv.org/abs/2104.09864)。

**SwiGLU**：对同一输入做 gate/up 双投影，SiLU × up 后做 down 投影；门控不等于可跳过的二值稀疏。参考：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。

**RMSNorm / LayerNorm**：缩放归一化、是否中心化不同，实际稳定性与计算收益依赖实现。参考：[RMSNorm](https://arxiv.org/abs/1910.07467) 和 [LayerNorm](https://arxiv.org/abs/1607.06450)。

**MoE**：路由器按 token 选择少量专家、各专家计算后合并；目标是以条件计算扩大可用参数容量，但通信/负载不均可能抵消速度收益。参考：[Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html)。

**AdaLN-Zero**：条件产生 shift、scale、残差门控；让 DiT 的变换依时间步/条件变化。参考：[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)。

**FlashAttention**：保持注意力数学目标，采用 IO-aware tiling 和在线 softmax 减少中间 HBM IO；不保证每个输入和设备上都更快。参考：[FlashAttention](https://arxiv.org/abs/2205.14135)。

**PPO / GRPO**：概率比率、优势和裁剪定义策略优化信号；需要对照奖励、KL、长度、clip fraction 和独立任务指标。参考：[PPO](https://arxiv.org/abs/1707.06347) 和 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。

## 实现与验证

- 纯 JS、无运行时网络请求的目录：../consensus_knowledge.js
- 前端入口：../index.html 的右侧节点 Inspector 的 ACADEMIC CONSENSUS 区域
- 正式来源仍保留为独立的 SOURCE EVIDENCE / RUNTIME EVIDENCE 展示
- 运行测试：node tests/consensus_contracts.js
- Chromium：python tests/browser_acceptance.py

后续可把论文分解为更细的 **claim-level** 证据边，并结合源码等价式、真实张量统计、最小控制实验确定某个机制是否在特定模型上发挥预期效用。

目前版本是人工整理的、可测试的**学术机制知识库**；并非自动完成文献综述或因果识别。
