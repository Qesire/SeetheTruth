# SeetheTruth · 积木深度知识体系（45 种）

## 宗旨：解释一个积木，而不只是给它贴上一个数学名称

每个积木在原理学习工作台的右侧 Inspector 中均拥有可展开的知识卡。为了帮助学习者在脑海里复现真实计算流，一个完整解析应回答：输入是什么、每一步发生什么、沿哪个轴操作、得到什么数学效果、为什么架构常用它、替代方案有哪些、真实实现最容易出现何种偏差、如何用最小对照检验。

### 知识卡的层级

1. **数学定义**：沿用积木已经固定的公式、输入输出端口和具体教学参数。为避免伪造，每项公式仅在自身的实现/合同边界内成立。
2. **直觉与信息流**：从 token、特征、Head、概率、梯度或状态的语义出发，先解释设计意图，再具体解释信息的变换路径。
3. **计算流**：沿顺序拆成不可再省略的数学变换；例如 GQA 的 Head 分组、QKᵀ、缩放、Mask、Softmax、加权 V，不与真正的 GPU kernel 划等号。
4. **业内常见做法**：记录真实普遍使用的融合、IO、Norm 位置、训练/推理行为、不同选型之间的取舍，并明确适用条件，不说成必然规律。
5. **常见误判**：专门区分近名概念，如 Sigmoid/Softmax、乘法/矩阵乘、RMSNorm/LayerNorm、SwiGLU/GEGLU、GQA/Token 稀疏、数学 Attention/FlashAttention。
6. **验证与来源**：继承课程已有的反例题与实验提示；重要机制接入源码图谱原有 Consensus 知识库的目标、作用链、假设和测量建议；附可直接打开的权威论文、PyTorch 或 Hugging Face 文档。

当前共有 45 张唯一知识卡、22 组权威参考源、若干重点机制扩展，以及针对关键模块的横向比较。卡片内容为离线静态资料，不依赖网络调用、外部 API 或 GPU。

### 三种证据等级不可混淆

| 等级 | 可以合理展示 | 不能声称 |
| --- | --- | --- |
| 教学数学合同 | 符号计算、轴与 shape、定义中的前提条件 | 已在指定模型中得到相同结果 |
| 公开学术共识 / 工程做法 | 有文献/官方文档依据的一般结构规律、常见替代实现 | 特定训练设置一定质量更好或一定更快 |
| 特定源码/真实运行证据 | 若从已有图谱进入，应核对源文件、配置、张量及实际数据 | 不根据名称、论文类别直接推测源码执行 |

## 三个应在课堂重点掌握的案例

**Attention 与 FlashAttention**。QKᵀ/√D、Softmax、PV 是数学描述；FlashAttention 的关键在减少显存层级间数据移动、利用分块和在线 Softmax，通常保持对应数学目标。它不是一种允许任意近似概率的不同 Attention。现有课程仍区分无 Mask 的单头 Attention 与可配置 GQA。

**GQA 与 KV Cache**。减少 KV Head 会影响缓存字节、读写与共享关系；但实际吞吐依赖 kernel、batch、序列长度、训练得到的权重及硬件。使用 Head sharing 并不等于已经实现 KV cache 的持久写入；可再进入独立的 Stateful Cache 课堂观察真正的 append 行为。

**Loss、Logits 和训练**。CrossEntropy 通常直接消费未归一化 logits；语言模型常要求标签错位、padding mask/ignore index 与正确 reduction。只有前向 Loss 标量仍未发生 backward 和 optimizer update。PyTorch CE 可以接受软标签，但目前的微型教学块只实现整数 ID 目标。

## 开发者扩展要求

新增积木必须提供至少：数学合同、信息流、适用场景、错误情况、可证伪检验、知识来源和证据等级。概念-only 是完全支持的：无需 GPU、真实矩阵或数值 kernel。

数据接口：block_knowledge.js 是以现有 45 个 `L.defs` 种类为键的离线纯文本知识库，`validate()` 拒绝漏项或悬空引用；consensus_knowledge.js 的深入机制作为独立附注，不取代任何真实源码证据。

CI 验收：node tests/block_knowledge_contracts.js 覆盖 45 唯一卡片与来源、相近算子的关键误判；python tests/browser_block_knowledge.py 检查 GQA、SwiGLU、Masked Softmax、LayerNorm 与 CE 的可见知识内容、来源链接与实际 Chromium 截图。测试在默认的原理推演模式中运行，完全不需要 GPU。

参考（完整逐项引用见 UI 的积木知识卡）：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)、[Grouped Query Attention](https://aclanthology.org/2023.emnlp-main.298/)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[Transformers KV Cache](https://huggingface.co/docs/transformers/main/kv_cache)。
