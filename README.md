# SeetheTruth — 用积木理解并执行大模型计算

SeetheTruth 是一个 **Scratch 式的大模型原理学习工作台**。在自由画布中连接张量积木，检查端口、形状和计算依赖，逐步追踪数学运算，查看对应论文与工程惯例；对于支持的计算子图，还可以运行微型数值计算、反向传播和参数更新，并**导出独立运行的 NumPy 前向程序或 PyTorch CPU 可训练程序**。

> 当前是公开测试版候选，不是完整大模型训练平台。无需 GPU，也不强制要求构建特定 Transformer 模板。模型源码分析、真实运行轨迹和 Profiler 是独立的可选研究功能。

**2026-10-20 推理系统时间实验：** [DeepInfer Worker ZOS / 非 ZOS 对照](deepinfer_worker_lab.html) 独立展示跨 Step 的 Scheduler、GPU Forward、上一轮延迟 Post-process、Sample、Flush 的可执行事件依赖图。它以用户输入的假设时长进行可证伪的逻辑调度对照，**不是 DeepInfer 的实测吞吐或框架复刻**。精确来源与限制详见 [时序重建与证据](docs/deepinfer-worker-events.md)。

**DeepInfer Decode 状态重建：** Worker 实验室现已将 C++ `prepare_decode_zos_inputs` 压缩块表、累计 Query Offset，与 `refresh_for_sampling` 的 MTP 槽位转移和 Golden Token 窗口绑定到同一真实计算 Step。全部为**可重复的小规模 CPU 参考算术**，并不执行 GPU 页分配或 DeepInfer 完整 Serving。参见 [源码与验证](docs/deepinfer-decode-blocktable-refresh.md)。\n\n## 立即体验

需要 Python 3.10+（启动网页不需要 PyTorch、Node 或 GPU）：

```bash
git clone https://github.com/Qesire/SeetheTruth.git
cd SeetheTruth
python atlas_server.py --port 8765
```

浏览器打开 **http://127.0.0.1:8765/learn.html**。当前完整学习工作台已进入 main；涉及新多模态积木的功能请以对应 PR 的合并状态和 CI 为准。

1. 选择一门引导课程，或从空白画布自由连线；端口、DAG、形状和参数有效性会进行验证，**不会因为不符合某种模型模板而禁止连接**。
2. 默认使用**原理推演**，只追踪运算意义、形状和依赖，不执行矩阵计算。
3. 主动切换**微型数值模式**后，可以观察中间张量、注意力、局部梯度及适用的训练事务。
4. 选中任意积木，查看来源论文、数学公式、计算步骤、工业惯例、误区和可证伪实验。论文解析也支持悬停预览。
5. 在导出选项中选 **NumPy Python**（独立前向计算）或 **PyTorch CPU**（可训练子集的真实 autograd、AdamW 与 checkpoint）。
6. 点击 **⌘ 导出 .py**，将受支持的完整图或选中上游分支编译为所选后端的独立 Python 文件；不支持的节点会明确拒绝，不生成假代码。

NumPy 导出另外需要 `python -m pip install numpy`；PyTorch CPU 导出需要 `python -m pip install torch`。后者的完整使用和真实执行合同见 [PyTorch 独立可训练代码](docs/pytorch-executable-training.md)。生成脚本包含可调用的 `run_graph(overrides=...)`，能够替换输入张量重新计算所有下游节点；其输出不是写死的演示结果。

**2026-10-20 DeepInfer 开源状态更正：** DeepSeek 已正式开源 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 真实离线推理框架及 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) API/Prompt 转换组件；本项目已有的 TopK/Sparse Attention/Paged KV/Greedy Verify 是独立的 CPU 教学积木，**不是对 DeepInfer 引擎的完整复刻**。详见 [今日源码审计与差异](docs/deepinfer-day0-source-audit.md)。

## 当前学习工作台功能

| 能力 | 已实现范围 | 明确限制 |
| --- | --- | --- |
| 积木库 | **134 种积木**：11 种基本计算积木 + 123 种扩展计算、状态和优化器积木 | 新算子需先注册正确的执行合同；不是每种积木都支持所有后端 |
| 学习内容 | **59 门引导课程**：工业 Qwen3 风格 GQA、第 39/40 课 2/3 次共享参数 Looped Transformer、第 41 课逐轮 CE 加权监督／保留优化器状态的深度切换，以及第 42 课 LoopFormer 启发的短深度学生/长深度 Detach 教师一致性实验，第 43 课 Loop 残差缩放 1 / 1/√N / 1/N 的真实 Loss、梯度范数和计算量对照；第 44 课 MoR 启发的逐 Token 深度预算（硬停止状态、密集计算语义）；第 45 课 Huginn/xLLM-Loop 源码启发的共享 Linear 输入注入与函数保持消融；另有空白自由画布 | 第 44 课仅控制数学状态是否更新，仍执行全部候选 Attention/FFN；不包含 MoR 学习式 Token Router、真实活跃行打包或 KV 稀疏内核，也不能据此声称加速 |
| DeepInfer 推理调度 | 按官方 Reference 数学实现 MTP remap+累积置信度+稳定排名+固定预算；YoCo SWA mask/compact 保留后缀可执行数学示例及 Query 行数对照 | YoCo 尚未实现 DeepInfer 的真实多 Chunk metadata、Yoco Pivots、ZOS Worker/GPU Kernel；不能据此推断线上部署吞吐 |\n| 推理系统基础 | 真实 CPU Top-K、Top-K 稀疏 QK/PV、Paged KV 逻辑到物理地址映射、贪心推测解码验收；四门新的可自由编辑课程可组合 FlashMLA/DeepSelect/DeepSpec 相关机制 | 属于源码机制教学抽象，不提供 DeepSeek 旧 vLLM Fork、FP8/FP4 GPU kernels、实际分布式 EP 通信、目标/草稿模型权重或 GPU 性能 |\n| 原生视频计算 | 真 5D [B,T,H,W,C] 与 [O,Kt,Kh,Kw,I] 因果 Conv3D、左侧时间补零和历史帧流式状态；逐节点 JS/NumPy/PyTorch 比较及可训练卷积核权重梯度 | 不是完整 Wan VAE；空间 valid、时间 stride=1、无完整归一化、上下采样或实际视频文件解码 |\n| 进阶多模态 | 可执行 HWC Conv2D（独立可训练 Parameter 权重）、HTK Mel 三角滤波与两步确定性 ε-DDIM；NumPy/PyTorch CPU 支持，卷积参数的真实 autograd 可单独验证 | 无完整视频 VAE、卷积填充/膨胀、真实 Whisper Slaney 滤波权重、去噪神经网络或生成质量指标 |\n| 多模态 | 图像 Patchify、音频真实 Hann-STFT 功率谱与对数压缩、视频 Tubelet，另有图像/音频/视频三路交叉注意力的可执行积木图；NumPy、PyTorch CPU 代码导出 | 采用受限二维载体、合成数据和明确轴映射；尚无真实图片/音频/视频解码、ViT/ViViT 权重、Whisper Mel 滤波及 Wan VAE 3D 因果缓存 |\n| 原理解释 | 详细数学/论文知识卡；第 38 课支持按积木悬停查看 Qwen3 真实源码、张量形状、公式、差异及最小证伪实验；可执行 LoRA 真实图重写 | 模型源引用不等于 checkpoint 复现；代码与硬件性能结论需独立验证 |
| 计算与训练 | 有界二维/四维微型数值、显式 KV Cache、GQA/MQA、MoE、next-token 对齐/可训练 Embedding/交叉熵、GRPO/PPO、局部和组合 VJP、多步优化器状态 | 不是任意算子的通用 PyTorch Autograd 或 GPU 训练 |
| 代码生成 | 已验证图可导出独立 **NumPy Python**；可选 **PyTorch CPU** 支持真实 autograd、AdamW 持久状态及 checkpoint | 包含已验证的 MoE 稀疏路由、共享专家权重梯度、MAE 训练；仍不是任意积木或大型模型的通用编译器 |
| 真实证据 | 可选 Python AST、PyTorch Runtime Trace / Profiler；真实小模型采集器和本地 JSON 溯源检查器 | 真实样本必须运行指定不可变模型版本，未复跑的数据仅结构验证，不伪造 kernel/吞吐量 |

### 三个关键语义边界

- **逻辑可执行**：允许自由组合；只有端口、形状、参数、拓扑和实际已实现的算子语义会限制运行。形状合法并不代表算法有训练价值。
- **解释、模拟、实测分离**：符号推演（SYMBOLIC）、确定性教学数值（SYNTHETIC）和带不可变模型修订/模块定位的真实采样（REAL_CAPTURE）在界面和导出中不能混同。固定修订的 TinyStories-1M CPU 采集及独立二次复跑已通过；用户本地导入但未独立复跑的 JSON 仍仅属于来源声明。
- **导出不能降级为伪代码**：遇到不支持的算子，应清楚报出积木 ID 和算子类型；不得用预计算常数、固定输出或空实现来假装可运行。

## 其他入口

- [Learning Studio](learn.html)：产品主入口、自由积木及代码导出。
- [Architecture Trade-off Lab](tradeoffs.html)：交互比较 GQA 缓存、MoE 参数激活比例与 Loop Transformer 参数共享的解析成本；不表示真实 GPU 加速。
- Learning Studio 的 **GRAPH COST** 折叠面板直接读取当前实际积木 DAG 的合法张量形状，核算明确覆盖的 MatMul/批量 MatMul/Linear MAC 和物理 Parameter 元素；未覆盖算子不计入总 FLOPs，不冒充 GPU 测速。
  - 同一面板另计融合 Attention/GQA 的 QK/PV GEMM、路由专家 Linear 投影 MAC 与纯函数 Cache 快照元素；可固定当前积木图并与后续修改后的实际图逐项比较差值。不会把这些算术口径当作显存峰值或性能测量。
- [训练状态教学](training.html)：独立的前向—损失—反向—梯度累积—AdamW/SGD 状态示例。
- [KV Cache 实验室](cache.html)：Prefill/Decode、缓存持久化及 Head 级数学可视化。
- [论文 Patch 实验](paper_lab.html)：模块修改和文献计算合同。
- [训练微流程](micro_training.html)：有界自动微分、MoE、PPO 示例。
- [真实小模型证据实验室](real_model_lab.html)：本地导入 pinned-revision 模型采集 JSON，显示参数/梯度/激活统计和证据等级；不把模拟数据冒充实测。也可在 [Learning Studio](learn.html) 的检查器中把观测关联到所选 Linear/MatMul/Parameter，**只读，不替换计算输入或模型参数**。
- [源码与结构图](index.html)：静态源码解释、全息结构与可选 Runtime Trace。
- [性能和张量观测](telemetry.html)：显式实测的 profiler 与可选数值采样。

## 验证与开发

无需训练大型模型，主要合同可以用 Node 22 检查：

```bash
node tests/release_surface_contracts.js
node tests/torch_codegen_contracts.js    # 另外需要 CPU PyTorch
node tests/torch_codegen_extensions_contracts.js  # torch + numpy
node tests/moe_executable_codegen_contracts.js    # torch + numpy
node tests/token_embedding_ce_codegen_contracts.js  # torch + numpy
node tests/causal_transformer_codegen_contracts.js # torch + numpy
node tests/industrial_gqa_tied_codegen_contracts.js # torch + numpy
node tests/loop_transformer_codegen_contracts.js     # CPU NumPy/Torch，原始 2/3 轮最终步 CE
node tests/loop_depth_supervision_contracts.js        # CPU NumPy/Torch，每轮 CE、深度加权梯度、深度调整保持参数和动量
node tests/loop_consistency_tradeoffs_contracts.js     # CPU NumPy/Torch，深度一致性、BPTT、AdamW、真实参数/MAC 数量
node tests/adaptive_token_depth_contracts.js            # CPU NumPy/Torch，逐 Token 停止、共享权重梯度、AdamW/断点续训
node tests/loop_input_injection_contracts.js      # NumPy/Torch 实际运行；共享参数/梯度/续训
node tests/architecture_why_contracts.js               # 论文与工业代码的设计动机、方案对比、失败条件
node tests/industrial_source_contracts.js          # Node
node tests/paper_graph_rewrites_contracts.js
node tests/model_trace_contracts.js
node tests/block_catalog_contracts.js
node tests/graph_readiness_contracts.js
node tests/graph_autodiff_contracts.js
node tests/python_codegen_contracts.js   # 还需 Python 3.10+ 和 NumPy
node tests/python_codegen_extensions_contracts.js
```

浏览器验收使用 `python -m pip install playwright`、`python -m playwright install chromium`，再运行 `python tests/browser_catalog_screenshots.py`。CI 对候选分支进行数值合同、JavaScript↔NumPy 中间量一致性、真实 Chromium 操作/截图、源码审计和静态行为验证。

## 发布状态和文档

截至 **2026-10-19**，新功能仍在 [Draft PR #21](https://github.com/Qesire/SeetheTruth/pull/21)，尚未创建正式 Release。清晰的 Alpha/Beta 发布门槛、尚未通过的条目与证据见 **[发布准备清单](docs/release-readiness.md)**。发布前还需要确定并提交开源许可证；仓库当前尚无 LICENSE。

设计与实现资料：[积木和模型组合](docs/composable-model-lessons.md)、[论文驱动扩展](docs/paper-driven-block-expansion.md)、[真实小模型样本规范](docs/real-model-sampling.md)、[可执行 Python 导出](docs/executable-python-codegen.md)、[统一图反向传播](docs/unified-graph-autodiff.md)、[PyTorch 真实训练导出](docs/pytorch-executable-training.md)、[LoRA 数学重写](docs/paper-driven-block-expansion.md)、[Next-token 真实计算流与梯度](docs/next-token-training-flow.md)、[源码对齐的 Qwen3 风格 GQA / QK Norm / RoPE / 可选权重绑定](docs/industrial-gqa-rope-tied.md)、[设计决策：工业界为什么这么做？](docs/architecture-design-rationale.md)、[Loop Transformer / Ouro / Huginn / MoR 的可执行循环深度及数学边界](docs/loop-transformer-recurrent-depth.md)、[逐轮监督与自适应深度的真实计算边界](docs/looped-multidepth-supervision.md)、[循环残差为什么要缩放：2026 论文与三组真实对照](docs/loop-residual-scaling-design-why.md)、[MoR 逐 Token 递归预算：设计动机、真实计算图与没有节省的 FLOPs](docs/mor-token-depth-computation.md)、[Huginn 原输入注入：真实源码、工业设计动机与可证伪对照](docs/loop-input-injection-design-why.md)、[架构优先级](docs/learning-first-architecture.md)。

项目遵守“可见效果须有测试、数值结论须有可重复实验、模型归因须有实际源码或运行证据”的原则。
