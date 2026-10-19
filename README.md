# SeetheTruth — 用积木理解并执行大模型计算

SeetheTruth 是一个 **Scratch 式的大模型原理学习工作台**。在自由画布中连接张量积木，检查端口、形状和计算依赖，逐步追踪数学运算，查看对应论文与工程惯例；对于支持的计算子图，还可以运行微型数值计算、反向传播和参数更新，并**导出独立运行的 NumPy 前向程序或 PyTorch CPU 可训练程序**。

> 当前是公开测试版候选，不是完整大模型训练平台。无需 GPU，也不强制要求构建特定 Transformer 模板。模型源码分析、真实运行轨迹和 Profiler 是独立的可选研究功能。

## 立即体验

需要 Python 3.10+（启动网页不需要 PyTorch、Node 或 GPU）：

```bash
git clone https://github.com/Qesire/SeetheTruth.git
cd SeetheTruth
python atlas_server.py --port 8765
```

浏览器打开 **http://127.0.0.1:8765/learn.html**。要体验本 README 描述的完整新功能，请切换到正在审查的 [PR #21 候选分支](https://github.com/Qesire/SeetheTruth/pull/21)；在其合并前，默认 `main` 仍是旧版。

1. 选择一门引导课程，或从空白画布自由连线；端口、DAG、形状和参数有效性会进行验证，**不会因为不符合某种模型模板而禁止连接**。
2. 默认使用**原理推演**，只追踪运算意义、形状和依赖，不执行矩阵计算。
3. 主动切换**微型数值模式**后，可以观察中间张量、注意力、局部梯度及适用的训练事务。
4. 选中任意积木，查看来源论文、数学公式、计算步骤、工业惯例、误区和可证伪实验。论文解析也支持悬停预览。
5. 在导出选项中选 **NumPy Python**（独立前向计算）或 **PyTorch CPU**（可训练子集的真实 autograd、AdamW 与 checkpoint）。
6. 点击 **⌘ 导出 .py**，将受支持的完整图或选中上游分支编译为所选后端的独立 Python 文件；不支持的节点会明确拒绝，不生成假代码。

NumPy 导出另外需要 `python -m pip install numpy`；PyTorch CPU 导出需要 `python -m pip install torch`。后者的完整使用和真实执行合同见 [PyTorch 独立可训练代码](docs/pytorch-executable-training.md)。生成脚本包含可调用的 `run_graph(overrides=...)`，能够替换输入张量重新计算所有下游节点；其输出不是写死的演示结果。

## 当前候选功能（以 PR #21 为准）

| 能力 | 已实现范围 | 明确限制 |
| --- | --- | --- |
| 积木库 | **117 种积木**：11 种基本计算积木 + 106 种扩展计算、状态和优化器积木 | 新算子需先注册正确的执行合同；不是每种积木都支持所有后端 |
| 学习内容 | **36 门引导课程**，另有空白自由画布 | 课程是教学组合，不代表完整复刻开源模型 |
| 原理解释 | 详细数学/论文知识卡、源码指针和论文悬停视图；可对选定 MatMul 执行真实 LoRA 低秩图重写 | 图重写必须重新通过数值/梯度合同，不代表 GPU 加速 |
| 计算与训练 | 有界二维/四维微型数值、显式 KV Cache、GQA/MQA、MoE、next-token 对齐/可训练 Embedding/交叉熵、GRPO/PPO、局部和组合 VJP、多步优化器状态 | 不是任意算子的通用 PyTorch Autograd 或 GPU 训练 |
| 代码生成 | 已验证图可导出独立 **NumPy Python**；可选 **PyTorch CPU** 支持真实 autograd、AdamW 持久状态及 checkpoint | 包含已验证的 MoE 稀疏路由、共享专家权重梯度、MAE 训练；仍不是任意积木或大型模型的通用编译器 |
| 真实证据 | 可选 Python AST、PyTorch Runtime Trace / Profiler；真实小模型采集器和本地 JSON 溯源检查器 | 真实样本必须运行指定不可变模型版本，未复跑的数据仅结构验证，不伪造 kernel/吞吐量 |

### 三个关键语义边界

- **逻辑可执行**：允许自由组合；只有端口、形状、参数、拓扑和实际已实现的算子语义会限制运行。形状合法并不代表算法有训练价值。
- **解释、模拟、实测分离**：符号推演（SYMBOLIC）、确定性教学数值（SYNTHETIC）和带不可变模型修订/模块定位的真实采样（REAL_CAPTURE）在界面和导出中不能混同。固定修订的 TinyStories-1M CPU 采集及独立二次复跑已通过；用户本地导入但未独立复跑的 JSON 仍仅属于来源声明。
- **导出不能降级为伪代码**：遇到不支持的算子，应清楚报出积木 ID 和算子类型；不得用预计算常数、固定输出或空实现来假装可运行。

## 其他入口

- [Learning Studio](learn.html)：产品主入口、自由积木及代码导出。
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

设计与实现资料：[积木和模型组合](docs/composable-model-lessons.md)、[论文驱动扩展](docs/paper-driven-block-expansion.md)、[真实小模型样本规范](docs/real-model-sampling.md)、[可执行 Python 导出](docs/executable-python-codegen.md)、[统一图反向传播](docs/unified-graph-autodiff.md)、[PyTorch 真实训练导出](docs/pytorch-executable-training.md)、[LoRA 数学重写](docs/paper-driven-block-expansion.md)、[架构优先级](docs/learning-first-architecture.md)。

项目遵守“可见效果须有测试、数值结论须有可重复实验、模型归因须有实际源码或运行证据”的原则。
