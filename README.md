# SeetheTruth

SeetheTruth 是一个面向大模型学习与研究的交互式全息计算图浏览器（MODEL ATLAS HOLOGRAPH）。

## 核心目标
- **结构→模块→算子→矩阵→张量**五级 3D 计算图与相机钻取。
- 细化 Attention / GQA / RoPE / KV Cache / FFN / AdaLN 的逻辑数据流、轴重排、广播、mask 与矩阵收缩。
- 解释每个节点的设计目的、数学函数、张量契约、训练和推理差异。
- 用 Python AST 对源码做**无执行**静态解析；明确标注源码事实、约定推断和教学模板。
- 训练、推理与 RL 的工作流是概念模型；不应冒充某个模型的真实执行轨迹。

## 使用
1. 直接打开 [index.html](./index.html) 体验教学模板（浏览器需要同时加载仓库内的 `tensor_subgraph.js`）。
2. 进入 **架构** 视图，依次进入 **Block → 算子 → 矩阵 → 张量**。
3. 选择 Attention / FFN / AdaLN，单击节点查看右侧数学解释，双击下钻。
4. 本地源码解析（解析器及服务端文件见仓库）：
   `python atlas_server.py --port 8765`，打开 `http://127.0.0.1:8765`。

## 新增：运行时实测与反向传播（Runtime Trace）

不再局限于配置/AST 推断。实验性 RuntimeTrace 在**用户主动运行的可信 PyTorch 脚本中**记录 ATen、module、shape、stride、dtype、梯度与显式 optimizer.step；导出图后可在前端钻取。

    python examples/runtime_trace_demo.py
    python atlas_server.py --port 8765

将产生的 examples/runtime_trace_demo.json 导入浏览器。需要安装 PyTorch；静态 AST 解析仍不依赖 torch。

使用方法、数据隐私与精度边界：[Runtime Trace 文档](docs/runtime-trace.md)。

**重要：** HTTP 服务器从不执行被分析的源码；RuntimeTrace 只有用户显式导入并在自己的脚本中调用才会运行。图内 ATen 是观察到的算子，不是 GPU kernel 发射或完整 Autograd DAG。

## 证据边界
3D 张量视图展示**逻辑张量运算**：实际执行可能使用融合 kernel、FlashAttention、tensor parallel 或无物化的 GQA 广播。符号形状不是运行时实测，不表示真实 kernel 数量或显存布局。要确认真实调用路径、dtype、stride 和实际分支，需要结合源码实现及 runtime trace。

## 项目结构
- `index.html`: 全息 3D 浏览器
- `tensor_subgraph.js`: Tensor micrograph 的教学分解与契约
- `atlas_parser.py`: Python AST 静态计算图解析器
- `atlas_server.py`: 本地受限源码分析 HTTP 服务
- `runtime_trace.py`: 显式启用的 PyTorch 动态观测器
- `docs/runtime-trace.md`: 采集、证据等级与限制
- `examples/`: 静态图测试源代码
- `tests/`: 一致性与安全测试

当前为研究原型，并非模型执行引擎。

## 运行效率与代表性张量可视化

新增 [Performance & Tensor Observatory](telemetry.html)：真实 PyTorch profiler 的 ATen self time、模块 Host 时间与 tensor shape/stride/dtype；显式开启数值采样可展示热力图（带符号/幅值/零值）、直方图、矩阵切片。

```bash
python examples/telemetry_demo.py
python atlas_server.py --port 8765
```

打开 http://127.0.0.1:8765/telemetry.html 并导入生成的 JSON。**默认不采集数值；性能测量与数值采样建议分成两次运行。** [技术与隐私说明](docs/telemetry.md)。

## 学术与业界机制共识（新增）

选择模块节点后，右侧新增 **ACADEMIC CONSENSUS** 知识层：典型数学公式 → 局部机制 → 预期终端效果 → 适用前提与测量建议，并提供原始论文链接。涵盖 Attention、GQA、RoPE、FFN、归一化、MoE、AdaLN、FlashAttention、语言模型损失、AdamW、PPO、GRPO。**与 AST 的源码事实和 Runtime Trace 的观测事实分离；不保证目标模型获得文献里的效果。**

知识条目见 [`consensus_knowledge.js`](./consensus_knowledge.js)，说明见 [机制共识文档](./docs/mechanism-consensus.md)，验证命令 `node tests/consensus_contracts.js`。

## 真实源码的矩阵与张量证据视图（实验性）

用 `python atlas_server.py --port 8765` 分析本地 Python 源码目录，再切换整体 / Block / 算子 / 矩阵 / 张量：源码导入后可以在当前调用层级和其嵌套模块中筛选有源码依据的运算（不再仅依赖教学模板）。已加载的显式绝对 `from module import Class as Alias` 可以解析同名类冲突及真实跨文件 `forward`，未知或未加载依赖会保留未解析证据。矩阵标签只识别明确的线性/GEMM 调用，不把自定义类名称臆断为 GEMM；所有源码图的张量 shape 仍标记为**未经运行验证**。完整范围、外部库限制、投影容量和验证方式见 [Source evidence drill](docs/source-evidence-drill.md)。


## 一次真实 Pass 的计算成本与源码证据

已支持在**同一个用户显式执行的 PyTorch Pass**中联合采集 RuntimeTrace 和
Profiler：调用 `same_pass_evidence.VerifiedPass`，为每次模块调用记录独占 ATen
CPU 自耗时、包含下级的成本、真实张量 shape/stride/dtype 与调用序号。
依赖父子调用关系自底向上汇总，避免把子模块的 ATen 成本再次作为父模块独占成本。
主全息图的运行模块 inspector 显示同次成本，Telemetry 页面也可展开逐次调用。
源码版本一致时仍可用此前的 SHA-256/行号验证为节点绑定源码证据。

快速 CPU 验收：`python examples/same_pass_demo.py`。
说明与计时口径：[Performance & Tensor Observatory](docs/telemetry.md)。
所有 Profiler 数字都属于**有插桩的单次测量**，不是未插桩模型吞吐率；
含子模块成本不能跨父子相加，CUDA profiler 归因时间不等于设备执行关键路径。

## Scratch 式大模型学习实验台（新增）

本项目新增 [Learning Studio](learn.html)：在一张可以拖拽、连线、修改参数的工作台上，自行拼出微型 FFN、残差、单头 Attention 与 Loss。选择积木可观察数学公式和设计意图；按“运行微型计算”后逐步查看 **真实由浏览器数值计算得到的小矩阵、shape、Attention 分数与 Softmax 权重**。可尝试断线挑战、修复输入端口、保存或导入本地课程工程 JSON。

```bash
python atlas_server.py --port 8765
# 浏览器打开 http://127.0.0.1:8765/learn.html
```

学习实验台的权重是教学用固定伪权重；输入由确定性种子生成，矩阵最大 8×16。**它不执行任何真实 LLM 权重，不包含训练反向传播、真实 GPU 用时或 RL rollout。** 原有 [全息源码/运行分析器](index.html) 和性能显微镜保持不变。升级方向、支持的积木、限制和逐阶段研究型课程设计见 [Learning Studio Roadmap](docs/learning-studio.md)。

验证：`node tests/learning_lab_contracts.js` 和真实浏览器 CI。


## 多维张量与 GQA / RoPE / Decode（教学实验）

新文件 tensor_ir.js 保存微型张量的命名轴、真实计算数值和可选择的 Head 切片。Learning Studio 新增 **任务 05 · GQA / RoPE / Causal Attention** 和 **任务 06 · Decode 读取已有 KV 前缀**。积木支持切换 Query Head 数、KV Head 数、因果 Mask 和 RoPE，在检查器内逐次展开 Q/K/V Heads、旋转、scores、Mask、Softmax probabilities 与输出拼接。

验证命令：node tests/semantic_tensor_ir_contracts.js。启动 python atlas_server.py --port 8765，在浏览器访问 http://127.0.0.1:8765/learn.html 。

这些是**真实计算的有界微型教学数值**，而不是运行真实 LLM 或 GPU。Decode 示例只读取提供的 K/V 前缀，尚未实现可持续更新的 KV Cache。数学合同及测试见 docs/learning-studio.md。

## Stateful KV Cache 学习与真实截图验收

新增 [KV Cache 状态机实验室](cache.html)。Prefill 将投影 K/V 实际写入独立 Session 的缓存数组；每一步 Decode 仅追加当前 token 的 K/V，之后使用完整已缓存前缀计算 GQA 注意力。页面展示缓存槽位、Key/Value 数值矩阵、逐 Head 注意力概率及数学 GEMM FLOPs。

测试 node tests/kv_cache_contracts.js 将逐 token 输出与完整 causal 前缀重算逐元素比较。通过 python tests/browser_cache_screenshots.py 使用真实 Chromium 截图并验证 DOM 中的缓存槽位、矩阵单元格和布局。四张 PNG 及 manifest.json 上传到 GitHub Actions 的 seethetruth-kv-real-chromium-screenshots artifact（14 天保留）。

启动：python atlas_server.py --port 8765，访问 http://127.0.0.1:8765/cache.html。

限定：这是确定性教学数值模拟，没有实际大模型权重、采样、真实 GPU latency，也没有复杂批次或 Paged Attention。实际缓存数组更新已实现，但不代表 PyTorch/HF/KV cache 的具体物理数据结构。
