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
