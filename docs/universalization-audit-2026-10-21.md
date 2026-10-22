# SeetheTruth 全站通用化审计与优先级（2026-10-21）

> 本文审计分为两种粒度：**149 个注册积木 / 22 个计算类别**是注册、知识卡及 CPU 解释器目录；**20 个主要功能页面**是用户入口。某页面复用 UI，不意味着其全部子算子都有相同细粒度数值证据。不能用“有积木”推导“能真实执行大型模型”。

## 当前通过核对的基本能力

| 基础层 | 实现与入口 | 可证明什么 | 尚不可证明什么 |
|---|---|---|---|
| 端口/形状/自由 DAG | `learning_lab.js`, `graph_readiness.js`, `stream_graph_blocks.js` | 已注册算子的输入、依赖、静态维度；部分 ND/流状态合同 | 任意输入满足模型语义、真实权重兼容 |
| 小张量实算 | `block_catalog.js`, `learning_lab.js`, `nd_tensor.js` | 小规模确定性 CPU 数值，节点间实际运算 | 不等于 GPU/kernel、真实 checkpoint |
| 计算透视核心 | `compute_microscope.js` + `compute_microscope_ui.js` | 依赖、1–5D 切片、Linear/MatMul 点积闭合、标准 Attention 内部 QK–Softmax–PV | 不能无条件映射 MLA、FlashAttention、各家 fused kernel |
| 源码级结构 | `source_evidence_levels.js`, `atlas_parser.py`, pinned Wan adapter | 静态源码关系、确定来源的模块排列；Wan FFN 小规模参考公式 | 不等于已经加载真实权重或重放运行 |
| 梯度与训练 | `graph_autodiff.js`, `nd_autodiff.js`, `train_state.js`, `engineering_training_pipeline.js` | 支持算子的局部 VJP、具体梯度/优化器更新和确切 Bigram 续训 | 非全模型 Autograd/分布式训练 |
| PyTorch/NumPy 导出 | `torch_codegen.js`, `python_codegen.js`, `torch_stream_lowering.js` | 白名单子图可生成可执行代码；独立 CPU 合同验证 | 注册积木不保证每种任意组合都可导出 |
| 事件与状态透视核心 | `state_microscope.js`, `state_microscope_ui.js` | KV 引用计数、Worker 因果依赖、训练检查点前后状态，来自三种独立执行器 | 不等于真实 GPU 分配、服务时延或实测吞吐 |
| 多模态前处理 | `multimodal_ops.js`, `multimodal_advanced.js`, `video_causal_conv3d.js` | 小尺寸图像、音频、视频形状及数值语义 | 真实大规模视音频模型 checkpoint 与硬件性能 |
| 论文/工业设计动机 | `block_knowledge.js`, `architecture_why.js`, `industrial_source_inspector.js`, `paper_hover.js` | 结构决策的公开知识、来源与反例，鼠标悬浮查询 | 不能将任意组合图自动等同于引用论文 |
| Android 离线教学 | `android/teaching-assets.txt` 与构建测试 | 指定教学页面复用同源 JS/CSS，离线可打包 | 无文件产物导出、无 GPU 训练、还需真实机型验证 |

## 20 个主要功能页面逐项结果

`universal_surface_audit.js` 为机器可检查的单一登记表，`tests/universal_surface_audit_contracts.js` 从**实际 HTML 的脚本依赖**反向检查声明。前端 `capabilities.html` 展示相同登记表。

| 状态 | 页面 | 判断与下一工作 |
|---|---|---|
| **通用张量/算子**（4） | `learn.html`, `wan_ffn_lab.html`, `micro_training.html`, `paper_lab.html` | 共享真正的 `compute_microscope` 数据合同与可视化；Wan 保留 Gate/Channel 反事实；微训练复用真实 Tape 的逐边 VJP、MoE Top-1 专家 GEMM 与 PPO Clip 梯度 |
| **通用状态/事件**（7） | `deepinfer_kv_lifecycle_lab.html`, `deepinfer_worker_lab.html`, `engineering_pipeline.html`, `training.html`, `deepinfer_scheduler_lab.html`, `deepinfer_prefill_lab.html`, `cache.html` | 共享 `state_microscope`；七者分别从 KV Engine、Worker Event DAG、Bigram Checkpoint、标量分析训练、Decode 调度资源/Worker 所有权、Prefill 源码策略、GQA 数学缓存的完全重放生成证据；后两项并不具有真实 GPU/多机语义 |
| **专用实现，尚未完成接口统一**（4） | `semantic_state.html`, `tradeoffs.html`, `capabilities.html`, `engineering_journey.html` | 数值引擎、前沿结构或原理解释可能已成熟，但尚未统一通用透视接口；不能统计为完成 |
| **入口/导航**（2） | `teaching.html`, `android_teaching.html` | 非计算运行页面；应复用导航/课程状态而非误接计算透视 |
| **PC 独有证据工作流**（3） | `real_model_lab.html`, `index.html`, `telemetry.html` | 模型实测与静态源码证据必须保持独立，不从文件格式或采样数据自动提升为真实计算证据 |

**注意口径**：上表是页面级别的通用 UI/trace 接入，不是对 149 个积木逐一声明“源码级实测”。标准 Attention 的 QK–Softmax–PV 细粒度钻取使用执行器的实际内部阶段；其它 Attention 变体若没有对应执行器证据只显示整体张量和上下游，禁止套用标准模型计算流。

## 当前最关键的通用化缺口（排序）

1. **剩余独立训练/论文与多模态复合算子的异构数值与状态合同**：把已有已验证事件和实际输出适配到共用 `state_microscope`，保留独立数学后端，不重复实现权重或推理规则。为训练 VJP/Optimizer 的每一阶段补 `before→grad→update`，不能只靠演示页面。
2. **将已有全部可执行积木按类别逐一审计 Micro Trace 的分辨率**：对于 ND Attention、Conv3D、ViT/Patch、STFT/Mel、DDIM、MoE、Loop、Paged KV，构建各自来源明确的子算子序列，配套独立标量/NumPy/PyTorch oracle 与形状/数值负例。
3. **真正的模型小样本运行证据接入**：重放时显式绑定代码 revision、配置、权重 hash、Token/Time、block index、dtype 和原始张量坐标。仅具汇总统计的现有真实小模型观测不具有整个矩阵操作数，必须明确标示只能“只读参考”。
4. **训练到服务的全链条状态机**：将 Codegen、Checkpoint、模型加载、Tokenizer、缓存、prefill/decode、调度、异常和资源释放串成可逐级进入的课程，支持故障注入和跨层校验。生产 HTTP/gRPC、吞吐和实时 GPU tracing 只作独立扩展。
5. **工业源码与论文追溯**：形成模型适配器注册协议，每个能力给出不可变源码 locator、计算顺序、输入/输出语义、适用变体、数学 oracle 与反事实。对 user-rewired DAG 保持“组合有意义 ≠ 复刻论文”的清晰界限。
6. **移动端回归**：所有教学算子共用数据合同，但 Android 功能不等于 PC 的文件保存/导出或真实模型执行。逐机型视觉与 WebView 性能仍缺实测。

## 质量门禁

每项新能力都应同时具备 (a) 数据类型与证据边界、(b) 所有必要输入来源、(c) 可执行且可复验的计算合同、(d) 支持的数学 oracle、(e) 不支持情况 fail closed、(f) PC 和 Android 页面可操作截图。不可把 README 描述、源码文件存在、UI 动画、成功打包或 shape-only 校验单独当作工业可用证明。

核心命令：
```bash
node tests/compute_microscope_contracts.js
node tests/state_microscope_contracts.js
node tests/universal_surface_audit_contracts.js
python tests/browser_compute_microscope.py
python tests/browser_state_microscope.py
python tests/browser_engineering_pipeline.py
node tests/android_asset_manifest_contracts.js
```

官方发布条件仍应审查整套 CI、APK 打包、全站静态行为测试及关键课程的浏览器验收，不能凭子集通过就合并。
