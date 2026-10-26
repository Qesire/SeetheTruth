# SeetheTruth — 用积木理解并执行大模型计算

**构建 → 执行 → 解释 → 验证。** SeetheTruth 是 Scratch 式 AI 模型学习工作台：自由连接张量积木，在同一计算图中检查端口、形状和依赖，追踪逐坐标运算、梯度及参数更新，并在支持的图上导出可执行 NumPy/PyTorch 程序。

原理推演模式**无需 GPU**。微型数值模式执行确定性的有界 CPU 张量运算。论文与源码证据、教学合成输入、真实模型采集及 GPU 实测采用不同证据层级。

## 立即体验

仅浏览教学页面需要 Python 3.10+，无需 PyTorch、Node 或 GPU：

~~~bash
git clone https://github.com/Qesire/SeetheTruth.git
cd SeetheTruth
python atlas_server.py --port 8765
~~~

打开 **http://127.0.0.1:8765/learn.html**，从左栏选择「从 FFN 开始」。

1. 阅读 `Linear → 激活 → Linear` 的图连接与张量形状。
2. 在学习模式中切换「微型数值」，点击「运行」，检查各节点输入输出。
3. 选中积木阅读公式、论文来源、工程设计动机和最小反例。
4. 切换「自由实验」，亲手构图，观察形状诊断与实际运行。
5. 在受支持的图上尝试 **NumPy Python** 或 **PyTorch CPU** 导出；**不支持的节点会明确拒绝**，不会输出假代码。

更详细的练习步骤：[第一次使用：15 分钟计算实验](docs/first-15-minutes.md)。

## 推荐探索路线

| 路线 | 入口 | 你将看到 |
| --- | --- | --- |
| 从基础开始 | [Learning Studio](learn.html) | FFN、Attention、残差、GQA、MoE 和自由 DAG 的逐节点计算 |
| 从工程问题开始 | [跨平台教学中心](teaching.html) | 多模态、推理调度、训练流水线、论文与工业实现解释 |
| 从真实源码结构开始 | [Wan2.2 I2V 单 Block](learn.html) | 选择 `nd_wan_i2v_full_block`，展开 Self/Cross/FFN、AdaLN e0–e5 与梯度链 |

Android 教学版使用同一教学课程数据，通过 [Android teaching APK](.github/workflows/android-teaching-apk.yml) 构建离线 WebView **debug APK 工件**；它提供学习交互，不提供代码导出、模型权重部署或 GPU 推理。

## 当前能力与可信边界

- **自由可组合图**：端口、DAG、命名张量轴、形状与状态约束、逐节点计算显微镜、观测回放。
- **数学与设计解释**：可展开的计算步骤、公式、论文链接、设计取舍和可复现的机制实验。
- **训练/推理**：限定图中的 Forward/VJP、优化器状态、KV Cache、Loop、MoE、GRPO/PPO 参考实验。
- **图像/音频/视频**：可执行基础积木、视频 5D 语义、源码顺序对齐的 Wan I2V 单 Block（115 节点合成 Float64 图）。
- **程序生成**：支持的图可生成可独立运行的 NumPy 前向或 PyTorch CPU 训练程序；算子登记、导出候选和**独立数值通过**是不同结论。查看源码实时生成的 [能力矩阵](capabilities.html)，而不是静态承诺任意模型可导出。
- **真实模型证据**：固定修订的 TinyStories-1M CPU 采集及独立二次复跑已通过；用户本地导入但未独立复跑的 JSON 仍仅属于来源声明。Wan 单 Block 的独立 NumPy/PyTorch 证据覆盖 FFN 闭包及其梯度边界，而非整套官方 checkpoint 或完整视频生成。

详情：[双层数学验证](docs/dual-tier-mathematics-verification.md)、[Wan Block 计算与证据](docs/wan-i2v-complete-block-verification.md)、[可信扩展包](docs/trusted-block-pack-installation-v1.md)。

## 专项入口

- [计算成本与设计取舍](tradeoffs.html)：已建模计算量和参数量，不是 GPU 实测。
- [训练状态](training.html) 与 [微训练](micro_training.html)：反向传播、梯度和参数更新。
- [KV Cache](cache.html)：Prefill / Decode 与缓存读取。
- [论文实验](paper_lab.html)：论文驱动的模块解释、图改写。
- [真实模型证据](real_model_lab.html)、[源码图谱](index.html)、[可选性能观测](telemetry.html)。
- [DeepInfer Worker](deepinfer_worker_lab.html)、[KV 生命周期](deepinfer_kv_lifecycle_lab.html)、[Scheduler](deepinfer_scheduler_lab.html)、[Prefill](deepinfer_prefill_lab.html)：源码启发的有界 CPU / 状态参考实验。

## 验证与参与

快速合同验证推荐 Node.js 22；独立数值验证需要 Python、NumPy、CPU PyTorch；真实浏览器验收由 GitHub Actions 执行：

~~~bash
node tests/release_surface_contracts.js
node tests/source_math_invariants.js
node tests/architecture_extension_spine_contracts.js
node scripts/sync_runtime_assets.js --check
~~~

查看 [Actions](https://github.com/Qesire/SeetheTruth/actions) 的确切提交验证结果，以及 [贡献指南](CONTRIBUTING.md)、[开发变更记录](CHANGELOG.md)、[发布验收清单](docs/release-readiness.md)。

**发布状态：** 当前是公开 Alpha 候选，尚未发布正式 GitHub Release，根目录也尚无 `LICENSE`。公开源码可查看，不等于自动获得第三方复制与再分发许可；由版权持有者确认适合的许可证后再正式发布。