# SeetheTruth Guided Engineering Journey — 从训练到推理

入口：PC / Android 共同使用 `engineering_journey.html`；学习者主动通过 12 个连贯检查点，不能仅凭滑动或选择“下一页”跳关。

## 教学合同

先讲清 **输入、计算、可观察证据、工程理由、常见失败、下一步**。每关需要选择机制判断，再运行实际数字计算；答错会提示原因，但不能解锁；错误注入不影响模型权重与进度。课程不保存进度或自动采集个人数据；Android 不能导出文件、代码、权重和日志。

## 每关交付物和验收

| 阶段 | 操作 | 可观察、可否证的结果 |
|---|---|---|
| 0 环境/复现 | 固定模型、随机种子、LR、Batch、Decay | 同种子初始化权重逐元素一致 |
| 1 数据准备 | 审核 train/validation、划分与授权问题 | 样本无重叠，token pairs 数量可重复计算 |
| 2 Tokenizer | 编码 BOS/EOS 并右移 next-token 标签 | `[BOS, 我, 爱, 数学, EOS]` 对齐 |
| 3 架构 | 追踪 W 的行索引和形状 | W: [8,8]，64 参数 |
| 4 初始化/优化器 | 检查 W/m/v 与 AdamW step | 初始化动量为零，梯度有非零范数 |
| 5 Forward | 计算 Logits 和稳定 Softmax | 概率和约为 1 |
| 6 Backward | 对照解析 CE 梯度与有限差分 | 绝对误差 < 1e-6 |
| 7 Training | 120 次有界 AdamW 优化 | 全训练集 CE 相对初始化下降 > 0.05 |
| 8 Evaluation | 分别计算 held-out CE、PPL 和 accuracy | 不更新权重的评估过程；识别泛化差距 |
| 9 Checkpoint | 保存 W/m/v/step、扰动并恢复 | 优化器状态和 Forward 逐元素恢复 |
| 10 Inference | 同一权重逐 token greedy/top-K/temperature | 可检查每步上下文、采样概率、EOS |
| 11 Serving/QA | 确定性、未知 Token、输出长度限额 | 有界的数学合同；不宣称真实 HTTP Serving |

## 数学实现边界

微模型定义：词表 V=8，参数 W∈R^(V×V)，输入上一 Token x∈[0,V)，Logits z=W[x,:]。取 p=softmax(z)，逐目标交叉熵 L=−log p[y]。Batch 解析导数：

```text
∂L/∂W[x,j] += (p[j] - 1[j=y]) / batch_size
```

使用 β1=0.9、β2=0.999、ε=1e−8 的 AdamW，包含解耦权重衰减。所有计算为浏览器/Node 的 Float64 真实算术，而不是服务器执行大型模型、预录结果或 CUDA 基准。数据集是内置合成中文语句，验证句子与训练句子互异，但这不能证明训练语料无现实世界版权/泄漏风险。

Bigram 只依赖最后一个 token，不存在 Attention、RoPE、Prefill、KV Cache、复杂 Batching、分布式训练、生成式安全对齐或大型模型质量；相关内容应跳转到现有积木、训练、KV 与 DeepInfer 课程。PC 工程导出请使用已有 `learn.html` PyTorch 编译器白名单流程，此微型课程自身不产出 Python 文件。特别地，Node 契约、PC Chromium、手机尺寸 Chromium、Android 真机分别是四个证据级别，**浏览器测试不等于 APK 真机验收**。

## 工程设计规则

1. **从操作走向可复现证据**：给出 seed、数据 split、版本，显示实际 Loss/gradient/update 而不是用固定文字冒充结果。
2. **错误要形成可纠正的因果链**：区分 shape、语义、数值稳定、梯度、优化器、保存恢复、Sampling、Serving 类型。
3. **学习者自主预测与试错**：每阶段的判断题和失败案例提供可反证的练习；必要时返回前一关检查。
4. **积木与工程流程双向导航**：课程负责项目生命周期，积木面板负责单个算子的输入、输出、数学、论文和可导出能力。
5. **分层而不夸大**：纯符号、合成数字模型、真实小模型捕获、GPU profiler 是不同证据等级。

## 验证

```sh
node tests/engineering_journey_contracts.js
python tests/browser_engineering_journey.py
node tests/android_asset_manifest_contracts.js
```

其中浏览器测试要求 Playwright Chromium；离线 APK 使用 `android/teaching-assets.txt` 作为精确资产白名单。

## 已接入的进阶工程训练状态机

[工程训练进阶课程](../engineering_pipeline.html) 共 6 个递进实验，真正执行 Shuffle/Epoch/Sampler Cursor、Microbatch 梯度累积、AdamW 单次更新、按 Optimizer Step 的 Warmup/Cosine 和梯度裁剪；同时使用 JSON 往返快照验证 **Microbatch 中断时的梯度缓冲、样本游标及优化器状态** 一并恢复。通过对照相同样本与相同有效 Batch 的训练验证数值等价，通过不中断/中断重启的两条支路验证整个状态一致，最终使用恢复的同一权重进行 Decode。参见 [数学及失败合同](engineering-training-pipeline.md)。

## 下一阶段

在同一任务合同下，增加“真实小模型工程轨”：受控下载/校验、数据许可证确认、可复现环境、PyTorch CPU/GPU 训练、断点续训、导出、API 形式的 Serving 与端到端 Trace。任何声称运行真实预训练模型或取得 GPU 性能的数据都必须有独立实测证据，而非从微模型外推。
