# SeetheTruth — 可执行训练状态机实验（Engineering Training Pipeline）

## 目标与证据口径

入口：PC 与离线 Android 共同使用 [engineering_pipeline.html](../engineering_pipeline.html)，紧接在 12 关基础工程教学之后。课程面向没有 GPU 的学习者，以真实 Float64 CPU Bigram 微模型呈现 6 个必须逐关通过的工程实践：

1. Epoch shuffle、Sampler 序列和 Cursor；
2. Microbatch 前向/反向、Gradient Buffer 累积但不更新参数；
3. 不同 Microbatch 划分与相同有效 Batch 的最终梯度/参数等价；
4. AdamW 的一次参数更新、Warmup/Cosine、梯度裁剪与训练/验证 Loss；
5. 中断时 `gradientSum`、`lossSum`、`samples`、`microSteps`、`indices`、`sampler.epoch/cursor`、`W/m/v/step` 同时序列化，并与未中断训练进行逐元素对照；
6. 恢复后的训练状态作为推理权重来源（Tokenizer、Dataset Fingerprint、模型 Step），比较 Greedy 与固定 seed 的 Top-K 采样。

**本实验不是生产系统的测量证据。** 模型为 8-token 的条件 Bigram 表，核心含交叉熵、解析反向和 AdamW，并不包含 Transformer、GPU、混合精度、分布式训练或网络 Serving。Android 只打包共享教材与数值引擎，不提供模型训练产物或下载。

## 数学和过程合同

训练集为固定的 24 个 Next-token Pair；Shuffle 使用基于 `samplerSeed` 和 `epoch` 的确定性 Fisher-Yates 排列。Cursor 表示下一条样本在当前 Epoch 的位置；跨 Epoch 时正确换序，不重复初始化已训练模型。可从 `epoch/cursor/samplerSeed` 推导下一个 Batch 的样本索引。

对于 `M` 个微批次，设第 (i) 批包含 (n_i) 条样本，输出局部平均梯度 (g_i)。累积的**样本加权**梯度为

\[
g = \frac{\sum_{i=1}^M n_i g_i}{\sum_{i=1}^M n_i}.
\]

该计算在所有 Microbatch 完成前不更新模型参数。完成后先执行 Global-Norm Clip：

\[
\tilde g = g\cdot\min\bigl(1, c / \lVert g\rVert_2\bigr),
\]

再进行**恰好一次** AdamW 更新（包含动量和偏差修正）。不应将微批次的局部平均梯度直接相加后忘记除以有效 Batch 大小，也不应在各微批次之间更改 W。

调度按 **Optimizer Step (t)** 定义。Warmup 使用 (lr_t=lr_0t/W)（(t\le W)）；余弦段从 (W) 到 (T) 逐步下降到 (minScale\cdot lr_0)。超出预设 (T) 后维持下界，不继续执行负的 Cosine 周期。Grad Clip 发生在 AdamW Step 之前；Clip 不是参数更新范数的限制。

## 精确续训

本地 JSON 快照结构：

```json
{
  "schema": "seethetruth-training-pipeline-v1",
  "dataSource": "synthetic-24-next-token-pairs-v1",
  "dataFingerprint": "deterministic-structural-identifier",
  "config": {"seed": 7, "samplerSeed": 19, "microBatchSize": 2, "accumulationSteps": 4},
  "model": {"schema": "seethetruth-bigram-training-v1", "weights": [], "firstMoment": [], "secondMoment": [], "step": 0},
  "sampler": {"epoch": 0, "cursor": 0},
  "pending": {"microSteps": 0, "samples": 0, "lossSum": 0, "gradientSum": [], "indices": []}
}
```

以上为关键字段示例，省略的模型超参数、词表等仍由实际快照保留。程序恢复前检查模型格式、Tokenizer、配置信息、数据身份、样本消耗计数、游标以及 Pending 样本索引关系；拒绝错误状态且保持原有模型不变。Fingerprint 只是由合成数据内容生成的**非密码学结构识别符**，不可作为来源真实性或签名验证。

**明确边界**：当前 DataLoader 是本地固定数据集，无 Worker 预取、异步 shuffle 或持久化磁盘 Iterator；恢复在相同 JS Float64 引擎中数值逐元素一致，但不能自动外推至多 GPU/多进程混合精度或硬件浮点非确定性。真实训练系统还需维护 RNG/Worker/Distributed Sampler 状态、性能配置、软件/硬件版本和原子 Checkpoint 存储。

## 验收指标

```sh
node tests/engineering_pipeline_contracts.js
python tests/browser_engineering_pipeline.py
node tests/android_asset_manifest_contracts.js
```

Node 中应验证：Shuffle 全排列且可重建；不修改权重的 Preview 与 Microstep；大 Batch/微批划分参数误差低于 1e-12；Warmup/Cosine 数值端点；裁剪前后范数；第 3 Step 后中断在 2/4 微批位置恢复，最终与连续 8 Step 模型、动量、游标和缓存严格相同；非法恢复应原子失败；恢复后 Seeded Decode 的结果一致。

Chromium 测试必须遍历 6 个关卡，包含答错无法跳关、所有真实数值通过、PC/390px 页面布局、进度恢复、模型状态与截图。Chromium 的安卓尺寸验收不等于真机验证。静态 CI、离线 APK 打包及 Android 安全约束依然必须通过。

## 下一步研究型工程轨道

积木解释链继续向真实代码细化：真实 PyTorch 模型参数、Autograd、DataLoader、GradScaler、Checkpoint 文件、Data/Model Parallel、Torch Compile/Kernel、Hugging Face 小模型源代码、HTTP/gRPC 服务生命周期与实时 Profiler。任何“已在真实模型中可用”的声称需要独立源代码或运行证据。
