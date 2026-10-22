# SeetheTruth 机制驱动通用化复审 · 2026-10-22

## 实际进展

这次复审的重点不是再加一个独立教程，而是把先前被拆散的计算证据真正接入同一个可展开的数据协议。

最新的 20 个页面级别清查为：**3 个通用张量/算子**、**7 个通用状态/事件**、**5 个专用实验页面**、**2 个导航**和**3 个 PC 证据工具**。这个清查包括正向的实际模块调用检查，不再仅凭引入 JS 文件计数。对应执行资产和 negative mutation cases 位于 `universal_surface_audit.js` 和 `tests/universal_surface_audit_contracts.js`。

### 1. 共享参数反向传播：真正看见梯度如何合流

`micro_training_flows.js` 的 `Tape` 原先只保留局部 VJP 统计量，无法解释“两个不同路径各自对 W 贡献了哪些元素”。现在 backward 记录每条边的**完整数值梯度矩阵**和**当次接收到的梯度**，并在 backward 节点记录**当时累计值**。

通过 `micro_training_trace.js`，通用 Compute Microscope 展开真实事件序列：
- Input/Parameter → MatMul×2 → Add → MSE；
- 每条真实逆向传播边的 `VJP(outputGrad)` 数值；
- 同一参数 W 两条反向路径的贡献相加及最终梯度。

数值验证用第二种途径重新调用边上的局部 VJP，同时检查两条贡献之和与 `Tape.gradients()` 匹配。不把仅有 loss 的实验假冒成反向传播可视化。

### 2. MoE：从路由索引深入到真正执行的专家矩阵乘法

`micro_training_trace.js` 用原执行器的 Top-1 + Capacity 结果产生：
`tokens → router logits → softmax → dispatch → selected expert GEMM / overflow zero → MSE → expert/input grads`。

真正被接收的 Token 可以进一步展开独立子算子：选定专家的 W、Token 向量、各输入通道乘积，以及当前输出坐标的完整闭合误差。容量不足的 Token 则走真实零输出分支，不显示虚假的 GEMM。更改 Capacity 与 Router Logit 的对照可以明确区分：**路由变动**与**专家自身的数值计算**究竟哪个先发生变化。

### 3. PPO：Ratio、Clip 的竞争分支与局部梯度

直接使用 `micro_training_flows.ppo()` 返回的旧策略概率、当前策略概率、概率比、裁剪比率、损失及 `dL/dlogits`，作为统一阶段序列。调节优势 A 的符号、新策略选定动作的 Logit 或 `epsilon`，观察梯度从非零到零的真实切换，并用被篡改的输出测试拒绝伪造的概率比。

## 反防御式的教学与研究解释标准

介绍每一个模块时，默认顺序为：

1. **真实结构与运算顺序**：先指明上游输入、张量轴语义、计算步骤及实际源码。
2. **机制假设**：说明设计如何改变数学行为，并至少提出一种可能竞争的机制。
3. **最小区分实验**：指出一个学生能直接修改的输入、分支或参数，以及两个假设给出的不同可测结果。
4. **观察与解释**：报告真实数值，区分主体效应与局部副作用；遇到劣化时从结果倒推机制，而非只罗列风险。
5. **下一步设计**：提出能够升级模型/算子/执行性能验证的具体步骤；证据范围只保留一处简明说明。

对用户开放的积木组合，不以“这不是官方完整模型”作为跳过数学重建的理由；能够运行和数值闭合的部分必须主动分析。另一方面也不允许把单一的模拟时间轴伪造成真实 GPU Profiler 指标。所有明确可检验的论断都对应具体执行器与测试。

`architecture_why.js` 现在将数学机制后的最小鉴别实验提前，优先于替代方案、失效机制和代价说明。

## 完成判据

- [x] 原始 Tape backward 的每条边保留真实 VJP 数值、梯度累计和节点顺序
- [x] 通用计算透视器支持 Tape 前向、反向和真正的共享 W 梯度合流
- [x] MoE 每个被派发 Token 的真实专家矩阵点积可数值审查；容量拒绝路径明确
- [x] PPO 通过实际数学执行器的概率/裁剪/梯度事件展开
- [x] 正例和伪造数值负例接入 Node 数值合同
- [x] PC/手机 Chromium 验收与截图任务加入 GitHub Actions
- [x] 清查表增加“源码中真正调用适配器”的第二重检查
- [ ] 最新提交全量 CI 全部通过（以 GitHub Actions 结果为准）

接下来的重点是 `paper_lab.html` 中 LoRA / DoRA / 在线 Softmax 的每次矩阵运算及中间状态；随后是 `semantic_state.html` 中具名轴/流状态转换。两者的现有数学引擎先复用，不另起一套教学模拟。

## 测试路径

```bash
node tests/micro_training_flows_contracts.js
node tests/micro_training_trace_contracts.js
node tests/compute_microscope_contracts.js
node tests/universal_surface_audit_contracts.js
python tests/browser_micro_training_trace.py
node tests/android_asset_manifest_contracts.js
```
