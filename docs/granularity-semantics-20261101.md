# SeetheTruth G0–G4 粒度与语义展开设计（2026-11-01）

状态：跨平台的粒度展示协议已作为候选提交；此文是下一阶段的系统 IR、课程合同和验证规格，并非已经完成的真实多域执行引擎。

## 1. 三条正交轴

- 架构问题 L：系统目标、架构决策、运行时与状态、内核数学、证据与观测。决定学习者在问什么。
- 语义粒度 G：G0 系统、G1 子系统、G2 组件、G3 算子/动作、G4 坐标/事件。决定观察的对象有多细。
- 时间跨度 T：静态拓扑、一次算子/动作、一个请求或训练步、整个采样轨迹、完整生成/训练/Agent 任务。时间跨度不能暗中由 G 推出。

形式上，学习上下文是 \(V=(L,G,T,\mathrm{scope},\mathrm{evidence})\)。改变其中一个坐标应当保持其余坐标可追踪，而不是丢弃原始运行、资源或源码身份。观测证据横贯所有粒度，不能用单独的“最末层课”取代。

## 2. 五级教学与积木的对应

| 粒度 | 学习问题 | 视觉对象 | 源与执行地位 |
| --- | --- | --- | --- |
| G0 系统 | 任务是否完成？吞吐、时延、质量、可靠性、成本的目标是什么？ | System / SLO 胶囊 | 课程目标，不自动执行 |
| G1 子系统 | 控制面、数据面、资源所有权如何划分？ | Pipeline / Service | 关联已有 Specialist Lab；未实现的部分标规划 |
| G2 组件 | 某个 Scheduler、FFN、Trainer、Tool Gateway 实际怎样工作？ | 可展开的功能组件 | 原生引擎、源码对齐或明确的模拟 |
| G3 算子/动作 | 精确的更新或功能机制是什么？ | Tensor Operator / Typed Tool Action | 已注册的张量算子才有图执行权；工具副作用不是 MatMul |
| G4 坐标/事件 | 哪个张量值、梯度路径、事件或状态版本引起差异？ | Coordinate / Event microscope | 必须引用实际执行的 Trace 或确定性教学事件 |

课程不直接等于积木。一个 Lesson Contract 应描述工作负载、学习目标、要决策的架构问题、焦点粒度、源实体、可复现输入、预期反例、证据等级和评分标准。一个课可以包含多个 G 层，但每一阶段最好只要求完成一个可证伪的设计判断。

例如“FFN 稀疏”不是背诵 GELU；它要判断静态还是逐 Token 掩码、W1/W2 哪些计算实际可省，以及最终生成视频是否维持运动与语义。

## 3. 分层对象的身份和边

长期对象合同拟采用：

- semanticId：既有课程、实验室、Composite、Operator 的源身份。
- viewId：一次 UI 展开；同一个源身份可在多门课程中投影。
- runId/sourceRevision：确实运行或载入源码时才存在。
- stateRevision/observationId：原生状态与 Trace 实际给出时才存在。
- parentId/portContract：真实父子端口或有明确标注的教学父子关系。

边类型不能统一称“数据流”。至少区分 tensor_data、control_dependency、state_transition、resource_ownership、temporal_dependency、observation_link 和 pedagogical_zoom。最后一种只是教学导航边，不是等价分解。

同理，视觉缩放、课程阶段推进和数值计算展开彼此独立。点击一块 Scheduler 胶囊不会自动创建名叫 Scheduler 的张量算子。

## 4. 展开正确性：可交换图，不是相同名称

给定父对象转移 \(F_P(s,x)=(s',y,e)\)、子图转移 \(F_C\)，输入注入 \(I\)、状态映射 \(J\)，以及输出与状态的折叠映射 \(\pi_y,\pi_s\)，只有在声明的工作负载域 \(D\) 中满足

\[
\forall (s,x)\in D:\quad
\pi_y(F_C^y(J(s),I(x)))=F_P^y(s,x), \qquad
\pi_s(F_C^s(J(s),I(x)))=F_P^s(s,x)
\]

才有资格称为父子功能的精确等价展开。含并发、随机性或外部侧效应时还必须定义 RNG、事件偏序、幂等性、资源占有与可恢复性的等价。近似方案应报告作用域、度量、误差界及其验证机制，而不是以一次 SSE 代替所有行为。

设计证据等级：NOT_CLAIMED（默认）、DESIGN_ONLY、SYNTHETIC_NUMERIC、SOURCE_PINNED、BOUNDED_VERIFIED、REAL_MODEL_CAPTURE、HARDWARE_MEASURED。真正的来源验证和速度测量要走原有统一观测协议。函数等价不代表 GPU 加速；局部数值接近也不代表视频质量等价。

一个反例：项目的通用 SwiGLU FFN 宏模块不能因为都叫 FFN，就被认作 Wan2.2 I2V GELU+AdaLN/e5 FFN 的数学等价实现。培训中的 AdamW 算子也不自动等价于有 rollout、reference policy、分布式参数同步的真实 LLM 后训练流程。

## 5. 三条路线如何共享粒度协议

**视频稀疏：** 完整视频与算力 SLO → 生成/去噪执行系统 → 专家、Block、FFN → W1/GELU/W2/e5 与稀疏选择 → Token/Hidden/Step 贡献。当前真实视频级系统未接入；局部 Wan FFN 有源码约束的合成数值；只有完整生成和硬件数据才可验证最终结论。

**后训练：** 能力与成本 SLO → rollout/训练/评估流水线 → trainer、optimizer、checkpoint → CE/DPO/GRPO/AdamW → 参数坐标和训练事件。已有流水线属于小模型 CPU 教学实践，不能误认为对目标 LLM 已完成后训练验证。

**Agent：** 任务成功率、权限、时延 → Planner/Executor/Verifier → 状态与 Tool Gateway → Tool Call → eventId、attempt、checkpoint revision。当前各粒度主要指向同一可注入故障的脚本状态机；不是四种不同的真实 Agent Runtime。Agent 的 G4 是事件而不是张量坐标。

## 5.5 首个可核验的跨粒度执行闭环：Scheduler/KV/Worker

新增的 G0–G4 **不是一套 System IR**。在已有统一 `learning_experience_core.js` 中，`projectExecutionGrain` 只验证被接受的 `deepinfer_scheduler/simulate` 执行可按原输入无损回放，再把任务委派给原 `state_microscope.js` 的纯只读 `schedulerGrain` 投影。

- **G0** 读取原 `value.summary` 和 `stateTrace.steps[].id`。
- **G1** 读取原 `value.decisions`、`value.snapshots`、`value.worker.events` 的范围和数量；三条来源并不是额外启动的执行器。
- **G2** 使用原 `stateTrace.steps[cycle+1]`、`value.snapshots[cycle+1].revision` 和 `value.executionProof[cycle].forwardEvent`。
- **G3** 展示 `value.decisions[cycle].actions` 的真实 KV 操作、请求 ID、修订号；不构造新的事件 ID。
- **G4** 可选择同一轮真实的 Worker Event ID 或某个原生 KV 操作来源路径，两者是兄弟观察分支而非假定的函数等价父子。

统一观察面板复用当前被接受的执行对象；统一 Core 为核验来源一致性，可以重新执行**同一套** Scheduler 源引擎的确定性回放。粒度投影自身不运行 Scheduler、不编辑 KV 状态，也没有第二套独立的实现。现有 `state_microscope.validate`、`replayLab` 和原生事件校验承担可信边界。G4 的 Worker 时间数据仍是**假设的逻辑时间，不是 GPU Profiler**。

已有 `tests/native_granularity_spine_contracts.js`、`tests/browser_deepinfer_scheduler.py` 对三个真实源引擎场景、完整来源映射、跨粒度状态一致性、篡改拒绝和 Chromium 界面进行回归；CI 结果需以最新提交检查。

## 6. 已落实与后续工程边界

本轮 G0–G4 由 granularity_lens.js 组织成三轨道、每轨五级的轻量视图：课程用原 Curriculum.lookup、实验室用 UnifiedCore.inspectLab、注册算子用 BlockCatalog 检查；相同来源的重复下钻标 SAME_LAB_REFOCUS / SAME_SIMULATION_REFOCUS，跨来源仅标 PEDAGOGICAL_ZOOM。所有导航的数值等价默认 NOT_CLAIMED，显示来源、覆盖关系和证据等级。

teaching.html 与 Android 离线教学共享粒度控制；Learning Studio 支持使用原有算子 ID 定位 G3，或打开原有 Composite 路径。新合同测试和 Chromium 浏览器测试核验真实链接、不重复注册引擎、跨平台加载以及轴切换。此处仅是**语义导航层**，并未新增统一系统图执行器。

下一工程切片应以已有 DeepInfer Scheduler → KV → Worker 为唯一真实适配样例：从原生事件投影到父系统图、定位子事件与物理 KV 所有权、故障注入后回放同一事件、原始 ID 与状态修订号往返不丢失。若某组件没有真实 Operator 级追踪，应清楚显示不可进一步展开，不生成伪算子或虚构速度。

验收原则：改变 G 不得改写原数据、状态、算子语义；无法证明数学等价的关系不能被标为精确分解；全局指标质量和硬件性能必须保留独立的真实数据来源。