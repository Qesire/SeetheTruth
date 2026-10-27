# 原生状态实验 ↔ 计算图课程：可重放教学衔接 v1

## 对象与适用范围

这一衔接记录的是**教学导航和证据来源**。Worker 的事件 DAG、KV 的资源状态转移与 Scheduler 的决策程序保持各自的原生数学语义。Learning Studio 的 Attention/Decode 课程仍通过 `LearningLab` 的可编辑张量 DAG 执行。跨页面操作只建立可检验的机制对应问题，不推断状态字段与 Attention 张量存在逐元素等价关系。

源入口为 `deepinfer_worker_lab.html`、`deepinfer_kv_lifecycle_lab.html`、`deepinfer_scheduler_lab.html`。根据原生领域选择现有课程：Worker → Attention(`attn`)；KV/Scheduler → Decode(`gqa`)。计算图节点必须出现在对应课程的真实 `LearningLab.preset` 图中。

## 执行和证据协议

1. 学习者首先通过原生页面得到一份可执行结果；可选地在 A/B 区完成有差异输入的对照。通过 `SeetheTruthUnifiedCore` / `SeetheTruthSpecializedLesson` 执行、回放和比较真实原生程序。
2. 点击「联系计算图积木」时，`learning_handoff.js` 调用既有 `observation_snapshots.create` 保存 B，使用 `at` 对当前被选中的原生事件生成核验光标；存在 A/B 时另保存 A。B 输入必须重新执行并证明选中事件实际存在，A/B 必须由同一原生引擎独立回放。
3. 完整信封仅包含 `schema`、`domain`、`active`、`baseline`、`question`，限制 180000 字符，使用单一固定键的**本标签页** `sessionStorage`，由学习者点击启动。URL 只携带导航标记，不携带原始状态值或输入。
4. `learn.html` 接收时调用 `restore` 对 A、B、事件光标和来源引擎进行重新验证，加载现有 `data-preset` 课程并用 `SeetheTruthUnifiedWorkbench.focusNode` 选中真实节点。原生事件作为来源说明，图课程内的数值需另按「运行」获取。
5. 返回源页面时，再次调用 `restore` 核验原生状态，打开选中的 B 显微镜事件；A/B 还可在只读视图中切换。返回页面**不覆盖当前原生控制器的输入**，因此恢复的是教学证据光标而非系统运行状态。
6. 原生输入变化、未知来源、非法事件光标、不同引擎的 A/B 配对、导入内容篡改或超限，都会拒绝证据恢复。快照 FNV 摘要用于普通文件完整性与可重复性诊断，不提供抵抗恶意伪造的加密认证。

## 可证伪测试

- `tests/native_ab_guide_contracts.js`：KV、Scheduler、Worker 的 A/B 与 Worker 的**不同输入同一输出**反例，逐项通过原生执行；验证正确光标、错误摘要和跨来源替换。
- `tests/browser_state_learning_guide.py`：实际浏览器在桌面和手机视口从原生 A/B 进入 Attention/Decode，定位存在的源事件，返回后在同一源引擎核验并切换 A/B 光标。所有运行路径必须无页面 JS 异常。
- `scripts/sync_runtime_assets.js --check`：HTTP 与离线 Android 资源同步；安卓课堂沿用相同静态教学实现，不提供代码导出或真实 GPU 性能承诺。

## 后续演进

当前首先落地的是**有来源、可返回的教学导航**。可逐步完善对具体数据依赖与资源事件的结构化关系解释（例如 KV block ownership 与 GQA K/V 读取所面对的不同层次），并将所提出的竞争解释组织为有界课堂问题。任何将机制关系提升为形式等价关系的工作，应提供明确映射函数、条件和对应的独立反例验证。

## 第二阶段：原生事件 ↔ 真实图端口与数值的机制显微镜

`learning_handoff.explain(core, packet, currentGraph, numericRecord?)` 每次调用都会重新验收原生 A/B 来源，然后对**当前正在编辑的** `LearningLab` 图调用 `Core.inspectGraph(..., numeric:false)`。只从实际图的 `GraphLearningPath` 读取 Attention/GQA 的 Q、K、V 上游节点 ID、端口及源张量形状。Worker 关联 `attention/attn`，KV 与 Scheduler 关联 `decode/gqa`；一旦相关节点在当前图中不存在或端口不完整，原有机制关联失效，提示学习者重新选课。其他有效端口重接线被按新图重新分析，不沿用旧端口结论。

选择「运行当前积木图并核验 Q/K/V」会通过已有 `#studyMode` 和 `#run` 运行原生小规模图引擎。数值轨迹由 `SeetheTruthUnifiedWorkbench.inspectAccepted()` 提供，并且必须匹配当前图、`Core.replay` 成功且目标节点有 `SYNTHETIC_NUMERIC` 数值证据。切换课程、修改图或执行无关分支时，原有数值资格随图身份变化而失效。

原生事件帧提供源事件 ID、标题、`dependencies`、原生 `changes` 字段和 A/B 所选事件是否改变。源实验室中的状态先后关系与计算图端口关系并列展示。对于 Worker，可比较资源依赖事件 DAG 与 Q/K/V 数值计算 DAG 的可并行条件；对于 KV，可区分物理缓存所有权与 K/V 逻辑张量内容；对于 Scheduler，可区分准入条件与一次已准入 Attention 的算术。这种对照是**机制层面的教学关系**，不会产生跨域张量等价断言。

测试扩展：`tests/native_ab_guide_contracts.js` 用三个实际实验引擎与 Worker 无效果反例逐一校验 Q/K/V 的实际图端口、数值重放资格、错误图证据与删除目标节点；`tests/browser_state_learning_guide.py` 在桌面和手机上核验端口列表、真实数值按钮、切换 FFN 导致机制关联失效、重选目标课程后按当前图恢复静态解释。

## 第三阶段：逐 Query / Head / Channel 的 P×V 数值证据

`learning_handoff.coordinate(core, packet, graph, numericRecord, selection)` 从当前图经 `Core.inspectGraph` 验证的数值运行记录出发，逐坐标列出 `Score[t,h,k]`、`P[t,h,k]`、`V[kv(h),k,c]` 和乘积 `P×V`。该接口要求已执行 `numericOk=true`、输入图与当前编辑图逐字段一致、所选操作类型与来源端口一致，并调用已有 `Core.replay` 与 `Core.inspectObservationStep` 核验目标输出坐标。

- 普通 Attention 使用原 `LearningLab` trace 的 `scores`、`probabilities` 与图中 `v` 端口来源节点；单头 `h=0`。
- GQA 使用原 `TensorIR.gqa` 返回的 `probabilities.tensor`、`scores.tensor`、`v_heads.tensor`、`context_heads.tensor` 和源因果掩码。原生 `Hq / Hkv` 整数分组规则决定 Query Head 到 KV Head 的映射；仅从已验证的原始阶段读取，不根据展示数据重造另一套 GQA 引擎。
- 一个可审计输出元素满足 `Out[t,h*Dv+c] = Σ_k P[0,h,t,k] * V_heads[0,floor(h/group),k,c]`。同时检查 `ΣP≈1`、Mask=0 所在项的 P 严格为零、分头 Context 与合并 Out 一致；超过 1e-10 相对尺度的误差拒绝该见证。该求和只校验观察值，并非又一次完整 Attention forward。
- 因果 `decode` 预设中只有最后一个 Query，`queryPositionOffset=Tk-Tq`，所以最后查询可读取全部已提供的前缀，可能没有被掩蔽的 Key。另用完整多 Query GQA 预设检验未来 Key 的严格零概率，而不是要求 Decode 场景凭空出现 masked 项。
- 结果严格标记 `SYNTHETIC_NUMERIC_REPLAY_VERIFIED`。它来自有界微型 CPU 教学计算；没有推断真实 GPU 速度、KV 物理块地址与单坐标值的一一对应关系。原生状态 A/B 仍独立保留和回放。

Learning Studio 的 `#nativeGraphCoordinateLab` 新增三个坐标下拉选择、原始 Score/P/V/P×V 表格、Query Head → KV Head 及核验误差。修改当前计算图或转入非数值模式时，原有坐标见证失效；重新执行后由最新轨迹恢复。浏览器测试覆盖桌面与移动视口、切换 head/token/channel，以及跨课程后失效；Node 测试覆盖普通 Attention、GQA、头共享、真正的 Mask 反例、越界坐标和错误图证据。
