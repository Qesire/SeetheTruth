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

## 第四阶段：Score 的分维来源、RoPE 与稳定 Softmax

新增 `learning_handoff.score(core, packet, graph, numericRecord, selection)`；复用已验证的 `coordinate` 上下文，在具体 `queryToken, head, keyToken, channel` 上沿**原执行记录**解析 Attention 分数。GQA 从 `TensorIR.gqa` 已生成的 `q_heads / k_heads / q_positioned / k_positioned / scores / probabilities` 读取，原生 Attention 从 Q/K 上游执行节点读取。它不在跨页教学层创建另一个完整 Attention 引擎。

每个真实特征位置 `d` 依次展示原值 `Q,K`、位置编码后 `Q',K'`、RoPE 偶/奇 pair 的原始旋转角度，以及缩放贡献：

```text
angle_q(pair) = (query_index + queryPositionOffset) / 10000^(pair/Dk)
angle_k(pair) = key_index / 10000^(pair/Dk)
Q'[2i]   = Q[2i]cos(angle_q) - Q[2i+1]sin(angle_q)
Q'[2i+1] = Q[2i]sin(angle_q) + Q[2i+1]cos(angle_q)
Score[t,h,k] = Σ_d Q'[t,h,d] * K'[k,kvHead(h),d] / sqrt(Dk)
P[t,h,k] = 0 (masked); otherwise exp(Score[t,h,k]-max(visible Score)) / Σ_visible exp(...)
```

每个 Rotated Q/K 元素和源码执行阶段逐值核对；缩放点积之和与原 `scores` 核对；行最大值仅取可见的 Key，且计算稳定 Softmax 后与原 `probabilities` 核对。未执行 RoPE 的单头 Attention 展示恒等位置变换。对 Causal 多 Query 的未来 Key，即使原始 QK Score 有数值，Attention 概率严格为零；单 Query 最后时刻的 Decode 前缀则可以全部可见。所有见证有界、可重放、因图结构变化而失效。

Learning Studio 通过 `#nativeGraphScoreKey` 指定 Key，通过已有 Query/Head/Channel 控件指定上下文，并在 `#nativeGraphScoreFeatures` 展示逐特征原始值、RoPE 结果及缩放乘积。此界面的证明身份是 `SYNTHETIC_NUMERIC_REPLAY_VERIFIED`，不是来自真实模型权重或生产 GPU 内核的测试结果。

新增 Node 正/反例：原始单头、GQA 分组、RoPE 旋转后坐标、Score 重构、稳定 Softmax 重构、未来 Key 掩码、非整数及越界 Key；浏览器覆盖桌面与手机 Key 下拉、分维贡献、数值模式切换与编辑图后的证据失效。

## 第五阶段：原计算图内的单坐标 Q/K 因果 A/B

新增 `learning_handoff.intervene(core, packet, graph, numericRecord, options)`，对当前已核验且可重放的 Attention/GQA 数值图构造**非破坏性、同拓扑双轨实验**：

1. 先通过 `Core.replay(numericRecord)` 核对来源，定位实际 Attention 的 `q` 或 `k` 输入端口和上游输出矩阵。
2. 将此矩阵复制进一个已经存在于积木目录、由 `BlockCatalog.execute` 计算的显式二维 `parameter` 节点，修改**仅目标 Attention 端口**的来源连线。构造 A，调用 `Core.inspectGraph(A,numeric:true)`；逐坐标证明目标 Attention 输出与原图严格数值等价（误差至多 `1e-10`）。
3. 深拷贝 A 为同拓扑 B，只在 `Parameter.values[token][feature]` 加一个有限且幅度不超过 1 的非零 `delta`；调用相同执行器执行 B，并通过 `Core.compareRecords(A,B)` 对原始算子及微型数值轨迹执行重放和差异比较。
4. 对已选 Query Token、Query Head、Key Token、输出通道分别用 `Handoff.score`、`Handoff.coordinate` 构造 A/B 数值证据；报告 `ΔScore`、`ΔP`、`ΔOut`、全输出最大绝对变化与改变坐标数。A/B 仍保留原生事件来源 ID，但计算图扰动不篡改原 Worker、KV、Scheduler A/B 状态实验。

### 关键因果限定

- 本实验是 **Q/K 投影输出的端口干预**，通过一份显式 `Parameter` 冻结端口输入。它不会更新 Linear 投影的权重，也不会暗示修改了所有共享上游节点的消费者；原始教学图本身保持不变。
- **局部零效应是正确结果**：例如 Q 的某个特征属于 Hq0，对 Hq1 所选 Score/概率不会产生改变；因果掩码排除的未来 Key 即使原始 Score 改变，也不改变该查询的概率和输出。测试应保留这两类负对照，避免用“所有 Δ 必须非零”替代因果分析。
- A 和 B 的差别只有显式 Parameter 矩阵中的一个元素；冻结后的 A 必须和原图相等，才能将 B-A 解释为该坐标的受控效应。仅限有界 Float64 CPU 教学模型；不代表原始大模型权重、GPU KV 地址或硬件性能。
- UI 的干预参数包括端口、投影 Token、投影特征、Δ；结果只在被接受的当前数值图上有效。编辑图、切换课程或离开 numeric 模式会撤销当前结果。

Node 契约扩展到普通 Attention 与 GQA 的 Q/K 单元素实际干预、来源等价、全图数值重放、Head 隔离、掩码负对照、越界与无效 Δ。Chromium 桌面/手机验收扩展到可执行干预 UI、输入校验、A/B 对照和证据撤销。

## 第六阶段：受控 Q/K 干预的完整因果传播证据

`Handoff.intervene(...).propagation` 把同一次原生图 A/B 回放扩展为严格的数据依赖解释，不另外计算一次 Attention forward。原 `Core.inspectGraph` 产生 A/B 全图 `session.trace.steps`；原 `Handoff.score`、`Handoff.coordinate` 已核验 Q/K、Score、Softmax 和 Value/Context。新结构按图引擎实际执行的拓扑序构造以下证据：

- **数据源**：显式 Parameter 对指定 Q/K 端口的一处数值扰动；本次图 A 在修改前必须重现原图全部已有节点与中间轨迹。A/B 图恢复被改动的坐标后，所有其他字段逐字段完全一致。
- **特征展开**：对所选 Query–Key、Head 的每一个特征 `d`，比较 RoPE 后 `Q'`、`K'` 与 `Q'[d]K'[d]/sqrt(Dk)` 的 A/B 差值；特征贡献之和必须等于观测的 `ΔScore`。
- **整行 Score → Softmax**：比较每一个 Key 的原始 Score、概率以及 Mask 可见性。概率的 A/B 行和变化必须约为零（因为每行归一化到 1），被因果掩蔽的未来 Key 始终具有零概率。
- **每 Key 的 `P×V` 变化**：同时对比 P、V 与其乘积；整行乘积增量和必须等于所选输出坐标的 `ΔOut`。这里绝不把某个 Key 的 Score 变化直接解释为相同量级的 Value 贡献变化。
- **真实 DAG 执行节点**：逐节点核对 A/B 的 ID、算子类型、形状、实际数值坐标；显示变化坐标数、最大绝对变化和最多六个坐标样本。其它 Query 或 Head 的真实变化由实际节点数值决定，不依据局部结果推断。
- **负对照**：Q 只影响所属 Query Head 的注意力；某个 Head 的目标 Score 可以完全不变，但全图其它 Head 确实发生变化。对早期 Query 被因果掩蔽的未来 Key，原始 QK Score 可以发生变化，当前 Query 的 P 和输出仍不变；后续能够访问该 Key 的 Query 则仍可能变化。以上被明确标记为局部隔离/掩码抑制，而不是误判为全图无效。

证据类型为 `seethetruth-attention-causal-propagation-v1`，保留数值阈值 `1e-12`（“可观察坐标改变”的 UI 判定）和误差重构阈值 `1e-10`（数值一致性校验）。二者语义不同；微小非零变化不等于可测的实际 GPU 效应。证据可追溯到原 Native event ID，但不构造“Worker/KV 事件等于张量坐标”的跨域连线。

Learning Studio 在 A/B 成功后显示 `#nativeGraphPropagation`：输入→Score→Softmax→输出四阶段、逐 Key 对照、逐特征 RoPE/Score 贡献、全 DAG 节点的变化/保持以及实例坐标。所有单元格使用 textContent；状态切换、无效干预、图结构或数值模式改变时，整体证据撤销。Node 数值契约和 Chromium 桌面/手机测试验证求和一致性、完整步数、GQA Head 隔离、Mask 局部抑制、交互和撤销。

## 第六阶段补强：计算图有向因果锥与原生端口证明

逐阶段 A/B 数值变化另外同 A/B 的实际图连接关系交叉核验：

- 每个 `propagation.graphSteps[i]` 附带真实 `incoming=[{from,port,sourceChanged}]`，取自本次临时 A 图的连接结构，不凭图形位置、用户自然语言或相似节点名称猜测依赖。
- 从唯一改动的 `ab_q_probe` / `ab_k_probe` Parameter 出发，沿有向输出边顺计算图拓扑传播，构建 `reachable`。所有实际发生数值变化的节点必须属于该集合；一旦因果锥外某个节点变化，实验**拒绝**给出因果结论。
- 被访问但没有变化的节点报告为“可达但数值保持”；不可达的节点报告为“因果锥外保持”。`causalReachableCount` 和 `changedOutsideCausalCone=0` 的含义明确限定在这次本地 A/B 教学图的数值执行，不构成对真实 GPU 调度或跨 Worker/KV 状态的归因。
- UI 展示每个节点的输入端口来源与来源节点是否变化，使单次实验同时支持局部算术机制证据与全局 DAG 传播证据的反向查验。静态契约要求所有变动节点在有向后继中，并检查真实 Attention/GQA 端口连线和上游输入未受到其他改动。
