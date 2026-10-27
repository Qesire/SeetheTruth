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
