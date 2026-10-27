# 状态实验 A/B → 计算图：逐坐标证据合同（2026-10-27）

## 统一学习路径

`semantic_state.html` 负责一次确定性状态实验的展示：数值结果、A/B 比较、实际坐标来源和异常分支。前端使用 `SeetheTruthUnifiedCore.executeLab("semantic_state","scenario",{name})`，并通过 `replay` 验证相同执行入口；原始算术由 `semantic_state_examples.js` 调用 `tensor_ir.js` / `semantic_state.js`。

`state_learning_handoff.js` 负责案例到已有积木图的**可审核、具名映射**。其 `validateGraph` 逐项检查节点 ID、算子类型和必需的连线端口。`coordinate` 给出同一次实验的数值元素来源；`coordinatePath` 检查高亮路径包含于所选积木的真实上游依赖闭包，且路径内部存在对应实际图连线。课程仍通过自身图执行器产生数值。

从实验到课程传递 `state_case`、`state_node`、`state_branch`、`state_index`。回到实验保留案例、分支与元素索引。这四个参数有白名单和有界整数检查；它们不能改写图计算内核或参数。

## 数学可解释性与四组实验

### 1. 流状态分叉（stream）

定义原状态 (S_0)、新增块 (u_A) 和 (u_B)：

\[
S_A=\mathrm{append}(\mathrm{fork}(S_0),u_A,t),\qquad
S_B=\mathrm{append}(\mathrm{fork}(S_0),u_B,t).
\]

对扁平索引 (i)，当前教学案例的形状为 ([1,3,2])，且 (mathrm{flat}(S_{A/B})[i]) 的来源是前缀相同位置，或者各自独立的 `appendChunks.a/b`。逐坐标验证输入值与输出值一致，并返回真实分支的 `prefix → init → fork_{a/b} → append_{a/b}` 路径（包含实际写入块的输入节点）。

### 2. 五维视频（video）

当前示例 (X\in\mathbb{R}^{1\times3\times2\times2\times2})：

\[
Y=\mathrm{concat}_T\bigl(\mathrm{slice}_T(X,0,1),
\mathrm{slice}_T(X,1,1),\mathrm{slice}_T(X,2,1)\bigr).
\]

对所有 24 个元素进行精确数值对照 (X_i=Y_i)。在课程 `extension_video_time_composition` 中，同一时间索引映射到 `early` 或 `late` → `join` 的可执行积木路径；课程以 `early[0:2]`、`late[2:3]` 组合，划分粒度不同但时间顺序一致。实验的整数输入与课程的正弦输入独立，因此仅对照计算机制和坐标索引规律。

### 3. 具名轴广播（axis）

\[
Y_{t,d}=X_{t,d}+b_d,quad X\in\mathbb{R}^{2\times3},\quad b\in\mathbb{R}^{3}.
\]

6 个合法输出坐标由原始 `input` 和 `bias` 精确验证，课程路径为 `x,bias → add`。错误的 `Tk` 轴在状态实验引擎中被拒绝；它没有数值输出坐标，不被包装为合法图结果。

### 4. 过期写入（reject）

当 `nextPosition=6`，请求 `position=5` 触发明确的状态不变量拒绝。原状态的 4 个元素可追踪；被拒绝请求没有新状态或数值输出。学习者可进入 `stream_state_branches`，修改合法 Append 的 `position` 自行复现异常路径。

## UI 层级和交互

- 第一层：案例选择与可证伪预测。
- 第二层：A/B 结果摘要和可展开的原始执行 JSON。
- 第三层：紧凑坐标网格，点击展示 `[axes:index]`、实际值、输入项和 A−B 差值。异常分支显示拒绝原因和可复现的条件。
- 第四层：对应积木高亮、上游依赖节点及课程独立输入说明，允许返回原坐标。
- 键盘操作：坐标按钮具备名称、选中态与焦点可视化；手机端按屏宽重排，遵从 `prefers-reduced-motion`。

## 证据与验收入口

- `node tests/semantic_state_contracts.js`：真实具名轴与流状态数值合同。
- `node tests/state_ab_learning_bridge_contracts.js`：四案例各坐标的原始输入来源、输出等式、图 ID/类型/连线、坐标路径。
- `python tests/browser_state_ab_bridge.py`：真实 Chromium PC/手机往返深链接，选中积木与路径高亮，保存 `state_ab_screenshots/`。
- `python tests/browser_catalog_screenshots.py`：完整积木工作台回归和截图。
- `node scripts/sync_runtime_assets.js --check`：Web 与 Android 离线资源发布闭包。

数值相等的口径限定为**同一次状态实验的原始输入与输出**；图课程使用自己真正执行的教学数据。浏览器截图与真实运行通过 CI 工件逐次审查，不能用静态界面构件替代端到端验收。
