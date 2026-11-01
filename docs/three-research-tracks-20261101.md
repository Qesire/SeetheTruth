# 三条研究主线：视频生成稀疏、LLM 后训练与 Agent 架构

更新：2026-11-01。入口：[共享教学中心](../teaching.html#researchTracks)（PC/Android）、[Learning Studio](../learn.html)。

## 共同底层：三种计算对象，一套证据契约

| 研究方向 | 基本对象 | 观察与实验 | 最终证据 |
| --- | --- | --- | --- |
| 视频模型稀疏 | 张量图与保留集合 \(S(x,t)\) | FFN hidden、门控写回、通道耦合、跨块传播 | 完整生成视频、质量和实际省算 |
| LLM 后训练 | 数据 → 策略 → Loss → Gradient → Optimizer | SFT、DPO、GRPO 的参考策略、奖励、有效 Token 和梯度 | 小模型训练、泛化与校准 |
| Agent 架构 | 状态／事件图 \(s_{k+1}=F(s_k,a_k,o_k)\) | 规划、工具、审批、重试、委派、验证与恢复 | 真实任务正确率、工具审计与延迟 |

三条路线共享图身份、坐标追踪、因果边、状态版本和证据等级。**Tensor DAG、Optimization DAG 与 Tool/Event DAG 仍是不同类型的计算对象**，不得把工具副作用伪装成 MatMul，也不能把标量偏好函数说成完成了 LLM 训练。

## A. 视频模型稀疏

### 已实现：真实源码拓扑上的合成 FFN 消融

实验复用 wan_ffn_trace.js 的 Wan2.2 I2V 固定源码拓扑和确定性合成 Float64：

\[
d_{t,c,h}=e_{5,t,c}W_{2,c,h}\phi((W_1z_t+b_1)_h),\quad
\hat y_{t,c}=y_{t,c}-\sum_{h\notin S_t}d_{t,c,h}.
\]

计算 \(\|\hat y-y\|_2\)、相对 gated FFN delta 的误差、耦合指标

\[
\rho_{\rm coupling}=\frac{\|\sum_{h\notin S}d_h\|_2^2}
{\sum_{h\notin S}\|d_h\|_2^2}.
\]

\(\rho<1\) 是抵消，\(\rho>1\) 是同向增强；这不是视频画质指标。交互选择静态／逐 Token 的 Mask，以及激活能量／门控后输出贡献评分。

静态结构化裁剪的理想 FFN 双 GEMM MAC 比例为 \(K/H\)；如果逐 Token 选择必须先完成整个 W1，W2 可跳过未选列，其忽略选择开销的比例仅为 \((1+K/H)/2\)。**门控后贡献评分本身读取全部 W2 通道项，属于事后诊断 oracle，不能据此声称真实加速**。当前输入 Token 不是视频帧。

### 下一阶段：真实视频闭环

在完整 Wan I2V 上以多来源／多 prompt／多 seed／多 Block／两专家配对保存 Dense 与 Sparse 生成结果、模型修订与推理参数。除 FFN 局部误差外，至少比较 VBench I2V 的主体背景与相机运动、语义事件一致性、光流／CoTracker 运动幅度、时间闪烁、连续帧诊断、跨块张量漂移，以及 GPU 实测 GEMM、延迟、峰值显存和吞吐。HWG 及其变体使用同源同 seed 对照并保留逐块 hidden、e5、W2、mask 和残差写回资产。

局部误差仅是提出候选的代理指标；**完整视频是方法评价的主证据**。

## B. LLM 后训练

### 已实现：DPO 与 GRPO 的解析小实验

DPO：

\[
\mathcal L_{\mathrm{DPO}}=-\log \sigma\{\beta[
(\log\pi_\theta(y_w|x)-\log\pi_{\rm ref}(y_w|x))-
(\log\pi_\theta(y_l|x)-\log\pi_{\rm ref}(y_l|x))]\}.
\]

输出对 chosen/rejected log-prob 的解析导数；reference 冻结，并用独立中心差分验证。

GRPO：展示单组 \(A_i=(r_i-\bar r)/\mathrm{std}(r)\) 与裁剪后的 PPO surrogate；组内奖励方差为零时优势显式为零。当前仅一组 3 个应答，每个应答相当于一个等权 Token，未包含长度 Mask、reference KL、rollout、优化器与参数训练。项目已有更完整的 GRPO/PPO 可组合 VJP 和训练事务。

### 下一阶段：从目标函数到训练闭环

连接 SFT 的 Token CE → Prompt/Completion/Preference 数据身份 → Reference log-prob → DPO 训练步 → 真实梯度与 AdamW；GRPO 增加 Group Rollout、奖励与 Verifier、旧策略重要性比值、KL、Mask、梯度截断、Checkpoint。独立校验前向、梯度和恢复轨迹，再使用真实小模型测试。评价包括任务正确率、偏好胜率、泛化、长度偏差、KL、奖励投机和推理成本，不能用训练 Loss 一项替代。

## C. Agent 架构

### 已实现：脚本化故障注入状态图

同一模拟任务对比：

- ReAct：Observe/Decide → Tool → 根据错误反馈 Retry。
- Plan-Execute：先计划再执行；本轮特定教学策略在首轮 Tool Read 失败时停止，并不表明这一架构必然失败。
- Supervisor：主管委派 worker-A，失败后改派 worker-B，由 Reviewer 核验再完成。

每个事件携带 ID、依赖 ID、Actor、Action、前后状态和结果。审批失败即阻断模拟写入；验证未完成不得 Finalize；原始输入可确定性重放。无真实 LLM 或外部工具调用，不宣称实际任务成功率和速度。

### 下一阶段：真实运行架构

在既有统一状态及观测协议之上增加 Typed Agent Event（任务/运行 ID、父边、工具 schema、输入哈希、状态修订、幂等键、审批证据、产物和 Verifier 结果）；逐步支持沙箱 Tool、并行 fork/join、超时、去重、Checkpoint、长期/工作记忆和权限隔离。故障矩阵覆盖部分成功、重复提交、过期状态、错误工具返回、Reviewer 误判与恢复。衡量任务完成率、恢复率、轨迹可复现性、工具精确率、越权率、Token 及时间成本。

## 本轮实现与证据边界

- 统一共享课程：research_video_sparse / research_llm_posttrain / research_agent_arch。
- 学习中心上的三个交互实验在 PC 与离线 Android 使用相同的 research_tracks.js、wan_ffn_trace.js 和 teaching_hub.js。
- FFN 的独立后验重构、DPO 中心差分、GRPO 零方差/clip、Agent 故障/审批反例进入 tests/research_tracks_contracts.js；双尺寸浏览器回归进入 tests/browser_shared_teaching.py。
- **源代码注册与合成数值通过，并不证明实际模型权重、GPU 加速、长视频质量或完整 Agent/后训练系统。** 后续研究继续按既有统一执行/观测/证据协议扩展，不建立并行且互不兼容的微内核。

相关入口：[Wan FFN 显微镜](../wan_ffn_lab.html)、[训练微流程](../micro_training.html)、[状态语义实验室](../semantic_state.html)。
