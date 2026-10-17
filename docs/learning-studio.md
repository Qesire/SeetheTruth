# SeetheTruth Learning Studio — 从源码浏览器到 Scratch 式大模型实验台

## 1. 愿景：Construct → Execute → Explain → Verify

SeetheTruth 原有的 AST/Runtime/Profiler 层保持独立、可溯源。新增加的学习环境为零依赖纯浏览器 **积木式 Tensor Graph 编辑器**。学习者从具体张量形状与运算出发，自行连接模块，逐步计算，定位形状或语义错误，最后再去真实模型源码与运行时证据中核对自己的理解。

不应直接把真实模型的前向过程宣称为“拖一拖积木就完成训练”。真正的训练需要 autograd、参数更新、状态管理、优化器等；RL 还包含 rollout、reward/advantage、sampling 与策略目标，不能用一个抽象积木替代为已完成的运算。

## 2. 已有可运行的最小学习闭环

入口：\`http://127.0.0.1:8765/learn.html\`，或离线同时保留 \`learn.html\` 与 \`learning_lab.js\`。

- **构建**：从 10 种教学积木拖放/单击添加，自由移动，明确输入端口、输出端口，连线、断线和删除。循环连线不合法；同一输入端口至多一个来源。
- **执行**：确定性、有界的二维小矩阵 CPU JS 运算。功能包括 Input、Linear、RMSNorm、SiLU、Add、Multiply、MatMul、Softmax、单头无 mask Scaled Dot-Product Attention、MSE Loss。
- **解释**：节点可查看设计意图、公式、注意事项、参数、输入输出 shape。逐步输出矩阵热力图；Attention 可展开 QKᵀ/√d 分数和 Softmax 权重。
- **检验**：断线挑战、形状检查、未连接提示、课程示例；具有独立 JS 数学/数值 oracle、Chromium 浏览器验收及恶意 JSON 拒绝测试。
- **创作**：撤销/重做、参数修改、工程本地 JSON 保存和导入。无服务器工程存储、无用户账户系统。
- **明确证据类型**：\`deterministic_browser_teaching_simulation_not_model_execution\`。权重由局部固定公式生成，输入由种子生成；不上传数值、不会执行模型源码，不做真实权重推理。

样例任务：
1. FFN 两层投影（观察 D→F→D）。
2. 残差（验证两路输入具有同形状；为什么保留原信号）。
3. Attention（Q/K/V、缩放、稳定 Softmax、加权汇聚）。
4. MSE Loss（计算标量；辨认“得到 Loss”与“已经训练”不是一回事）。

### 数学真实性和限制

所有教学积木实际在浏览器中对上限 8×16 的二维矩阵做数值计算，结果不能外推到大型 Transformer 的速度或质量。单头 Attention 处理 Q∈R^{Tq×Dk}、K∈R^{Tk×Dk}、V∈R^{Tk×Dv}，输出 R^{Tq×Dv}；沿最后一轴做数值稳定 Softmax，并强制 K/V 序列长度一致。

**Linear** 使用固定且可复算的微型权重与偏置；**RMSNorm** γ=1；**Loss** 只有前向。FLOPs 仅统计明确约定的线性/矩阵乘法 GEMM 部分，不构成硬件测量或完整运算开销。

### 互操作与安全

学习工程为 \`seethetruth-learning-lab-v1\` JSON，不能等同于已证实的 \`model-atlas-graph-v1\` 或观测型 \`seethetruth-runtime-v1\`。二者保留各自身份。导入时验证节点种类、ID、端口、边数、坐标、参数数值和 DAG 约束；拒绝循环、非法维度与重复占用的输入端口。网页只在本地解释小矩阵；没有 eval / 动态导入或执行用户上传的 Python。

## 3. 课程和引擎的后续路线

**阶段 A（已完成首版）**：单张量、投影、门控、残差、单头 Attention、Loss 的拖放工作台和逐步调试。将抽象公式还原成可检查的数值例子。

**阶段 B（待实现）**：教学用**张量形状 IR**，明确 \`B,T,H,D,Q,K,V\` 等轴的物理/语义区别；支持多 rank、广播、einsum、RoPE、causal mask、KV Cache、GQA/MQA、SwiGLU 与 AdaLN 的微型数值实验；对同一层的不同实现比较 FLOPs、内存访问等形式指标。必须每项附形状、数值和退化情形测试。

**阶段 C（待实现）**：训练编程积木（可微算子 forward → autograd backward → optimizer step）和真实的小型梯度检查：对 MSE、Linear、LayerNorm/RMSNorm、Attention 逐步扩展，记录权重、梯度、激活、损失，并与有限差分作数学验证；将 Prefill、Decode、Cache 生命周期作为真实状态机。不能把 Loss 前向误标为训练完成。

**阶段 D（待实现）**：RL 专题：policy/ref/reward、采样、logprob、advantage、KL、PPO/GRPO 等；引入显式批次维度、group、mask、token 权重和状态更新。目标是精确重构独立教学推导，再与指定代码实现比较，而不是画一条概念箭头充当实际 RL。

**阶段 E（待实现）**：以现有源代码 AST、运行时 Module/ATen、Profiler 证据将现实模型模块**映射到等价教学积木**，逐个标注“确切代码支持 / 一般数学解释 / 未能解释”。提供学习者自行解释真实模块的练习和可否证差异追踪。只接受可验证的来源位置、张量、参数；不由名称猜测实际网络。

**阶段 F（待实现）**：完整图形编程 UX：磁吸式端口、复合模块折叠、课程关卡评估、辅助提示、协作分享/本地作品、可访问性键盘连线、大图场景布局；任何 UX 扩展仍以正确数学与工程测试为先。

## 4. 验收方法

- \`node tests/learning_lab_contracts.js\`：运算性质、残差相加、Softmax 权重总和、Attention 加权和独立验算、非方阵乘法、误差、有限性及 DAG 验证。
- \`python tests/browser_acceptance.py\`：真实 Chromium 课程切换、插口连线、参数变更、撤销、矩阵步进、JSON 拒绝、断线修复。
- \`.github/workflows/ci.yml\`：静态单测、原有源码/运行时回归及教学解释器合同。
- 原有真正的模型源码审计和运行时性能测试不得被教学功能替换或减弱。

这个升级不是抛弃 SeetheTruth 研究型工具，而是建立两条有桥梁、且证据分明的路径：**由积木学习原理** 与 **由源码/运行数据验证实现**。


## 5. 多维语义轴 Tensor IR 与可展开 GQA

新模块 tensor_ir.js 将微型数值张量的 shape、命名轴与 Float64 数值绑定。最多 5 维，每维最多 32 个元素，单个张量最多 4096 个元素。学习实验不执行实际模型权重或 CUDA Kernel。

**任务 05：GQA / RoPE / Causal Attention**：从 Q/K/V 投影开始，建立两个 Query Head 对一个 KV Head 的分组共享结构。内部逐次展示 Head 拆分、旋转、缩放点积、因果掩码、Softmax、Value 汇聚及合并输出。可以调整 heads、kvHeads、causal、rope 参数。

**任务 06：Decode KV 前缀读取**：一个 Query token 与三个已提供 K/V token 交互。教学使用位置偏移 Tk−Tq 进行对齐；这是已有 KV 前缀的只读情景，不包含缓存追加、生命周期或物理存储更新。

### 数学约束

输入：
- Q: [Tq,Hq×Dk]
- K: [Tk,Hkv×Dk]
- V: [Tk,Hkv×Dv]

展开成 [B,Hq,Tq,Dk] 与 [B,Hkv,Tk,Dk/Dv]；当前 B=1。必须 Hq % Hkv == 0。设 g=Hq/Hkv，则 Query Head h 使用 KV Head floor(h/g)。

对于 Query Head h，scores 是 Q_h 乘 K_floor(h/g) 的转置，并除以 sqrt(Dk)。Softmax 在允许的 Key token 范围上计算，最后 P 乘以对应 V_head。因果模式允许的 key 为 j <= Tk-Tq+i。**被掩码排除的位置概率严格为 0**，并不是把近似小概率当作零。

RoPE 采用相邻两个通道组成一对的正交旋转，并按 token position 指定角度。Dk 必须为偶数；旋转前后成对范数保持不变。真实模型可能使用其他等价旋转布局，必须针对具体源码再验证。

### 展开与观测

Inspector 展示 q_heads、k_heads、v_heads、q_positioned、k_positioned、scores、masked_scores、causal_mask、probabilities、context_heads、merged_output，每个阶段保留命名轴。对于具有 Head 轴的张量可选择 Head 0、1 等不同矩阵切片。

GEMM FLOPs 只覆盖可明确计算的 QK^T 与 P×V 乘加部分：2*Hq*Tq*Tk*(Dk+Dv)，不包含 Mask、Softmax、RoPE 或设备实现开销。这是教学运算量，不是实测 profiler 时间。

### 验收与限制

Node 数值测试覆盖 Head 拆装、轴置换、RoPE 范数不变、均匀注意力、精确 Causal Mask、不同 GQA 组的 KV 路由、Decode 对齐和异常输入。Chromium 验证参数修改与 Head 切片交互。

当前尚无真正可更新的 KV Cache 状态机、自动微分、Optimizer、FlashAttention 内核、完整多 batch 或实际 GPU 执行证据。与既有 AST/Runtime/Profiler 工具保持证据分层。
