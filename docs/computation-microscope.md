# SeetheTruth 通用计算透视合同（Core Compute Microscope）

## 产品约定：基本能力，而不是 Wan 专用页面

计算透视必须出现在 **主自由积木工作台** 的右侧 Inspector 中，并可由模型特化教学页复用。每个积木至少揭示该步的拓扑执行位置、真实上游连线/端口、输入/输出张量形状、数学操作、为什么这样设计、限制；当已有经过本项目校验的数值执行结果时，再展开张量坐标、实际浮点值、Linear/MatMul 的输入通道乘积和偏置。对所有不支持逐项重建的算子，不编造 dot-product 或输出值。

实现拆分：

- `compute_microscope.js`：纯数据模型、证据等级、边/形状/数值合同、矩阵乘加逐坐标闭合验证。**Node 和浏览器共享**。
- `compute_microscope_ui.js` + `compute_microscope.css`：统一只读渲染器；导航、依赖、数学、源码锚点、1–5 轴张量切片、颜色符号与坐标、GEMM 逐项计算；使用安全 DOM 文本接口，不执行上传内容。
- `SeetheTruthComputeMicroscope.learning(graph,run,L)`：主编辑器实际 `Learning.evaluate()` 和 `ConceptFlow.evaluate()` 输出的适配器。前者标成教学数值；后者**仅静态形状，无数值/假梯度**。
- `SeetheTruthComputeMicroscope.wan(data)`：固定 Wan 源码与受控 Float64 微型算术的适配器。Wan 专页不仅保留其 e3/e4/e5、残差旁路和 hidden-channel 删除的特化操作，也挂载了与普通积木完全相同的共用 Inspector。
- 所有适配器必须保留真实步骤顺序和每个输入的来源；不能将不同模型里“名字类似”的 FFN 脑补成同一套图。

## Trace schema v1

```text
schema = seethetruth-computation-microscope-v1
title = source-or-graph-title
evidenceTier = STATIC_SHAPES_NO_EXECUTION
             | EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS
             | PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64
origin = actual graph executor / explicit adapter identity
source? = {url: immutable github.com/owner/repo/blob/<40-hex-sha>/...,
           tier: PINNED_CODE_STRUCTURE_ONLY, revision, path}
steps = [
  {id, kind, title, formula, why,
   dependencies: [{from,port}],  // 必须指向更早已执行步骤
   out: {shape,values},          // values=null 意味着只推断形状
   operands: {name: {shape,values}},
   dot?: {type:dot,orientation:out_in,input,weights,bias},
   substeps?: [{id:parent_id:suboperator,kind,title,formula,why,out,operands,dot?}] // 有实算证据的内部阶段
   }
]
```

对所有数字输出：形状 rank 为 1–5，元素数有界、数值为 finite Float64、严格拒绝 ragged tensor。支持浏览最后二维的行/通道，允许显式选择前置轴的切片；列宽超出渲染上限时只显示前 24 列和前 20 行，并告知只截断 *可视化*（算术使用完整界内张量）。

对 Dot：输入 `X[T,D]`，权重 `W[H,D]`，Bias `b[1,H]`，输出 `Y[T,H]`。校验 `Y[t,h] ≈ b[h] + Σ_k X[t,k]W[h,k]`，容差是相对于输出大小的 1e-9，验证所有坐标；点选后显示所有维度参与的完整重建误差，表格最多展开前 64 项。MatMul 通过显式转置 B 适配同一公式。Linear 教学权重来自现有 `Learning.evaluate` 的固定、确定性哈希函数；此适配器只说明在**当前小模型解释器**里的权重，不能冒充原模型 checkpoint。

## Attention 作为复合算子的可展开示范

通用核心不把每个 Attention 整体当成一个无法透视的黑盒。使用主编辑器内实际 `Learning.evaluate()` 返回的 Attention `internals.scores` 和 `internals.probabilities`，Adapter 有独立数学核对的三个子步骤：

1. `QKᵀ / √d`：`Scores[Tq,Tk]`，将 Q 的输入除以 √d 再用 K 的 `[Tk,D]` 进行逐通道点积重建，对每个 query/key 坐标闭合；
2. 对 Scores 最后一维执行稳定 Softmax，展示真实概率矩阵及每 Query 的归一化情况；
3. `Probabilities × V`：逐输出坐标验证对所有 Value token 的加权求和。重建后的矩阵应严格对应当前 Attention 积木的输出。

对应浏览器内部按钮、局部矩阵、乘加表格必须直接来自执行器记录；没有内部运行证据的模型或算子不会自动被改写为 QK/PV。不同类型的 MLA、SWA、Chunk/FlashAttention、Diffusion Video Attention 需按真实计算流另行适配，不准仅依名字套用标准 Attention 结构。

## 不能跨越的证据边界

静态 shape 成立不等于数值前向成功。教学 Float64 成功不等于大型权重、Torch Autograd、CUDA kernel、FlashAttention、Tensor Core、服务时延或质量得到真实验证。Wan 适配器的源码结构有固定 Git commit，但内部数值仍然是合成输入/权重/Attention 边界；只有将来源绑定的真实 model forward/activation 通过另外的 provenance 校验和运行复核才能提高证据等级。普通模型的导入 JSON 只读统计数据不可自动成为完整 operands，不能生成虚假的逐元素 GEMM。

未知 evidence tier、无效依赖、错误维度/NaN/Inf、声称数值却缺少真实数组、与重算矩阵乘法不符的结果必须 fail closed；对超出后端支持的维度/真实模型不提供伪数据。新增模型适配器必须实现相同完整合同并通过负例审计。

## 最小工程验收

```bash
node tests/compute_microscope_contracts.js
python tests/browser_compute_microscope.py
node tests/wan_ffn_microscope_contracts.js
node tests/android_asset_manifest_contracts.js
```

四种互异组合积木：普通 FFN、SwiGLU、Attention、Residual；每种均需至少静态与数值两个证据档位。Linear/MatMul 全坐标重算闭合，模拟篡改输出失败；Wan 九阶段源代码 pin 与 dot 适配；4D/5D slice 数学检查。浏览器检查 PC 主编辑器由 Numeric 切换 Concept、GEMM 数值点击、替换为 SwiGLU，Android 尺寸的 Wan 通用面板与源码链接。Android 打包同一份 JS/CSS，无额外网络/模型训练/产物导出。

## 后续延伸接口（尚未实现）

1. 对现有所有 ND Attention、卷积、频谱处理、Loop 等内部多阶段算子构造语义正确的 operator substeps（不是盲目扁平）；
2. 数学计算和实际源码映射拓扑双层关联、真实设备读取已验证的模型 Trace，然后才可升级为真实权重的张量操作数；
3. 反向 VJP、Kernel tiling 和线程映射、显存访问顺序的独立取证与可视化，同时保持 CPU 教学为默认；
4. 任意模型积木的热插拔 adapter registry，并为每类 adapter 维护源码版本、数学 oracle、反事实测试和浏览器回归。
