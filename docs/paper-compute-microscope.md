# Paper-to-Operator Microscope：从论文改写到逐项可核算的计算流

SeetheTruth 的论文学习单元共用 `Compute Microscope`，不再停在“某论文把一个积木替换成另一积木”的图标层。本单元先运行已有的 `paper_patches.js`，保存必要的真实中间值，使用 `paper_compute_trace.js` 转为公共 Trace 合同，并把每一步与设计机制、可操作对照关联。

## LoRA：低秩训练参数如何改变实际前向

[Y=XW_0+\frac{\alpha}{r}(XA)B]

执行次序：
`X,W0,A,B → XW0 → XA → (XA)B → (α/r)(XA)B → Y`。

- **可钻取**：`XW0`、`XA`、`(XA)B` 每个输出坐标的全部输入通道乘积和权重。
- **机制变量**：`r` 控制可训练更新的中间维度，`α/r` 控制更新幅度；改变 `α=0`，输出应严格回退到基座。
- **最小对照**：用相同 `X,W0,A,B` 比较 `α=0,1,2,-2`，检查输出增量是否线性缩放、冻结基座是否完全不动。比较大/小 `r` 必须同时改变实际矩阵形状，不能仅改一个字符串参数。

## DoRA：权重方向与幅度不是同一自由度

[
V=W_0+\frac{\alpha}{r}AB,qquad
W_{\mathrm{eff}}[:,j]=m_j\frac{V[:,j]}{\|V[:,j]\|_2},qquad
Y=XW_{\mathrm{eff}}
]

执行次序：
`A,B → AB → ΔW → V → 每个输出列的 L₂ 范数 → W_eff → XW_eff`。

- **可钻取**：`AB` 的每项乘加、`V` 每个权重坐标、列归一化数值、有效权重、`XW_eff` 的最终乘加。
- **最小对照**：固定方向 `V`，只改变 `m_j`；对应有效权重列范数应跟随 `|m_j|` 改变。固定 `m`、改变 `A/B` 时，输出差异只能通过方向矩阵和后续重归一化传递。
- **计算语义**：DoRA 修改权重的表示法及最终进入线性层的有效权重；不会直接把 LoRA 的输出残差当作 DoRA。

## Online Softmax：改变计算顺序而保持数学结果

每个 Query 依次遍历 Key/Value tiles。设 tile 分数 `s`、历史最大值 `m_old`、历史归一化常数 `l_old`、历史加权累积器 `a_old`：

[
m'=\max(m_{\rm old},\max s),\quad c=\exp(m_{\rm old}-m'),\quad p_j=\exp(s_j-m')
]
[
l'=c\,l_{\rm old}+\sum_jp_j,\qquad a'=c\,a_{\rm old}+\sum_jp_jV_j,\qquad o'=a'/l'.
]

首块 `m_old=-∞` 对应 `c=0`，由参考执行器执行初始化分支；可视化没有将 `-∞` 偷换成某个伪造的有限数值。

- **可钻取**：每个 Query/Tile 的真实 `QK_tileᵀ/√d` 点积、最大值更新、未归一化指数权重、分母、Value 累积器、局部输出。
- **独立数值校验**：`paper_compute_trace.js` 另走一次完整 `QKᵀ→稳定 Softmax→PV` 数值计算，对照 `paper_patches.onlineAttention` 的输出；tile 宽度为 1、2、3 时都要满足逐元素误差阈值。
- **最小对照**：只改变 Key Tile 长度，不改变 `Q,K,V`；观察递归内部过程和 Tile 次数怎样变化，以及最终输出是否保持一致。这一对照将*计算安排*与*数学函数*区分开，不把编排改变误认为模型功能发生改变。

## 通用接口与测试

`paper_compute_trace.js` 不独立实现第二套假前向：对输入调用已有执行器 `paper_patches.js`，核对传入的全部结果及中间值，再把它们映射到仓库共享 `compute_microscope.js` 和 `compute_microscope_ui.js`。在可完整分解的矩阵乘法上，公共合同独立重算每一个 `XW` 坐标。对于在线归一化阶段，保留逐项标量状态与 Value 矩阵，不会伪装成 GEMM。

```bash
node tests/paper_patches_contracts.js
node tests/paper_compute_trace_contracts.js
node tests/universal_surface_audit_contracts.js
python tests/browser_paper_compute_trace.py
node tests/android_asset_manifest_contracts.js
```

当前方法比较重点是数值机制和计算次序；显存真实分配、GPU tile occupancy、特定 FlashAttention-2 fused kernel 的资源使用属于独立的内核运行证据层，后续应当用实际 profiler 记录补入，而不是从微型递推数据推断。

接下来从这个公共算子数据合同扩展至：激活重算时的前向重演与梯度精确一致性，以及 MoE 不同容量/损失选择的局部反向传播研究。
