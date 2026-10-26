# SeetheTruth · 第一次使用，15 分钟入门

采用 **问题 → 预测 → 实际操作 → 数值证据 → 下一问** 的学习路径。进入 Learning Studio 后，顶部的四步引导会定位真实计算节点、检查运行状态和展示实际张量形状；也可以随时自主探索其他课程。

## 一、提出 FFN 的结构问题

~~~bash
git clone https://github.com/Qesire/SeetheTruth.git
cd SeetheTruth
python atlas_server.py --port 8765
~~~

打开 http://127.0.0.1:8765/learn.html。左侧「从 FFN 开始」对应 `data-preset="ffn"`，默认展示 `Linear → 激活 → Linear`。观察为什么需要中间通道和非线性，选中各个节点查看 shape、输入来源及下游依赖。

## 二、执行计算来检验结构推断

将学习模式选择器 `#studyMode` 改为微型数值（`numeric`），点击 `#run`。状态栏 `#status` 显示执行状态。选中算子，按阶段检查张量与计算；修改可编辑参数后再次运行。

目标：让 `Y=W₂ φ(W₁ X+b₁)+b₂` 的每一项都能对应到一个明确的计算节点。

## 三、观察 SiLU 和下投影，然后迁移到 Attention

选择「Attention 自己拼」（`data-preset="attention"`），逐一观察 Q/K/V、缩放点积、Softmax 和 Value 汇聚。接着选择「可训练参数 · 真正闭环」（`data-preset="trainable_core"`），寻找 Forward、Loss、Backward、AdamW 更新及更新后的 Forward。

目标：分辨 *知道张量形状*、*真正执行前向*、*已验证某个 Loss 路径梯度* 和 *真实修改参数*。

## 四、展开一个源码顺序对齐的 Wan Block

选择「Wan2.2 I2V · 完整 Self / Cross / FFN Block」（`data-preset="nd_wan_i2v_full_block"`）。当前示例是 115 节点的单 Block；在数值模式运行后沿着这些真实节点追踪：

`after_self → after_cross → norm2_ffn → ffn_gelu_tanh → after_ffn → loss`

查看 [完整 Wan Block 数学验证](wan-i2v-complete-block-verification.md)，理解为何原生整图有限差分与独立 NumPy/PyTorch FFN 闭包验证是不同证据。

## 五、自由图和导出

切换「自由实验」（`data-preset="blank"`），从基础算子构建自己的小图，练习断线与形状修复，然后在数值模式执行。对具备代码生成支持的子图，选择 **NumPy Python** 或 **PyTorch CPU** 输出真正可执行程序；遇到不支持的算子时查看明确拒绝原因。

**完成标准：** 能解释一个节点的输入、输出、函数、形状、来源、直接下游，以及本次运行里哪些数值被计算、哪些结果获得独立验证。

反馈请按 [CONTRIBUTING](../CONTRIBUTING.md) 记录课程 ID、步骤、预期结果、实际行为、浏览器环境与最小输入。