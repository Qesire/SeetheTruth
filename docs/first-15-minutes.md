# SeetheTruth · 第一次使用，15 分钟入门

采用 **问题 → 预测 → 实际操作 → 数值证据 → 下一问** 的学习路径。进入 Learning Studio 后，顶部的四步引导会定位真实计算节点、检查运行状态和展示实际张量形状；也可以随时自主探索其他课程。

## 一、提出 FFN 的结构问题

启动本地服务：

~~~bash
git clone https://github.com/Qesire/SeetheTruth.git
cd SeetheTruth
python atlas_server.py --port 8765
~~~

打开 http://127.0.0.1:8765/learn.html。顶部会显示四步学习引导，左侧默认课程是「从 FFN 开始」（`ffn`）。

**研究问题：** 为什么相同的 Token 经过上投影后通道从 4 扩展到 8，最后还要投影回 4？

点击引导中的「定位上投影」，查看 `up` 的输入来源、矩阵运算、输出形状。先预测：如果将中间通道数改变，哪些后续节点的形状也应改变？

## 二、用实际计算检验结构预测

在引导第二步点击「切换数值模式」，然后点击下方「运行」。执行完成后，现有计算图引擎将产生真实的教学数值轨迹，引导显示实际来源于本次运行的：

`x [3,4] → up [3,8] → down [3,4]`

**观察任务：** 在 `#studyMode` 选择数值并不等于已运行；必须等待 `#run` 得到本次 `inspectAccepted()` 的实际结果。用输入与输出形状核对第一步的预测。

## 三、追踪 SiLU 与下投影

第三步定位 `act`：结合计算公式 `SiLU(z)=z·sigmoid(z)`，观察上投影输出在激活后有哪些坐标发生改变。第四步定位 `down`：比较输入、中间通道与输出的实际形状，并解释两次投影各自的作用。

**新问题：** 如果移除中间激活，哪些数值关系会改变？可以先写出一个猜想，再利用自由画布构造简单的线性对照。

完成 FFN 引导后会进入「Attention 自己拼」（`attention`），按 Q/K/V → Scores → Softmax → Value 加权继续理解模型；然后进入「可训练参数 · 真正闭环」（`trainable_core`），观察 Forward、Loss、Backward 与 AdamW 参数更新。

**学习证据：** 区分形状推演、实际前向数值、梯度路径以及参数更新；让每一种结论都对应可选中的图节点与真实运行记录。

## 四、展开一个源码顺序对齐的 Wan Block

选择「Wan2.2 I2V · 完整 Self / Cross / FFN Block」（`data-preset="nd_wan_i2v_full_block"`）。当前示例是 115 节点的单 Block；在数值模式运行后沿着这些真实节点追踪：

`after_self → after_cross → norm2_ffn → ffn_gelu_tanh → after_ffn → loss`

查看 [完整 Wan Block 数学验证](wan-i2v-complete-block-verification.md)，理解为何原生整图有限差分与独立 NumPy/PyTorch FFN 闭包验证是不同证据。

## 五、自由图和导出

切换「自由实验」（`data-preset="blank"`），从基础算子构建自己的小图，练习断线与形状修复，然后在数值模式执行。对具备代码生成支持的子图，选择 **NumPy Python** 或 **PyTorch CPU** 输出真正可执行程序；遇到不支持的算子时查看明确拒绝原因。

**完成标准：** 能解释一个节点的输入、输出、函数、形状、来源、直接下游，以及本次运行里哪些数值被计算、哪些结果获得独立验证。

反馈请按 [CONTRIBUTING](../CONTRIBUTING.md) 记录课程 ID、步骤、预期结果、实际行为、浏览器环境与最小输入。

后续课程的提问与关键节点从同一张实际计算图推导。引导的来源与完成条件见 [交互引导协议](guided-learning-contract.md)。