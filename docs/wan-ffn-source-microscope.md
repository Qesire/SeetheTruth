# Wan2.2-I2V-A14B FFN：从官方源码到每一次点积

## 1. 可核查研究对象（禁止用“标准 Transformer”猜测）

本实验只绑定官方 [Wan2.2 @ 1ea34ff](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff) 的：

- [`wan/modules/model.py` L183–259](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py#L183-L259)：`WanAttentionBlock` 初始化及 Forward 顺序，`norm2`、FFN、时间调制和残差作用点。
- [`wan/modules/model.py` L465–490](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py#L465-L490)：time_embedding→time_projection→`e0` 形成→逐 Block 调用。
- [`wan/configs/wan_i2v_A14B.py`](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/configs/wan_i2v_A14B.py)：**C=5120、FFN H=13824、40 blocks、40 attention heads**；高噪声/低噪声阶段使用不同 checkpoint。

此处限定为官方核心 Wan I2V 路径；不要混入 `animate`、`s2v` 的特化 Block，也不要不经核对将 Diffusers 版本的实现细节断言为此处的源码行为。

**重点纠正：** 此分支的 FFN 是 *single-up GELU MLP*，不是 SwiGLU；其模块声明是 `Linear(C,H) → GELU(approximate='tanh') → Linear(H,C)`。AdaLN/time modulation 在 FFN 外部，残差在 FFN 后部。以上是已验证的代码结构事实；不代表本实验加载了官方训练后权重。

## 2. 实际计算顺序与张量语义

设某个 Block 在 Self-Attention residual 写回后的隐藏状态为 `x_self[B,L,C]`，Cross-Attention 产生的输出为 `cross_delta[B,L,C]`，时间投影 `e0[B,L,6,C]`，块参数 `M[1,6,C]`。先获得广播条件：

\[
[e_0,e_1,e_2,e_3,e_4,e_5]=\operatorname{chunk}_6(e0+M)
\]

其中公式左侧带下标 `e_i` 表示切分结果，不应与 time projection 的变量 `e0` 混淆。代码中 `e[3]` 对应 FFN shift，`e[4]` 对应 FFN scale，`e[5]` 对应 residual output gate。实际执行在 Block 内还有 Self-Attention 的 norm1+条件+Attention+gate，实验将此阶段视为明确的前置边界，而不是伪造其内部数值。

接下来依次计算：

\[
\begin{aligned}
x_c &= x_{self}+\operatorname{CrossAttn}(\operatorname{norm3}(x_{self}),context),\\
n&=\operatorname{LayerNorm}(x_c),\\
a&=n\odot(1+e_4)+e_3,\\
u&=aW_1^\top+b_1, \qquad W_1\in\mathbb{R}^{H\times C},\\
h&=\operatorname{GELU}_{\tanh}(u),\\
y&=hW_2^\top+b_2,\qquad W_2\in\mathbb{R}^{C\times H},\\
\delta&=y\odot e_5,\\
x_{out}&=x_c+\delta.
\end{aligned}
\]

这里的 `LayerNorm` 是 **无学习的 affine 参数、epsilon=1e-6** 的 WanLayerNorm，沿最后的 C 轴；中间 `float()` 和 `torch.amp.autocast(...dtype=torch.float32)` 的计算精度约束来自源码。具体 CUDA mixed precision kernel 是否/何时融合必须另行测量。浏览器数值实验用 Float64，不能宣称精确复刻 GPU 的舍入。

第一个 Linear 的单个 hidden 输出可钻取：

\[
u_{t,h}=b_{1,h}+\sum_{c=0}^{C-1} a_{t,c}W_{1,h,c}.
\]

下投影单个 output channel 同理：

\[
y_{t,c}=b_{2,c}+\sum_{h=0}^{H-1} \operatorname{GELU}(u_{t,h}) W_{2,c,h}.
\]

两个 Dense Linear 的解析乘加 FLOPs（每 multiply-add 为 2 FLOPs）约为 `4·B·L·C·H`。仅计算 projection；忽略 LayerNorm、GELU、modulation、bias、访存、launch、Tensor Core tiling 和 pipeline。官方 C/H 下，单 Token FFN GEMM 约 **283,115,520 FLOPs**，单 Block 两次投影权重与 bias 共 **141,576,704** 元素。不能直接把这个理论 FLOPs 当成实测延迟或者实际可节省比例。

## 3. 可视化教学协议

**[计算透视入口](../wan_ffn_lab.html)** 有 9 个按源码顺序的阶段（条件、Cross residual、Norm、Scale/Shift、Linear1、GELU、Linear2、Gate、Residual），每阶段能观察：

1. 输入和输出形状（小张量数值演示 vs 真实 Wan 官方 C/H），以及对应 `model.py` 的行号和表达式；
2. 逐 token/channel 矩阵热力图，点击格子可看到真实数值。两次 Linear 能进一步展示 `input[t,k] × W[out,k]` 每一项、bias 和数值重建误差；
3. 反事实操作：设置 `Gate 系数=0` 时，最终输出严格等于 Cross-Attention 后 x；变更 Scale/Shift，检查作用点。反事实改变的是教学控制，不是 Wan 官方修改后的 checkpoint；
4. 数据类型与来源边界：Cross-Attention 的增量是 **明示外部输入**，由确定性种子生成，不是伪装成真实 FlashAttention QKV 输出；调制与 FFN 的演示权重也不是官方权重。

## 4. 可证伪测试

```bash
node tests/wan_ffn_microscope_contracts.js
python tests/browser_wan_ffn.py
node tests/android_asset_manifest_contracts.js
```

Node 独立 reference 要从逐 Token 标量算式重新计算九步中的最终输出：多组 T/C/H 与 seed 数值一致，Linear1/Linear2 每个可选坐标点积完全闭合，LayerNorm 均值近似 0，GELU(tanh) 非线性不被错写成 ReLU 或 SwiGLU。无输出门控时 residual 恒等，对任意门控符号做线性对照。负样例拒绝非法 shape/seed。Chromium 桌面和 390px 手机尺寸分别截图验证源码映射、矩阵点击、Dot 解剖、参数改变与反事实效果。Android 仍无 Internet 权限、无导出和 Python 执行。

## 5. 进一步深化的方向（不是已经实现）

- 完整 Block 的自注意力/交叉注意力入口，包括真实 QKV、RoPE、Mask 与数据来源；沿着 `norm3` 和 Cross-Attention return 的代码路径继续可视化。
- 使用用户指定的真实模型权重与运行时 Trace，而不是合成数值：要核对官方/用户加载配置、dtype、timestep、block index、高低噪声分支及真实 activation/weight tensor，之后才允许标成实际 Wan 运行。
- 在 FFN 内增加训练时的反向图（GELU derivative、W1/W2 gradient）和 kernel/tiling 可视化，分清算术代价、带宽、调度效率与稀疏策略的真实加速条件。
- 以现有多模态积木为宿主提供“源码级展开”能力，逐篇论文复用相同的输入/输出/计算序列证据合同，而不把特例写死为统一架构。
