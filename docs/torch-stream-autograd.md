# 流状态积木：PyTorch CPU 可微计算合同

Issue #42 / 2026-10-21 / 对应 PR #48

## 从计算流重建状态

令输入前缀 P 为 [2,3]，逻辑 token 绝对起点 origin=10，最大容量=5。状态初始化产生 version=0，nextPosition=12 的不可变逻辑状态。

两条独立后缀 A、B 形状均为 [1,3]，分别从该前缀 Fork 并在绝对位置 12 追加。两个分支的逻辑 nextPosition=13，version=1；历史前缀应相等，未来后缀可不同。

取窗口 [11,13) 时，对应数值输出：
- R_A = concat(P[1:2], A)，shape [2,3]
- R_B = concat(P[1:2], B)，shape [2,3]

Torch 导出器产生真正独立的 GraphRuntime.forward() Python 代码。类型信息与位置元数据单独存放；数值张量使用 clone、torch.cat 和切片，不调用 detach、不把中间输出硬编码，也不就地修改上游版本。

## 梯度必须如何汇聚

给两个分支不同的目标 Y_A、Y_B，都是 [2,3]；使用逐元素均值平方误差之和：

L = ||R_A - Y_A||_F² / 6 + ||R_B - Y_B||_F² / 6

因此有：
- dL/dP[0] = 0：这一行没有进入两个窗口；
- dL/dP[1] = (P[1] - Y_A[0]) / 3 + (P[1] - Y_B[0]) / 3；
- dL/dA[0] = (A[0] - Y_A[1]) / 3；
- dL/dB[0] = (B[0] - Y_B[1]) / 3。

这个等式能够区分两类错误：分叉执行时误做 detach 会丢失共享前缀梯度；把 Fork 当成覆盖原数组则可能破坏另一分支的历史状态。两种错误都不能通过仅检查两个分支的 shape 被发现。

## 独立计算与实验

运行命令：

    node tests/torch_stream_autograd_contracts.js

测试使用真正的 Python 3 / PyTorch CPU Float64 解释器，并做以下检查：

1. 对照 JS 工作台执行的十节点状态示例，逐节点比较前向数值、状态版本与绝对位置。
2. 对可训练前缀、可训练 A/B 后缀，用不依赖任何状态积木的两组 torch.cat + F.mse_loss 构造独立参考，逐坐标比较三个 Parameter 的 autograd 梯度；未读取行的梯度严格为零。
3. 执行两次真实 AdamW 更新、检查 Loss 降低；写入并用 weights_only=True 恢复参数和优化器动量状态。
4. 验证动态重算、Fork 内存隔离；拒绝过期写入、超容量、越界读取、特征宽度不一致、轴名错误、时间元数据损坏。
5. PC Chromium 真实操作验证：工作台可以导出独立 PyTorch 文件；Android 教学界面仍隐藏全部导出入口。

## 工程上必须保留的边界

- JS 学习画布现有流状态积木只是前向计算与形状/状态静态校验；Torch 真实梯度发生在导出的独立 Python CPU 图内，不能冒称 JS 图反向执行已经覆盖这些算子。
- 本次共享前缀的两支损失属于一次 DAG forward 的反向传播，不是跨多个真实推理调用的状态回放，也不是循环时间展开的 BPTT。
- Fork 在 PyTorch 中复制小型 CPU 张量且可回传梯度，不是物理 KV Cache 的引用计数、GPU 显存复用或 COW 分页，不得由此推断吞吐与内存性能。
- 状态载体限 [T,D]。虽然 tensor_ir 的独立数学实验室已有 rank 1–5 的具名轴语义，但在自由画布中统一处理高维状态端口、保存/恢复及动态图仍是未来独立任务。

关键实现：stream_graph_blocks.js、torch_stream_lowering.js、torch_codegen.js。
可参考 PyTorch 官方 Autograd (https://docs.pytorch.org/docs/stable/autograd.html) 与 Hugging Face KV Cache 文档 (https://huggingface.co/docs/transformers/cache_explanation)。
