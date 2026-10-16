# Runtime Tensor Trace（实验性）

## 设计边界

- **静态解析**：atlas_parser.py 只读 Python AST，永不运行模型代码。
- **运行时追踪**：runtime_trace.py 由用户在自己的可信 PyTorch 脚本里显式开启，服务器不调用。
- 只序列化 ATen 名称、module 名称、shape、dtype、stride、device、梯度元数据与参数版本是否改变。**不会保存张量数值、梯度值、权重或者 GPU 指针。**
- 追踪到 ATen != CUDA kernel 启动；梯度 hook != 完整 autograd DAG。
- max_events 限制事件，meta.truncated/dropped_events 报告截断。插桩可能影响性能与编译/自动微分行为。

## 使用

安装 PyTorch，运行以下命令：

    python examples/runtime_trace_demo.py
    python atlas_server.py --port 8765

访问 http://127.0.0.1:8765 ，点击 Graph JSON 导入生成的 examples/runtime_trace_demo.json。

在可信训练代码中：

    from runtime_trace import RuntimeTrace
    with RuntimeTrace(model, max_events=600) as trace:
        out = model(batch)
        loss = loss_fn(out, target)
        trace.backward(loss)
        trace.step(optimizer)  # 调用一次真正的 optimizer.step
    trace.save("observed_graph.json")

**不要**在训练代码已经执行 optimizer.step 之后再次调用 trace.step，以免重复参数更新。对于不需要更新的推理，直接将 forward 包在 trace 中即可，推荐 no_grad。

RL 训练时真实 rollout、reference logprob、policy loss 只有在执行对应代码后才会被观察到；不会仅凭名称断言 PPO、GRPO 或环境交互。

## 证据级别

| 级别 | 证据 |
|---|---|
| 教学模板 | 经典架构的逻辑数学分解 |
| 配置 | 声明的超参数及推断 |
| AST | 源码中存在某个语法/调用 |
| Runtime Trace | 单次执行中观察到的 ATen 和张量元数据 |
| Profiler（未实现） | 实际设备 kernel、耗时与显存行为 |

本阶段未实现 torch.compile 展开、GPU kernel 时间线、分布式跨设备收集或完整梯度雅可比链。
