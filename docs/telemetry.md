# Performance & Tensor Observatory

TelemetrySession 是完全显式启用的 PyTorch profiler + 有界张量采样功能。静态分析服务器绝不执行模型源码。

## 运行

    python examples/telemetry_demo.py
    python atlas_server.py --port 8765

打开 http://127.0.0.1:8765/telemetry.html ，导入 examples/telemetry_demo.json。

在可信的用户训练脚本中：

    from telemetry_capture import TelemetrySession
    with TelemetrySession(model, capture_values=False) as profile:
        output = model(batch)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    profile.save("profile.json")

数值采样默认关闭。开启 capture_values=True 时仅抽取有限的激活值、直方图和最多 20×20 的矩阵切片；不保存权重或完整张量。GPU→CPU 复制可能同步设备，导致性能数据受到干扰。对于科研基准，建议分离无采样性能测量和采样型探索两次 pass。非受信代码绝不可由分析服务器自动执行。

## 证据与口径

- operator.self_cpu_ms: torch.profiler 对自 CPU 时间的聚合，不是端到端延迟。
- operator.self_device_ms: 如果 profiler 观测到 CUDA，统计设备归因时间；不能被称作 GPU utilization。
- module.host_wall_ms: Python pre/post hooks 包围的 Host 执行时间，含子模块，不可逐层求和，也不是 GPU 完成时间。
- gpu_memory.peak_allocated_bytes_since_last_reset: 上次外部 reset 以来的峰值，未必是本次新增峰值。
- tensor.logical_bytes: numel × element_size，可能共享 storage。
- Histogram、sample_zero_fraction 和 Matrix: 有界统计样本，非整个张量精确分布。高阶张量显示前导轴固定为 0 后的最后两轴间隔采样。
- Profiling 本身会增加内存/时间开销；单次 pass 不应当作为稳健统计结论。
- 硬件 MFU/HFU、GPU kernel 启动时间线、实际多卡通信和 GPU 关键路径尚未实现。

数据契约 schema 为 seethetruth-telemetry-v1，包含 meta、operators、modules、snapshots、gpu_memory。前端的示意样本明确标为 SYNTHETIC。

后续应把 Module path、ATen 代价与 3D graph 的真实节点进行同源映射，并增加多 step/timestep 对比；当前是独立观察视图。


## 同一次执行：SOURCE / ATen / Module Profiler 联合证据

新接口 `same_pass_evidence.VerifiedPass` 将 `RuntimeTrace` 与
`TelemetrySession` 放入**同一次、由用户自己发起的可信模型执行**。它不会自动
调用模型，也不会默认采集张量值。可以直接运行 CPU 最小示例：

```bash
python examples/same_pass_demo.py
python atlas_server.py --port 8765
```

示例输出 `examples/same_pass_graph.json` 和
`examples/same_pass_telemetry.json`。前者导入 SeetheTruth 主界面；
后者导入 Telemetry 页面。这两个文件的 `run_id` 一致，来自同一插桩上下文。

在真实模型中：

```python
with VerifiedPass(model, source_root=source_directory,
                  capture_values=False) as evidence:
    with torch.no_grad():
        result = model(example_input)   # 仅执行用户写出的这一次 forward
bundle = evidence.save(graph_path="run-graph.json",
                       telemetry_path="run-telemetry.json",
                       static_graph=static_ast_graph)
```

### 计时方法与父子关系

PyTorch Profiler 的 ATen CPU 事件按照 `cpu_parent` 归给**最近的**
`SeetheTruth.module/.../invocation/N` 注释；每个算子事件仅归给
**一个独占模块调用**。再通过真实的嵌套模块调用树进行自底向上聚合，
得到每个模块的两种不同成本：

| 字段 | 口径 |
|---|---|
| `exclusive_self_cpu_ms` | 直接归属该模块调用的 ATen 自 CPU 时间；不含子模块 |
| `inclusive_attributed_cpu_ms` | 该模块本身及下层模块的 ATen 自 CPU 时间 |
| `exclusive_operator_events` | 独占 ATen 事件个数 |
| `inclusive_operator_events` | 包含下层模块的 ATen 事件个数 |
| `host_wall_ms` | Python Hook 包围的 Host 时间；不是 GPU 时间 |
| `exclusive_self_device_ms` | 若有 CUDA，仅为 Profiler 对 ATen 的设备时间归因 |
| `inclusive_attributed_device_ms` | 含下层模块归因设备时间；**不是 GPU 关键路径** |

**可加和的是同一执行中各模块调用的 exclusive 部分**（前提是
Profiler event parent 归属成功），而父子模块的 inclusive 时间会重叠，
不能在总成本中重复加和。模块以 `name`、`type`、`invocation_index`
及输入输出的 `shape/stride/dtype` 核实匹配；不同执行轮次或签名不一致
将拒绝声称精确关联。`module_calls` 保留每次调用，`modules` 保留聚合视图。
丢失归属的 `aten::` 事件计入 `meta.unattributed_operator_events`。

这种归因仍不等于 CUDA Kernel 的设备执行区间、异步流重叠程度、
端到端时延或未插桩的真实性能。一次 Pass 的 Python/Profiler 仪器化
会引入可测量干扰；如需设备实际关键路径，须另行获取 CUDA Activities
的关联事件和同步时序，不能直接将 CPU 计时当 GPU 耗时。

当前假定同步、单线程、平衡的 `nn.Module` 前后 Hook；
`torch.compile`、重入多线程、跨进程/分布式和异常中断的调用
不在精确归因承诺内。Runtime 图截断时停止同次 Pass 关联。
静态 AST → Runtime 的 SHA/行号匹配仍独立执行，不因此声称
源码中所有数学步骤都被实测。
