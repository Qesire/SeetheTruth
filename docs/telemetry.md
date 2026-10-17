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
