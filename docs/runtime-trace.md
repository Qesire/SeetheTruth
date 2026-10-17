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


## SOURCE ↔ RUNTIME 归因：逐调用的源码证据（2026-10）

本功能默认关闭；只有可信、由用户控制的 Python 脚本显式提供
`RuntimeTrace(model, source_root=source_directory)` 才采集运行帧的源码位置。
输出仅包含相对于 `source_root` 的 Python 文件名、行号、SHA-256，
不序列化目录绝对路径或源码文本。追踪不执行额外的 forward、backward 或 step，
但额外的堆栈检查和文件哈希会影响性能，建议先用 CPU 小模型验证。

建议在**同一份未修改的源码快照**上生成两个图。静态侧：

```python
from atlas_parser import parse_paths
from pathlib import Path
import json
root = Path("my_code").resolve()
g = parse_paths([str(root)], entry="Toy.forward")
Path("static.json").write_text(json.dumps(g, ensure_ascii=False))
```

运行时侧，在自己明确允许执行的代码中插入：

```python
with RuntimeTrace(model, record_gradients=False, source_root=root) as trace:
    with torch.no_grad():
        _ = model(example_input)
trace.save("runtime.json")
```

然后**在运行代码外部**执行：

```bash
python source_runtime_join.py --static static.json --runtime runtime.json --out joined.json
```

在 SeetheTruth 中导入 `joined.json`，查看每一个有证据的 Runtime 节点上的
SOURCE ↔ RUNTIME 匹配。所有来源信息都经过 HTML 转义处理。

匹配是保守的，并要求同一个 AST 节点具有完全一致的
**文件相对路径 + SHA-256 + 行号**，还必须满足唯一的算子语义，
或（Module Hook）一致的 `self.<attr>` 调用名称和真实 `module_type`。
重复执行一个调用点会生成多条运行事件，`invocation_index` 保留调用顺序；
绝不会用调用顺序替代源位置校验。

状态包括 `matched`、`unmatched`、`missing_location`、
`source_mismatch` 和 `ambiguous`。当静态/运行任意图发生截断时，
**整次匹配会被阻止**。没有匹配的 ATen 不代表运行错误，
可能由框架内部、动态调用、未提供的源文件或源码行多个操作造成。
本功能不推导未执行分支、CUDA Kernel、完整动态控制流或 GPU 用时，
也不将来自另一单独 Profiling 会话的统计数据强行关联到某个调用点。
源码归因不是训练机制、模型设计意图或完整 Autograd 证据。
