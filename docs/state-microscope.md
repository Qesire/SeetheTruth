# 通用事件／状态显微镜 · State Microscope V1

SeetheTruth 的“深入计算”并不等于所有功能都画成矩阵热力图。运行时的 Scheduler 决策、前缀缓存、Optimizer State、Prefill Chunk 和事件资源依赖，必须共用**可验证的事件／状态转移合同**，而不是人为伪造张量层。

## 接口

统一由 `state_microscope.js` 定义：

```js
{
 schema: "seethetruth-state-microscope-v1",
 title: "...",
 evidenceTier: "...",                // 严格枚举
 steps: [{
   id: "node:index",
   index: 0, title: "...", why: "...",
   dependencies: ["earlier-step-id"], // 仅可引用更早出现的已验证步骤
   before: { /* typed state fields */ },
   after: { /* typed state fields */ },
   changes: [{key,before,after}],   // 必须与真正的结构化前后差分逐字段一致
   timing: null                    // 假设事件仅传显式 {start,end,resource}
 }],
 disclaimer: "..."
}
```

校验拒绝：未知证据级别、重复事件、指向未来的依赖、缺失解释、NaN/Inf、过深/过大的状态、不一致的 `changes`、不合法时间区间。通用的 `state_microscope_ui.js` 只用 DOM textContent 渲染，不执行来自 JSON 的 HTML / JS。浏览器只负责渲染；数值 oracle 属于来源适配器，不能在面板里伪算。

## 已有 7 个真正接入的页面与 6 个来源适配器

| 页面 | 适配器 | 验证根据 |
|---|---|---|
| `deepinfer_kv_lifecycle_lab.html` | `kv(run, K)` | `deepinfer_kv_lifecycle.js` 每次操作通过引用计数和物理空闲队列合同；每步独立验证 |
| `deepinfer_worker_lab.html` | `worker(run)` | `deepinfer_worker_events.js` 的实际逻辑事件；逐条校验前驱、CPU/GPU 单资源互斥和假设时长 |
| `engineering_pipeline.html` | `training(snapshots)` | 真实 Bigram 教学训练器产生的模型/采样游标/待提交梯度的快照；是教学器的状态差分，不代表训练大模型 |
| `training.html` | `scalar(run)` | `training_flow.js` 的精确标量 Forward/Backward/AdamW 算术事件；纯概念模式关闭数值透视 |
| `deepinfer_scheduler_lab.html` | `scheduler(run,K)` | Decode Scheduler 决策必须与 `KV.revision`、FIFO、refs、Worker 序列顺序、Dispatch Block Table 以及特定 Forward Event 同时一致 |
| `deepinfer_prefill_lab.html` | `prefill(execution,P)` | NORMAL 的 ALL/ANY 与第二轮投票，及 IMG/PAD Chunk 决策，分别用官方源码对照的 CPU 参考引擎重新计算、全结果对比。**两步是独立决策，不添加假前驱** |
| `cache.html` | `cache(snapshots,Cache)` | 从相同配置和空状态重启真实小张量 GQA/RoPE Session，逐步骤重新产生完整 Snapshot 并进行全字段 byte-equivalent JSON 比对，包括 Key/Value 数值、形状、Token Cursor、Attention 结果与历史 |

因为一条通用镜头可以呈现多种不同类型的有效证据，适配器数不等于模型能力数。实际 GPU KV 分配、跨 Rank 分布式投票、Tensor Core、Attention 加速、生产吞吐、真实模型权重和物理时延均**未执行**，严禁从上述参考数值推断。

## 对“真正通用”的工程要求

1. 先确定真实执行器的语义边界、状态结构、事件排序，再实现 adapter，禁止复制一套独立模拟替代原运算。
2. 为输出和依赖选择最精确的校验：KV 拥有 refs 守恒；Scheduler 必须把派发与 Worker 接收绑定；Prefill 必须重算完整决策；数学 KV 必须重放所有 K/V 内容，而不是只比较计数。
3. 任何只包含 `shape` 或只有吞吐摘要的外部文件不能伪装成已执行矩阵算术；真实设备测量必须由独立的 Provenance 签名和数据采集链升级证据等级。
4. 不支持的功能继续列在 `universal_surface_audit.js` 中，明确将其归为 specialized。任何将旧页面迁移进通用平台的改动必须同时提供正例、故意伪造的数据负例、浏览器操作、Android 资产检查，并将审计状态改成真实模块引用。

## 验收入口

```sh
node tests/state_microscope_contracts.js
node tests/universal_surface_audit_contracts.js
python tests/browser_state_microscope.py
node tests/android_asset_manifest_contracts.js
```

这些不是完整系统发布的唯一 Gate；整个 CI、Android 离线 APK 打包、工作台浏览器验收仍然是发布阻塞项。
