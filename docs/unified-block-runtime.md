# SeetheTruth 的统一积木运行时（Unified Block Runtime）

**设计原则：一切论文方法、模型计算和多模态操作首先是可组合的积木，不是一个新的独立页面。** 专用实验页可以保留为教学入口，但它所使用的核心数值和计算解释必须由自由画布中的模块复用。PC 和 Android 教学版加载相同的模块包、数学验证和透视结果。

## 分层架构

```text
独立模块包 ──(definitions / recipes / numeric execute / trace)──┐
旧版原生积木 ───────────────────────────────────────────────────┤
显式流状态积木 ─────────────────────────────────────────────────┤
                                                               ▼
         Block Registry → Graph JSON → Port/Shape Inference → DAG
                                                               │
                                                               ▼
                       CPU Numeric Executor / Concept Interpreter
                                                               │
                      ┌────────────────┴─────────────────┐
                      ▼                                  ▼
                Compute Microscope                 State Microscope
             1–5D/逐坐标/点积复核                 时间、事件、因果与状态
                      │                                  │
                      └──────────────┬───────────────────┘
                                     ▼
                          PC Web / Android Teaching
```

### 语义化端口：不仅比较形状，也比较轴的身份

需要严格轴语义的模块可以在 `def` 上额外声明可选的 `portAxes` 与 `outputAxes`：

```js
{
  ports: ['x', 'bias'],
  portAxes: {x: ['T','D'], bias: ['One','D']},
  outputAxes: ['T','D']
}
```

此合同由 **同一模块注册器**在 `infer` 与 `execute` 阶段实施。上游结果的 `axes` 被写入计算图的执行记录，随后透传给 `ConceptFlow`、`GraphReadiness` 和最终的数值执行器。输入没有对应轴标注时，不允许直接绕过正常图运行器执行有轴要求的模块。

**验证案例**：一个 `3×3` 的 `axis_named_transpose` 结果实际拥有 `['D','T']`，把它接到要求 `['T','D']` 的 `axis_time_mean` 时，即使数值形状完全相等，概念推导、运行就绪检测和真实数值执行也都会拒绝。该案例独立于数据取值，直击“形状一样却误接轴”的机制性错误。

### 模块的最小完整合同

模块包位于独立的 `*_block_pack.js`，且在 `block_extension_registry.js` **集中安装**。UI 及 `learning_lab.js` 通过公共接口调用，不新增每个 `kind` 的 `switch`：

| 接口 | 输入 / 输出 | 目的 |
|---|---|---|
| `pack.name` | 不重复的稳定标识符 | 避免不同学术方法使用同名算子 |
| `pack.defs` | `kind → {title,en,category,ports,params,formula,why,lesson,caution,source}` | 积木形状、端口、默认参数、数学机制和来源 |
| `pack.validate(kind,params)` | 参数验证 | 绑定的参数域与执行合同一致 |
| `pack.infer(kind,params,inputShapes)` | 推导输出 1–5D 形状 | 在未运行数据时判断是否可连接 |
| `pack.execute(kind,params,portValues)` | `{values,internals?,flops?}` | 从真实上游连线执行完整小张量，而非预设视频或硬编码结果 |
| `pack.trace(kind,params,inputs,run,node)` | 计算显微镜子算子数组 | 保留来自同一执行器的真实操作数、公式和逐坐标映射 |
| `pack.recipes` | `[{id,title,summary,graph:{title,nodes,edges}}]`（可选） | 模块包提供组合课程；无需手改自由画布课程列表 |

注册器统一检验重名、端口元数据、源链接、数值界限及静态推断／真实输出一致性。输入数据的张量形状由实际嵌套数值检测，不凭用户传入的 `shape` 声明；子算子继续经过 `Compute Microscope` 的数值闭合验证。示例工程读取时**深拷贝**，修改图和参数不会污染已安装模块包内的原始示例。

**重要的设计区别：** `pack` 是*数学原语及其合同的发布单元*；`recipe` 是*如何连接一组原语的图*；`page` 是*如何教学与解释这个图的界面*。新增研究论文时，优先询问能否重写成已有原语的组合图；需要新原语才扩展一个 pack；无需因此增加新的执行器或 UI。

## 现有跨领域积木包

| 积木包 | 积木种类 | 已实际实现的计算 | 可迁移的机理实验 |
|---|---:|---|---|
| `semantic_video` | 3 | 五维 `[B,T,H,W,C]` 数据源、按 `T` 切片、沿 `T` 拼接 | 检查完整帧块的顺序、边界和每个像素原值 |
| `paper_operators` | 3 | LoRA `XW₀+(α/r)XAB`、DoRA 有效权重、分块 Online Softmax | 冻结基座／调整低秩更新；权重列幅度／方向分离；改变 Tile 宽度但结果不变 |
| `named_axis` | 5 | `[T,D]` 源、按 `D` 生成 Bias、严格轴广播、`T/D` 置换、沿 `T` 均值 | 两条转置路径（注册积木 vs 原生 `Transpose2D`）最终数值差分必须全零 |

合计 **11 种新增独立模块包积木**。与原生 11 种、原有计算库 123 种及流状态 4 种一起，当前共有 **149 种注册积木，22 个类别**。这些数字是代码注册与已实现引擎的实际统计，不代表每种积木都有 GPU 内核、完整反向传播或代码生成后端。

## 已完成的通用学习流程

新 `named_axis` 模块包把 `extension_named_axis_composition` 作为 recipe 自带安装。用户直接在已有自由编辑器中选择该任务，连接状态完全可编辑：

```text
                         ┌─ axis_named_transpose ─┐
axis_tensor_source ─┐     │                        ├─ subtract → 逐坐标全 0
                    ├─ axis_named_add ─────────────┤
axis_feature_bias ──┘     │                        └─ 原生 transpose2d ───┘
                          └─ axis_time_mean → [1,D]
```

具体数值由 `TensorIR`、`SemanticState` 及原生 `transpose2d` 独立计算：
1. 每个 `[t,d]` 位置必须满足 `Y[t,d]=X[t,d]+bias[d]`。
2. `axis_named_transpose` 与原生 `Transpose2D` 产生相同数组，差分为全零。
3. 时间均值只沿 `T` 累加，不能错误地在 `D` 上求平均。
4. Bias 宽度不等于 `D` 时，运行必须明确拒绝，而非猜测轴身份。
5. recipe 装载返回独立副本，改变 `seed` 或 Bias `scale` 后可重算，不污染注册源。

## 模块发布验收门禁

新的模块包在合并之前，至少通过：
- **合同**：有效输入、不同参数、断开的端口、非法形状和非法维度的正负例。
- **数值**：单模块输入／输出 oracle、内部点积或索引映射的独立复核，以及可区分两种机理的参数对照。
- **组合**：新模块与旧模块互连，验证前后模块都使用真实执行器，不能只在专用页面运行。
- **运行时**：静态推导、数值结果、公共显微镜输出和知识卡共享同一个积木 `kind`。
- **客户端**：PC Chromium 页面、Android 教学模式和离线 APK 资产必须包含同一模块包，无 Android 专用假数据实现。

```bash
node tests/axis_block_pack_contracts.js
node tests/block_extension_registry_contracts.js
node tests/block_catalog_contracts.js
node tests/stream_graph_blocks_contracts.js
python tests/browser_block_extension_registry.py
node tests/android_asset_manifest_contracts.js
```

下一步的扩展遵循同样的注册/校验/数值/透视协议：`audio` 的 STFT→LogMel、`video` 的真实局部卷积/时空打包、`attention` 的 KV 读写和张量轴变换、`training` 的逐边 VJP／优化器等。首先复用仓库中的数值后端，只有缺失真正原语时才增加独立模块包。
