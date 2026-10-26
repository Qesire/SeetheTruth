# SeetheTruth 领域覆盖图谱与五维独立数值证明

## 可浏览入口

打开 [积木能力与领域图谱](../capabilities.html)。页面的领域卡片和筛选来自当前运行时的三种实际来源：

- `SeetheTruthCapabilities.audit()`：受安装的算子 kind、类别和 NumPy/PyTorch **资格**；
- `SeetheTruthCapabilities.matrix()`：固定 fixture 的源码绑定 PyTorch CPU **实际案例证书**；
- `SeetheTruthUnifiedCore.inventory().labs`：全部受注册的专项实验室、入口、底层引擎和声明证据。

页面按研究用途提供五个**可交叠**导航领域：语言模型与 Transformer、图像/音频/视频与时空计算、训练/优化/后训练、推理引擎/缓存/调度、源码/论文/运行证据。每个领域显示实际匹配的注册 kind 数、可导出候选数、已测案例种类数以及对应专项实验室链接。单击卡片筛选同一页的真实注册表；其下还保留每一个原生算子类别，数量总计应与注册表一致。

领域标签的关键词是导航分类，不是假设所有该领域模型均可执行、可微或已由真实模型验证。例如现有视频语义引擎提供真实的有界 `B,T,H,W,C` 操作，但这**不等于** Wan 任意视频生成网络已拥有 PyTorch 导出支持。专项实验室的注册证据、实际执行证据和模型证据继续分别标注。

## 五维计算流：真实执行与数学定义

对于有界的 `X[B,T,H,W,C]`，选取 `[1,3,2,2,2]`、每通道偏置 `b[C]` 和逐时间门控 `g[T]`：

```text
  X[B,T,H,W,C]        b[C]         g[T]
          \             /             /
           named-axis add             /
                   \                  /
                    named-axis multiply
                             |
             slice_T([0:2]), slice_T([2:3])
                             |
                          concat_T
                             |
                reorder B,C,H,W,T
                             |
                inverse reorder B,T,H,W,C
                             |
                    mean of all Y^2
```

逐坐标定义：

```text
  U[b,t,h,w,c] = X[b,t,h,w,c] + beta[c]
  Y[b,t,h,w,c] = U[b,t,h,w,c] * gain[t]
  L = sum(Y^2) / N
  D[b,t,h,w,c] = 2 * Y[b,t,h,w,c] / N
  grad_X[b,t,h,w,c] = D[b,t,h,w,c] * gain[t]
  grad_beta[c] = sum_(b,t,h,w) D[b,t,h,w,c] * gain[t]
  grad_gain[t] = sum_(b,h,w,c) D[b,t,h,w,c] * U[b,t,h,w,c]
```

`tests/semantic_5d_source_oracle.js` executes the existing `TensorIR/SemanticState` engine, then separately reconstructs every output with flat scalar loops. It checks the complete forward values, time slicing/concatenation and both axis-reordering passes; wrong named-axis reinterpretation, bad dimensions, invalid permutation and nonfinite inputs are explicit negative cases.

`tests/python_semantic_5d_reference.py` independently implements scalar loop formulas, executes separate native **NumPy CPU** and **PyTorch CPU** programs and compares every observed value and all original-input, bias and time-gain gradients. Every parameter coordinate receives a separately evaluated central finite difference. The Python backend uses real `torch.float64`, with `torch==2.6.0+cpu` pinned in CI; the Heavy job does **not** depend on the Fast job.

**Proof boundary:** these are actual CPU numerical and reverse-mode oracles for the existing named-axis 5D semantic engine, **not** generated Python exporters for `semantic_video_*` (these kinds are not advertised as backend export eligible). The independent oracle's function is to compare the same mathematical operations with the existing engine. Future expansion to generated 5D code must first implement legitimate exporter lowering and a genuinely executable, graph-scoped parity witness.

## What remains

1. Move from a single named `B,T,H,W,C` case to adversarial shapes and alternate axis orderings; verify immutable temporal streaming and replay at the same granularity.
2. Extend the same independent CPU proof workflow to complete, actually composed Wan Block computations, including AdaLN, Cross-Attention, FFN and accumulated gradients.
3. Expand per-operator source-pinned certificate cases without treating a declaration, export allowance or same-engine test as general numeric correctness.
4. Keep the domain map source-linked; new registered specialist IDs must be assigned to one navigational domain, while native categories remain automatically enumerated.

### Verification

```bash
node tests/semantic_5d_source_oracle.js
node tests/semantic_5d_cross_backend.js  # Python 3.11, NumPy, torch 2.6.0+cpu
python tests/browser_capabilities.py      # Playwright Chromium installed
```

CI runs the Fast/Heavy tests in `.github/workflows/math-verification.yml`, and the real PC/mobile browser contract in the existing static acceptance suite.
