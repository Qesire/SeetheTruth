# SeetheTruth VLM · 发布候选审查记录

**日期：** 2026-11-11　**目标：** 本地 Web 教学预览 / 离线 Android 测试版  
**来源分支：** `feat/vlm-unified-learning-20261111`，PR [#105](https://github.com/Qesire/SeetheTruth/pull/105)  
**上轮已验证提交：** `65f5011663bbecb869f34b228f671f55b2ca4738`。此 SHA 的 8 项 PR Actions 全部成功；任何新提交需使用其最终 SHA 重新证明。  
**发行决策：** 代码具备 Beta 教学候选条件，**尚不能正式发布开源 GitHub Release**。根目录 `LICENSE` 缺失、尚未合并主分支、无发布 tag/Release。移动端工作流仅产出调试签名 APK；尚无真实设备兼容性保证。

## 适合发布的内容

SeetheTruth 是供学习与实验使用的本地张量计算工作台。现有功能包含自由组合的已注册积木、符号形状推导、微型数值、独立 NumPy/PyTorch 受支持子图导出、来源知识卡、PC/Android 同源教学内容及科研域证据边界。

本轮 VLM 增加 **4 个可执行有界数值积木**、**4 条图课程**及 **1 个跨平台共享专题**，覆盖视觉前缀、Cross-Attention、视觉 Token 池化、文本 Query 对 Patch Top-K 选择和 Dense/稀疏 Attention 读出差异。计算输入是确定性合成 Float64；视觉投影与语言 logits 是教学函数，并非真实预训练模型。对这四类 VLM 扩展算子的完整图，NumPy/PyTorch 导出仍明确拒绝；独立的 CPU 数学 oracle 属于特殊固定案例，不代表通用编译支持。

## 可以交付的初次使用路径

### Web 桌面

在分支合并前，测试人员进入仓库后执行：

```bash
git fetch origin
git switch --track origin/feat/vlm-unified-learning-20261111
python atlas_server.py --port 8765
```

随后在本地浏览器进入以下地址：

- `http://127.0.0.1:8765/teaching.html?course=vlm` — VLM 专题，从统一课程入口了解机制
- `http://127.0.0.1:8765/learn.html?mission=extension_vlm_early_fusion` — 视觉前缀与因果注意力
- `http://127.0.0.1:8765/learn.html?mission=extension_vlm_query_patch_selection` — Top-K 视觉 Patch 与 Dense/稀疏误差对照

工作台选择**微型数值模式**、执行图并检查原始节点张量。无需 GPU，不需下载模型权重。通用页面继续可由 `learn.html`、`teaching.html` 和 `capabilities.html` 访问。导出按钮对不支持的完整 VLM 图会拒绝生成伪代码。

### Android 内测

使用本分支同一提交的 [Android teaching APK 工作流](https://github.com/Qesire/SeetheTruth/actions/workflows/android-teaching-apk.yml) 产出的 `seethetruth-android-teaching-debug-apk` 测试工件。它包含同源教学资产，不含 Python 运行时、模型权重、下载功能或互联网权限；实际安装仍需用户设备允许测试 APK。请明确标记 **Debug / 教学预览 / 非 Play Store 上架版**。

## 上轮已通过的证据和范围

| 检查 | 上轮结果 | 证明范围 |
| --- | --- | --- |
| Static Analysis CI | PASS | 注册、源码/数值合同、统一图入口、资产与基本约束 |
| Static Behavior Acceptance | PASS | 真实 Chromium PC/手机视口、课程跳转、VLM 图运行和原有页面回归 |
| Dual-Tier Source Mathematics | PASS | 独立 NumPy/PyTorch CPU Float64 小图复核、源码数学与误差 |
| Real CPU PyTorch Training Codegen | PASS | 既有受支持图的训练代码导出及测试（非 VLM 全图导出） |
| Training Microflows Contracts | PASS | 现有训练状态与微型优化流程 |
| Android teaching APK | PASS | Debug APK 编译、离线资源白名单与包完整性 |
| Pinned Real-Source Operator Audit | PASS | 既有固定源码算子审计 |
| Real-model Static Concordance | PASS | 已有真实模型来源静态校验，**非 VLM 真实模型执行** |

以上结果严格属于旧 SHA `65f50116`。后续变更必须重新通过最终提交的全部检查。当前 PR 合并、Release 以及实际用户测试均不属于这些 CI 的已验证结果。

## 发布门槛与尚未完成的事项

| 目标 | 门槛 | 状态 |
| --- | --- | --- |
| 内部测试版 | 受控测试人员、同一提交的全量 CI、明确 debug 签名与适用范围 | 候选已具备；仍应安排手动首开与设备试用 |
| 公开 Alpha/教学预览 | 根目录开源许可证或明确授权条款、最终主分支/Tag CI、版本和发布说明、干净环境首启检查 | **阻塞：授权未决定；没有 GitHub Release/Tag** |
| Beta 教学质量 | Android 真机触屏连线、布局/缩放、首次使用任务完成检查、反馈渠道、重要导入/导出兼容性 | 部分浏览器用例已有；实际设备和外部体验仍待完成 |
| 稳定 v1.0 | 版本化图合同、升级兼容和长期维护、完整受支持子图保证、外部复现 | 不在当前发布范围 |
| 预训练 VLM 推理 | 模型权重、图片解码、真实语义任务、质量和性能评价 | 单独的后续科研里程碑 |

**许可选择**：仓库所有者需自行决定具体许可证并确认权利来源；不替所有者生成法律授权声明。即便源代码已公开可读，也不等于自动获得再分发许可。

## 推荐发行定位

先以 **`v0.1.0-alpha.1`（候选名称，尚未打标签）** 标识“学习工具内测版”，给出本地安装与课程说明、debug APK、代码/报告来源、已知限制与反馈渠道。现有 `docs/release-readiness.md` 的 Alpha/Beta 定义继续有效；本文件补齐 VLM 增量的可发布证据与真实阻塞。正式公开之前先完成许可选择、最新 SHA 全量测试、最终主分支和 tag 的测试、首次使用手测与 GitHub Release 的明确创建。

## 发布时检查清单

- [ ] 本次 PR 最终 HEAD 的 8 条必需 Actions 全绿（不沿用旧提交）
- [ ] 合并 PR 后主分支再次全绿，再创建正式 tag 和可验证的发布记录
- [ ] 所有者确认开源/分发授权并提交根目录 `LICENSE` 或等效正式条款
- [ ] 全新机器运行 `python atlas_server.py --port 8765`，无页面错误、无横向溢出
- [ ] 桌面与 390px 手机首开：VLM 课程、数值实验、返回与失败路径可见；真机触屏另测
- [ ] 旧图导入、保存/恢复和受支持的代码生成无回归
- [ ] 版本号、发布日期、更新摘要、已知限制、测试工件与反馈入口全部明确
