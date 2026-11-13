# SeetheTruth · 发布候选界面与表达整改

2026-11-13 · 从 PR #105 的 `32615a9a869d6c84584cd16f761882e3c766d6dd` 建立独立整改分支。内容针对 **内部教学 Alpha 候选**；不改变算子数学或发布授权，不把候选分支称为公开 Release。

## 已落地：可见任务路径

| 实际位置 | 本次更改 | 可检验结果 |
| --- | --- | --- |
| `teaching.html` 手机首屏 | 当前课程详情与实验入口放在课程筛选和长目录之前；课程 CTA 移到标题下面 | 深链 `?course=vlm` 打开后，第一屏可以找到对应计算图入口，数学解释和理解检查仍完整保留 |
| `teaching_curriculum.js` | 11 门课程以实际待解问题命名；VLM 理论以 Patch → 融合 → 读出 → 选择/误差组织 | 术语仍在理论、反例与证据中；`Core.inspectCourse` 仍指向相同引导图或专门实验室 |
| `teaching_hub.js` | 按课程 ID 提供能预测目标的操作文案，例如“打开 VLM 视觉前缀计算图” | 文案描述的是**打开**，而不是声称已执行；`safeTarget` 的源注册不变 |
| `learn.html` | 在执行按钮旁显示“尚未运行 / 推演完成 / 合成数值已执行”、当前步骤 ID 和张量形状 | 状态只从现有 `run / step / selected / studyMode` 推导；切换模式或变更图后不保留旧的数值成功提示 |
| Scheduler 与 Wan FFN 页面 | 用请求如何获取 KV 内存、FFN 如何逐坐标计算来命名研究任务 | 原源码路径、合成 Float64 边界、数学推导、反例和操作未移除 |
| `README.md` | 将含混的“公开测试版候选”改为“内部教学 Alpha 候选，尚未公开发行” | 与实际无 LICENSE、无最终 Tag/Release 的状态一致 |

## 已落地：中文视觉证据门禁

旧 CI 的 Chromium 截图存在中文汉字均为方框的现象，因此其 40 张首屏图只可证明 DOM/布局和导航，无法证明中文字号和文字换行。

- `scripts/ensure_cjk_font.py` 在 Linux Runner 检查 Noto Sans CJK SC；缺失时从上游 `Sans2.004` 发布标签取简体中文 OTF，安装至临时用户字体目录并通过 fontconfig 检查。字体不提交到仓库、不打包 Android，授权归属见上游 Noto CJK SIL OFL 1.1。
- `.github/workflows/static-acceptance.yml` 在运行截图之前执行字体预检，不再依赖会随机失败的 Ubuntu apt 字体镜像。
- `tests/browser_sitewide_ui_audit.py` 使用 Canvas 将两组等长汉字栅格化，比较非透明像素的差异，直接拒绝“不同汉字却渲染为相同方框”的截图结果；另外对教学中心、工作台、源码首页和 Wan FFN 检查 320px 窄屏。
- 新增首轮任务验证：VLM 深链应指向实际视觉前缀图；工作台原理推演之后再切换数值模式必须撤销旧执行证据，真实数值运行后才显示成功状态。

## 验证记录与下一门禁

本地通过：`node tests/sitewide_ui_contracts.js`、`node tests/teaching_curriculum_contracts.js`、JS 语法检查、Python 语法检查、`python scripts/ensure_cjk_font.py`。使用系统 Chromium 与本地 CSS/HTML 静态渲染的 320px、390px、1440px 首屏没有页面级横向溢出；本地中文栅格差异探针得到 1107 个不同像素（阈值 200）。

本地执行环境阻止 Chromium 访问 `127.0.0.1` HTTP，因此 **实际 HTTP 全站测试与新任务路径尚未本地通过**，必须以 GitHub Actions 对当前提交的真实工作流结果判断。静态 `set_content` 截图不能代替正式动态页面截图，更不能代替 Android 真机。

公开发布依然需要由项目所有者决定许可证和第三方权利、审查合并候选、在最终 `main`/Tag 提交重新验收并准备分发工件。Beta 仍需要 Android 真机和非作者贯通任务的实测。不要为通过门禁而添加不相关的模型算子。

### GitHub Actions 运行环境修订（2026-11-13）

首次整改提交 `83d635f` 的 CI 实际下载并安装了 Noto Sans CJK SC，但浏览器尚未执行：`ubuntu-latest` 已升级至 Ubuntu 26.04，固定的 Playwright 1.55.0 在安装 Chromium 时返回 `does not support chromium on ubuntu26.04-x64`。现将浏览器验收 job 明确固定到 `ubuntu-24.04`，需以新提交的完整 HTTP 验收结果确认；此前失败不属于页面渲染结论。

### 中文标题不应成为课程身份合同（2026-11-13）

Ubuntu 24.04 的完整 HTTP Chromium 已验证 20 页双视口、中文栅格、320px 和首次执行状态，随后旧版 `tests/browser_unified_learning_core.py` 因假设兜底课程标题必须包含“积木”而失败。课程功能回退到第一门注册课本身正确；把断言改为对比 `#detailTitle` 与 `SeetheTruthCurriculum.lessons[0].title` 的实际身份，不再把自然语言字面词条当作导航契约。其余旧测试保持不变，需要由新的完整 CI 再验收。

### 计算画布与状态条的可见区域几何（2026-11-13）

最新 HEAD `e4a064c` 的完整 Chromium 行为验收通过前 35 个步骤，但第 36 步的 1600×1000 积木目录场景在 Mask Softmax 图检查中测到最后节点底边 `769.625`、画布视口底边 `762.765625`，超出容许 `+3px` 约 3.86px。此时新执行状态条占用了额外垂直空间。针对已观察到的原因，将桌面状态条竖向 padding 从 9px 收为 4px、行高从 1.5 改为 1.35，保留字体 12px、真实执行身份与全部信息；手机 10px padding 未改变。保持原节点完整可见的严格断言，需用最终完整 Chromium 回归复核。
