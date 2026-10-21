# SeetheTruth Android · 教学专用 APK

这是 SeetheTruth 的纯教学 Android 版本，不是桌面模型开发平台移植。

**统一课程入口（2026-10-21）**：Android 默认打开 [共享教学中心](../teaching.html)，它与 PC 读取相同的 [teaching_curriculum.js](../teaching_curriculum.js) 和数学计算模块。旧 `android_teaching.html` 保留兼容入口。无需维护两份知识卡，新增/修改课程随下次 APK 构建同步。详见 [双端同步设计](../docs/teaching-crossplatform.md)。

**收录**：Learning Studio 课程与积木解释（包括图像、音频、视频、语言模型）、论文知识卡、模型架构成本取舍，以及 DeepInfer Worker/KV/Scheduler 等浏览器交互教学页。

**不收录**：PyTorch/NumPy 程序导出、文件上传或保存、源代码分析服务器、模型权重、GPU 推理、实验产物、训练结果。JavaScript 的小规模数学演示是课程内容，不代表实际运行 DeepInfer。

## 自动构建 / 下载 APK

GitHub Actions 工作流 [Android teaching APK](../.github/workflows/android-teaching-apk.yml) 在 PR 和 main 更新后执行：

- Android Gradle Plugin 8.9.2、Gradle 8.11.1、Java 17、Android SDK 35。
- 构建 `android/app/build/outputs/apk/debug/app-debug.apk`，校验 APK ZIP、静态 HTML/JS 存在及不可携带 .py 或模型权重。
- 工件名：`seethetruth-android-teaching-debug-apk`，在对应工作流运行的 Artifacts 下载。
- Debug APK 由 Android 工具自动签名，只适合内部安装测试。正式对外发布需应用签名、版本管理和 Android 真机/模拟器回归。

## 本地构建

在已安装 Android SDK 35 和 Java 17 的机器上执行：

```bash
gradle -p android :app:assembleDebug
```

**离线设计**：WebViewAssetLoader 从 APK 加载 HTTPS 同源离线静态资源；应用 Manifest 不声明 INTERNET 权限，不运行服务器。外部论文链接仅在用户点击后交给系统浏览器。WebView 禁止文件访问、上传与下载，不暴露 JavaScriptInterface。课程页通过 `android_teaching_mode.js` 隐藏桌面的导出/导入控件并优化移动端布局。

**资产策略**：由 [显式审核清单](teaching-assets.txt) 精确选择离线 HTML/JS/CSS 资源；`node tests/android_asset_manifest_contracts.js` 检查直接网页依赖，APK CI 校验真实 APK assets 与清单完全一致，不允许根目录通配收集。Python 服务、模型权重等不进入 APK。

**当前代码隔离边界**：共享的积木课堂 HTML 仍静态引用桌面 NumPy/PyTorch JavaScript 编译器，因此若干编译器脚本在清单中，仅用于维持同源页面依赖完整性；Android UI 与原生 WebView 禁止导入、导出和文件下载。后续应把课堂运行脚本与桌面编辑器导出脚本真正拆包。

**已知限制**：主积木画布原本为桌面交互设计，小屏设备的连线精度/拖拽体验仍需 Android 设备手势验收。Android APK 是否真正构建成功，请以 CI 为准，而不是仅凭源码文件存在。
