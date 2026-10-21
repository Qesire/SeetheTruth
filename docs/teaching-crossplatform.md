# Single-source PC / Android teaching contract

Updated 2026-10-21.

## Scope

- The desktop full editor remains at `learn.html`. It retains executable NumPy/PyTorch exports and the full research workflow.
- The end-to-end learning path `engineering_journey.html` is also shared across PC/Android, includes an executable 8-token bigram CPU training engine, and guides 12 gated checkpoints through evaluation, checkpoint restore and inference. All state remains volatile; the Android edition does not export artifacts. It is not full Transformer training or GPU Serving.\n- The shared course homepage is `teaching.html`. It reads **exactly one** `teaching_curriculum.js`, which provides the topic ordering, mathematical explanations, industry-design rationale and falsifiable self-check prompts. Both PC and Android run this identical source, so explanations do not fork.
- The already-existing native Android project at `android/` builds an **offline educational APK** with `WebViewAssetLoader`. It now launches `teaching.html?edition=teaching`. The asset Gradle Sync picks up the same top-level HTML/JS/CSS used by the PC site; no second Android source tree is maintained.
- In teaching mode, the shared graph workbench remains capable of pedagogical structural checks and small numeric calculations, but hides artifact/code exports, imports and study trace uploads. The native WebView also disables file selection and downloads. No Python backend, model weights, GPU engine, local server or Internet permission are packaged.
- External academic references open by deliberate user gesture in the system browser; they are not required for in-app lessons.

## Why a single knowledge layer

The theoretical interpretation of a block must agree across devices: e.g. a DeepInfer ALL vote minimizes idle Prefill participation, an ANY point gives a lone rank a chance, and a *second* vote prevents zero-work Prompt cycles. The Android app cannot rewrite this into a different explanation merely to fit a smaller screen. Only presentation, navigation, and output restrictions may vary by platform.

Each shared lesson has:
1. A precise computing-flow or mathematical statement.
2. An engineering design motivation with its real tradeoff or limitation.
3. A falsifiable check question plus an explicit answer.
4. A destination to the same interactive exercise on both devices.

Topics include base tensor graphs, paper rewrites, image/audio/video tokenization, Loop Transformer/MoE/GQA costs, KV caching, DeepInfer Prefill/Decode, Scheduler/KV/Worker and ZOS status updates. All source mathematical engines are existing, separately tested shared web modules; this PR does not claim to have implemented additional DeepInfer GPU kernels.

## Compatibility and test gates

- `node tests/teaching_curriculum_contracts.js` checks the shared content, link destinations, source-only APK manifest, native bridge policy and the absence of Android-side output actions.
- `python tests/browser_shared_teaching.py` compares the **same DeepInfer Prefill mathematical/design explanation** on desktop and 390-pixel mobile Chromium, tests lesson filtering, self-check reveal, and verifies the mobile graph can still open and run but does not display exports.
- Existing `node tests/android_teaching_contracts.js` protects the legacy Android portal and build policy, and now also checks the shared homepage target.
- The existing `android-teaching-apk.yml` builds the single `android/` native Gradle project, validates that new teaching assets are truly in the APK, and publishes a debug APK workflow artifact. An artifact is not a release-signed APK, and desktop/mobile Chromium checks are not equivalent to on-device touchscreen/emulator evidence.

Release rule: do not merge if APK packaging, source consistency, or Chromium teaching tests fail. Existing inference and numerical suites remain intact.

## Known usability boundaries

The full graph editor was designed for desktop and is presented in a stacked scrollable layout in mobile teaching mode. Small-screen dragging and pin-to-pin wiring still need real-device gesture tests, including Android System WebView variations and density-dependent hit targets. Until then, label it a *teaching preview*, not polished Android-native interaction.
