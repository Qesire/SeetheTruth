# Single-source PC / Android teaching contract

Updated 2026-10-21.

## Scope

- The desktop full editor remains at `learn.html`. It retains executable NumPy/PyTorch exports and the full research workflow.
- The shared course homepage is `teaching.html`. It reads **exactly one** `teaching_curriculum.js`, which provides the topic ordering, mathematical explanations, industry-design rationale and falsifiable self-check prompts. Both PC and Android run this identical source, so explanations do not fork.
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

## 2026-11-01 research-first expansion

The same shared curriculum now includes **13 topics**, including video-model FFN sparsity, LLM DPO/GRPO post-training mathematics, and Agent state/event architecture. The new research lane on teaching.html reuses Wan FFN's pinned-topology synthetic trace and research_tracks.js for bounded numerical and scripted control-flow exercises. PC and offline Android use exactly the same sources and controls. It is not a real video-generation, LLM-training, or tool-execution system. See [three research tracks](three-research-tracks-20261101.md).

## Architecture / Infra-first learning (2026-11-01)

The shared PC and offline Android landing now begins with a **three-track, five-level architecture atlas**: System/SLO → Architecture/Topology → Runtime/State → Kernel/Math → Observability/Evidence. Each layer resolves through the existing shared curriculum and explicitly declares whether the example is a design prompt, synthetic numeric exercise, scripted state machine or real evaluation proposal. It does not create a second graph executor or simulator.

In Learning Studio, a dedicated **system-module shelf** links to the actual Scheduler/KV/Worker, Training Pipeline and research exercises; only registered tensor operators appear as draggable graph blocks. Four architecture-role filters derive directly from the underlying block category and reveal actual input port names without changing source semantics, free composition, backward/coding eligibility or hardware-evidence boundaries. The PC and Android app share the same architecture_perspective.js and static asset policy. New checks live in tests/architecture_perspective_contracts.js and expanded tests/browser_shared_teaching.py.

See [architecture-first product architecture](architecture-first-infra-learning-20261101.md) and [three research tracks](three-research-tracks-20261101.md). This is a presentation and curriculum slice; a generic executable typed System IR is a separate future engineering milestone.

## Compatibility and test gates

- `node tests/teaching_curriculum_contracts.js` checks the shared content, link destinations, source-only APK manifest, native bridge policy and the absence of Android-side output actions.
- `python tests/browser_shared_teaching.py` compares the **same DeepInfer Prefill mathematical/design explanation** on desktop and 390-pixel mobile Chromium, tests lesson filtering, self-check reveal, and verifies the mobile graph can still open and run but does not display exports.
- Existing `node tests/android_teaching_contracts.js` protects the legacy Android portal and build policy, and now also checks the shared homepage target.
- The existing `android-teaching-apk.yml` builds the single `android/` native Gradle project, validates that new teaching assets are truly in the APK, and publishes a debug APK workflow artifact. An artifact is not a release-signed APK, and desktop/mobile Chromium checks are not equivalent to on-device touchscreen/emulator evidence.

Release rule: do not merge if APK packaging, source consistency, or Chromium teaching tests fail. Existing inference and numerical suites remain intact.

## Known usability boundaries

The full graph editor was designed for desktop and is presented in a stacked scrollable layout in mobile teaching mode. Small-screen dragging and pin-to-pin wiring still need real-device gesture tests, including Android System WebView variations and density-dependent hit targets. Until then, label it a *teaching preview*, not polished Android-native interaction.
