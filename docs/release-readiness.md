# SeetheTruth · Release readiness and evidence gates

**Assessment refreshed 2026-10-21:** shared PC/Android curriculum PR #40 is merged into `main`. The dated milestones below are historical. This is a release decision record, not a published Release.

Multimodal foundation candidate (2026-10-20): add 4 shape-validated, numerically executable image/audio/video blocks and 4 guided graphs, including three-way Cross-Attention. Tests must prove JS, NumPy, PyTorch forward parity and Chromium interactions before marking accepted. Semantic scope is channel-last image/time-major video flattening and finite Hann-window STFT; not native media decoding, Mel filtering, 3D VAE, source checkpoint replication or audio/video generation. See `docs/multimodal-computation.md`.\n\nSecond multimodal executable increment (2026-10-20): Conv2D with explicit weight edge, numerical HTK-triangular Mel power aggregation, deterministic DDIM epsilon/eta=0 reverse step. Adds 3 lessons and 3 registered operators for 126 kinds / 52 lessons. No claim of end-to-end visual/audio generation, Wan VAE reproduction, Whisper filterbank identity or full gradient support in the teaching interpreter; PyTorch Conv2D weight gradients require independent executable verification.\n\nThird multimodal math increment: Native rank-5 causal Conv3D uses IR tensor axes [B,T,H,W,C], persistent kernel [O,Kt,Kh,Kw,I], mathematical temporal left padding and an immutable last-(Kt-1)-frames streaming state. Release gates must compare full execution against arbitrary chunk partitioning and all three JS/NumPy/PyTorch implementations, plus actual PyTorch parameter gradient and real browser evidence. No claim to complete Wan VAE behavior or GPU speedup.\n\nInference mechanism expansion: add four shape-validated deterministic CPU operators and four new courses covering stable index Top-K, post-index sparse Attention, paged KV logical/physical lookup, and greedy speculative draft acceptance. Requires JS/NumPy/PyTorch independent parity plus Chromium acceptance, source-backed design notes, and explicit nonclaim of reproducing the **2026-10-20 released DeepInfer offline engine** or its actual DeepSelect/FlashMLA/DeepEP kernels. DeepInfer is now public, but its README excludes a complete KV cache service, prefill/decode disaggregation and cross-DP load balancing. Inventory 131 blocks and 57 lessons.\n\nDeepInfer source-derived increment (2026-10-20): official MTP variable query budget rank reference implemented exactly on CPU; bounded YoCo SWA suffix mask and compact contracts with explicit reduced-query logical footprint, NOT full upstream per-chunk metadata/actual device FLOPs. Add lessons 58–59, JS/NumPy/PyTorch numerical parity and Chromium screenshots. Scope 134 blocks/59 lessons.\n\nDeepInfer Worker quantitative ZOS correction: source C++ `prepare_zos_inputs` remap/overlap/padding, TileLang `compute_seqlens_k_zos_ref` accepted-token update and optional global MTP confidence Q allocation. Adds an explicitly edited fixture-backed Q/K state table alongside abstract Worker CPU/GPU events. Does not claim actual GPU metadata transfer or `refresh_for_sampling` side effects. CI requires Node oracle, PyTorch CPU independent parity, Chromium visual and invalid-fixture checks.\n\nDeepInfer decode KV metadata increment: exact `prepare_decode_zos_inputs` compressed BlockTable integer packing, offsets, cu_seqlens_q, defaults/no-default behavior, and source-equivalent one-shot MTP slot refresh with golden-token window and opaque structured-output transfer. Integrated with one explicitly-fixtured ZOS correction step, not physical memory allocations; CPU PyTorch reference and Chromium screenshot tests required.\n\nDeepInfer decode-only Scheduler→KV→Worker integration: C++ `_schedule_decode` output-length priority, one-free-block guard, victim selection, recompute WAITING versus structured-output stop and explicit re-admission. Worker events generated only from Scheduler-owned physical KV requests. A separate CPU teaching lab now covers NORMAL ALL/ANY Prefill voting and image-safe chunk math. Full end-to-end voting integrated with the decode-only lifecycle, real model sample feedback and measured GPU time remain excluded. Node adversarial source tests plus Chromium screenshots are merge gates.

## Proposed release boundary

**v0.1.0-alpha = learning-first, local web app** with freedom-first graph construction, shape contracts, paper-grounded explanations, finite educational tensor execution and independent NumPy export for supported graphs. Source/runtime/telemetry are opt-in research features. Neither production LLM training nor full checkpoint inference is a release requirement.

**v0.2.0-beta** adds tested onboarding and accessibility/responsive paths, actual pinned tiny-model `REAL_CAPTURE` fixtures and artifact provenance, documented user-driven bug reporting and backward compatibility testing.

**v1.0** requires a stable versioned graph/export contract, arbitrary supported-block composition with systematic forward and reverse support, reproducible checkpointed training backend or explicitly narrower documented guarantees, sustained non-author release testing, and clear compatibility/security maintenance process.

## Verified candidate inventory

- `learning_lab.js`: 11 original kinds + 123 `block_catalog.js` definitions = **134 registered kinds**. NumPy/PyTorch CPU compiler eligibility is separately measured by `node scripts/report_capabilities.js`, not assumed from this count.
- `learn.html`: **59** guided preset identifiers, plus a `blank` canvas. These are verified by `tests/release_surface_contracts.js`.
- Graphs: bounded to 80 nodes / 176 edges in the current `learning_lab.js` constants, DAG and port/shape validation; the *chosen upstream closure* may be runnable while unrelated incomplete nodes remain on the canvas. Never require a model-specific template.
- NumPy codegen: full and selected-branch exports, source tensor overrides, fail-closed unsupported operations, intermediate-value parity against JS. For supported differentiable subgraphs, the separate PyTorch CPU backend now emits real training/AdamW/checkpoint code; unsupported ops fail closed.
- Training: bounded explicit parameter identities and supported VJPs; differentiate one-step optimizer function blocks from persistent multistep training transactions.
- Evidence: a real pinned TinyStories-1M CPU module trace and a second fresh, numerically identical replay are archived by GitHub Actions; the real Chromium inspector can render its weight/gradient/activation samples. The importer continues to distinguish source assertions from independently reproduced evidence.

- `tradeoffs.html` (standalone analytical experiment) and `architecture_tradeoffs.js` are now served from the same local server and tested in release-surface contracts; estimates are not profiler measurements.

- Learning Studio now exposes exact graph-derived physical Parameter and known GEMM MAC counts for any fully shape-valid DAG; tests reject incomplete graphs. Unknown operators make this a partial arithmetic report, never total FLOPs or GPU latency.

## Release blockers: required before public alpha

| Gate | Current evidence | Completion criterion |
| --- | --- | --- |
| P0 · Integrate candidate | PR #40 merged on 2026-10-21; prior learning changes are in main | Repeat all mandatory CI on the exact release tag SHA |
| P0 · Legal distribution | No `LICENSE` at repository root | Owner explicitly chooses compatible open-source license and commits it; do not assume a license |
| P0 · Stable smoke path | Seven mandatory GitHub Actions workflow runs on candidate HEAD reported success; browser suite in Static Behavior Acceptance | Repeat all mandatory CI workflows on final merged SHA, manually open `/learn.html` as first-time user in a clean environment and confirm no JS console errors or overflow |
| P0 · User-facing truth | New source-driven inventory and public capability page distinguish registered vs exporter-eligible vs gradient-tested | Keep every UI claim within its source and evidence level; check codegen fail-closed |
| P0 · Release artifact | No published GitHub Release | Cut explicitly labeled prerelease `v0.1.0-alpha`, include release notes, dependency/startup instructions, example project JSON and reproducible evidence links |

## Non-blocking for alpha; mandatory before related claims

- **Actual model measurements:** a pinned TinyStories-1M CPU forward/backward trace has now run twice and replayed exactly on [the recorded CI workflow](https://github.com/Qesire/SeetheTruth/actions/runs/39217748548), with real Chromium screenshot artifacts. This covers **one linear module**, not complete LLM equivalence, GPU throughput, MoE routing or real RL rollouts. The local file importer still labels unattested user-supplied JSON as structural verification only.
- **Arbitrary codegen coverage:** both backends fail closed on unsupported kinds. Full educational MoE routing and bounded CPU Autograd/AdamW for three shared expert banks are supported; this is not generic production-scale PyTorch training.
- **Large tensor performance:** simulation is bounded and CPU-browser-oriented. GPU validation and actual model inference are not required for the learning-first alpha.
- **Paper coverage:** modules and paper *patches* differ; both require mathematical contracts, shape/axis semantics and primary sources. A symbolic patch is not an executable operator.
- **Interaction hardening:** verify keyboard navigation, focus states, smaller screens, resize, local file import bounds, visual scroll behavior and README onboarding in real browsers before beta.

## Mandatory evidence before release tagging

1. `node tests/release_surface_contracts.js` — source-of-truth catalogue count, README consistency and visible public entrypoints.
2. `node tests/graph_readiness_contracts.js` and `node tests/branch_execution_contracts.js` — unconstrained valid DAG + incomplete parallel branches.
3. `node tests/python_codegen_contracts.js` and `node tests/python_codegen_extensions_contracts.js` — actual Python/NumPy execution and per-node cross-language parity; `node tests/moe_executable_codegen_contracts.js` additionally verifies dynamic MoE source overrides and CPU Torch shared-gradient/optimizer state consistency.
4. `node tests/train_state_contracts.js`, `node tests/nd_train_state_contracts.js`, `node tests/moe_graph_autodiff_contracts.js`, `node tests/torch_codegen_extensions_contracts.js` and `node tests/loop_depth_supervision_contracts.js`, `node tests/loop_consistency_tradeoffs_contracts.js`, `node tests/architecture_why_contracts.js` — fresh forward on each optimizer step, gradient fan-in, state/undo/save/restore, finite difference.
5. `python tests/browser_catalog_screenshots.py` and screenshot manifest — **actual headless Chromium**, click/drag/repair/train/export/import, browser scenarios; inspect screenshots and errors rather than relying on PNG existence.
6. Existing `Static Analysis CI`, `Static Behavior Acceptance`, `Real-model Static Concordance`, `Pinned Real-Source Operator Audit`, `Training Microflows Contracts`, `Real CPU PyTorch Training Codegen`, and `Pinned TinyStories CPU Real-Model Capture` all success on the same final SHA.

For each failing gate: record input graph, exception or numerical discrepancy, which feature regressed, and the exact PR SHA. No “mostly green” release claim. Keep user-visible claims aligned with the strongest reproducible evidence tier.

## Suggested release note outline

**2026-10 Loop WHY extension:** explicit two-loop residual step-size ablation comparing `1`, `1/sqrt(N)`, and `1/N`, with all parameters shared, pre-Norm not misrepresented as DeepLoop Post-LN. Real JS/NumPy/PyTorch gradient and checkpoint parity in `tests/loop_residual_scaling_contracts.js`; source-linked parameter/compute/gradient tradeoffs and fail-closed attribution after custom rewiring.

**What works:** Scratch-like free graph editing, 134 registered block kinds, 59 curated lessons, formula/paper inspector and hover, bounded numerical steps/gradient/optimizer flows, NumPy standalone export for supported graphs, optional code/evidence exploration.

**Not included:** generic trainable Python/PyTorch graph export, full-precision real models, GPU timings, actual RL rollout, unsupported operator VJPs or every paper patch being executable.

**Upgrade compatibility:** saved graph schema remains `seethetruth-learning-lab-v1`; verify save/import against the exact release branch before claiming future compatibility. The app must not execute uploaded JSON as source code.

**Known data provenance:** synthetic/qualitative results must never be promoted to observed model evidence without pinned capture and real tests.

## 2026-10-21 capability evidence gate

The executable `block_capabilities.js` inventory and `capabilities.html` compute kind coverage from registered definitions, knowledge-card validation, and the current NumPy / PyTorch CPU compiler allowlists. Run `node scripts/report_capabilities.js --json` to list the exact unsupported kinds. **Do not** interpret an allowlist match as proof that an arbitrary composed graph passes shape/semantic validation, backward differentiation, PyTorch training, or GPU parity.

Android teaching asset scope is explicitly reviewed in `android/teaching-assets.txt`; its build must not silently package every repository-root HTML/JS file. Offline demo code has no model backend or runtime file export. Some JS compiler helpers remain direct dependencies of the shared Learning Studio until the teaching editor is fully split from the desktop export UI.

Remaining prerelease blockers: license choice by the owner, final main/release SHA acceptance, clean first-run browser inspection, release notes and an explicitly published prerelease. A debug-signed Android workflow artifact is not a store release.

## 2026-10-21 named-axis/state proof increment

`semantic_state.js` adds separate typed Float64 rank 1–5 numerical contracts and immutable logical stream state, not a general graph-level persistent operator or actual GPU KV Cache implementation. The same `semantic_state.html` mathematics works in PC Chromium and Android offline assets. Run `node tests/semantic_state_contracts.js` and `python tests/browser_semantic_state.py`; only legacy 134 blocks/59 presets are counted. This is P1 issue #42 foundation; compiler/VJP and versioned graph state operators remain pending.

## 2026-10-21 typed logical stream graph blocks

Four `stream_state_*` operators were registered and integrated into the free editor. `stream_graph_blocks.js` provides tagged state-vs-tensor port checks, static temporal inference, immutable numerical state transitions, and forked branch execution. The same contract is used by `learning_lab.js`, `concept_flow.js`, and `graph_readiness.js`. Inventory is now **138 blocks** / 59 guided presets. **NumPy and PyTorch compiler allowlists remain unchanged** (no state export, no VJP); real GPU BlockTable, COW sharing or throughput remain out of scope. Node and Chromium/Android regression tests gate this increment.
