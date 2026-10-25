# SeetheTruth foundation and extensibility assessment — 2026-10-25

## Decision

**The shared foundation is real and usable for bounded educational computation. It is not yet a universal, plug-and-play foundation for all prospective operators, models, backends and stateful experiments.** Build new modules on the existing contracts; invest in the seams below before proliferating specialized controllers.

This is an audit of **actual source, observable behavior, declared capabilities and CI**, not a claim that independent model/GPU parity has been established. Numbers below come from `tests/architecture_extension_spine_contracts.js` on PR #86's validated source (not a stale README count). They update the earlier 2026-10-24 inventory.

## Measured 2026-10-25 inventory

| Dimension | Source-derived observation | Interpretation |
| --- | ---: | --- |
| Registered graph operators | **150** | Distinct kinds, graph metadata and bounded teaching execution |
| Independent installed extension kinds | **8** in **2 packs** | Generic registry/shape/numeric extension path exists; installed packs are chosen at build time |
| Reachable guided graph entries | **67** | Real recipe identity checked in `Core.activateGraphLessons`; not evidence every generated backend works |
| Shared PC/Android curriculum | **10** tracks | Single `teaching_curriculum.js` source |
| Registered specialist domains | **17** | Real source-labelled identity and route |
| Registered specialist execution actions | **23** | Derived from actual dispatch, *not* 23 GPU implementations |
| Shared observation UI pages | **7** | Learning Studio, Wan FFN, four DeepInfer labs and independent training timeline |
| NumPy export-eligible operator kinds | **121/150** | Eligibility, not universal executable graph parity |
| PyTorch export-eligible operator kinds | **115/150** | Eligibility, not universal executable graph parity |
| Universal per-kind gradient proof | **Not certified** | Current capability field is `requires_graph_test`, not a per-kind proven guarantee |
| Model checkpoint / GPU timing evidence | **Not universal** | Specialized teaching CPU engines and source-derived synthetic traces must remain distinct |

## Computation and identity spine

```text
Shared PC browser / Android teaching / guided courses / freely edited graph
             | identity, native evidence and source references
             v
  SeetheTruthUnifiedCore (learning_experience_core.js)
             |                           |
             | graph session             | explicit specialist actions
             v                           v
  LearningLab + GraphSession        Real domain engine
  indexGraph / operator contracts   state machine, Wan FFN, Bigram ...
  shape, trace, bounded numeric       |
             |                       | only when native trace exists
             +----------+------------+
                        v
      observe() / replay() / inspectObservationStep()
      compareRecords() / native state & numeric microscopes
                        |
      observation_snapshots.js: input recipe, source
      evidence, deterministic replay and typed cursor
                        |
              unified_observation_ui.js
            selected current accepted record
```

The underlying mathematical objects remain separate: a tensor DAG with typed ports and values; an event DAG with causal dependencies; a state transition `s[t+1] = F(s[t], a[t])`; and a real CPU training checkpoint with weights, optimizer moments, sampler and pending gradient buffer. The *shared identity, actions, observation, replay, evidence and UI* are the contract—not one fabricated tensor algebra.

## Architectural assessment

| Layer | Present strength | Structural limit / evidence missing |
| --- | --- | --- |
| Graph identity/topology | **Established**: `LearningLab.indexGraph` drives topo ordering, branch slicing, port identity | All imported external model structures still require an explicit lowering or adapter |
| Operator semantics and extensible bricks | **Established, statically extensible**: `block_extension_registry.createRegistry` validates arbitrary candidate packs with independent numerical/shape semantics | The active `LearningLab` loads **two built-in packs statically**; independently created third-pack registries do not automatically install into the active graph runtime |
| Capability model | **Partial**: registered/knowledge, NumPy/Torch eligibility distinguished | `graphAutodiff` is `requires_graph_test` for every kind, `modelParity` is `not_claimed`; backend coverage is not homogeneous |
| Specialist identity / execution | **Improved in PR #86**: 17 lab identities and 23 actions exposed from the same real dispatch object as trace adapters | An action registration or declared native trace alone is not a successful per-scenario runtime test; each new action needs at least one native oracle |
| Computation and state microscope | **Established for supported frames**: actual coordinate proof or genuine event before/after with causal edges | Not all 23 operations or 17 domains have native trace frames; some evidence-oriented routes are intentionally observational |
| Replay and checkpoints | **Strong within supported CPU reference engines**: exact input replay, branch lineage, selected state cursors, KV and Bigram native continuation | Checksum is non-cryptographic; no universal external process/GPU checkpoint semantics or cross-backend state conversion |
| Web / Android | **Verified subset**: common curriculum, shared observation UI, offline asset list, Chromium phone/desktop testing | `<script>` dependency order, static HTTP route and offline asset lists remain explicit bootstrap maintenance points; adding a JS module requires coordinating them |
| Actual-code fidelity | **Partial**: JS numeric oracles with NumPy/PyTorch parity tests for selected operator families | 150 registered operator kinds **does not mean** 150 PyTorch lowering/gradient implementations or industrial-kernel parity |
| Release/engineering confidence | **Significant**: multiple native contract tests, CI syntax/semantic/browser checks and APK build | These tests do not establish universal GPU behavior, visual UX depth in every lab, or arbitrary graph codegen success |

## High-value gaps, in priority order

**P0 — One source of truth for specialist operations and observation.** Previously the real dispatch and the separate `eventAdapters` map were independently maintained. In PR #86 every runner owns its native `observeState` or `observeCompute` handler. `Core.describeActions(id)` and `Core.inspectLab(id).operations` derive names, source binding and observation type from the same actual dispatch table. This minimizes silent integration drift without introducing a new execution system. Verified by positive native state/compute cases, negative no-trace cases and exact replay.

**P0 — Treat installation as a real capability rather than a name.** A new independent registry pack can validate shape and execute values, but cannot be loaded into the graph by a runtime-only call. A future trusted pack-installation seam should register new definitions, frontend assets and tests with deterministic collision resolution and no external untrusted JS execution. Until then describe the system as **statically extensible**, not a dynamic plugin marketplace.

**P1 — Typed per-operation and per-operator capability records.** Extend the same manifest to express explicit availability and *verified* coverage for numeric execution, shape inference, trace, VJP/autograd, PyTorch/NumPy lowerings, state rewind and replay. Do not derive a `verified` flag merely from registration. Existing `modelParity: not_claimed` is appropriate and should remain unchanged absent independent proof.

**P1 — Make browser/offline dependency installation declarative.** Current tests now check all seven observation consumer pages for script existence/order, server delivery and Android manifest equality where packaged. A future build manifest should generate HTML ordered scripts, server routes and offline assets from trusted dependency declarations; keep the tests as a guard while migrating.

**P1 — Bridge graph training to shared state lifecycle.** `training_session.js` and `train_state.js` already represent graph optimizer transactions with before/after execution IR, while `engineering_training_pipeline.js` has exact native checkpoint/restore. The next integration should link *actual graph node IDs, parameter edges and optimizer transactions* into shared state observation, before offering a single backward/forward click-through interface. Do not invent a hidden graph/GPU state.

**P2 — Scale verification against independent backends and richer teaching UX.** Grow model-backed small-tensor oracles and per-operator gradient/codegen coverage, extend UI consumption to specialists with genuine frames, and validate more true Android WebView behavior. Avoid making a large-model GPU run the default educational dependency.

## Extension recipe for contributors

1. **Source first.** Identify real operator function or state transition, input/output ports, named axes, condition/branch semantics, and source revision. Write the minimal equivalent mathematics and a counterexample.
2. **Add a block through the real graph path.** For a new installed pack implement `defs/validate/infer/execute` (+ `trace` as needed); register it as a trusted installed pack in the existing extension bootstrap. Build a valid graph and prove shape/numeric parity. Do not create a second graph engine.
3. **Or add a specialist action.** Add one entry to `domainRunners` with `module/method/run`. If a **native** microscope frame is available, attach `observeState` or `observeCompute` **on the same entry**. `Core.describeActions` will expose its actual capability. If absent, report `observation:'none'` rather than fabricate data.
4. **Wire the real teaching surface.** Use `specialized_learning_adapter.js` and current accepted controller input, plus the shared observation/snapshot presenter. Preserve source-specific rich widgets.
5. **Make cross-platform packaging reproducible.** Serve the exact scripts, load them in dependency order, include in Android teaching assets, and validate native engine results, Chromium PC/mobile interaction, negative cases and final-head CI before merging.

## Independent proof used for this audit

- `tests/architecture_extension_spine_contracts.js`: compares all registered lab actions against their actual dispatch metadata, validates real state/compute/native replay, insists no-trace operations stay no-trace, checks seven shared UI routes and their script/server/Android dependency chain, and instantiates an entirely new standalone pack against the generic extension contract.
- `tests/unified_learning_core_contracts.js`: real registered graph presets, specialist routes, original engine dispatch and static/numeric distinctions.
- `tests/unified_pack_registry_contracts.js`, `tests/universal_microscope_contracts.js`, `tests/unified_observation_contracts.js`, `tests/state_checkpoint_contracts.js`, `tests/native_training_continuation_contracts.js`: operator pack, graph and coordinate inspection, state and checkpoint semantics.
- Chromium and APK CI: browser behavior is not inferred from static source presence; check run results are valid only for the final SHA.

## Decision criteria for a genuinely robust universal foundation

Call the architecture **broadly and reliably extensible** when (a) any new installed operator can enter graph semantics through one declarative source and pass independent oracles, (b) any new specialist is registered once for execution, trace, evidence, replay and user-facing launch, (c) cross-platform script/asset wiring is generated or formally dependency-checked, (d) backend/gradient/trace eligibility is a verified granular capability, and (e) both positive and adversarial native/browser tests pass on the same release commit.

PR #86 resolves the first part of (b) and adds a live guard for (c). It does **not** claim (a), (d), or (e) universally complete.
