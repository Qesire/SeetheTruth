# SeetheTruth unified learning core v1 — 2026-10-24

## Architectural result

A single **typed learning contract**, `learning_experience_core.js`, now sits above the already shared graph interpreter and above specialized domain engines. This is integration and interface stabilization, **not** a wholesale replacement of each specialist numerical implementation.

```
PC Learning Studio       PC/Android Shared Teaching       Dedicated labs
         |                         |                          |
         +------------ SeetheTruthUnifiedCore ----------------+
                     schema: learning-experience-v1
                  identity + typed domain + evidence
                            /           \
            arbitrary LearningLab DAG    domain engine adapter
              GraphSession / IR           KV state / CPU-GPU timeline
              GraphLearningPath           Scheduler / source microscope
              operator contracts          training / source trace
                            \           /
                  explicit provenance / capability
```

## Why one system must admit several mathematical substrates

A freely composed tensor program is a directed acyclic graph `G=(V,E,θ)` with tensor-valued ports, shape/axis contracts and conditional reverse-mode rules. KV management and a request Scheduler instead form a transition system `s[t+1]=F(s[t],a[t])`; Worker execution has resource-constrained event edges and causal timestamps. Treating such state machines as a fake tensor operator would erase important invariants. The common layer therefore shares **identity, execution entrypoints, teaching descriptions and evidence contracts**, not a false numerical algebra.

All 146 native/extension operators pass through the existing `operator_contracts.js` inventory; an operator being registered is not proof of all NumPy/Torch lowerings. `inspectGraph(graph,{numeric:false|true,targetId?})` calls the existing `GraphSession` and `GraphLearningPath`, preserving target slicing and unexecuted static status. `activateGraphLessons` registers the real HTML studio missions and installed pack recipes, then `openPreset` rejects an unregistered preset. The contract audits **59 guided mission graphs, one blank editable canvas, two independently registered pack recipes and one source-state shortcut = 63 reachable graph entries**.

The ten PC/Android explanation tracks now project from the same `teaching_curriculum.js` into typed `curriculum` records. Every other shipped root HTML learning page has an explicit `specialized` contract via `specialized_learning_adapter.js`: 17 specialist modules, each with a concrete engine source file, domain kind and declared evidence boundary. Cross-platform lesson cards now consume the core inventory rather than a separate presentation-specific list.

## Execution maturity, not a false all-green capability flag

| Substrate | Shared metadata/route | Executed through common API | Actual compute |
| --- | --- | --- | --- |
| All 146 native/pack bricks | Yes | Yes: complete and selected-output graph inspection | Existing bounded LearningLab engine |
| 59 guided graphs + blank + 2 pack recipes + state shortcut | Yes | Yes: same graph engine | Real preset graphs; no template whitelist on free editing |
| Ten shared curriculum tracks | Yes | Linked to graph or specialist domain | A description alone has **no** execution evidence |
| DeepInfer Prefill | Yes | `executeLab(vote/chunk)` | Source-derived deterministic decision mathematics |
| DeepInfer Worker | Yes | `executeLab(simulate/compare)` | Event DAG with explicitly assumed abstract durations |
| DeepInfer KV | Yes | `executeLab(replay)` | Physical-block ownership reference simulator |
| DeepInfer Scheduler | Yes | `executeLab(simulate)` | Decode-only Scheduler/KV/Worker reference |
| Remaining 13 specialized pages | Yes | Independent lab application only | Original calculators, source trace or analytical models; unified adapter has **no** automatic execution claim |

The labels `NOT_EXECUTED`, `STATIC_ONLY`, `PEDAGOGICAL_DESCRIPTION`, `SYNTHETIC_NUMERIC`, `ABSTRACT_TIMELINE`, and `SOURCE_DERIVED_DETERMINISTIC` must never be implicitly promoted into actual checkpoint execution or GPU performance. `inspectLab` intentionally returns `NOT_EXECUTED`; only `executeLab` calls a registered domain function and returns its real result.

## Stability checks

- `node tests/unified_learning_core_contracts.js`: coverage of **every** registered brick, every HTML mission, installed recipes, ten shared courses, all non-entry standalone HTML routes, engine source existence, Android assets, server routes, static/numeric graph evidence, branch identity, and live parity against the original four domain engines.
- `node tests/graph_learning_path_contracts.js`: graph-route contract. The adjacent patch fixes a hidden target-slice defect where node parameters were previously retrieved by `index.ordered[i]` rather than node ID.
- `python tests/browser_unified_learning_core.py`: Chromium learning studio graph count, Android-width shared curriculum, real source lab adapters and no browser exceptions.
- Existing codegen, gradient, shape/axis, source-pinned, streaming and Android build tests are preserved.
- CI is valid only at the **final PR head SHA**; a previous successful run is not transferred to later edits.

## Shared verified event/state microscope

The new `executeLab` adapter directly invokes original source-derived Prefill/Worker/KV/Scheduler reference engines. Prefill exposes an explicit `inspect_pair` that independently evaluates vote and chunk (no invented order); the other three produce the same validated `stateTrace` event contract via `state_microscope.js`. A shared read-only interactive UI (`unified_state_lab_ui.js`) runs these adapters directly inside their specialist pages, showing state-before, state-after and proven causal links. Timings remain assumed abstract ticks, never measured GPU latency. Independent state-engine proofs remain in place.

## Extension interface

1. Add a new computational brick to a reviewed native catalog or a pack with `defs`, `validate`, `infer`, `execute`, optional `trace`, and published source. It is automatically checked through the existing operator inventory and full arbitrary-DAG inspector.
2. Add a guided graph via `LearningLab.preset` or a registered pack recipe, surface it once in the studio mission list. The common loader binds the actual graph and its contracts.
3. Add a new standalone lab with a declared domain, source engine, evidence level and HTML route. The exhaustive root-HTML coverage test must fail until it is registered. Domain `executeLab` operations are **explicit**, bounded and independently oracle-checked; no generic unsafe method invocation.
4. A future domain adapter should supply normalized observational events (state versions, input/output identity, causal edges, typed tensors where applicable), while independent domain calculation stays within its own verifier.

## AutoResearcher mechanism audit

Pinned start: SeetheTruth main after PR #66, merged at `95ec58d955f4a59d91de75ca5ffb3321e7d39d4e`. Method: reconstruct the actual graph and the independent domain state machines; identify whether "unification" means equivalent arithmetic or merely a common data contract; derive the latter with explicit non-equivalence; discriminate with negative cases (unregistered modules, incomplete graphs, shape-preserving parameter changes and invalid domain actions). A single executable trace is stronger evidence than a label; an independent backend parity oracle is stronger than a same-engine duplicate computation.

**Not yet done**: all specialized execution engines using one typed trace event schema, shared replay/undo across domains, a unified view of all specialist internal steps, broad schema migration tooling, and independent parity for each future adapter. Until then describe this as a stable **v1 integration seam candidate**, not completion of a universal end-to-end runtime.

## Restored expert engineering curriculum

PR #49 had independent tested 12-stage Bigram engineering introduction and 6-stage training pipeline; these were selectively transplanted as `engineering_journey_{engine,ui}.js` and `engineering_training_pipeline.js`/`engineering_pipeline_ui.js` together with the shared `state_microscope.js`. They now have typed identities `engineering_journey` and `engineering_pipeline`, the same PC/Android course inventory, explicit offline assets and original numerical/Chromium acceptance. These are *specialized training state machines*, not extra tensor operator kinds. Full-parity claims require their dedicated tests at the current final head.
