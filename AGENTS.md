# SeetheTruth — contributable learning runtime

## North star

A learner can build **any well-typed runnable graph**, inspect each actual tensor/operator computation and its mathematical reason, then enter source-pinned guided lessons on PC or offline Android without learning a second incompatible abstraction. Explore creatively; use exact source and numerical proof before asserting an implementation is correct.

Apply [AutoResearcher](https://github.com/Qesire/AutoResearcher) selectively: actual computation reconstruction → minimal equivalent mathematics → decisive counterexample/test → code/teaching change → independent evidence. Begin with the strongest new mechanism or useful implementation. Avoid bureaucratic gate proliferation; the gates below are actual release evidence, not barriers to exploration.

## Extension target

`learning_experience_core.js` provides the versioned *learning identity/observation contract*. Do not add a fresh incompatible registry, duplicate operator semantics, separate Android lesson database or ad-hoc fake numerical trace.

- **Free graph**: an arbitrary DAG with typed input ports, axes, states and operator kinds, validated by `learning_lab.js` / `operator_contracts.js` / `graph_session.js`. Add blocks to existing registered native/pack definitions. Preserve shape/numerical/gradient/codegen eligibility as **distinct** capabilities; don't declare support merely because a kind exists. Preserve independent NumPy/PyTorch numeric oracles.
- **Graph lesson**: add the actual recipe to `LearningLab.preset` or `block_extension_registry.js`, then register the visible or hidden launch ID in the studio's `UnifiedCore.activateGraphLessons`. Do not accept the fallback FFN as proof a new recipe exists. Branch slices retain node-ID parameter binding.
- **Specialist domain**: state machine `s[t+1]=F(s[t],a[t])`, event schedule, pinned-source model capture and tensor DAG are *not* the same arithmetic. Add a typed entry in the core `labs` manifest and matching HTML `specialized_learning_adapter.js`. For a shared `executeLab` action, map a **real existing engine function** explicitly and test exact parity; do not use arbitrary dynamic method calls.
- **Observability**: emit a vetted `state_microscope.js` frame for stateful engines; use actual events, causal edges, state revisions and declared evidence labels. Assume no GPU timing unless real profiler proof exists. A descriptive card is not a successful execution.
- **Curriculum**: teach computation, mathematical mechanism, design tradeoff, failure counterexample and a meaningful experiment. Course cards on PC and Android come from `teaching_curriculum.js` via the unified core; engineering journey and training pipeline retain their verified numerical engines.

## Shared observation and replay

When adding graph or specialist computation, use the existing `SeetheTruthUnifiedCore.observe/replay/inspectObservationStep/compareRecords` contract instead of inventing a second trace envelope. Preserve native `stateTrace` or `computationTrace` and verified causality; replay the exact accepted input before coordinate inspection. Static-only routes must report no numeric trace. See [unified observation protocol](docs/unified-observation-v1.md).

## Shared browser observation consumer

For a new browser microscope or specialist controller, reuse `unified_observation_ui.js` with a callback to the **last accepted engine record**. The consumer calls `SeetheTruthUnifiedCore.observe/replay/inspectObservationStep/compareRecords`; it must not have its own numerical execution path. Keep existing rich teaching UIs and coordinate/VJP witnesses, and test desktop + Android viewport behavior. Re-register server route and Android offline assets for any shared JS additions.

## Portable experiments and branch lineage

Build new experiment snapshots through `observation_snapshots.js` (`create / restore / describe / fork`). A branch is a **complete alternate engine input**, checked through the same source and evidence identity; the parent checksum is descriptive lineage, never proof of external attestation or reversible side effects. Use the common observation UI for download/import and explicitly verify source engine output before marking a transported record as accepted. Browser handoffs use opt-in per-tab session storage and a registered lab route, not duplicate numerical fixtures.

## Native state cursors and local replay

For stateful lessons, preserve the original validated `stateTrace`. Use `Snapshots.at(Core,snapshot,stepId)` to represent an exact teaching-state cursor; `Snapshots.restore` must verify the full source run and selected step before either microscope selects it. For KV local CPU semantics only, `Snapshots.resumeKV` replays a prefix of original operations and checks its state against the complete accepted run. Other domains may add prefix-state semantics only with an actual source engine replay oracle, never by assigning an arbitrary `after` dictionary as mutable device state.

## Acceptance for each change

1. Pin source revision, focal graph/state and what is being claimed. A non-obvious equivalence needs the actual operator or state transition, not a model-family shortcut.
2. Add a small positive oracle and an adversarial negative contract. Check off-axis coupling, missing ports, invalid training state, stale traces, arbitrary graph choice and determinism as relevant.
3. Verify unchanged `tests/unified_learning_core_contracts.js`, graph codegen/gradient tests, and source-driven learning surfaces. Register new standalone HTML in the exhaustive root-page coverage test, server routes and Android asset list.
4. Render changed interactions in real Chromium at desktop and Android viewport; Android stays teaching-only. Preserve existing screenshots or add new acceptance when the UI changes.
5. For an interactive specialist, replay the **actual last accepted controller input** and compare the result with the controller’s observed output before exposing a shared microscope; independent canned fixtures are not evidence of integration. Keep source-engine arithmetic separate from state/event presentation.
6. Keep success claims scoped to results from **the final commit SHA**. A queued Action, static registry entry or same-engine consistency check is not real CPU/GPU end-to-end evidence. Only merge validated changes.

See `docs/unified-learning-core-v1.md` for current coverage and remaining domain migrations.
