# Unified Coordinate Focus v1 · native evidence and graph selection

## Contract

`coordinate_focus.js` is a small, dependency-free adapter around **selection state**, not an additional evaluator. It publishes `seethetruth-coordinate-focus-v1` records:

```js
{
  schema, domain, executionId, nodeId,
  shape, coordinate, flatIndex, axes,
  evidence, value?, branch?, path, note
}
```

`shape` and `coordinate` are integer rank-1–5 arrays with bounded total elements (at most 16,384). Coordinates are checked elementwise before flattening. Named axes must be distinct and match tensor rank. `value` is permitted only for a recognized numeric provenance; static focus cannot create a numerical value. Records and coordinate lists are immutable. One `store()` instance serves selection updates and subscribers on each page; page-to-page links deliberately carry only bounded location parameters, not invented serialized runtime states.

Evidence and scope remain explicit:

| Source | Focus domain | Value | Provenance |
|---|---|---|---|
| `semantic_state_examples.js` via unified `executeLab` | `semantic_state` | Exact native Float64 tiny tensor entry | `EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS` |
| Current free-form DAG, shape readiness / shape-index interaction | `guided_graph` | Omitted | `STATIC_SHAPES_NO_EXECUTION` |
| Actual cell in `compute_microscope_ui.js` output tensor | `computation_microscope` | Original `CORE.dot` result | Native frame's MICRO or source-pinned synthetic tier |
| `unified_observation_ui.js` graph tensor after `inspectObservationStep` | `unified_observation` | Native replay-checked `detail.value` | Numeric frame tier |

Two separately accepted graph records are compared through `SeetheTruthUnifiedCore.compareRecords`. Direct per-element `CoordinateFocus.compare` is for the immutable *same state-execution* A/B result, using matching domain/execution identity, shape, coordinate and evidence. A state laboratory's values and a lesson's seeded graph values remain separate examples. Their operator topology may align without any cross-sample equality claim.

## User interaction and API

1. The state laboratory's per-coordinate grid calls `CoordinateFocus.fromState` and publishes it through `SeetheTruthCoordinateFocusSession.select`. Matched A/B coordinates reuse `CoordinateFocus.compare`.
2. The graph handoff replays the same canonical scenario, checks nodes and edges, and reconstructs `fromState` from the URL's `state_case`, `state_branch`, and `state_index`. Its dependency chips navigate within the guided graph. Source state identity remains the original state execution.
3. In Learning Studio, the computation microscope's optional fourth callback fires on user cell selection with the original frame, step, coordinates and `CORE.dot` detail; the workbench publishes `fromMicroscope` and does not recompute another value.
4. `window.SeetheTruthUnifiedWorkbench.focusCoordinate(nodeId, coordinate)` validates against `SeetheTruthUnifiedCore.inspectGraph(...).path.steps`, updates the native shape-index map, and publishes **static-only** coordinate identity. External lessons may use this method without introducing model-specific branches.
5. The shared graph observation panel uses replay-verified `inspectObservationStep` to publish `fromObservation`, retaining the selected node and coordinate. The original unified observation engine remains the numerical authority.
6. `window.SeetheTruthCoordinateFocusSession.current` and `window.SeetheTruthUnifiedWorkbench.coordinateFocus` expose the current normalized selection for compatible teaching UI modules. Changes are local to the active page.

## Coordinate drilldown and training transition

`coordinate_drilldown.js` resolves the shared selection against the original computation engine, rather than introducing a second element propagation engine.

- `semantic_state` returns **mechanism correspondence** only. State experiment coordinates and seeded course graphs are distinct numeric samples.
- `guided_graph` static selections call `element_lineage.inspect` with the actual graph, shape and output index; no synthetic scalar value or local derivative is attached.
- `computation_microscope` selections require the same bounded numeric graph execution, original canonical microscope frame and selected value. `coordinate_vjp_bridge.inspect` authenticates the entire frame and uses a **unit output seed**, while `element_lineage.inspect` supplies the original scalar graph. All three selected values must agree.
- `unified_observation` selections re-run `inspectObservationStep` on the accepted graph record before admitting numerical VJP. Observations from another graph or changed input do not qualify.
- `training_parameter` is a **separate transition type**. It is produced only by `fromTraining(accepted, Core.replay(accepted))` after the original `training/coordinate_proof` engine supplies a bounded parameter coordinate, before/after optimizer values and whole-loss gradient. The inspector retains `parameter.after` as the selected value and explicitly separates whole-loss gradient from any selected intermediate-path contribution. The training proof UI remains the arithmetic authority.

In the Learning Studio inspector, the user selects a native coordinate and clicks **追踪此坐标的实际依赖 / 局部 VJP**. The existing element-lineage view expands its actual upstream coordinate graph, and an informational report describes supported/truncated VJP frontiers. The button runs graph exploration only on user interaction, not on every mouseover, which avoids needless repeated compute. All display values are derived from the existing engines.

## Validations

- `node tests/coordinate_focus_contracts.js`: rank/index/axes/finite value limits, same native A/B delta, replayed graph observation, native training transition and controller notifications.\n- `node tests/coordinate_drilldown_contracts.js`: original static element lineage, numeric unit-seed VJP, stale-graph rejection, and state/optimizer distinctness.
- `node tests/state_ab_learning_bridge_contracts.js`: exact state elements, accepted graph node and port topology, A/B navigation and execution handoff.
- `python tests/browser_state_ab_bridge.py`: desktop/phone real Chromium state → graph → same coordinate route and screenshot evidence.
- `python tests/browser_universal_microscope.py`: actual microscopic numeric matrix cell emits common focus.
- `python tests/browser_unified_observation_ui.py`: static graph coordinate selection followed by replayed numerical upgrade and on-demand VJP.\n- `python tests/browser_training_coordinate_proof.py`: original optimizer parameter before/after and whole-loss gradient translated into the shared focus readout.
- `node scripts/sync_runtime_assets.js --check`: Web and Android offline delivery agreement.

Extension criterion: an additional model, optimizer or trainer should first expose its own accepted native execution record and coordinate evidence. It can then provide a small `from...()` adapter without altering graph math or publishing a second global selection protocol.
