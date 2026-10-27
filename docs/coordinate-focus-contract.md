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

## Native scalar propagation and optimizer timeline

`coordinate_journey.js` receives an **already accepted** `coordinate_drilldown` report. It does not recalculate graph values, numerical derivatives, VJP or optimizer updates. Its step sequence is constructed from the original `element_lineage.vertices/edges`:

1. Seed the selected output coordinate with sensitivity (1).
2. Traverse the actual coordinate-dependency DAG in reverse computational direction. For every native edge (u\to v), present its port, rule and local derivative (J_{u,v}).
3. Calculate each displayed path contribution (s_u J_{u,v}), or preserve `null` if a required numerical derivative is not evaluated. Sum the contributions only after all incoming dependencies have arrived at the same upstream element.
4. Verify every accumulated sensitivity against the **native** `element_lineage` result and check the selected root and edge/node counts against `coordinate_vjp_bridge`. A mismatched report is rejected.
5. Label each element as an upstream element, a genuine multi-path merge, or an actual source coordinate according to the original dependency graph. Unknown rules and truncated frontiers remain visible.

This is a walk through an authenticated bounded **unit-seed** VJP, not a generic whole-loss gradient or an execution profiler. The timeline can be large for attention and full FFN blocks; the interface renders one step at a time and provides previous/next buttons and an accessible range control.

For native training proof, the same stepper instead displays: parameter before, **whole-loss gradient**, optimizer state (including native first and second moment entries where available), parameter after/delta, full loss before/after, and separately the selected intermediate-path contribution. The loss change is a property of the complete training transaction and is not attributed to the isolated parameter element.

The inspector clears previous selection and timeline when a graph changes, a different recipe is opened, or a parameter-coordinate proof fails. Accepted state experiments are kept distinct from graph numeric samples.

### UI and tests

- `coordinate_journey_ui.js`: bounded single-step detail view using ordinary DOM text, reverse dependency labels, mobile wrapping and reduced-motion preference.
- `tests/coordinate_drilldown_contracts.js`: topological count, shared-source sum, native sensitivity cross-check, static analytic/unknown derivative separation, optimizer moment and loss evidence.
- `tests/browser_universal_microscope.py`: real Chromium numerical focus, timeline buttons, range control and screenshots.
- `tests/browser_unified_observation_ui.py`: static analytical Jacobian and numerical graph stepper on PC/mobile.
- `tests/browser_training_coordinate_proof.py`: parameter timeline and actual shape-derived invalid coordinate with stale-proof invalidation; screenshots are archived separately.

## Scalar-to-canvas cross-highlighting (v1)

A step in the authenticated reverse-coordinate timeline now maps to **exactly one existing directed graph edge and input port**:

- `coordinate_journey.js` preserves the downstream element index (`sourceIndex`) and upstream index (`index`) from the original `element_lineage` vertices.
- `coordinate_journey_canvas.js` validates each step's identity against the active `graph.edges`. For a reverse contribution `out@j → input@i`, it highlights the actual *forward* wire `input → out : port` while displaying the reverse sensitivity direction separately.
- The canvas renders SVG paths with `data-edge-from`, `data-edge-to`, `data-edge-port` attributes, using native DOM SVG creation instead of interpolating untrusted ids into HTML strings. Position/zoom recomputation preserves the active edge identity.
- The corresponding upstream node displays `[i]`, the downstream node displays `[j]`, and the original input port receives a separate highlight. The element-lineage panel highlights precisely the two native vertex keys and expands the pertinent operator group.
- A user-requested **定位画布积木** action brings the upstream node into the viewport. Step changes update the graph without forcing navigation or mutating numeric results.
- A graph edit, preset switch, or a new focused coordinate clears the previous visual overlay alongside its provenance. Independent state-scenario records never claim a numeric wire relationship with an unrelated lesson.

Mathematical authority remains with `element_lineage` and `coordinate_vjp_bridge`; this module is **coordinate navigation**, not an alternative differentiation or trace engine.

Validation: `tests/coordinate_drilldown_contracts.js` checks exact port identity and stale/missing-wire rejection. `tests/browser_universal_microscope.py` checks SVG edge identity, highlighted upstream/downstream scalar cells, node/socket styles, and explicit locate navigation in Chromium.

## One native training transaction: forward → backward → optimizer

The coordinate journey now relates an **accepted training parameter** `θ_j` to a selected intermediate activation `y_i`, the full scalar loss `L`, and the optimizer state. All displayed numbers originate from the same replay-verified `training/coordinate_proof` record.

The native engines independently produce three quantities:

```text
selectedAdjoint      = ∂L/∂y_i                 [whole-loss graph_autodiff]
parameterUnitVJP     = ∂y_i/∂θ_j               [unit-seed scalar element_lineage]
selectedContribution = (∂L/∂y_i)(∂y_i/∂θ_j)  [coordinate_vjp_bridge]
wholeLossGradient    = ∂L/∂θ_j                [whole-loss graph_autodiff]
```

The source proof checks the factorization numerically. The coordinate-focus adapter checks it again and will reject a record with inconsistent values. A selected contribution can be **zero**, **numerically verified**, or **unavailable**; these states remain distinct. The full parameter gradient need not equal the selected contribution because other intermediate coordinates may contribute.

An accepted selected-output route is stored as immutable `transition.selectedRoute`, with its verified node ID, coordinate, forward value, Loss adjoint, unit VJP, and native status. The training stepper presents these phases in causal order:

1. Parameter before the update.
2. Native selected intermediate forward value, located on its real graph node.
3. Native whole-loss adjoint at that intermediate coordinate.
4. Factorization and contribution through the selected output to the parameter.
5. Entire Loss gradient at the parameter.
6. Native optimizer moment changes and update type.
7. Parameter value after update.
8. Entire Loss before/after, located on the true source Loss node.

A training proof without a selected intermediate retains a shorter route and does not synthesize missing stages. A missing derivative is reported as *unevaluated* rather than zero. Each phase highlights its authentic graph node; the Loss stage does not reuse the parameter index as though it were a Loss coordinate. Editing the graph, parameter selection or route invalidates previously accepted evidence.

Regression contracts: `tests/training_coordinate_proof_contracts.js`, `tests/coordinate_focus_contracts.js`, `tests/coordinate_drilldown_contracts.js`, and Chromium `tests/browser_training_coordinate_proof.py`.

## Validations

- `node tests/coordinate_focus_contracts.js`: rank/index/axes/finite value limits, same native A/B delta, replayed graph observation, native training transition and controller notifications.\n- `node tests/coordinate_drilldown_contracts.js`: original static element lineage, numeric unit-seed VJP, stale-graph rejection, and state/optimizer distinctness.
- `node tests/state_ab_learning_bridge_contracts.js`: exact state elements, accepted graph node and port topology, A/B navigation and execution handoff.
- `python tests/browser_state_ab_bridge.py`: desktop/phone real Chromium state → graph → same coordinate route and screenshot evidence.
- `python tests/browser_universal_microscope.py`: actual microscopic numeric matrix cell emits common focus.
- `python tests/browser_unified_observation_ui.py`: static graph coordinate selection followed by replayed numerical upgrade and on-demand VJP.\n- `python tests/browser_training_coordinate_proof.py`: original optimizer parameter before/after and whole-loss gradient translated into the shared focus readout.
- `node scripts/sync_runtime_assets.js --check`: Web and Android offline delivery agreement.

Extension criterion: an additional model, optimizer or trainer should first expose its own accepted native execution record and coordinate evidence. It can then provide a small `from...()` adapter without altering graph math or publishing a second global selection protocol.
