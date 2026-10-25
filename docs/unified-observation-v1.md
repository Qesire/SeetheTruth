# Unified Observation and Replay Contract (v1 extension)

The project has **one observational envelope**, not one fictional compute engine.
`SeetheTruthUnifiedCore` owns identity and dispatch; `LearningLab`, the bounded N-D graph executor, Wan FFN, the state microscope and specialist domain engines remain the sources of numerical or event truth.

## Public interfaces

| API | Accepted record | Behavior |
| --- | --- | --- |
| `observe(record)` | inspected free graph or accepted specialist execution | Returns versioned trace inventory; preserves genuine causality/shape/precision evidence |
| `replay(record)` | same | Repeats accepted input through original engine and compares trace/value/metadata; static graphs are revalidated, not executed |
| `inspectObservationStep(record,{kind,stepId,coordinate?})` | same | Replays first; state step returns before/after/dependencies; numeric tensor step calls the existing `compute_microscope.dot` on an exact full-rank coordinate |
| `compareRecords(left,right)` | two graphs or two same-operation specialist executions | Independently verifies both and reports exact value/trace equality and IDs of changed recorded steps |
| `observeExecution`, `replayLab`, `observeGraph`, `replayGraph`, `compareExecutions` | legacy/domain-specific explicit surfaces | Retained for existing consumers |

Records use `seethetruth-observation-v1`, a reference envelope. For a tensor step, the public inventory preserves its **real** upstream `from` and `port`; state step dependencies reference the actual recorded event IDs. No invented dependency edges and no implicit promotion of static/source-derived data to checkpoint inference or GPU measurement. A computation step advertises shape and whether numerical values exist; actual values/coordinate proofs are read from the source frame only on request.

An operation without a valid native trace returns `traceAvailable:false` and an empty source list. A source-only, unexecuted graph yields the same status: its absence of numerical steps is not an error or a disguised successful run. `compareRecords` uses `valueEqual:null` when two static graphs have no values to compare.

## Source-backed examples

```javascript
const Core = require('./learning_experience_core.js');
const L = require('./learning_lab.js');
const graph = Core.inspectGraph(L.preset('residual'), {numeric:true});
const observation = Core.observe(graph);
const coordinate = Core.inspectObservationStep(graph, {
  kind:'tensor_computation', stepId:'sum', coordinate:[0,1]
});
const replay = Core.replay(graph);
const unchanged = Core.compareRecords(graph,graph);

const wan = Core.executeLab('wan_ffn','trace',
  {options:{seed:17,tokens:3,dim:4,hidden:8}});
const wanDot = Core.inspectObservationStep(wan, {
  kind:'tensor_computation', stepId:'up', coordinate:[1,2]
});
```

## Semantics and constraints

1. **Same source arithmetic.** The microscope never recalculates the original operator from a second synthetic fixture. `compute_microscope.dot` validates original displayed arithmetic, including GEMM terms where supported.
2. **Replay first.** Coordinate inspection and comparisons re-execute and verify accepted graph or specialized input before serving a result. This detects stale numeric values, changed event history and tampered evidence declarations.
3. **Typed boundaries.** Tensor DAG coordinates and scheduler/KV state transitions are separate proof types with the same versioned identity envelope. A state event cannot be selected using a tensor coordinate.
4. **Evidence stays local.** Trace tiers come from the source microscope; `replayVerified` means deterministic equality within its declared reference engine, never real-model or hardware profiling attestation.
5. **No unverified undo.** Diffing immutable accepted records does not imply a reversible external side effect or a shared cross-engine checkpoint format.
6. **Capability is granular.** Static composition, numeric execution, gradients, code generation, coordinates and state replay are distinct obligations. The envelope makes no universal backend eligibility claim.

## Verification

Run `node tests/unified_observation_contracts.js` plus `node tests/unified_learning_core_contracts.js` and `node tests/universal_microscope_contracts.js`. CI runs the new observation test in the evidence-contract job, and preexisting Chromium/Android tests still guard the teaching surfaces. New specialist engines must register typed sources and pass replay, stale-state, invalid-coordinate and provenance negative tests before claiming universal observability.

## Remaining migration

The v1 bridge supports graph microscope coordinates and those specialists that already have verified native frames. The next improvements are: UI-level shared coordinate selection across Graph/State/Wan, stable version IDs for long workflows, explicit state checkpoints for reversible **local** simulations, and reliable independent backend parity for new adapters. Do not model these as completed merely because an observation record can be constructed.

## Shared browser consumer (PR #82)

`unified_observation_ui.js` provides one **read-only DOM consumer** of these APIs. A host page supplies only its existing accepted execution via `readAccepted()`. The panel never mutates parameters, issues its own independent numerical approximation, or upgrades an evidence tier. It presents:

- A replay-verified current execution and trace/source inventory, using the source engine and evidence carried by the accepted record.
- Real state-transition dependencies or real compute-graph `{from,port}` dependencies (no guessed edges), with full-rank tensor coordinate selection delegated to `inspectObservationStep`.
- An immutable JSON snapshot of an explicitly pinned baseline and a `compareRecords` delta across the same domain, with static-only null numerical parity.
- Visible rejection of stale, malformed or absent accepted source records.

The PC Learning Studio, Wan FFN source microscope, and DeepInfer Worker are the initial UI consumers. Existing specialized visualization remains in place; the common panel is not a replacement for element lineage, local VJP, or Worker timeline. Registered HTTP and Android offline teaching assets include the same UI source.

Browser acceptance: `python tests/browser_unified_observation_ui.py` must exercise 1366px and 390px viewports, numeric graph coordinates, static/no-execution status, Wan gate-change comparison, Worker state inspection and original-source evidence without page errors. A future lab must provide a validated accepted-record provider rather than a canned alternate fixture.
