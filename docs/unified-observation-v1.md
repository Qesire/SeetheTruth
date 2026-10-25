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

## Portable execution snapshots and experiment branches (PR #83)

`observation_snapshots.js` extends the accepted execution protocol with portable **input recipes**, exact source identities and a deterministic result/trace checksum. A snapshot never transports an independently authenticated tensor trace: the original graph engine or specialist engine must regenerate the result on import before it can be marked `replayVerified`.

`Snapshots.create(Core, accepted, label?)` records a bounded `seethetruth-observation-snapshot-v1` JSON object (maximum 192 KiB). `Snapshots.restore(Core, snapshot)` re-executes the original graph/source action, checks evidence identity and the native result/trace checksum, and returns a verified record. `Snapshots.describe(Core, snapshot)` resolves the registered owning route **without executing** and always marks it as unverified. `Snapshots.fork(Core, parent, completeChangedInput, label?)` runs a complete alternate source input in the same domain, verifies it, and returns an independent snapshot with a `parent` checksum reference. Parent refs form *experimental lineage*, not access authority or verified external side-effect rollback.

The checksum is a bounded JavaScript FNV-1a32 *diagnostic* for accidental drift. It is **not cryptographic**, and it is neither a signature of file origin nor a statement about GPU execution. Replay through the original engine is the evidence mechanism. A malicious author could forge an internally consistent snapshot, so evidence remains limited to the independently available reference engine.

`unified_observation_ui.js` has the same controls on the Learning Studio, Wan FFN and DeepInfer state labs: export a verified baseline to JSON; import a baseline; make a changed-input experimental branch; compare against the active controller result. Importing a snapshot on another lab displays its identity and **UNVERIFIED** status. An explicit click follows the registered source route and uses a short-lived, same-origin `sessionStorage` handoff in that browser tab; the destination must replay before exposing a baseline. Files remain locally owned, and no cloud persistence or external observation store is involved.

Acceptance: `node tests/observation_snapshots_contracts.js`, `python tests/browser_unified_observation_ui.py`, and existing graph, FFN, KV, browser/mobile and evidence tests. Negative cases cover edited source inputs without matching replay, altered result checksums, inflated evidence, malformed parent links, non-finite numbers, oversized files, unknown engines, and inappropriate numeric promotion.

## Native state-step checkpoints and bounded KV continuation (PR #84)

Portable v1 snapshots now optionally contain a validated `cursor` for a genuine `state_transition` trace. `Snapshots.at(Core, portable, stepId)` replays the source experiment and records only `stepId`, exact step index, native evidence tier, and a deterministic digest over the returned state-step proof. `Snapshots.restore(Core, withCursor)` replays the full original engine first, verifies the full-run checksum **and** source-step digest, then returns `checkpoint` with authenticated-within-this-engine `before`, `after`, `changes` and `dependencies`. The digest is diagnostic, not a cryptographic attestation.

`unified_observation_ui.js` now supports **export selected state step**. A local import of such a snapshot restores the unified observation selection and its native state microscope to the verified step. It restores a **read-only teaching view**; the page controller, CUDA state and external scheduler are unchanged. Existing end-result snapshots remain importable.

For DeepInfer KV BlockManager, `Snapshots.resumeKV(Core, withCursor)` additionally uses the actual KV reference engine to replay the exact prefix of operations ending at the selected step. It checks both native prefix state and the original complete run at that index, then applies each remaining operation to the reconstructed CPU state using the original `BlockManager.advance()` and independently checks every resulting state against the full replay. It returns verified intermediate suffix states and the remaining original operations. This is a local CPU model operation: it does not reconstruct physical GPU blocks, production token streams or an external process. The v1 state cursor also works for Worker and Prefill traces, but these are **observational rewind**, not mutable runtime execution rewind.

Acceptance: `node tests/state_checkpoint_contracts.js`, `python tests/browser_unified_observation_ui.py`. Negative contracts cover nonexistent steps, shifted indices, changed before/after data, malformed cursor/digest, tensor trace in a state slot, absent KV cursor, and source drift.

## Native Bigram training recovery: microbatch, optimizer, sampler (PR #85)

The existing `engineering_training_pipeline.js` already owns the real bounded Bigram training transition, including exact 8×8 weights, first/second moment arrays, pending gradient sums, sample indices, deterministic shuffle cursor, learning-rate schedule and optimizer.step. The unified core now supports `executeLab('engineering_pipeline','timeline',{config,actions})`, where `actions` is a bounded sequence of 1–24 explicit `micro` or `optimizer` operations. Only the original `microStep()` and `optimizerStep()` are called; an invalid optimizer operation while accumulation is incomplete is rejected by the native engine. Each action produces a complete original-engine checkpoint, and the existing state microscope independently validates that checkpoint sequence as `train:i` events. Its shared `before/after` projection now exposes the consumed sample indices plus weighted diagnostics for weights, AdamW first/second moments and the pending gradient buffer, in addition to optimizer step and sampler cursor. These diagnostics are for coordinate-free teaching comparisons; the native checkpoint remains the authoritative complete state.

`Snapshots.at(core, Snapshots.create(core, timelineRecord), 'train:i')` exports a cursor over any verified training checkpoint, including partially accumulated gradients. `Snapshots.resumeTraining(core, cursor)` verifies the complete source timeline and the exact prefix, calls the native `Pipeline.create(config).restore(checkpoint[i])`, and then invokes every remaining recorded `microStep/optimizerStep`. Each intermediate checkpoint is compared against the original full accepted timeline, not just the final weights. Restoration is local bounded CPU arithmetic; there is no claim of PyTorch/GPU optimizer restoration or real training accelerator checkpoint portability.

The interactive `engineering_pipeline.html` now includes a distinctly labeled independent **native training timeline** with the original model/optimizer settings, buttons for one microbatch and one optimizer step, and the same shared observation/snapshot interface. It does not pretend to mirror the main six-lesson controller. The learner can export any `train:i` step, import it into the owner page, verify pending gradient/optimizer/sampler state and inspect the re-executed suffix. The original lesson controller is deliberately left unchanged by read-only imports. The same view is available in offline Android teaching assets.

Validation: `node tests/native_training_continuation_contracts.js` plus `python tests/browser_native_training_timeline.py`. Negative cases include early optimizer step, unknown action, oversized timeline, mutated inputs and state cursors, and attempt to apply mutable recovery to scalar/legacy observer-only traces.
