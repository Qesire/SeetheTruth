# Free graph training ↔ unified state lineage (v1)

This feature joins **the exact original small-graph training transaction** with the existing unified state/observation/snapshot protocols. It does not add a new optimizer, independent VJP, model weights or GPU executor.

## Source execution graph

For an edited, well-typed LearningLab graph `G` and explicit training options `u`:

```text
G + options
   │
   ▼
train_state.step(G,u)                           source of numeric truth
   ├─ LearningLab.evaluate(G_scope)             before Forward and tensors
   ├─ GraphAutodiff.backward(G_scope)           reverse VJP to loss
   ├─ Optimizers.step(parameter,gradient,state) true chosen bounded CPU update
   └─ LearningLab.evaluate(G'_scope)            updated Forward
   │
   ▼
training_session.step(G,u)
   ├─ beforeIR, afterIR                          stable node/port identity
   ├─ transaction: previous, next, VJP records
   └─ graph G' + per-parameter update records
   │
   ▼
node_timeline.fromSession(G, transaction)
   ├─ inputSources(node,port)                    original graph edges
   ├─ forwardValue, updatedValue                 computed original values' stats
   ├─ gradient, localRule, onLossPath            source-calculated backward
   └─ optimizerState(first/second moments)       true stored parameter state
   │
   ▼
StateMicroscope.graphTraining({session,timeline})
   └─ seethetruth-state-microscope-v1             native typed transition steps
       ├─ graph:forward:*                        original DAG order
       ├─ graph:backward:*                       reverse Loss Path only
       ├─ graph:optimizer:*                      actual trainable parameters only
       └─ graph:updated:*                        original DAG order after update
             │
             ▼
Core.executeLab('training','graph_step',{graph,options})
   ├─ Core.observe / replay / inspectObservationStep / compareRecords
   ├─ ObservationSnapshots.create / restore / at(stepId)
   └─ Learning Studio shared read-only observation UI
       ├─ chosen graph node → native training phase
       └─ selected training phase → exact graph node
```

The mathematical computation is `h_i=f_i(h_{parents(i)},θ_i)`, `L=ℓ(h_target)`, reverse-mode `g_i = Σ_(j∈children(i)) J_(j←i)^T g_j`, followed by `(θ_i',s_i')=OptimizerStep(θ_i,g_i,s_i)`, and `h_i'=f_i(h'_{parents(i)},θ_i')`. The *actual* source for each `f_i`, `J^T` and optimizer step is the existing repository implementation. The state microscope extracts validated identities, gradients, summary statistics and causal phase edges only.

## Causal links and semantics

- Forward dependencies are the actual `ExecutionIR.inputSources` port → source-node edges. When a node has two or more ports from the same source, its inventory retains all ports; the **event dependency set** intentionally lists the ancestor event once.
- Backward events exist **only** for nodes reported `onLossPath` by the original `GradientInspection`. They depend on the node's original forward plus already-produced consumer backward events; no unused node receives an invented VJP.
- Optimizer events exist only for update records from `training_session.step`. They depend on that parameter's backward event; their before/after states derive from true persisted optimizer moments, step counts and source gradient/update norms.
- Updated-forward events follow graph topology again and depend on updated parent outputs, plus the actual parameter optimizer event where applicable.
- A source-specific `graphLinks[nodeId]` map connects these phase IDs to the same native graph node. The map is part of the replay-verifiable state trace; external code must not treat it as free-floating inferred provenance.

## Browser integration

The Learning Studio now has a source selector: **当前计算图** versus **已执行的训练步骤**. A training selection always calls the same `Core.executeLab('training','graph_step')` with the original graph and optimizer options for the chosen step. Its node timeline and final graph are compared with the studio's previously accepted execution before the observer is allowed to show them.

Select a node in the canvas and click **定位所选积木的训练轨迹**; the observer chooses the native optimizer event if present, otherwise its backward event, otherwise its forward event. Conversely selecting and inspecting a native training event highlights the original corresponding node in the canvas. This is read-only navigation: inspecting does not silently perform a second optimizer update in the visible graph.

The shared observation/snapshot interface can replay graph-training source inputs, pin them as baselines, compare different learning rates, export a selected transition cursor and restore the native teaching phase. The FNV snapshot digest is a deterministic drift check, not cryptographic provenance.

## Limits / intended next work

- This contract is for bounded typed CPU graph training; it is **not** PyTorch model state, optimizer device checkpoint or generalized backpropagation-through-time.
- Values displayed in the state view are **source-computed summaries**, while actual source forward tensors and VJP/optimizer results remain in `record.value.session` and `record.value.timeline`. Do not label the summaries independent numerical oracles.
- The state trace is bounded to at most 70 training nodes and 250 events, matching the native microscope's resource constraints.
- Training UI still has its existing one/two-to-eight-step controller; the shared observer selects an existing step and verifies it, but does not migrate the entire training UI to a replacement controller.
- Subsequent expansion should support VJP coordinate proof and a typed, reusable phase-link contract, with actual small-tensor verification for new operator families. Do not default arbitrary registered blocks to gradient-verified.

## Acceptance

`node tests/graph_training_state_lineage_contracts.js` checks actual numerical session identity, exact port topology, reverse path, update states, updated forward, source replay, portable checkpoint roundtrip, altered source/cursor rejection and five optimizer families.

`python tests/browser_graph_training_lineage.py` exercises real desktop and mobile Learning Studio training, switches the shared observer, navigates source nodes in both directions and checks previously selected training steps.

Both are required alongside the existing graph, source/VJP, state microscope and Android packaging CI suites.
