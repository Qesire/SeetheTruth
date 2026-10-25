# Native training parameter-coordinate proof (2026-10-25)

## Research object and computation flow

The learner selects a genuinely executed training transaction for a small, editable directed acyclic graph, parameter node `p`, coordinate `i`, and optionally an intermediate node `z) coordinate `j`.

The single executable source remains:

```text
L.validateGraph(raw)
  → LearningLab.evaluate(G)                    authentic Forward tensors
  → GraphAutodiff.backward(G, trace, targetId) full-loss reverse gradients
  → train_state.step(G,opts)                   original per-optimizer update
  → TrainingSession.step(G,opts)               actual transaction and before/after IR
  → TrainingCoordinateProof.inspect(G,opts,{parameterId,index,viaNodeId,viaIndex})
       ├─ dL/dp[i]                            true whole-loss gradient coordinate
       ├─ (p_before[i],p_after[i],Δp[i])        real committed CPU optimizer output
       ├─ Adam/SGD/etc. moment before/after    exact stored original optimizer state
       └─ CoordinateVJP.inspect(z[j])          independent *source-derived path* view
            ├─ local VJP edges and source indices
            ├─ dL/dz[j]                        real full-loss adjoint
            └─ dL/dz[j] · dz[j]/dp[i]          through **one chosen** coordinate
```

This is **not** another optimizer or reverse-mode engine. The graph's actual original loss and per-parameter gradients are recomputed through the original functions, and the source transaction supplies the updated parameter and persistent optimizer moments. The checker compares exact gradient/update norms and optimizer iteration identity with those source records.

## Mathematical distinction

For an intermediate output `z` that lies between trainable parameter `p` and scalar loss `L`, the contribution through one coordinate `z_j` is

```math
c_{j,i} = \frac{\partial L}{\partial z_j}\,\frac{\partial z_j}{\partial p_i}
```

whereas the whole-loss gradient is `g_i = ∂L/∂p_i`. In a complete cut where **every** influence of `p_i` on the loss crosses the chosen tensor `z`, `g_i = Σ_j c_{j,i}`. Otherwise there may be bypass paths; consequently `c_{j,i}` cannot be relabelled `g_i`.

The local trace carries the exact participating source coordinate, local derivative and term where `element_lineage.js` has a native rule. If an otherwise completely known selected-coordinate dependency tree omits `p_i`, `c_{j,i}=0` is a **verified zero selected path**. This never implies `g_i=0`.

If the selected coordinate is truncated, hits an unsupported operator, encounters an intentionally detached/discrete path or cannot be safely paired with the whole-loss adjoint, its path status is **PARTIAL_OR_UNAVAILABLE_PATH** with no fabricated numerical contribution. The whole-loss gradient may still be valid through `GraphAutodiff`.

Parameter updates expose `θ_{t+1,i}-θ_{t,i}` and the native first/second moment states. The numerical update is not inferred from the selected local contribution: every optimizer updates from the **whole** gradient and its own state.

## UI and protocol

The Learning Studio adds a parameter-coordinate inspector below the unified graph training observer. The user selects a parameter node ID, parameter coordinate, intermediate node ID and intermediate output coordinate. The action `Core.executeLab('training','coordinate_proof',...)` is a normal registered specialist action and the caller replays it with `Core.replay`. Its observation capability is correctly `none`: the returned proof contains verified numeric source evidence, but does not fabricate a new state/tensor microscope frame.

The UI does not execute a new optimizer step on the visible canvas. It starts from a previously executed training step and uses exactly that step's input graph and training options. The proof is cleared on invalid input to avoid displaying a stale result.

## Independent verification and boundaries

1. `tests/training_coordinate_proof_contracts.js` validates source transaction identity, six native optimizer rules, all four coordinates of a 2×2 projection cut, finite-difference gradient and per-path chain rule; incorrect source, missing node, out-of-range coordinates, incompatible objectives and tampering are rejected.
2. `tests/torch_training_coordinate_parity.js` compiles the same multi-path graph through the existing standalone PyTorch exporter and runs real CPU Float64 `torch.autograd` in Python. All eight trainable parameter coordinates are compared against the independently evaluated Python gradients.
3. `tests/browser_training_coordinate_proof.py` checks real desktop/mobile Learning Studio training, source proof and invalid-coordinate rejection.

PyTorch parity is established only for the validated fixture and supported lowering; registration does not upgrade all 150 operators to autograd-verified. Device or pretrained model training is not executed.

## Next extension

The remaining higher-value interface is a verified **per-coordinate backend capability certificate**, recording the chosen graph, source revision, shape bounds, executed independent backend, tolerance and negative cases. That evidence can feed the capability explorer without promoting an allowlist to an unearned verified status.
