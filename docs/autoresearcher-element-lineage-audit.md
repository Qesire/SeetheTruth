# AutoResearcher mechanism audit: element-wise lineage through activations and normalization

Date: 2026-10-23. Object pinned before the intervention: `Qesire/SeetheTruth@3458de6050b7e7180b1558b26a8041ce1f405045`. Audit method: AutoResearcher `AGENTS.md`, `.agents/skills/mechanism-research/SKILL.md` (MAP_FLOW / DERIVE / DISCRIMINATE / REPAIR), and `core/OBJECT_RECONSTRUCTION.md`. This is an engineering mathematical audit, not a claim of new scientific novelty.

## Decision and source reconstruction

Goal: a learner selects one scalar of an arbitrary valid editable graph and follows its **actual coordinate-level dependency and sensitivity** backward across nodes, including entire channels coupled by a normalization, with correct status when values are not available.

Relevant computation: `learning_lab.js` validates and orders the graph and produces bounded `evaluate` steps. `graph_readiness.js` derives node shapes; `shape_index_rules.js` exposes index remaps; `element_lineage.js` descends per-coordinate dependencies and sums sensitivities across shared upstream vertices. `activation_math.js` already defines the forward functions and analytic VJPs for SiLU, erf/tanh GELU and several other activations; `block_catalog.js` provides row-wise QK Norm and LayerNorm; `learning_lab.js` provides row-wise RMSNorm.

**Observed gap:** `element_lineage.js` previously stopped at an otherwise executable `silu` brick, even though the actual analytic derivative existed in `activation_math.js`. For row normalizations it lacked the `D×D` coupling of the last feature dimension. Numeric trace identity previously checked ordered node IDs and shapes, not numeric correspondence; same-shaped parameter edits could give stale local derivatives a misleading bounded-numeric label.

## Reconstructed mathematics

For an elementwise activation `y_i=f(x_i)`, the only input dependency is coordinate `i` with local factor `f'(x_i)` from the *same* activation formula used in the runnable teaching stack. In structural mode the dependency is still exact, but the local derivative is unknown until a numeric input is supplied. At nondifferentiable kinks (e.g. Leaky ReLU at zero, HardSwish at ±3 and ELU with α≠1 at zero), the scalar library supplies an explicit branch-gradient convention, not a mathematically unique derivative.

For a last-axis row `x ∈ R^D`, define `r = sqrt(D^{-1} Σ_k x_k² + ε)`. The complete RMSNorm / QK Norm Jacobian is

```
y_i = x_i / r
∂y_i/∂x_j = δ_ij/r − x_i x_j/(D r³)
```

For LayerNorm define `μ=D^{-1}Σ_k x_k`, `z_i=x_i−μ`, `s=sqrt(D^{-1}Σ_k z_k²+ε)`. Then

```
y_i = z_i/s
∂y_i/∂x_j = (δ_ij−1/D)/s − z_i z_j/(D s³)
```

This is **not** a diagonal scaling: a scalar output depends on every feature in its normalization row. A forward trace with unchanged shapes but changed parameters need not have unchanged values or derivatives.

## Controlled discrimination and intervention

Implemented on the feature branch:
1. Dispatch scalar activation derivatives through the existing `activation_math.js` implementation rather than fork a second formula; both base and registered `nd_` aliases use identical scalar rules.
2. Add exact last-axis coordinate expansion and local Jacobian weights for `rmsnorm`, `qk_norm`, `layernorm`. Shared-coordinate reverse accumulation remains unchanged.
3. When the caller supplies a numeric trace, recompute the **bounded deterministic** teaching graph and compare the kind and value of every ordered step. Reject partial/stale/altered traces; keep truly source-free calls structural. This is a provenance/consistency guard **within the same evaluator**, not independent numerical or model-runtime verification.
4. Add independent central-difference checks: perturb upstream parameter coordinates, execute the full graph, and compare the resulting scalar derivative to the lineage VJP. Include off-diagonal normalization terms, same-shape parameter mutation, tampered values, and missing-value negatives.

Minimum evidence commands: `node --check element_lineage.js` and `node tests/element_lineage_contracts.js`; CI also executes both as part of its evidence-contract job. Browser acceptance verifies the unchanged user-facing inspector separately.

## Remaining distinct opportunities

- **High-value next mechanism:** teach the *rank structure* of normalization sensitivity by displaying the diagonal term and low-rank correction as separate graphical dependencies. This connects Jacobian sparsity, feature coupling and computational cost without altering the execution graph.
- Expand coordinate-exact rules for other executable operators (e.g., composed GLU gates, Attention projections, pooling, masked operations), only after their port semantics and actual VJP are reconstructed; a generic approximate rule would incorrectly claim dependency completeness.
- Preserve two independent evidence axes: deterministic teaching-number consistency (this branch) versus independent Python/Torch forward/backward parity and actual model capture. No GPU performance, checkpoint-faithful normalization, end-to-end training or new-paper originality claim follows from the change.

## Candidate gates and result recording

The change is acceptable only when the updated Node contract passes on the exact branch head, existing CI arithmetic tests stay green, and static Chromium teaching acceptance remains unchanged. A started workflow is not a completed passing workflow. Capture final CI run URLs and merge SHA in the PR, not by inventing performance metrics.