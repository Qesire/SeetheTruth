# Unified graph dependency index (2026-10-22)

## Change

`learning_lab.indexGraph(validatedGraph)` is the common DAG dependency plan used by the core topological ordering, selected-output ancestor slicing, and readiness/port resolution. It preserves graph insertion order for tie-breaking, never imposes an architecture whitelist, and rejects cycles. Its results are `ordered`, `parents`, `children`, and `incomingByPort`.

- `order` now projects `indexGraph(...).ordered`.
- `branch_execution.slice` obtains ancestors from the same index.
- `graph_readiness.check` obtains execution order and port-source lookup from the same index.
- `tests/unified_graph_dependency_contracts.js` covers topology, fan-out, selected branch, missing ports, and cycles. CI runs it.

## Next integration seams

1. Define a versioned operator contract with port semantics, shape inference, numeric evaluation, gradient eligibility and codegen lowering as separate capabilities; compose the existing catalogs without claiming all backends support each kind.
2. Build one immutable execution trace format that preserves graph node identity, tensor axes, state versions and evidence type (symbolic, deterministic numeric, actual capture).
3. Route PC studio, Android teaching, specialized DeepInfer labs and paper cards through adapters to these contracts, not duplicated model templates.
4. Retain intentionally independent numerical oracles and backend parity tests, so sharing metadata does not conceal correlated arithmetic bugs.

## Verification

`node tests/unified_graph_dependency_contracts.js`

`node tests/graph_readiness_contracts.js`

`node tests/branch_execution_contracts.js`

Full CI and browser acceptance should pass before merging. This change is a structural deduplication, not a new operator or an assertion of GPU/runtime parity.
