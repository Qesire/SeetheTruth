# Shared DAG lowering plan (NumPy / PyTorch)

## Purpose

Both executable exporters start from the **same editable brick graph**, not from a predefined LLM, diffusion, or training architecture. This change unifies the preparation phase while leaving backend-specific arithmetic independent.

```
LearningLab validated DAG
  ├─ optional upstream target slice
  ├─ canonical topology and input-port source index
  ├─ fail-closed operator eligibility check for the selected backend
  └─ bounded numeric teaching forward trace
       ├─ NumPy expression lowering → standalone Python
       └─ PyTorch expression / autograd lowering → standalone Python
```

`graph_lowering_plan.js` provides `prepare(graph, {targetId, supportedKinds, unsupportedMessage})` with `graph`, `sorted`, `edges`, `forward`, and `steps`. The planner uses `LearningLab.indexGraph` as the authoritative port/dependency index, and `LearningLab.evaluate` as the bounded numerical oracle. It does not compile expressions, declare operators supported, inject templates, or relax shape requirements.

## Contract

1. **Free composition.** No graph-template or architecture allowlist is imposed. Arbitrary user-wired DAGs remain valid if individual bricks' contracts are satisfied.
2. **Target slicing.** A selected output is compiled together with its upstream dependencies; unrelated incomplete or backend-ineligible nodes are excluded.
3. **Operator support first.** Each backend supplies its existing `SUPPORTED` set and its original diagnostic string; unsupported kinds are rejected before numerical evaluation.
4. **Shared provenance.** Exporters consume identical ordered node IDs, input-port sources, validated shapes and numerical trace. Generated code must still pass the existing backend execution/parity suites.
5. **Backend differences remain explicit.** NumPy and PyTorch are separate lowering implementations; eligibility does not promise equivalence to a full pretrained model or guarantee that every possible graph executes.
6. **Cross-platform closure.** Every new browser-side planner dependency is listed in the local server route table, Android offline asset manifest, and correct script load order.

## Required regression evidence

- `tests/shared_lowering_plan_contracts.js`: planner and both exporters have the same node ordering, port map, trace shapes, and target slice; unsupported kinds fail closed and malformed sockets remain errors.
- Existing `tests/python_codegen_contracts.js`, `tests/torch_codegen_contracts.js`, stream-lowering parity and training codegen suites remain unchanged and must pass.
- Browser, Android offline, and asset loading tests must remain green.

## Next integration point

Move shared lowering *operator semantics* only after inspecting corresponding NumPy/PyTorch expressions and establishing numerical and gradient oracles. Do not deduplicate merely because two implementations have similar syntax. The shared plan provides the stable seam for independently verifiable backend lowerings.
