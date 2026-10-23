# Universal computation microscope — main-compatible extraction

## Scope

This patch extracts the **core** from draft PR #49 into the current production free-form graph. The microscope is mounted in `learn.html`, with no new isolated execution page. It reuses the exact existing `Learning.evaluate` numerical result or `ConceptFlow.evaluate` static result. The inspector displays the same source graph's nodes in topological order.

### Contracts

- All numeric outputs are **bounded rank 1–5** tensors with explicit row-major shape and finite values. Static runs contain shapes but **no numeric values**.
- Reconstructs original `linear` and `matmul` with actual input values and deterministic parameter formula; original `attention` exposes the actual QK scaled scores, stable softmax and PV result, including GEMM dot-sum explanations.
- Registered `pack.trace` observations are carried through a shared nested substep contract. `axis_named_transpose`, `semantic_video_slice` and `semantic_video_concat` expose exact input flat-coordinate mappings that must reproduce their displayed output. Each output coordinate is selectable on desktop and mobile, including rank-5 leading axis selectors.
- Structural dependencies may refer only to already-computed upstream nodes. Revisions that alter a GEMM output, index mapping, source indices, recorded shape, tier, or dependencies fail validation. Provenance may not be escalated to `REAL_MODEL_VERIFIED`.
- Other operators still expose the real input/output tensors, declared formulas and dependency chain. They do **not** claim a valid GEMM reconstruction, automatic VJP, or an unimplemented operator-specific microscope.
- Supported explanation tiers: symbolic/no-execution and bounded synthetic numerical execution. Existing `Wan` adapter in core is dormant: full pinned-source teaching UI promotion requires a separate source verification. This PR does not claim actual checkpoint activations or CUDA timing.
- Only a read-only UI component is added; no GPU runtime or unsafe user-authored executable script imports.

## Acceptance and coverage

`node tests/universal_microscope_contracts.js`: native FFN, attention, residual, SwiGLU, named-axis transpose, 5D video slicing and rejoin; dot and exact index reconstruction; rejected tampered coordinates, untrusted source tier, non-causal dependency, and wrong numerical evidence.

`python tests/browser_universal_microscope.py`: Chromium pointer navigation, real arithmetic breakdown, attention substeps, 5D axis selection, static fallback and mobile named-axis lessons. Desktop server and offline Android load the same JS/CSS.

All existing graph, source, training, codegen and Android CI remain required. The wider issue #61 remains open for complete per-kind numerical microscope and independent Wan source-adapter promotion.
