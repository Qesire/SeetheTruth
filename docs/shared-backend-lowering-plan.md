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

## Portable arithmetic lowering, 2026-10-23

`portable_operator_rules.js` defines a single semantic expression emitter for the 20 operators whose emitted Python tensor arithmetic is identical in NumPy and PyTorch: matrix multiplication, elementary binary arithmetic (including 4D broadcast add), scalar scale/bias/temperature/square, low-rank projection, weighted sum and row bias, deterministic slice/cache read, and DDIM epsilon update. Each exporter now calls this rule emitter rather than repeating the formula. Unsupported kinds still follow their own backend lowering with the established fail-closed checks.

The DDIM coefficients are explicitly `sqrt(alphaPrevious/alphaT)` and `sqrt(1-alphaPrevious) - sqrt(alphaPrevious*(1-alphaT)/alphaT)`. These are computed as JS numeric constants from the validated graph parameters, preserving the original emitted arithmetic. Shared rules do not imply the operators are validated against a downloaded diffusion model.

The portable layer makes no claims about PyTorch gradient semantics for `detach`, in-place operations, reductions, broadcasting edge cases outside validated shapes, or stream-state graph mutation. Backend-specific operators retain independent lowering. We require actual CPU NumPy/PyTorch forward agreement with the bounded LearningLab interpreter, and a trainable-graph Torch gradient comparison against the bounded VJP, in addition to unchanged full exporter regression suites.

## Normalization and attention semantics, 2026-10-23

The shared `normalization_rules.js` register is the mathematical specification used by both code generators for eight operators: `softmax`, `nd_softmax_last`, `rmsnorm`, `qk_norm`, `layernorm`, `attention`, `log_softmax`, and `row_logsumexp`. All reductions are over the final tensor axis; `rmsnorm`/`qk_norm` place epsilon **inside** the square root. `layernorm` uses population variance (not Bessel-corrected sample variance). `attention` scales QK logits by the square root of the query's last dimension and softmaxes over keys. Stable softmax/logsumexp remain backend-native; the shared semantic contract does not force NumPy and Torch to use the same implementation.

The register does not expand accepted tensor shapes, allow arbitrary attention masks, or claim performance/fidelity for real model kernels. It preserves the existing difference between NumPy explicit LayerNorm and `torch.nn.functional.layer_norm` while holding their *mathematical* contract fixed. The bounded executor remains the forward oracle; new tests execute independent emitted programs in actual NumPy and PyTorch and use finite differences of the bounded JS loss to check Torch autograd gradients for seven kinds. Rank-four softmax is checked separately. Existing full CPU codegen and browser suites are retained.

## Index-preserving tensor transformations, 2026-10-23

The shared `shape_index_rules.js` now specifies both export expressions and a pedagogical **index-to-source mapping** for eight shape operations: transpose, 4D permute, reshape, flatten, singleton-axis expansion, broadcast addition, keepdim reduction, and grouped KV-head repeat. Its pure `analyze(node,inputShapes)` produces the output shape, an illustrative output coordinate, corresponding input coordinates, and the reverse-mode interpretation. This is not a model-template whitelist; it applies to arbitrary valid user-wired graphs.

Rules follow the actual bounded interpreter: reshape/flatten preserve row-major flat positions; permute changes axis order; broadcast reuses inputs along singleton axes and sums their gradient contributions; reduce-sum keeps the reduced axis with size one and broadcasts the incoming VJP across it; GQA repeat-KV maps `outputHead -> floor(outputHead / repeatFactor)`, with backward summing contributing query-head gradients. NumPy and Torch exporters now consume the same mathematical indexing contract, while tensor storage and autograd remain native to each backend. The shared Execution IR verifies output shapes against the numerical trace and attaches `shapeTransform` to matching nodes.

Learning Studio uses the same static rule set to expose a selected brick's output shape, example coordinate, exact source coordinates and reverse-mode explanation **without needing numeric mode or a GPU**. Static inspection never changes the computational graph. New contracts verify actual JS value lookup at each reported index, static-readiness and IR shape identity, web/Android asset closure and independently executed NumPy/Torch results; the GQA test compares PyTorch autograd to existing VJP with an independent finite-difference spot check.

## Interactive index interrogation and reverse contribution ranges, 2026-10-23

The selected block's shape inspector now exposes one coordinate control per **output axis**. Changing a coordinate invokes the same pure `ShapeIndexRules.analyze(node,inputShapes,{outputIndex})` used for static IR correspondence; it never re-executes the graph and never changes connections or optimizer state. The API rejects negative, nonintegral, out-of-range and wrong-rank indices and keeps backward-compatible example coordinates when no explicit index is requested.

The returned `sourceIndices` enumerate precisely the inputs read by that output coordinate (all members of a keepdim reduction; both operands of broadcast-add; one source element for bijective reshape/permute; one shared KV-head position per repeated output head). `reverseContributions` provide *analytically derived* closed output-index ranges and counts for each selected source coordinate. This makes the reverse accumulation semantics concrete without allocating every output index: broadcasted length-one axes contribute over the full corresponding output range, and a repeated KV head receives the sum of gradients over its factor-sized query-head group. Reduction outputs read multiple input elements; each input element receives gradient from one reduced output position.

Learning Studio renders shape and index provenance in default concept mode; after a separate explicit bounded numeric run, it can also show output and source values taken from that run's trace. Axis selections are held per node, reset when switching presets and clamped when shapes change. No tensor-valued gradient is manufactured by the static inspector: accumulation ranges express the mathematical VJP contract, and the independent GQA VJP/finite-difference tests remain in place.

CI checks all bounded output coordinates for numeric source equivalence and analytically reported reverse fanout, invalid index rejection, and real Chromium coordinate switching, restoration, broadcast/reduction/Repeat-KV explanations and optional numeric evidence.

## Cross-graph scalar lineage (2026-10-23)

The read-only `element_lineage.js` explorer descends the *user-wired canonical DAG* for any selected output element. It reuses current tensor shapes from GraphReadiness and canonical input-port ownership from LearningLab's index. Each dependency edge identifies one concrete upstream coordinate, its input port, the summation term where applicable and a locally validated VJP multiplier. Shared ancestor coordinates are memoized and receive the **sum of all paths' contributions** in reverse topological order.

Initial exact rules cover eight shape/index operations, elementwise arithmetic and scalar transforms, 2D/4D matrix multiplication and stable Softmax's full last-axis Jacobian. Matrix multiplication creates one A/B coordinate pair per inner K; Softmax connects a selected output to every logit in that row. Broadcast and repeat-KV preserve multiplicities rather than pretending that a reused input element is unique. No model architecture whitelist is introduced.

There are deliberately two evidence levels: **structural**, where indexing and constant local derivatives are known without executing a tensor, and **bounded-numeric**, where a validated execution trace provides actual source/output values and variable-dependent derivatives. A missing derivative is represented as unknown, never as zero. A node with no exact local rule is an explicit boundary; a bounded explorer that exceeds the vertex/edge display budget identifies truncation and does not claim to have followed the full graph.

Learning Studio offers an on-demand, per-coordinate `ELEMENT LINEAGE` inspector. The output is a grouped path/element listing and small directed dependency edge sample. Selecting another graph or changing a coordinate invalidates the old selection; rendering never mutates connections, weights, optimizers, or the numeric trace. Regression tests compare multi-path derivative aggregation to finite differences, validate 4D batched MatMul with a genuine perturbed N-D parameter, cover unsupported operator boundaries and the Softmax Jacobian invariant, and interact with the rendered browser view.

The explorer computes the sensitivity of **one selected output element** to selected upstream elements, not a whole-model training loss or a substitute for backpropagation training. Larger model runtime instrumentation remains separate.
