# Static behavior acceptance — simulation first

Policy: a successfully rendered diagram, AST parse, or single profiler capture is not proof of mathematical or runtime correctness.

## Executable gates

- tests/test_static_contracts.py: deterministic synthetic Python Attention, transform, RL-like update, branches, loops, no-execution sentinel, repeatability and source location. Runtime and shape claims must remain unverified.
- tests/tensor_static_contracts.js: exact numerical-shape contracts using B=2, S=6, H=64, Q-heads=8, KV-heads=2, F=192 for GQA prefill/decode, DiT, SwiGLU, GELU and MoE; checks graph edges.
- tests/browser_acceptance.py: headless Chromium through local HTTP; 4 architecture presets × 5 3D levels, training/inference/RL, config errors, nested graph navigation, exact-path telemetry join, tensor heatmap/histogram modes and adversarial JSON.
- tests/test_telemetry_oracles.py: CPU PyTorch checks known signed matrix statistics, logical strides, noncontiguous views, and bounded sample size.

## Issues surfaced by these tests

1. A valid user node named root collided with the navigation sentinel root. Graph navigation now separates action from node ID and regression test covers it.
2. Imported telemetry allowed malformed matrix cells, shapes and inconsistent histogram counts, so malformed data could reach Canvas. Import validates finite numeric counts/shape and histogram invariants before rendering.
3. Importing invalid Config while viewing an annotated source graph used to clear internal graph state before input validation; importing is now transactional, and browser regression re-enters a nested node after the invalid import.
4. Initial browser acceptance tests had incorrect async assumptions and one case-sensitive assertion. Tests now wait for file reads and assert exact observable labels.

## Not validated by this gate

- Arbitrary real model source correctness and dynamic branches.
- CUDA kernel timing, fusion, memory layout and model FLOP utilization.
- Training/RL algorithm correctness in unseen implementation variants.
- Whether a bounded activation sample represents a large tensor population.
- Source-to-runtime attribution across different model versions and runs.
- Multi-GPU execution or realistic large-model performance and visual frame rate.

Static acceptance means the documented **simulated** cases pass with correct provenance and without crashes or untrusted Python execution. Research-grade completeness is not yet claimed.

## 2026-10-17: truncation and gated-residual correctness gate

- The parser now reports `truncated_nodes`, `truncated_edges`, `dropped_nodes`,
  `dropped_edges` and `graph_limits` alongside the conservative `truncated` flag.
  Dense fan-out can reach the edge limit before the node limit; a graph with
  missing edges must never advertise itself as complete.
- The teaching MoE graph must have both the original `x` and combined experts
  feeding the residual output; the teaching DiT AdaLN-Zero graph must expose
  six distinct condition slots and both independently gated Attention and MLP
  residual additions. These are structural and mathematical *teaching* checks,
  not claims about real GPU launches.
- A pinned Meta DiT source AST test independently asserts the two additions
  each contain `gate_msa/gate_mlp.unsqueeze(1) * self.attn/self.mlp(...)`.
  Imported real-source graphs still use source navigator rather than seamless
  five-level teaching views; external `timm` Attention/MLP internals remain
  unresolved until their exact pinned source is provided.
