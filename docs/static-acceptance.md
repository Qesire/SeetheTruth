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
3. Initial browser acceptance tests had incorrect async assumptions and one case-sensitive assertion. Tests now wait for file reads and assert exact observable labels.

## Not validated by this gate

- Arbitrary real model source correctness and dynamic branches.
- CUDA kernel timing, fusion, memory layout and model FLOP utilization.
- Training/RL algorithm correctness in unseen implementation variants.
- Whether a bounded activation sample represents a large tensor population.
- Source-to-runtime attribution across different model versions and runs.
- Multi-GPU execution or realistic large-model performance and visual frame rate.

Static acceptance means the documented **simulated** cases pass with correct provenance and without crashes or untrusted Python execution. Research-grade completeness is not yet claimed.
