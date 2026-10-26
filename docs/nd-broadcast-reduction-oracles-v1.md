# Independent 4D Broadcast–Reduction Gradient Oracles (2026-10-26)

## Actual computation

This test exercises the existing ND learning graph and its source code, not a separate numerical simulator:

```text
nd_parameter X [B,H,T,C] ────────────┐
                                    ├─ nd_broadcast_add U [B,H,T,C]
nd_parameter β [1,1,1,C] ────────────┘
                                    │
                                    ├─ nd_broadcast_mul V [B,H,T,C]
nd_parameter γ [1,1,1,C] ────────────┘
                                    │
                                    ▼
                       nd_reduce_sum(axis=2, keepdim)
                                    │
                             S [B,H,1,C]
                                    │
nd_input target Y [B,H,1,C] ─────────┴─ nd_mse L [1,1]
```

The mathematical function is

```math
U_{bhtc}=X_{bhtc}+\beta_c, \qquad
V_{bhtc}=U_{bhtc}\gamma_c, \qquad
S_{bh1c}=\sum_t V_{bhtc}, \qquad
L=\operatorname{mean}\big((S-Y)^2\big).
```

Reverse-mode accumulation must preserve the original `[B,H,T,C]` tensor coordinates and sum over every broadcasted singleton axis. For the shared channel parameters,

```math
\frac{\partial L}{\partial\beta_c}
=\sum_{b,h,t}\frac{\partial L}{\partial U_{bhtc}},
\qquad
\frac{\partial L}{\partial\gamma_c}
=\sum_{b,h,t}\frac{\partial L}{\partial V_{bhtc}}U_{bhtc}.
```

The sum over (t) in the forward pass introduces a broadcast in the backward pass: each original token of a batch/head/channel receives the same upstream (partial L/partial S_{bh1c}). The graph's actual VJP rules (not this document) are implemented by `nd_autodiff.vjp`, used by `graph_autodiff.backward`.

## Source-derived fixtures

The new independent fixtures come from `tests/fixtures/nd_broadcast_reduction_graphs.js`. Both are real `LearningLab.validateGraph` graphs:

- `nd_token_reduce_head_broadcast`: X `[1,2,3,2]`, β/γ `[1,1,1,2]`, reduction along Token axis, target `[1,2,1,2]`.
- `nd_batch_reduce_channel_broadcast`: X `[2,1,2,3]`, β/γ `[1,1,1,3]`, Token reduction, target `[2,1,1,3]`.

The initial independent run revealed a real backend gap: **the source 4D `nd_broadcast_mul` had native LearningLab/ND VJP implementations but was absent from NumPy/Torch lowering allowlists**. This PR now adds `nd_broadcast_mul` to the shared `portable_operator_rules.js` multiply expression and to both explicit exporter supported-kind registries. No separate numerical arithmetic implementation is introduced. `tests/portable_operator_runtime_parity.js` executes the same `gated` node through actual independent NumPy and PyTorch to verify the portable mapping.

The independent oracle compiles each exact validated graph using the existing `torch_codegen.js`, executes its standalone Python code on **PyTorch 2.6.0+cpu, Float64**, retains intermediate output adjoints, and compares every original JS forward tensor and available full-loss gradient coordinate with the generated PyTorch interpreter. Centered finite differences separately check every trainable parameter coordinate against `GraphAutodiff`.

In addition, a second algebraic invariant reconstructs each channel's gradient by explicitly summing upstream contributions across Batch/Head/Token. This distinguishes an unbroadcast error (e.g. retaining one token or one head, or overwriting accumulated gradients) even if the final Loss happens to match.

## Scope and evidence boundary

Because the exported code and portable operator rules change, the existing source-pinned certificate artifacts must be regenerated through the **real PyTorch CPU evidence generator** and checked with `--check` on the final SHA. Old source-digest evidence is deliberately invalid.

This is a **standalone independent numerical regression** in the pinned PyTorch CI workflow, not yet an entry in `operator_certificates_data.json`. The v2 certificate ledger still reflects its own four previously executed test graphs. Only after this test passes should it be integrated into the source-pinned ledger with the correct N-D index enumeration, per-fixture source SHA and differential tolerance.

These examples test rank-4 singleton-axis broadcasting and a keepdim reduction. They do **not** prove every broadcast combination, arbitrary shapes, production tensor kernels, non-contiguous strides, mixed precision or GPU parity.

## Run and acceptance

```bash
node tests/torch_nd_broadcast_reduction_parity.js
```

The test requires the pinned Python/CPU PyTorch environment; the executable workflow is `.github/workflows/torch-export.yml`. It fails on absent PyTorch gradients, shape mismatch, per-coordinate error or finite-difference mismatch, and leaves the input graphs unchanged. Existing evidence, Chromium PC/mobile and offline Android jobs remain release gates.
