# Hierarchical teaching · from Decoder Layer to every matrix operation

## Learning-first goal

SeetheTruth now exposes **hierarchical composite learning views**, not additional unverified numerical operators. The existing **45 primitive blocks** and their primary-source deep knowledge cards remain the single vocabulary. Macro levels are organizational containers: explaining how primitives compose into a decoder sublayer, how attention/FFN are formed, and where normalized/residual flows merge. No GPU, pretrained weights or Python environment is required.

## Drilldown model

Level 0: **Pre-Norm Decoder Layer** is a named teaching configuration, not a universal architecture. Its contract uses `x -> RMSNorm -> {Q,K,V} -> GQA(RoPE/causal) -> Linear -> Add(x) -> RMSNorm -> {gate,up} -> SwiGLU -> down -> Add(x1)`. The decoder overview displays six conceptual stages with shape results obtained through `concept_flow.js` on the *same* graph, rather than author-written shape labels. It does not fabricate GPU kernels or measured latency.

Level 1: **GQA Attention** breaks Q/K/V projection, KV Head sharing, attention and Output projection into four stages. Another view **SwiGLU FFN** breaks Gate, Up, SwiGLU and Down. **Normalization** and **Residual** explain their own roles in the decoder's computational flow. Navigation uses clickable macro stage cards, scoped breadcrumbs, and return-to-parent. The canvas supports zoom-in, zoom-out, fit-to-viewport and auto-centering the selected math operator at readable zoom.

Level 2: **attention_math** is deliberately **single-head, without RoPE/Mask/cache**. It literally exposes `Q -> K transpose -> QK^T -> multiply scalar (1/sqrt(Dk)) -> Softmax(Key axis) -> PV -> Output Linear`. The shape chain is `[Tq,Dk] × [Dk,Tk] -> [Tq,Tk] -> [Tq,Dv]`. This is an isolated mathematical lens; it is **not** claimed to be the entire GQA/RoPE/causal implementation. The `ffn_expanded` view decomposes the gated primitive into two Linear paths, SiLU, Elementwise Multiply, and Down Linear.

## Why this is meaningful rather than visual nesting

All views use the existing graph schema, `L.validateGraph`, `L.order`, `L.evaluate` for optional tiny numerical examples, and `C.evaluate` for default qualitative mode. `composite_atlas.js` contains structural views and hand-authored *explanations*, but obtains each stage's output shape by following its actual typed graph node. There is no second copy of numerical operator formulas or weights.

Independent `tests/composite_atlas_contracts.js` is a strict merge gate: all eight views are shape-compatible between concept and numeric interpreters; all stages point to real nodes; the full 14-node decoder graph has two distinct residual merges and ends in original T×D shape; expanded SwiGLU equals collapsed SwiGLU numerically at every output element; expanded single-head QK/Softmax/PV equals the original Attention operator numerically at every output element. The test also verifies a drilldown DAG has no recursive cycles or dangling child stage.

Real-browser `tests/browser_composite_hierarchy.py` starts the actual local server in Playwright Chromium, enters the decoder overview, expands GQA and QK/PV, chooses an individual MatMul to inspect its mathematical card, navigates back and expands FFN to SiLU×Up Hadamard Multiply, and finally verifies returning to guided course and opt-in micro numerical execution. It captures five screenshot PNG frames, including a true near-100%-zoom QK matrix view to a named GitHub Actions artifact (14-day retention). No AI-generated mockup can substitute for the real screenshots.

## Public design conventions and evidence limits

The reference decoder chooses RMSNorm Pre-Norm, grouped attention with RoPE and causal masking, SwiGLU FFN and residual adds. Other popular networks legitimately make different choices: GELU MLP vs SwiGLU, LayerNorm vs RMSNorm, MHA/MQA/GQA, Post-Norm vs Pre-Norm, multi-axis spatial/temporal or diffusion blocks. Every macro view labels its scope. Specific execution graph, optimization and performance require source/config evidence in the existing source explorer or an optional future adapter.

Default **concept mode** requires no tensor allocation and shows transformations, shape and conditions only. A nested view can optionally be executed with deterministic small matrices; no large GEMM or GPU validation is a product requirement.

Primary background: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [GQA](https://aclanthology.org/2023.emnlp-main.298/), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), [RoFormer](https://arxiv.org/abs/2104.09864).
