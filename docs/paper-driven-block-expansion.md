# Paper → operator patch → micro-execution: next-stage design

## Why a paper idea is often not a standalone block

Model innovations may modify existing modules without introducing an independently named module. Represent these as versioned **patches on an execution contract**, not merely adding names to the palette.

Patch schema: source (paper, code commit where available), target operation, transformation scope, graph rewrite, parameter identity and sharing, forward and reverse rules, state transitions, scheduling/memory semantics, preconditions, changed/preserved invariants, example tensors, failure cases, and level of proof (implemented numeric / symbolic only / full-model trace).

### Five independent patch axes
1. **Algebra:** change exact function, approximation or parameterization (LoRA, DoRA, RMSNorm replacement).
2. **Topology:** add or remove branch; shared parameter and gradient fan-in; expert dispatch (Switch).
3. **Execution schedule:** reorder mathematically equivalent operations and memory lifetime (FlashAttention-2 online normalization).
4. **Training rule:** custom VJP, detach/stop-gradient, loss composition, optimizer, clipped objective, RL advantage.
5. **Data/state contract:** KV caching, positional offsets, quantization scales, sparse masks, RNG seeds and recomputation.

Each paper patch must say whether it changes the mathematical function, its gradients, the numerical implementation, or only the hardware schedule. This prevents a FlashAttention tile scheduler from being misrepresented as a new attention distribution.

## Implemented in paper_patches.js
- LoRA: output = XW0 + (alpha/r)XAB, trainable A/B, frozen W0.
- DoRA: column-wise direction normalization and trainable magnitude; **forward only** (no custom normalization VJP).
- Online softmax streaming attention: exact unmasked single-head educational attention under finite precision; tiling, row max, row denominator and accumulator; **not** FlashAttention-2 GPU implementation.
- Switch top-1 and activation checkpoint registered as **semantic patch contracts** only. MoE top-1 numerical branch runs in micro_training_flows.js; it is not yet connected to the patch compiler.

## Prioritized further paper reads and blocks
| Research | change type | smallest executable primitive | must explain |
|---|---|---|---|
| RoPE / YaRN / NTK interpolation | position transform | phase interpolation / rotary pair | position-dependent phase, extrapolation assumptions |
| FlashAttention 1/2/3 | execution schedule | tile / online softmax / backward recompute | memory traffic vs FLOPs vs numeric parity |
| LoRA / DoRA / QLoRA | weight parameterization | low rank update, normalizing direction, dequantize-on-use | which params receive gradient, quant scale and storage |
| GQA / MLA | state & parameter sharing | KV head lookup / latent projection | cache size, shared KV gradient accumulation |
| SwiGLU / GeGLU | FFN operator replacement | gated affine paths | distinct up/gate/down weights and VJPs |
| Switch / Mixtral / DeepSeek MoE | topology & routing | dispatch, gather, combine, aux balancing | routing overflow, gradient paths and expert capacity |
| DPO / PPO / GRPO | training rule | sequence log-prob, ratios, detach, group normalize | objective distinctions and gradient stopping |
| Speculative decoding | control flow | draft/verify/accept/reject | correctness of target distribution |
| Diffusion scheduler / CFG | loop & state | timestep transition, conditional branch mixing | denoiser output parameterization and guidance effects |
| Sparse FFN / dynamic masking | compute graph rewrite | mask selection, proxy, exact retained path | selection overhead, dropped-path approximation and mask stability |

## Acceptance before catalog promotion
A title/formula is NOT sufficient. Each catalog primitive needs executable finite numeric examples or clearly marked symbolic-only status; exact shapes and axis semantics; forward trace; backward dependency or explicit non-differentiability; state and optimizer implications; invariants and counterexamples; paper references; test cases; enough explanatory prose for a user to reconstruct the computation. Distinguish canonical paper equations from simplified demos. Do not label derived micro-experiments as observed model statistics.

## Integration order
1. Introduce patch registry alongside the current blocks without breaking saved project JSON.
2. Build a *before / after* graph diff showing replaced subgraph and retained parameter IDs.
3. Connect evaluated patch outputs to the existing matrix details and statistics panels.
4. Add train/backward capabilities only for primitives with verified reverse rules.
5. Import real-model traces later as optional data sources, not as a precondition for static teaching.
