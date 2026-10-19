# Open-model composable lessons (bounded 2-D teaching simulations)

These are recipes made from ordinary palette blocks. They are **not** model checkpoint execution, hardware profiles, or entire architecture emulations. Hover over any node in the main Scratch-like learning editor to inspect its math, citations, edge cases, and engineering interpretation.

## 13: `mla_lite` — MLA-inspired one-head KV/Q compression and decoupled RoPE

Reference: DeepSeek-V3 Technical Report, §2.1.1, https://arxiv.org/abs/2412.19437 ; DeepSeek-V2 https://arxiv.org/abs/2405.04434 .

- `X:[3,4]` represents three tokens with four channels.
- `W_DKV:[4,2]`; `c_KV = X W_DKV:[3,2]`.
- The exact same `c_KV` node fans out into `K_content = c_KV W_UK:[3,2]` and `V = c_KV W_UV:[3,4]`.
- **Independent positional key:** `K_rope = RoPE(X W_KR):[3,2]`, concatenate `K = [K_content,K_rope]:[3,4]`.
- Query latent `c_Q = X W_DQ:[3,2]` feeds **two** paths: `Q_content = c_Q W_UQ:[3,2]`, `Q_rope = RoPE(c_Q W_QR):[3,2]`; concatenate `Q:[3,4]`.
- One-head teaching Attention is `softmax(Q Kᵀ/√4) V:[3,4]`. **This is not full multihead inference.** No latent cache is materialized and no learned model weights are loaded.

This lesson teaches the *split of content/position pathways* and *shared compressed latent*. It deliberately leaves KV cache implementation, head-level sharing, and a complete DeepSeek checkpoint outside scope.

## 14: `moe_route` — bias, group filtering, masked expert Top-k

Reference: DeepSeek-V3 §2.1.2, https://arxiv.org/abs/2412.19437 ; exact official Gate.forward: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py .

- Input token representations `X:[3,4]` feed a teaching router `S:[3,4]`.
- First create *affinities* `A = sigmoid(S)` as an explicit, replaceable activation block. Independently input `b:[1,4]`; `S' = A+b` is used for **selection only**.
- A `group_topk_mask` partitions experts and emits a `[3,4]` eligibility mask.
- `masked_router_topk(S',mask,k=2)` emits `I:[3,2]` integer expert indices.
- `gather_routed_bank(experts:[4,4],I)` returns `[6,4]` illustrative *expert prototype vectors* (token-major flattened), **not** outputs of real expert FFNs.

The group score sums the top two biased affinities per group. `router_gate_weights(A,I,mode=identity,normalize=true)` takes the selected **unbiased** affinities and normalizes them; `expert_bank_mix` then weights teaching prototype vectors. This reproduces the selection-vs-gating split in the cited inference source, but not its route-scale, genuine expert FFNs, or distributed dispatch.

## 15: `grpo_signals` — log policy probabilities and group advantages

Reference: DeepSeekMath, https://arxiv.org/abs/2402.03300 .

- `logits:[2,4]`, `tokenIds:[2,1]`; `selected_logprob:[2,1]` and `log_softmax:[2,4]`.
- Independently, `rewards:[2,4]` has each row interpreted as one prompt's four candidate responses. `group_advantage:[2,4]` normalizes each row independently.
- **These branches are deliberately not connected into a loss.** There is no rollout, old policy, reference policy, clipped objective, mask, or optimizer in this lesson.

The denominator is `sqrt(populationVariance+epsilon)`, a chosen stable teaching convention. The original equation is a group standardized reward; exact epsilon placement is implementation-specific.

## 16: `moe_expert_ffn` — actual per-token gated FFN arithmetic with shared experts

Primary sources:
- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- DeepSeek-V3 inference `Expert.forward`, `Gate.forward`, `MoE.forward`: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
- Mixtral of Experts: https://arxiv.org/abs/2401.04088

This lesson uses **only bounded deterministic teaching tensors**; "actual" means the expert matrix multiplications are evaluated, **not** checkpoint samples or published-model inference.

1. `X:[3,2]` → `Router Linear:[3,4]` → `Sigmoid Affinity:[3,4]`.
2. `affinity + selection_bias:[1,4]` → Group-Limited Mask (two groups, keep one) → Top-2 expert indices `I:[3,2]`.
3. `router_gate_weights(affinity,I,mode=identity,normalize=true):[3,2]` extracts **unbiased affinities** for the selected experts.
4. `dispatch_routed_tokens(X,I)` produces `Z:[6,2]` with row `n=t·2+k`, without needing a GPU scatter kernel.
5. Three independent parameter banks `W1,W3,W2:[4·2,2]` each store four disjoint expert-specific `[2,2]` matrices. When several tokens choose expert `e`, they consume the **same W_e** rows.
6. `ExpertLinear(Z,I,W1):[6,2]` and `ExpertLinear(Z,I,W3):[6,2]` feed an ordinary `SwiGLU:[6,2]`; `ExpertLinear(SwiGLU,I,W2):[6,2]`.
7. `routed_weighted_reduce(values:[6,2],gates:[3,2])→Y_routed:[3,2]`.
8. Independently, all input tokens pass through the unconditioned **shared expert** branch `Linear → SwiGLU → Linear`. The final output is `Y_routed + Y_shared:[3,2]`.

This mirrors the **structure of the routed-vs-shared MLP and the multiplicative gating operation** visible in the official implementation. It omits actual source parameters, distributed expert ownership, communication, compressed quantized weights, routing capacity, and kernel-level packing/sorting.

### Local VJP proof, separate from the web canvas

`expert_ffn_vjp.js` provides exact bounded local reverse rules. In particular, for a bank with independent expert weights:

```
dW[e,d,h] = sum_(t,k: I[t,k]=e) Z[t*K+k,d] * dY[t*K+k,h]
dZ[t*K+k,d] = sum_h W[e,d,h] * dY[t*K+k,h]
dX[t,d] = sum_k dZ[t*K+k,d]
```

The routed combine obeys `dV[t*K+k,d]=g[t,k]*dY[t,d]` and `dg[t,k]=sum_d V[t*K+k,d]*dY[t,d]`. `SwiGLU` applies the true `SiLU` derivative and product rule. Finite-difference checks compare these VJPs to the same forward blocks. **The full page does not yet automatically backpropagate across the entire assembled graph**; no gradients are claimed through discrete Top-k indices.

### Acceptance

- All 22 freely editable nodes have numeric/qualitative shape parity.
- Every forward route uses the corresponding expert weight bank, not a generic prototype.
- Shared parameter identities aggregate gradients from repeated expert selections.
- Browser screenshot shows routed and shared arms; inspector surfaces DeepSeek and Mixtral sources.
- Record real-model traces only through the separate provenance-controlled capture path, never by relabeling deterministic teaching inputs.

## 17: `mla_cache` — two mathematically equivalent paths through compressed MLA cache

Official implementation: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py (the `MLA.forward` method, particularly `attn_impl == "naive"` vs optimized `else`). Technical reports: https://arxiv.org/abs/2405.04434 and https://arxiv.org/abs/2412.19437 .

### Independently composable cache operations

- Input old **already normalized** KV latents `C_old:[2,2]` and one new latent `C_new:[1,2]`, representing decode position `start_pos=2`.
- `cache_write_segment(C_old,C_new,start=2,capacity=8)` yields an **immutable** snapshot `C_cache:[3,2]`; `cache_read_prefix(length=3)` emits the valid history and current-token prefix. The 8-row capacity is an educational bound, **not** a fabricated allocated capacity claim.
- `pe_old_raw:[2,2]` gets `RoPE(offset=0)`; `pe_new_raw:[1,2]` gets `RoPE(offset=2)`. Both travel through a distinct PE cache write/read path. Q positional features undergo the same offset-2 rotation.
- The real optimized implementation caches `kv_norm(kv)` and the rotated `k_pe`, not the full materialized K/V tensors. We provide the **already normalized** latent as an input to isolate this mechanism; no transformer checkpoint is loaded.

### Explicit reconstruction vs absorbed matrices

One-query / one-head teaching shapes: `C:[3,2]`, `Q_nope:[1,2]`, `W_k:[2,2]`, `W_v:[2,2]`, `Q_pe:[1,2]`, `PE:[3,2]`. Define:

```
K_nope = C W_k
S_explicit = Q_nope K_nope^T
Q_abs = Q_nope W_k^T
S_abs = Q_abs C^T
S_position = Q_pe PE^T
P_explicit = softmax((S_explicit + S_position)/sqrt(4))
P_abs = softmax((S_abs + S_position)/sqrt(4))
V_explicit = C W_v
O_explicit = P_explicit V_explicit
O_abs = (P_abs C) W_v
difference = O_explicit - O_abs
```

These two paths are **algebraically equal in exact arithmetic**. The graph computes both with identical deterministic small matrices and exposes the score, probability, and output intermediates as separate nodes, with a final difference node expected to be approximately zero. The symbolic `sqrt(4)` is the two-content-plus-two-position channels teaching dimension; long-context YaRN or model-specific scale adjustments are **not** represented.

### Execution semantics and caveats

- Only the single **last-position decode query** is present. All three keys are legal causal predecessors; for multiple simultaneous Prefill queries one additionally needs the correct causal masking per position.
- Cache write/read are *pure mathematical snapshot blocks*: no mutable process-wide state, cross-run persistence, GPU allocation, page tables, or eviction. The graph explicitly wires one previous snapshot into one updated snapshot.
- The 2-D plane omits batch and head axes; actual DeepSeek-V3 also supports multiple heads and tensor parallelism. This does **not** reproduce its physical kernel implementation.
- Associativity can differ slightly under float arithmetic or per-tensor quantization. Therefore the assertion is within a numerical tolerance, not strict bitwise identity.
- Nonlinearities and RoPE cannot simply be moved across arbitrary projection weights; the positional dot-product stays an independent branch.
- Source-classification: **SYNTHETIC teaching inputs**, executable forward computation, mathematical relationship grounded in cited source. Do not label these cache contents as sampled model activations.

### Verification

The unit contract checks append, overwrite-with-suffix-preservation, capacity failure, invalid read, immutable old snapshot and a nonsquare query-weight absorption oracle. The full graph test compares every coordinate of content scores, combined scores, probabilities, and outputs. Chromium visual acceptance exercises the freely editable 35-block lesson and checks the primary-paper inspector.

## 18: `grpo_objective` — composable PPO clip and DeepSeekMath GRPO training loss

Primary references:
- PPO: https://arxiv.org/abs/1707.06347
- DeepSeekMath (introduced GRPO): https://arxiv.org/abs/2402.03300
- DeepSeek-R1 (GRPO practice): https://arxiv.org/abs/2501.12948

This **20-block executable lesson** treats two sampled responses (`G=2`) of two token actions each (`T=2`). Four rows of seeded teaching logits `[4,V=4]` and one shared integer action-ID vector `[4,1]` feed *three separate* `selected_logprob` blocks: current, rollout-old, reference. The three roles cannot be interchanged.

1. `selected_logprob(logits,actions):[4,1]` → `response_token_reshape(G=2):[2,2]`, separately for current, old and reference. Each value is the log probability of the same sampled action under its corresponding policy.
2. `rewards:[1,2]` represents the rewards of **two responses to one prompt**. `group_advantage:[1,2]` centers and normalizes only within that prompt's response group. A transpose gives `A:[2,1]`.
3. `policy_importance_ratio(current,old):[2,2]` computes `r=exp(logπnew−logπold)`. `response_advantage_broadcast(A,tokenGrid):[2,2]` replicates the *response-level* A to each token without inventing token-level GAE.
4. `ppo_clipped_surrogate(r,A,epsilon=.2):[2,2]` calculates `min(rA,clip(r,0.8,1.2)A)` **including negative-advantage behavior**. It is the *maximization* surrogate, not the final optimizer loss.
5. `sampled_reference_kl(current,reference):[2,2]` implements `expm1(logπref−logπcurrent)−(logπref−logπcurrent)`. This sampled estimator is **not** the full categorical KL sum.
6. `completion_length_mask:[2,2]` emits two tokens in the first response, one valid token in the second; the last token of the second response is treated as padding.
7. `masked_grpo_objective(surrogate,KL,mask,beta=.04):[1,1]` computes **a mean of per-response valid-token means**, so one longer response is not implicitly counted as extra independent response evidence. The separate `scale(factor=-1)` block produces the optimization loss `L=-J`.

### Reverse-mode analysis: exact gradients and explicit gradient barriers

The main Scratch-style workbench provides **opt-in** `∂ GRPO 反向梯度` after a successful *numeric* 20-block lesson run. It calls `grpo_backward.js` on the **current** assembled graph, without changing the graph or performing an optimizer step. Each selected block's inspector displays `dLoss/dOutput` and, for clipped surrogate nodes, the binary saturation map. Moving/rewiring the graph invalidates its cached forward and backward traces.

For a sampled action `y`, a current policy logit vector `z`, and `l=log softmax(z)_y`:

```
d l / d z_j = [j==y] - softmax(z)_j
r = exp(l_current - l_old)               => dr/dl_current = r
d = l_reference - l_current
KL_sample = exp(d) - d - 1               => dKL_sample/dl_current = 1-exp(d)

s(r,A) = min(r*A, clip(r,1-eps,1+eps)*A)
ds/dr = A when unclipped is selected, or inside interval
      = 0 when the clipped branch wins outside the interval
L = -(1/G) Σ_g [ Σ_t mask[g,t]·(s[g,t]-β KL[g,t])/validTokens[g] ]
dL/ds[g,t] = -mask[g,t] / (G*validTokens[g])
dL/dKL[g,t] = β·mask[g,t] / (G*validTokens[g])
```

At a nondifferentiable clipping breakpoint the microscopic derivative convention chooses the raw `rA` branch; central finite difference tests deliberately avoid this breakpoint. The **same current logprob** receives contributions from the ratio and sampled KL branches, so reverse-mode accumulation **adds both gradients** before the selected-logprob VJP. The unit tests perform independent central finite difference perturbations to *each of the 16 current-policy logits* to check the complete composed loss and confirm the masked padding token has exactly zero gradient.

**Stop-gradient boundaries are part of the algorithm, not a numerical side effect:**

- The `old` input to the policy ratio is the policy that sampled the completion; its gradient is stopped.
- The `reference` input to the sampled KL estimator is frozen. It is not the same as `old`.
- Reward and group advantage are sampled rollout information; no policy gradient passes through them. The `advantage` input to clipped surrogate is detached.
- Discrete `labels` specify chosen token IDs and do not have a gradient.
- All unused/blocked branches display zero adjoint matrices; this is not a general proof that parameters elsewhere in a full network should be frozen.

**Scope boundary:** This is **scoped reverse-mode autodiff for compatible GRPO/PPO teaching DAGs**. It is not arbitrary-block autograd, differentiable sampling, backpropagation through model weights, or an optimizer update. If the user rewires the loss through an unsupported differentiable operation, the module rejects the graph rather than silently fabricating a gradient. The UI displays the proven subset; the earlier expert local VJP contracts remain separate.

### Explicit reductions and differences between research implementations

- The selected reduction is `J=(1/G) Σ_g ((Σ_t mask_g,t·[surrogate_g,t−β KL_g,t]) / Σ_t mask_g,t)`. Some contemporary systems normalize tokens across the entire batch, use different clipping ranges, vary KL handling, or omit the reference KL. Those variants require modifying/rewiring the corresponding blocks, **not** calling them identical GRPO.
- Reward advantage uses `(R−mean)/sqrt(populationVariance+epsilon)` as a **declared teaching convention**. The exact epsilon and standard-deviation convention can differ across source and paper.
- The current, old, and reference logits are deterministic **independent synthetic inputs**, not snapshots of a real trained model. This lesson shows the arithmetic as if those three policy forward passes had already happened. No rollout sampling, verifier scoring, training optimizer, full autograd, or actual policy update occurs.
- The explicit `completion_length_mask` demonstrates reduction semantics, not EOS/tokenizer processing. All valid-response mask rows must contain at least one 1.
- Numerical tests cover ratios at 1 and 1.5, both signs of advantages, the nonnegative sampled KL estimator, invalid ratio/mask bounds, and a direct hand-summed oracle for the complete 20-node graph.

## Acceptance contract

For every recipe:
1. Each component must remain in the palette and permit editing/reconnection.
2. Node graph edges must be shape-safe; numerical forward and qualitative shape results must agree at *every node*.
3. A hovered paper explanation must cite an appropriate primary source and distinguish textbook mechanics, paper-specific choice, and toy simplification.
4. No REAL_CAPTURE attribution is permitted for deterministic seeded teaching matrices.
5. Browser layout, hover accessibility, and screenshots must be validated separately from Node numerical tests.

## Free-form graph authoring: operational validity, not model conventions

The editor is a general-purpose **educational tensor DAG**, not a fixed model assembler or a model-family validator. You may independently combine blocks from Transformer, MLA, MoE, PPO/GRPO, optimization and simple arithmetic into a structure that no published model uses.

**Constraints that matter to execution:**

1. Every graph node must refer to a known executable block and have structurally valid parameters.
2. The graph must be acyclic, with exactly one source connected to each input socket. An upstream output may feed any number of consumers, including several ports of one consumer.
3. Before a **complete graph** is executable, all required input sockets must be connected, and operators' declared input and output tensor dimensions must be compatible. Integer indices, masks, exponent limits and other value-dependent contracts are checked by the actual bounded numeric execution.
4. Incomplete graphs and temporarily incompatible tensor dimensions **remain editable**. They are not rewritten to fit any standard architecture. The live readiness report marks `missing`, `blocked` and `shape-error` nodes, with specific cause and inferred output shape on each ready node. Errors link back to the offending block.
5. Clicking **检查可运行性** performs the full shape check; in numeric mode it additionally executes the complete CPU micro-tensor graph. It never requires a predefined model name or a specific sequence of blocks.

A student can, for example, freely connect `Input[3,4] → Transpose[4,3]` while sending the same original `Input[3,4]` into one MatMul operand. A temporary `Input[3,4] × Input[3,4]` mismatch is shown but kept on the canvas; replacing only the right-hand input with `Transpose[4,3]` yields a valid output `[3,3]`, which can then feed Softmax. This is a **deliberately non-template-specific example**.

The number of nodes and numeric tensor dimensions retain bounded in-browser safeguards. Those are resource/safety limits on a teaching implementation, not judgments about whether a novel architecture is academically sound. A mathematically runnable graph may be scientifically ineffective; this tool shows mechanisms and execution, not a quality certification.


## Selected-branch execution in a partially built graph

A free-form Scratch workbench often contains experiments that are not yet connected. The **Run selected branch** action resolves the chosen output block's transitive upstream dependency closure and runs only that induced, still independently validated DAG. For example, the user can finish `Input → Transpose + MatMul → Softmax` while leaving an entirely separate `ReLU` block unconnected: running the whole graph reports the missing ReLU input; running the selected Softmax branch succeeds and returns the actual numeric output without deleting or changing the ReLU block.

This is not a relaxation of operator semantics: every selected branch still requires its own inputs, shapes, legal value-dependent indices/masks and an acyclic dependency structure. Incompatible shapes within the selected branch cause normal errors; unrelated errors are excluded from the execution scope. The run trace makes the branch target explicit.

### Nonbinding socket previews

After choosing any output socket, each visible candidate input is decorated with a **predictive** classification: compatible, incompatible, pending (insufficient upstream shape information), or invalid (for example a cycle). The preview is calculated by tentatively replacing only the target input and propagating shape constraints through its affected descendants; existing errors in unrelated branches do not contaminate this prediction.

These colors are **advisory**. Users may connect a shape-incompatible route and correct dimensions later. Structural violations (cycles or invalid ports) remain blocked. This avoids restricting original architectures or nonstandard module order.

### Branch-local training

When a user selects an MSE loss (or a graph has one unambiguous MSE loss), the optimizer can use that scalar's upstream dependency slice. Only reachable Parameter blocks are updated via the same immutable Forward → VJP → AdamW → Forward transactions; **the full graph JSON remains the saved editing project**, including unfinished blocks elsewhere. State export, step history and atomic undo preserve the unfinished branches. If a selected output is not a supported differentiable scalar loss, the optimizer rejects it explicitly instead of claiming it was trained.

### Practical execution boundaries

This does not introduce runtime control-flow cycles, unbounded tensor memory, implicit dimension broadcasting or silent auto-repair. The graph remains a bounded CPU educational DAG. The standard full-graph Run and selected-branch Run remain separate actions, so there is no ambiguity about what was actually evaluated.

## 25: `nd_attention_lesson` — executable N-dimensional tensor calculations

The Scratch workbench now provides **89 executable block kinds**, including eight explicit multi-axis operators. Four-dimensional teaching tensors are real nested Float64 arrays, not shape-only labels. The bounded CPU interpreter supports rank 2–4, axis sizes 1–8 and at most 512 elements per N-D tensor. This replaces neither real GPU kernels nor framework tensor storage, but reproduces small-scale numerical results.

**Axis semantics are user-defined.** In lesson 25 the chosen convention is `[B,H,T,D]`: Batch, Head, Token and channel. A nonstandard graph can use these axes differently; only sizes and explicit index operations determine execution. Operations never guess which axis is Token or automatically fix incompatible dimensions.

Available N-D blocks:

| Block | Exact math contract |
|---|---|
| `nd_input` | Deterministic synthetic source `[B,H,T,D]` |
| `nd_permute` | Explicit permutation of four axes, e.g. `[B,H,T,D]→[B,T,H,D]` |
| `nd_reshape` | Row-major reinterpretation (equal element count), from 2D or 4D to 4D |
| `nd_flatten` | Flatten first three dimensions `[B,H,T,D]→[B·H·T,D]` |
| `nd_expand4d` | Insert two leading singleton axes `[T,D]→[1,1,T,D]` |
| `nd_broadcast_add` | Right-aligned broadcasting with equal or singleton axes; explicit elementwise addition |
| `nd_reduce_sum` | Sum along selected axis, retain dimension of size 1 |
| `nd_batched_matmul` | Batched `[B,H,M,K]·[B,H,K,N]→[B,H,M,N]`, broadcast on B/H only |

The lesson constructs the following genuine nine-node DAG:

```
queries: [1,2,3,4] ─┐
                      ├── batched MatMul [1,2,3,3] ─┐
keys:    [1,2,4,3] ─┘                                │
bias:    [1,1,3,3] ─────────────────────── broadcast Add [1,2,3,3]
                                                      │
                                token-axis Sum [1,2,1,3]
                                                      │
                                   Flatten [2,3]
                                                      │
                                   Expand [1,1,2,3]
                                                      │
                                Permute [1,2,1,3]
```

It does **not** claim to be an entire Multi-Head Attention block, because the K transpose, scaling, masking and Softmax must be explicitly added to construct that algorithm. Its purpose is to prove composable multi-axis arithmetic with no fixed model-family template.

The inspector shows each output's complete rank/dimensions and allows selection of leading-axis indices (e.g., Head 0 vs Head 1), displaying the corresponding genuine 2D tensor slice. Choosing another Head does not recalculate or replace the tensor—it changes only the viewer.

The unit test verifies numerical values for every Head via an independent loop, broadcasting correctness, sum vs mean, exact reshape/permute round-trips, 4D→2D→4D value preservation, and intentional shape failures; concept and numeric shapes agree for all 89 kinds and the new composition. The Chromium acceptance exercises the real Head selector and checks that Head 0 and Head 1 display different numbers. The original 81 2D block kinds and 24 existing lessons are preserved.

Primary implementation-semantics references: [PyTorch Broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html), [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html), [torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html). The present CPU interpreter intentionally does not model GPU storage strides, autograd through new N-D blocks, CUDA kernels, or hardware latency.


## 26: User-assembled causal multi-head attention — no opaque model shortcut

The 26th workbench lesson is an editable nine-node `nd_causal_attention` graph composed from the same free-form N-D blocks. **Nothing requires the architecture to match a known model**; the lesson is a verified example students may rewire.

```text
Q[1,2,3,4] ----------------------------------+
                                             v
K[1,2,3,4] → permute[0,1,3,2] → [1,2,4,3] → BatchedMatMul
                                              | scores[1,2,3,3]
                                              v
                                          Scale by 0.5 (1/√4)
                                              |
                                              v
                                     Causal Score Mask
                                              |
                                              v
                              Last-axis Softmax → P[1,2,3,3]
                                              |
V[1,2,3,4] ----------------------------------+→ BatchedMatMul → Context[1,2,3,4]
```

Exact contracts:

```text
Q, K, V: [B,H,T,D]
S[b,h,q,k] = Σ_d Q[b,h,q,d]*K[b,h,k,d] / sqrt(D)
S_mask[b,h,q,k] = S[b,h,q,k] if (keyStart + k) <= (queryStart + q) else -1e9
P[b,h,q,k] = exp(S_mask[b,h,q,k] - max_j S_mask[b,h,q,j]) /
             Σ_j exp(S_mask[b,h,q,j] - max_l S_mask[b,h,q,l])
Y[b,h,q,d] = Σ_k P[b,h,q,k]*V[b,h,k,d]
```

In the bounded example `B=1,H=2,T=3,D=4`, the Key permutation swaps time and channel axes, not batch/head axes. `Scale` uses the explicit factor `0.5` corresponding to `D=4`. The first Query can attend only to Key 0; masking rows are checked for exact zero future probabilities. Each head and each Query is normalized independently. Head 0 and Head 1 have different synthetic scores, independently selectable in the inspector.

- `nd_scale`: scalar multiplication, does not infer the model head dimension.
- `nd_causal_mask`: explicit absolute `queryStart` and `keyStart` offsets; supports different Query/Key sequence lengths when the first Query has at least one valid Key. Uses finite `-1e9` for blocked positions as a small Float64 teaching approximation, not a bitwise implementation of an attention kernel.
- `nd_softmax_last`: numerical max-subtracted softmax strictly along the last axis (Key dimension), never across Heads.

The new numeric acceptance checks both Heads against an independent Q/K dot-product + masked softmax + weighted V oracle, and separately checks the non-square cached-decoding case with query absolute positions 2–3 and keys 0–3. The expected GEMM arithmetic main-term is `144 + 144 = 288 FLOPs` for this tiny example; it is not a throughput measurement. Chromium validates both Head probability views and the final Context tensor.

Boundaries remain explicit: no K/V cache state machine in this lesson, no automatic Rotary embeddings, no fused FlashAttention kernel, and no N-dimensional backward engine yet. These can be assembled or implemented separately with precisely documented semantics. Relevant published reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762); implementation semantics: [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).


## 27–28: Grouped-Query and Multi-Query Attention with four-axis reverse-mode gradients

Lessons `nd_gqa_attention` and `nd_mqa_attention` are **fully editable compositions** of the same public tensor operators. They do not install specialized opaque "GQA kernels" or constrain independent Scratch experiments to Llama-like or DeepSeek-like model recipes.

- GQA: four Query Heads, two Key Heads and two Value Heads, with mapping `KVhead=floor(Qhead / 2)`.
- MQA: four Query Heads, one shared Key Head and one shared Value Head, with mapping `KVhead=0` for all Queries.
- Both use `[B=1,T=3,D=2]`, construct real `Q Kᵀ / sqrt(D)`, causal mask, last-axis softmax, weighted `PV`, and end in a separately connected `nd_mse` tensor loss.

These compositions expose a fundamental training mechanism that a pure shape calculator would miss:

```
dL/dK_shared[b,hkv,t,d] = sum over Query Heads mapped to hkv of dL/dK_expanded[b,hq,t,d]
dL/dV_shared[b,hkv,t,d] = sum over Query Heads mapped to hkv of dL/dV_expanded[b,hq,t,d]
```

The existing Scratch reverse engine now accepts rank-4 output gradients and dispatches local rules through `nd_autodiff.js` rather than inventing 2D flattened derivatives. Supported exact N-D VJPs cover: batched MatMul (including head/batch broadcast), head repetition, permute/reshape/flatten/expand, N-D broadcast-add, axis sum, scale, masked score stop gradients, last-axis softmax and multi-axis MSE. The original 2D block reverse rules remain available without modification of their mathematical behavior.

**Scope:** `nd_input` is a deterministic 4D teaching source, not a trainable model weight or optimizer state. The backward inspector can display `∂Loss/∂Q`, `∂Loss/∂K` and `∂Loss/∂V`, and it explicitly detaches the MSE target. There is no persistent 4D parameter update in these two lessons yet.

Verification includes an independent reference formula for masked scores/attention output, *every element* of Q/K/V checked by central finite differences through the full GQA and MQA loss, plus numerical consistency checks for individual N-D local VJPs. The implementation also rejects illegal `Hq/Hkv` ratios, rather than silently repeating incompatible head groups.

Research groundings: Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/) (EMNLP 2023); [PyTorch SDPA enable_gqa semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention), notably `Hq % Hkv == 0` and equal K/V Head counts. Physical KV cache savings and fused GPU-kernel performance are not measured by this CPU learning tool.

## 29–30: Persistent N-D Parameter and a real micro-training transaction

Lessons `nd_trainable_gqa` and `nd_trainable_mqa` use the **same 13-node free-form GQA/MQA computation graphs** as lessons 27–28. The only change is replacing their Q/K/V synthetic input sources with three explicit `nd_parameter` nodes that store full nested tensors and AdamW moment state.

The four-axis parameter shape is `[B,H,T,D]` with bounded dimensions. The Parameter JSON includes `params.values` and `params.optimizer={t,m,v}`; the node ID remains its stable identity through undo, redo, save and import. Manual editing of the values resets the AdamW moments. No model-family-based validation is introduced: any well-formed tensor graph using this Parameter node may be optimized when its selected scalar-loss path has supported adjoint rules.

The added `nd_adamw_micro.js` uses the same bias-corrected, decoupled AdamW mathematics as the pre-existing 2D optimizer. Its immutable transaction evaluates a selected MSE/ND-MSE loss, calls the unified reverse graph, applies independent moments to every reachable 2D or 4D Parameter node, re-runs the forward computation, then returns a complete validated new graph. A user can still leave unrelated blocks unconnected on the same canvas.

Example small-CPU losses with `lr=0.005`, `weightDecay=0.01`:

- GQA (4 Query / 2 KV Heads): initial `0.7704344436801055`, after two AdamW steps `0.7549565860316719`.
- MQA (4 Query / 1 KV Head): initial `0.6479845511236489`, after two AdamW steps `0.63602880824813`.

These are observed numerical outputs for deterministic synthetic tensors, not large-model loss curves, model-quality claims or guarantees of convergence.

Contract tests verify AdamW's first-step and second-step moment/bias correction analytically for every tensor element; zero-gradient decoupled weight decay; three simultaneous parameter updates through full GQA and MQA graphs; immutability of old versions; target/shape safety; serializing a two-step graph and reproducing the identical next update; and an unfinished unrelated side branch surviving a parameter update. Browser tests explicitly perform two-step training, inspect per-parameter optimizer counters and Loss curves, undo, export/reimport and continue training.

The N-D VJP and optimizer features remain deliberately bounded to `Float64` CPU teaching tensors with four axes of 1–8 elements and at most 512 elements. Real trainable-model components still require actual datasets, parameter initialization, backends, optimizer schedule and distributed learning before they can claim full training. Reference: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101); [PyTorch Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html).

## 31 — 4D RoPE, pure KV Cache read/append, cached GQA decoding

This 21-node lesson extends free-form CPU tensor graphs with three independently reusable primitives:

| Primitive | Actual bounded operation | Important constraint |
|---|---|---|
| `nd_rope` | Adjacent-pair planar rotation over the first `rotaryDim` channels with absolute `offset+t`; the rest are copied unchanged | `rotaryDim` even and ≤ D. Adjacent-pair is explicitly **not** every real model's split-half RoPE convention |
| `nd_cache_read` | Read `X[...,start:start+length,:]` from axis 2 and materialize a nested tensor | Start and length must describe a real contiguous window |
| `nd_cache_append` | Concatenate `cache` and `segment` on Token axis 2 while preserving B, KV Head and D | No side effects, no implicit broadcast, bounded combined Token length |

The lesson begins with three deterministic synthetic tensors: `Q=[1,4,3,4]`, `K=V=[1,2,3,4]`. It then explicitly splits the Key and Value series into a **prefill prefix of 2 Tokens** and the **new Token at absolute position 2**. New query comes from the same Token index. The keys are rotated *before* being appended:

```text
full K [1,2,3,4] → Read prefix [1,2,2,4] → RoPE(offset=0) ──┐
                  → Read current [1,2,1,4] → RoPE(offset=2) ─┴─ Append → K_cache [1,2,3,4]
full V [1,2,3,4] → Read prefix ────────────────────────────────┐
                  → Read current ───────────────────────────────┴─ Append → V_cache [1,2,3,4]
full Q [1,4,3,4] → Read token 2 [1,4,1,4] → RoPE(offset=2) → Q_current
K_cache → Repeat KV heads 2→4 → Transpose → QKᵀ → Scale → Causal Mask(queryStart=2) → Softmax
V_cache → Repeat KV heads 2→4 ────────────────────────────────────────────────────────┘ → PV → context [1,4,1,4]
```

**Numerical acceptance:** The cached output for absolute Query position 2 agrees elementwise with `Read(FullSequenceCausalAttention(Q,K,V),start=2,length=1)`. Independently, `Append(RoPE(K_prefix,0),RoPE(K_new,2))` agrees with `RoPE(K_full,0)`, and raw Value cache concatenation agrees with the original Values. This is a correctness invariant, not a latency measurement.

All three N-D primitives have reverse-mode rules as well: RoPE uses the transpose/inverse of its local rotation; Cache Read scatters gradients back into the selected Token interval, filling zero outside; Cache Append splits the downstream gradient into prefix and segment. The acceptance checks 96 full-graph Q/K/V source gradient coordinates using central finite differences, including prefix/current branch merging. As always, other unconnected blocks may coexist on the canvas, and custom topologies are accepted if they can execute.

**Important epistemic boundary:** The cache here is explicit **functional snapshot data** in a DAG and is not a mutable stateful decoding engine, external cache store, or GPU paged attention implementation. Student-controlled append calls do not constitute measured runtime savings. Standard HF dynamic caches have per-layer state and update it across actual autoregressive model calls; our composition teaches the same tensor dependency without claiming a full runtime.

References: [RoFormer — Rotary Position Embedding](https://arxiv.org/abs/2104.09864); [Hugging Face KV cache implementation](https://huggingface.co/docs/transformers/cache_explanation); [GQA](https://aclanthology.org/2023.emnlp-main.298/).

## 32–33 — Independently composable activation functions and optimizer rules

The catalogue expands to **110 executable building blocks** and **33 guided lessons**, while preserving shape-only readiness, editable DAGs, local paper hover cards, and CPU-only numerical execution. All new operators remain independent primitives, not hard-coded model-family templates.

### Activation families: mathematical distinctions

The library already provided ReLU, SiLU, Sigmoid, Tanh, Softplus, `gelu_tanh`, SwiGLU and GEGLU. The extension deliberately adds **distinct** operations rather than duplicating existing names:

| 2D kind (plus four-axis counterpart where shown) | Formula | Analytic derivative | Relevant caveat |
|---|---|---|---|
| `gelu_erf` / `nd_gelu_erf` | `x Φ(x)` | `Φ(x)+xφ(x)` | Gaussian-CDF GELU distinct from `gelu_tanh`; erf is high-accuracy bounded CPU quadrature |
| Existing `silu` / new `nd_silu` | `xσ(x)` | `σ(x)+xσ(x)(1−σ(x))` | SwiGLU also requires independent Gate/Up branches |
| `relu_squared` / `nd_relu_squared` | `max(x,0)^2` | `2x` if `x>0`, else `0` | Sparse output does not automatically imply sparse GEMM |
| `mish` / `nd_mish` | `x tanh(softplus(x))` | `tanh(sp)+xσ(x)(1−tanh(sp)^2)` | Stable softplus prevents overflow at large positive inputs |
| `leaky_relu` | `x` if nonnegative, else `slope*x` | `1` or `slope` | Positive-side subgradient selected at zero |

Every new activation has real value execution. The graph reverse engine routes 2D/4D VJPs through the same parameter identity and optimizer transaction; analytic derivatives have central finite-difference checks. Lesson `activation_gallery` sends one synthetic input along independent branches so students can inspect each function and replace the branch freely.

Primary references: [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415), [Mish](https://arxiv.org/abs/1908.08681), [Activation function comparison (JMLR 2024)](https://jmlr.org/papers/v25/23-0912.html).

### Four optimizer rules: operational differences, not only labels

The training toolbar provides **AdamW**, **Momentum SGD**, **RMSprop** and **Lion**, all connected to the *same* immutable `Forward → loss → reverse VJP → optimizer update → Forward` engine for both 2D and 4D Parameter blocks. Every rule records per-parameter update norms, scalar loss before/after, and serialized `optimizer.kind`, `t`, `m`, `v`. Legacy graphs without `optimizer.kind` are interpreted as AdamW.

| Optimizer | Teaching update | State memory | Decay semantics |
|---|---|---|---|
| AdamW | Bias-corrected `m / (sqrt(v)+eps)` | First+second moments | Decoupled |
| SGD Momentum | `m_t = μm_{t−1}+g+λθ`; `θ -= ηm_t` | Momentum | Coupled L2 |
| RMSprop | `v_t=αv_{t−1}+(1−α)(g+λθ)^2`; `θ -= ηg_eff/(sqrt(v)+eps)` | Squared gradient average | Coupled L2, non-centered, no momentum |
| Lion | `θ -= η sign(β1*m_prev+(1−β1)*g) + ηλθ`; `m_t=β2*m_prev+(1−β2)*g` | First momentum only | Decoupled |

For fairness these algorithms cannot be interpreted as a controlled quality benchmark merely because they receive identical synthetic inputs and learning rates. Lion's sign step often requires a different LR scale; no claim about large-model quality is implied. The four new `opt_*` **palette blocks** independently accept `parameter` and `gradient` sockets and return the one-step updated values from **zero optimizer history** (and expose moment/update matrices). These are forward-only, composable *functional previews*. By contrast the toolbar runs true multi-step transactions with independent optimizer state and undo/redo persistence. The `optimizer_compare` guided lesson wires all four previews to the same parameter and gradient input; it never silently modifies the source node.

**Safe switching:** Changing the toolbar optimizer on a graph that has Parameters explicitly resets the moments and step counters, preserving tensor values and recording an undo snapshot. It is not correct to reinterpret a previous RMSprop square average as AdamW's `v` or Lion's momentum. Saving and loading a graph recovers its stored optimizer kind. Each subsequent step recomputes the true graph VJP instead of reusing previous gradients.

Unit tests enforce scalar function derivatives, 4D local VJPs, optimizer recurrences and state persistence for all four algorithms, a composed 4D Mish activation + MQA trainable graph, invalid parameter values, and a cross-algorithm switch requiring reset. Chromium tests exercise the toolbar choices, JSON restore, and independently built activation/optimizer comparison DAGs.

Optimizer references: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html), [PyTorch RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), and [Lion / Symbolic Discovery (NeurIPS 2023)](https://arxiv.org/abs/2302.06675).
