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
