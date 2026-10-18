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
- `masked_router_topk(S',mask)` emits `I:[3,1]` integer expert indices.
- `gather_rows(experts:[4,4],I)` returns illustrative *expert prototype vectors*, **not** outputs of real expert FFNs.

The group score sums the top two biased affinities per group. `router_gate_weights(A,I,mode=identity,normalize=true)` takes the selected **unbiased** affinities and normalizes them; `expert_bank_mix` then weights teaching prototype vectors. This reproduces the selection-vs-gating split in the cited inference source, but not its route-scale, genuine expert FFNs, or distributed dispatch.

## 15: `grpo_signals` — log policy probabilities and group advantages

Reference: DeepSeekMath, https://arxiv.org/abs/2402.03300 .

- `logits:[2,4]`, `tokenIds:[2,1]`; `selected_logprob:[2,1]` and `log_softmax:[2,4]`.
- Independently, `rewards:[2,4]` has each row interpreted as one prompt's four candidate responses. `group_advantage:[2,4]` normalizes each row independently.
- **These branches are deliberately not connected into a loss.** There is no rollout, old policy, reference policy, clipped objective, mask, or optimizer in this lesson.

The denominator is `sqrt(populationVariance+epsilon)`, a chosen stable teaching convention. The original equation is a group standardized reward; exact epsilon placement is implementation-specific.

## Acceptance contract

For every recipe:
1. Each component must remain in the palette and permit editing/reconnection.
2. Node graph edges must be shape-safe; numerical forward and qualitative shape results must agree at *every node*.
3. A hovered paper explanation must cite an appropriate primary source and distinguish textbook mechanics, paper-specific choice, and toy simplification.
4. No REAL_CAPTURE attribution is permitted for deterministic seeded teaching matrices.
5. Browser layout, hover accessibility, and screenshots must be validated separately from Node numerical tests.
