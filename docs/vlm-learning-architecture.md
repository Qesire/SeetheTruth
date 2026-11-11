# VLM in SeetheTruth: one computation spine, three executable graph lessons

**Status (2026-11-11):** A bounded, synthetic-Float64 visual-language teaching extension is installed in the **existing** graph engine. This document separates the executed mechanisms from future checkpoint- and task-dependent VLM research.

## 1. Where VLM belongs

SeetheTruth already exposes typed graph execution, native attention, visual patchification, token embedding, graph-to-backend identity, numerical inspection and a PC/Android curriculum. VLM connects those capabilities, rather than requiring an independent tensor interpreter or lesson database.

```text
                 SeetheTruthUnifiedCore
              identity / evidence / replay
                             │
     ┌───────────────────────┴───────────────────────┐
     │             existing LearningLab DAG          │
     │                                              │
image → Patchify → projection → visual tokens ─┐     │
                                                ├─ fusion → decoder → text logits
token IDs → embedding → text tokens ───────────┘     │
     │                                              │
     ├─ installed vlm_block_pack (3 operators)      │
     ├─ native operators: matmul, attention, mask    │
     ├─ three guided graph recipes                  │
     └─ unified coordinate/math observation         │
                             │
         shared PC/Android teaching curriculum
```

The single substrate is the existing graph runtime. Source/state traces and imported real checkpoints continue to have independent evidence semantics. A lesson card is **PEDAGOGICAL_DESCRIPTION** until the learner actually executes the graph.

## 2. Minimal mathematics and axis identity

For an image with height \(H\), width \(W\), patch sizes \(P_h,P_w\), and model feature width \(d\):

\[
N_v=\frac{H}{P_h}\frac{W}{P_w},\qquad
R=\operatorname{Patchify}(I)\in\mathbb{R}^{N_v\times P_hP_wC},\qquad
V=P(R)\in\mathbb{R}^{N_v\times d}.
\]

Here `P` is a bounded synthetic learned-style `linear` operator in the existing graph. The present lesson **does not execute a pretrained vision encoder** or its real weights. Full ViT/SigLIP/ConvNeXt style image towers would require pinned checkpoint source, preprocessing, learned weights and independent numerical parity.

For prompt token IDs \(z\), `embedding` yields \(T=E(z)\in\mathbb{R}^{N_t\times d}\). The visual-prefix route concatenates in **token** order:

\[
X=[V;T]\in\mathbb{R}^{(N_v+N_t)\times d},\qquad
Q=XW_Q,\ K=XW_K,\ U=XW_V,
\]
\[
A_{ij}=\operatorname{softmax}_{j\le i}\left(
 \frac{Q_iK_j^\top}{\sqrt d}\right),\qquad
O=AU,\qquad
Y=O[N_v:N_v+N_t,:].
\]

The teaching graph explicitly constructs the Q, K, V paths, scaled score matrix, lower-triangular `causal_mask`, `masked_softmax`, weighted sum, text-only readout and language-logit projection. `vlm_text_tail(prefixTokens=N_v)` locates text rows in the *post-attention* context, preserving absolute source-row indices. The teaching version has a single attention stage, not a full multi-layer autoregressive VLM decoder, tokenizer, or generation sampler.

**Important axis distinction:** the visual prefix and text suffix have different provenance despite both having `[T,D]` carriers; connecting them requires the same feature width, not identity of meaning. Trace metadata records visual and text source ranges and element-wise index maps.

### A second way to connect image and text

The cross-attention lesson keeps independent text queries and image keys/values:

\[
Q_t=TW_Q,\quad K_v=VW_K,\quad U_v=VW_V,\qquad
T'=T+\operatorname{softmax}\left(\frac{Q_tK_v^\top}{\sqrt d}\right)U_v.
\]

The **executed** graph calls the existing `attention` operator directly on preprojected token matrices; it is a minimal two-modality cross-attention and residual, **not** an assertion that it implements Flamingo, BLIP-2 Q-Former or any checkpoint-specific gating. Unlike visual-prefix self-attention, cross-attention does not automatically make image tokens part of the autoregressive KV stream. That difference is the teaching comparison.

## 3. Installed blocks and runnable lessons

| Identifier | Mathematical contract | Direct source trace |
|---|---|---|
| `vlm_token_concat` | `concat(visual,text,axis=T)` with matching \(D\); \(N_v+N_t\le8\) | Every output element points to its original visual/text element |
| `vlm_visual_pair_pool` | \((V_{2j,d}+V_{2j+1,d})/2\) for even \(N_v\) | Pooled source pairs and coefficients (1/2) |
| `vlm_text_tail` | `context[prefixTokens:,:]`, leaving at least one text token | Every output element points to the original context row |

All use finite Float64 arrays and the existing trusted block-pack interface (`defs/validate/infer/execute/trace`). Unrecognized shapes, mismatched visual/text widths, odd pooling length, impossible prefix boundary and nonfinite values are rejected. No extra executor, exporter allowlist or permanent graph-template restriction is created.

| Guided graph ID | Purpose | Actual working path |
|---|---|---|
| `extension_vlm_early_fusion` | Visual-prefix causal language computation | 4 visual + 2 text → 6-token QKV / masked attention → 2-row language logits |
| `extension_vlm_compressed_prefix` | Lossy visual-token reduction | 4 visual tokens → 2 pooled visual + 2 text → 4-token attention → 2-row logits |
| `extension_vlm_cross_attention` | Text queries read separate visual memory | 2 text queries, 4 visual keys/values → text residual → logits |

These recipes appear in the regular **Learning Studio** via the trusted extension manifest and the dynamic recipe list. Any operator can still be used in arbitrary well-typed graphs; recipes are suggestions, not a whitelist. The new `vlm` curriculum topic in the PC/Android teaching center deep-links to `extension_vlm_early_fusion`. The teaching center's guided-graph link also respects its existing `edition=teaching` mode.

## 4. G0–G4: reusable granularity, not another microscope

This progression uses the established level-of-explanation convention while retaining the original graph/node/port identities.

- **G0 — user task:** image + question → text prediction. State the task and the actual test fixture.
- **G1 — architecture:** vision encoder / projector / fusion / causal decoder / output head. Distinguish prefix fusion and cross-attention.
- **G2 — module:** Patchify, Linear, token concatenation, QKV, mask, softmax, weighted sum, text readout. Navigate to the original graph nodes.
- **G3 — tensor and axis:** `[N_v,d]`, `[N_t,d]`, score shape `[L,L]`, exact causal visibility; compare 6-token and 4-token prefill representations.
- **G4 — coordinate:** selected input pixel/visual feature → projected token → concatenated offset → attention weight → text context. Where the computation supports index maps, follow actual numeric source elements; for learned mixing, follow the actual computation and its independent mathematical oracle rather than inventing a direct copy lineage.

The current VLM pack directly supplies coordinate source maps for concatenation and text tail. Future projector/attention derivative attribution should reuse existing coordinate/VJP machinery, with independent finite-difference or PyTorch checks when a new claim is introduced. A schematic does not provide evidence of causal relevance, semantic grounding or checkpoint fidelity.

## 5. The main experiments

### A. Test the causal boundary

For visual-prefix attention, the probability matrix is exactly lower triangular after `masked_softmax`. Mutating a future value token must leave earlier output rows unchanged. This distinguishes a real causal mask from a screen that merely draws one. Probe both the `[6,6]` uncompressed and `[4,4]` compressed masks.

### B. Compression and prefill cost

The toy graph compresses adjacent visual tokens:

\[
V'_j=(V_{2j}+V_{2j+1})/2.
\]

For fixed text length \(N_t=2\), visual token count changes from 4 to 2, making total length \(L\) change from 6 to 4. An ordinary dense attention score matrix shrinks from 36 to 16 elements, and the lower triangular causal region from 21 to 10 elements. This is an **analytical teaching comparison**, not a demonstrated wall-clock speedup: projections, data movement, KV traffic, kernels and token scheduling affect real latency. Pair pooling can destroy fine-grained spatial information; quality retention needs image-grounded evaluation.

### C. Fusion topologies

Run both prefix and cross-attention graphs under the same synthetic shapes. The former lets causal text positions attend to preceding image and text positions; the latter uses text Query to read a distinct visual K/V bank. Observe exact shapes and source paths, then introduce separate architecture-specific real weight checks only when available.

### D. Aligned evidence for VLM research

Future modules can reuse the same graph plus evidence contracts for:
- **Visual grounding:** object references, patches, bounding boxes and answer tokens; track coordinate provenance separately from genuine learned spatial alignment.
- **Training/post-training:** frozen visual tower, trainable projector, instruction SFT, multimodal preference/reward optimization; display parameters, masks, loss tokens and gradient flow with valid cross-backend oracles.
- **Video/efficient inference:** multiple frames, temporal stamps, high-resolution patch counts, adaptive pruning and token budgeting; link motion/temporal score to visual representation without assuming quality.
- **Visual Agent:** observation → perception/grounding → action proposal → tool step → new observation. Treat action/state transitions as stateful evidence domains; do not force them into a tensor operator.

The VLM addition therefore links video-model sparsity, LLM post-training and Agent architecture through **shared vision-language representation and observation**, while keeping their distinct execution/state semantics.

## 6. Acceptance and evidence boundaries

The repository test `tests/vlm_learning_contracts.js` checks:
- three genuinely executable graph recipes and independent symbolic shape agreement;
- every output's finite numeric values, visual/text source ordering and text-row location;
- token-compression arithmetic against a separately constructed pairwise-mean oracle;
- causal probability zeros and a deliberate future-token mutation counterexample;
- elementwise source-index maps for exact concatenation and text slicing;
- negative contracts (mismatched width, oversized sequence, odd pool, illegal prefix, wrong source edge);
- shared course identity, studio deep link and approved PC/Android static routes.

The CI workflow invokes this alongside existing trusted-pack and unified-learning contracts. Publishing scripts remain the sole source for deriving installed modules, offline Android assets and static routes. To regenerate deterministically after changes, use:

```bash
node scripts/sync_installed_block_packs.js --check
node scripts/sync_runtime_assets.js --check
node tests/vlm_learning_contracts.js
node tests/unified_pack_registry_contracts.js
node tests/unified_learning_core_contracts.js
```

**Current proof scope:** installed contract + small synthetic deterministic CPU graph when tests run. No visual file decoding, actual pretrained VLM weights, independent NumPy/PyTorch export parity for the three new extension operators, cross-modal semantic accuracy, GPU benchmark, end-to-end training or production agent actions. Do not mark these capabilities verified without their respective results.

## Source anchors

- LLaVA visual instruction tuning and visual-prefix projector: https://arxiv.org/abs/2304.08485 and https://llava-vl.github.io/
- BLIP-2 bootstrapped language-image pre-training with a Q-Former bridge: https://arxiv.org/abs/2301.12597
- Flamingo interleaved visual-language cross-attention design: https://arxiv.org/abs/2204.14198
- Qwen2.5-VL multimodal processing and dynamic-resolution modeling: https://github.com/QwenLM/Qwen2.5-VL
- Source of exact `torch.cat` behavior: https://pytorch.org/docs/stable/generated/torch.cat.html

Each anchor gives a model-family mechanism to compare, not proof that the teaching blocks reproduce those models.
