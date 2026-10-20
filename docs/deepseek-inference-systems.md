# DeepSeek-inspired inference systems: executable primitives and faithful evidence tiers

**Source status (2026-10-20, corrected following the same-day release).** DeepSeek **released its real internal DeepInfer inference framework today**, including runtime, C++ scheduler, model execution, sampling, optimised operators and distributed-parallel machinery: https://github.com/deepseek-ai/DeepInfer . This supersedes earlier pre-release claims that only independent kernels were available. The released edition explicitly supports **offline inference** for DeepSeek-V4.1-Flash on **NVIDIA SM100 / Huawei Ascend 950**. DeepInfer's own README explicitly states that a full online inference service still requires an external KV cache system, VE/Prefill/Decode disaggregation and cross-DP load balancing; therefore **released inference framework ≠ complete turn-key online serving deployment**.

DeepSeek simultaneously announced the **0.2.0 update** of the separate API/prompt/protocol adaptation library https://github.com/deepseek-ai/deepseek-recipe (its official changelog records an earlier 0.1.0 release on 2026-09-10). This Rust/Python library handles Messages / Chat Completions / Responses formats, DeepSeek V4/V4.1 prompt rendering, tokenization, multimodal image preprocessing, streaming and output parsing; the caller still supplies inference execution, tool execution and HTTP transport.

Prior historical position (now superseded): https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md

For full initial source-level audit and reproducible inspection paths see [DeepInfer day-zero source audit](deepinfer-day0-source-audit.md).

Do not conflate **DeepSeek Harness** (an agent/plugin execution harness) with a model inference/serving engine; or **DeepSpec** (draft-model training + speculative acceptance evaluation) with the serving engine.

## Component map and why these engineering choices exist

| Official component | Concern | Why it exists | What this workbench actually executes |
| --- | --- | --- | --- |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) (released 2026-10-20) | Offline inference engine / C++ scheduler / worker / model executor | Interleave scheduling, variable-length speculative verification, CP/EP/EPLB and source-grounded execution | **No full DeepInfer service replica yet**: current SeetheTruth only executes a bounded component-level CPU graph |\n| [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) | API adapters / prompt rendering / tokenization | Separate wire protocols from model execution and make input/output semantics reproducible | No recipe-compatible protocol server or tokenizer integration yet |\n| [DeepSelect](https://github.com/deepseek-ai/DeepSelect) (2026-09-10 v1.0) | Sparse Attention Lightning Indexer and top-k sampler | DSA selects a small subset of context; fast Top-K is a bandwidth/selection workload, not a GEMM | Deterministic, tie-stable Top-K indices over actual tiny scores |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (current API) | Fused sparse MLA decode/prefill, quantized paged cache | Avoid full context QK work, improve KV residency and fused IO | Exact selected-index QK/softmax/PV math, separate physical-page address lookup |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | MoE all-to-all dispatch/combine and overlap | Expert weights live across EP GPUs; tokens must move to expert owners and back | Existing routed-expert math; **no** simulated RDMA/NVLink byte-accurate network kernel |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | Dense, quantized, routed, fused expert GEMM | Tensor Core efficiency, quantization and reduced launches | Existing real Gemm/MatMul and graph arithmetic counts, **not** low-level FP8 hardware behavior |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) | Draft model training and decoding evaluation | Several draft tokens may be checked in one target pass; gain depends on acceptance and costs | Greedy draft-target matching, accepted prefix, bonus-token metadata |

**Compatibility warning:** FlashMLA's README explicitly announces a **2026-09-30 breaking change** removing earlier Hopper/DeepSeek V3/V3.2/V4.0 compatibility and changing FP8/FP4 KV representation. A teaching page table is not proof of compatibility with any particular FlashMLA version. DeepSelect's README additionally distinguishes BF16 sparse indexer workloads from FP32 sampler workloads; our CPU Float64 sort does not reproduce its 2–20× GPU benchmark results.

## Executable mathematical contracts

### Stable Top-K
`scores[T,V] → selected_indices[T,K]`, `1≤K≤V`. For equal values, the smallest index wins the tie; output consists of integer indices stored in the micro-workbench's Float64 matrix carrier. No GPU speedup is inferred.

### Selected-index sparse Attention
`q[Tq,D], k[Tk,D], v[Tk,Dv], indices[Tq,K] → context[Tq,Dv]`.

For each query t, select exactly the provided, unique Key indices, calculate `q_t·k_i / sqrt(D)`, use maximum-subtracted Softmax only on those indices, and sum weighted values. Optional causal contract rejects any index greater than `queryOffset+t`. The operation is **exact on the chosen subset**; it is not mathematically equivalent to dense attention unless the subset includes every key. The selection scores/Indexer learned by DeepSeek-V3.2 are not reconstructed.

Source: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py

### Paged KV read (no write)
`cache[P*pageSize,D]`, `blocks[logicalPages,1]`, `positions[T,1]`. Logical position p maps to physical token row
`blocks[floor(p/pageSize)] * pageSize + p%pageSize`.
Physical page IDs and positions must be valid integers; checked out-of-range locations error. This is a generic cache abstraction: no KV quantization, no allocator, no FP8 scaling, no attention heads, no asynchronous stream.

### Greedy speculative verification
`draft[K,1], target[K+1,1] → acceptedMask[K,1]`.
The first `j` where draft token `d[j]` differs from the target's *greedy* token for the same assumed draft prefix terminates acceptance. All earlier draft positions are accepted, and all later positions rejected. If `j < K`, emit target token `t[j]`; if all draft tokens match, the target's extra `t[K]` is the bonus token. The standalone JavaScript module exposes `greedyMetadata` to demonstrate these control outcomes; its ordinary DAG output is the accepted-prefix mask.

This is **not** general distribution-preserving speculative sampling, which uses acceptance probabilities and a residual distribution. It is not full Draft training, tree verification, or server rollback logic.

## Free graph assembly

Lessons 54–57 include: dense-vs-DSA selection, explicit paged cache mapping, greedy draft/target comparison, and compositional paged KV → Top-K → sparse attention. Existing Graph Cost deliberately counts none of the unspecified cache or communication hardware costs. Knowledge-card hover references official upstream, not just model-name marketing.

The graph engine's current bounded 2D carrier is a transparent limitation. For real FlashMLA reproduction, the next phase must unify native multidimensional typed IR, paged cache dtype/scale layout, variable sequence lengths, physical allocations, and actual kernel measurements. For EP, a separate routed dispatch/combine and communication/compute overlap event model is required; simulated estimates must remain visibly distinct from GPU measured timings.

## Tests

- `tests/inference_ops_contracts.js`: independent stable-tie oracle, sparse weighted-softmax oracle, causal/out-of-bounds/duplicate rejection, logical→physical page reordering and greedy mismatch/bonus semantics, plus all four graph lesson readiness contracts.
- `tests/inference_codegen_parity.js`: run every node of all four graphs through standalone NumPy and standalone CPU PyTorch, compare to JavaScript; no GPU required.
- `tests/browser_multimodal.py`: real Chromium numeric execution, source-card proof, independent generated Python downloads and screenshots.
- GitHub CI gates preserve existing training, video, static, and real-source regressions.

This increment is an **inference computation teaching layer**, not a reconstruction of today's open-source DeepInfer offline engine or a performance evaluation. DeepInfer's currently released offline inference runtime is a new verifiable upstream reference; its missing full-service integration components must not be claimed as included.
