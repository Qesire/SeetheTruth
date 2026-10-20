# DeepInfer · day-zero source audit (2026-10-20)

> **Correction:** DeepSeek open-sourced **DeepInfer, its real production-derived inference framework**, on 2026-10-20. Earlier claims in this project that *no unified DeepSeek inference engine source had been released* are now outdated. This file documents what is **actually released**, what still needs external serving infrastructure, and exactly what SeetheTruth does and does not implement.

## Release identity and practical scope

| Artifact | Official upstream | Released scope |
| --- | --- | --- |
| DeepInfer | https://github.com/deepseek-ai/DeepInfer | Apache-2.0, **offline inference** for DeepSeek-V4.1-Flash, actual model/runtime/sampling/kernel/scheduler code, multi-device execution |
| deepseek-recipe | https://github.com/deepseek-ai/deepseek-recipe | MIT, Rust libraries and Python bindings for API format conversions, V4/V4.1 prompt rendering/token encoding, tool/stream/output parsing and image preprocessing; **no model inference / tool execution / HTTP transport** |
| DeepSelect / FlashMLA / DeepGEMM / DeepEP | Repos in the deepseek-ai GitHub organization | Building blocks integrated by DeepInfer; their interfaces and supported hardware must be checked against the engine's pinned revision |

Verified against the public `deepseek-ai/DeepInfer` repository on 2026-10-20, especially `README.md` and the source paths below. The first search-indexed commit snapshot was `8b16149741a4935e1dec8cea27853f273b374c5b`. Revalidate against current `main` for subsequent code changes.

**Supported model:** DeepSeek-V4.1-Flash. **Supported accelerators:** NVIDIA SM100 and Huawei Ascend 950; NVIDIA and Ascend have distinct requirements and capability flags. Source build requires Linux, GCC >=13, Python 3.12, CUDA 13 on NVIDIA or CANN 9.2 on Ascend. The offline example is `python examples/offline_inference.py`. It downloads `deepseek-ai/DeepSeek-V4.1-Flash-1017` unless `DEEPINFER_HF_MODEL_DIR` points to a local checkpoint.

**Missing for a full, continuously available online inference service (per README, as of release):** an external/completed KV cache system, VE/Prefill/Decode disaggregation, and prefill/decode load balancing across DP replicas. Some scheduler/BlockManager components are present in source; do **not** confuse that fact with a complete production service distribution being included.

## Reconstructed actual source-level execution flow

1. **Offline caller:** `examples/offline_inference.py` and `deepinfer/entrypoints/llm.py` set up prompts, sampling parameters, the model and device groups. The public `LLM` entrypoint instantiates `deepinfer.engine.llm_engine.LLMEngine`.
2. **Engine state and scheduling:** `deepinfer/engine/llm_engine.py` imports `Scheduler`, `BlockManager`, `FlopsEstimator`, `SchedulerOutputs` and `process_MTPInfo` from `deepinfer._C.engine`. Its `_create_scheduler()` passes block size, block manager, state manager, data-parallel process group and `yoco_truncate_len` to C++ scheduling. This is a real compiled scheduler, **not** merely a Python list of requests.
3. **Worker pipeline:** `deepinfer/worker/worker.py` describes `execute_model_step` as **Forward → Post-process → Sample**. Under zero-overhead scheduling (ZOS), decode postprocessing can be deferred and overlapped with the next forward step. This creates cross-step state dependencies absent from the current SeetheTruth single-DAG interpreter.
4. **Variable-length speculative decoding:** `deepinfer/worker/worker.py` imports `compute_seqlens_q_by_confidence_rates` from `deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py`. The relevant setting is `DEEPINFER_ENABLE_MTP_VARLEN=1`; `DEEPINFER_USE_NEXTN_DECODE` defines a maximum draft length (default **5**). The scheduler/worker must handle varying token counts and accepted-prefix outcomes; do not equate a simple greedy accept mask with DSpark's confidence-scheduled verification.
5. **SWA bounded replay:** `deepinfer/environ.py` defines `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` = `off | mask | compact`. `mask` retains all query tokens but changes the visibility mask; `compact` actually drops excluded consumer-query work. With replay enabled, `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` must cover `window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS`. Official V4.1 README example: `compact` with **140 retained tokens**. `deepinfer/worker/model_runner.py` enforces this lower-bound invariant. `examples/yoco/yoco_equiv.py` tests mask/compact metadata and model-output comparisons: the optimization is **bounded computation**, not an arbitrary sliding-window truncation.
6. **Deterministic and structured generation:** README documents `DEEPINFER_BATCH_INVARIANT=1` for NVIDIA, and deterministic sampling with a fixed `SamplingParams.seq_seed` and variable-length draft disabled. Structured output supports regex, JSON Schema and EBNF through `SamplingParams.so_args` (see `examples/_structured_output.py`). Hardware support and algorithmic guarantees depend on the configuration.
7. **Parallel execution:** Attention DP/balanced CP, MoE EP/EPLB, DeepEP communication and DeepGEMM/FlashMLA kernels are integrated across engine and `deepinfer/model_executor`. Hardware execution and transport are not equivalent to simply multiplying small matrices on CPU.

### Source file anchors

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md)
- [Engine scheduler/step](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/engine/llm_engine.py)
- [Worker step overlap](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py)
- [Variable-length confidence-scheduled MTP](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py)
- [Feature flags and SWA replay constraints](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/environ.py)
- [YoCo equivalence harness](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/yoco/yoco_equiv.py)
- [Official offline inference example](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/offline_inference.py)
- [Recipe README and API/prompt conversion examples](https://github.com/deepseek-ai/deepseek-recipe/blob/main/README.md)

## Correct SeetheTruth evidence status

**Already truly executable in SeetheTruth:** stable tiny Top-K, explicitly selected sparse QK/softmax/PV, logical-to-physical KV page read, greedy speculative accepted-prefix mask, pure tensor algebra and limited graph arithmetic evidence. These remain legitimate *independent mathematical teaching models*, validated with JS/NumPy/PyTorch node parity; they **are not extracted kernel-exact DeepInfer runtime code**.

**Not yet implemented or validated against real DeepInfer source behavior:**

- C++ `Scheduler`, asynchronous ZOS Worker execution, and temporal dependencies across model steps
- confidence-to-variable-draft-length scheduling and the associated target verification performance model
- YoCo `off/mask/compact` bounded-replay exact query/mask and metadata equivalence
- EP/EPLB topology, RDMA/NVLink message counts and true communication/compute overlap
- precise V4.1 quantized KV storage, physical memory allocator, FP8/FP4 layouts, and distributed prefill/decode
- batch-invariant random sampling using stable per-sequence seed and concurrent-batch equivalence
- structured-output constrained automata and full API request/prompt/token lifecycle via `deepseek-recipe`
- actual SM100 or Ascend 950 kernel correctness/performance and official DeepInfer checkpoint inference

These are **specific next engineering tasks**, each requiring source-derived states, inputs/outputs, units, exact tensor semantics and small falsifiable numeric regressions. Avoid inventing a GPU deployment claim or treating a stand-alone CPU simulator as full source parity.

## Recommended implementation order

1. **YoCo bounded replay (first):** model `mask` and `compact` as separate query/mask operations, prove equivalent retained-suffix outputs subject to the exact `KEEP_TOKENS >= window + backtracking` invariant, and separately count removed consumer-query MAC.
2. **Confidence-scheduled variable MTP:** inspect the exact TileLang implementation and worker metadata; implement transparent host-side toy schedule decisions plus a sequence-length contract, distinguish acceptance statistics from throughput.
3. **ZOS pipeline:** represent Scheduler → Worker Forward → deferred Post-process → Sample as explicit temporally ordered events, with resource and synchronisation hazards; compare overlapped vs serial schedules using symbolic durations, not invented GPU timestamps.
4. **EP/CP and deterministic sampler:** reconstruct collectives, token ownership, and per-sequence RNG/counter state before pretending to model performance.
5. **Full stack boundary:** bind deepseek-recipe request mapping to actual offline DeepInfer interface, then add virtual online-service components with **simulated** status until source support is available.

This audit corrects the evidence base; it does not claim any new implementation is complete.
