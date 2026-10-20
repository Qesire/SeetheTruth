# DeepInfer Prefill · NORMAL votes and image-safe chunking

Official release source: https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp

## Verified NORMAL engine decision path

The C++ Scheduler uses:
- ANY attempt when decode_step != 0 and decode_step % prefill_any_attempt_interval == 0.
- ALL attempt when decode_step % prefill_attempt_interval == 0.
- When both apply, ANY takes precedence.
- Per-rank **want** requires cache usage <= GPU_CACHE_PREFILL_WATERMARK and a schedulable waiting queue head.
- ANY uses an allreduce MIN of !want, ALL uses MIN of want. All ranks must choose the *same* step type.
- A second vote verifies that at least one rank **actually scheduled** a prefill. Otherwise, scheduling falls back to Decode; rank-specific workless prompt steps only occur when another rank did schedule real Prefill.
- Successful prompt resets decode_step to zero. A fallback does not.

The pure `deepinfer_prefill_policy.js::vote` implements the decisions for explicitly supplied rank states. It computes every rank's waiting predicate and actual scheduling eligibility but **does not** execute c10d/DP allreduce.

## Chunked Prefill boundary

The source's `_schedule_prefill` starts from remaining work = token length minus cached tokens. It trims to CP-aligned free token budget, respects explicit checkpoint positions, and then applies `_adjust_chunk_size` to avoid splitting an image's contiguous PAD body. At a proposed boundary on IMAGE_PAD, the first eligible request can expand to the first token **after** the image; if that exceeds budget, shrink to the image start. Later requests shrink rather than extend. C++ permits a boundary exactly **at** IMAGE start.

The JS lesson models token categories TEXT, IMG, PAD, AUDIO, VIDEO without claiming real Token IDs. It produces q, remaining work, CP padded charge, WAITING status while chunked, RUNNING when complete, and rejects zero-progress first requests. It checks the same image-boundary logic, but it **conservatively rejects** any CP padded footprint that would exceed the teaching budget; users should not interpret this additional guard as a byte-for-byte C++ Scheduler implementation.

## Test oracles

- At decodeStep=4, ALL interval=2, ANY interval=4: ANY wins; one rank with work produces one prompt and one empty-prompt rank.
- If all first-stage wants succeed but no rank can actually schedule, second vote falls back to Decode, avoiding zero-progress cycles.
- For [TEXT, IMG, PAD, PAD, PAD, TEXT, TEXT] and a budget of four, the tentative cut falls inside IMAGE_PAD; expanding exceeds budget, so it shrinks to Q=1 (before IMG). With budget five, full image included, Q=5.
- CP alignment with seven text tokens and alignment 2 charges eight slots.

**Not implemented:** full waiting queue, prefill cache block allocation, fork state copies, GPU watermarks from live BlockManager, actual multimodal token IDs, model Forward, distributed processes or throughput. The existing Decoder-only Scheduler→KV→Worker lab is independent until a future unified NORMAL-engine state machine.

Executable Node tests: `tests/deepinfer_prefill_policy_contracts.js`.
Chromium teaching test: `tests/browser_deepinfer_prefill.py`.
Both desktop and Android teaching edition link the standalone page `deepinfer_prefill_lab.html`. It is an explanatory feature, not a model result-producing feature.
