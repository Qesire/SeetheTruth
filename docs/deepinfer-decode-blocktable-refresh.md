# DeepInfer Decode KV BlockTable & sampling-slot state — source audit (2026-10-20)

## Exact official implementation paths

- [`csrc/engine/zos.cpp::prepare_decode_zos_inputs`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/zos.cpp)
- [`csrc/engine/zos.cpp::update_metadata_for_sampling_zos`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/zos.cpp)
- [`csrc/engine/sequence.hpp::SequenceMetadata::refresh_for_sampling`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/sequence.hpp)
- [`csrc/engine/sequence.hpp::MtpState::take_draft_info_slot / next_golden_tokens`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/sequence.hpp)
- [`deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py)

These paths are source-level evidence for deterministic *integer metadata calculation*. They do not prove SeetheTruth can run DeepInfer's model or its C++ library.

## Packing semantics: not an invented page table

Given each SequenceMetadata's uncorrected Q and K, `block_size`, `max_seqlen_q`, `block_table_compress_ratio`:

```text
tokens_per_block = block_size / block_table_compress_ratio
max_k = seqlen_k - seqlen_q + max_seqlen_q
compressed_k = floor(max_k / block_table_compress_ratio)
n_blocks = ceil(compressed_k / tokens_per_block)
```

**Exception:** placeholder rows always use one block. Non-placeholder rows can require *zero* blocks when `compressed_k=0`; they must not be silently forced to one. A malformed ratio, missing reserved Block Table entry, or exceeding `max_num_blocks` is rejected. Data structures mirror the source:
- `block_tables` is all sequences' selected physical block IDs concatenated.
- `block_table_offsets` has length `num_sequences+1` with leading zero and cumulative physical block counts; it **is not** a dense sequential logical block mapping.
- `cu_seqlens_q` is the cumulative query-token count **after each sequence** (length `num_sequences`, not prefix with a leading zero).
- `state_ids` and `is_placeholder` retain independent per-sequence values.
- `seqlens_q` and `seqlens_k` are present only if `need_default_seqlens=true`; otherwise these source optional buffers are absent.

The constructor **does not allocate or free** GPU blocks. Its inputs already contain externally reserved block IDs. We do not claim to reproduce Scheduler `_append_slot()`, BlockManager, token-input packing, compressed attention kernels, engram token gathering or CUDA Graph capture.

## Source-equivalent MTP / sampling metadata transition

`update_metadata_for_sampling_zos` assigns `Q_corrected` and uses `seqlenK=Q_corrected+seq.cache.num_tokens` for the sampler (rather than reusing the earlier decode K correction). It refreshes metadata only for requests that map to a prior non-placeholder request.

The C++ refresh method mutates the selected Sequence metadata using:

1. `last_output_token = seq.mtp.last_output_token`
2. `mtp_info_read_slot = seq.mtp.take_draft_info_slot()`: obtain `mtp.info_slot` if `has_draft_info`, otherwise 0, then **clear** `has_draft_info`.
3. If `mtp.info_slot != 0`, update `mtp_info_write_slot` to that slot. When zero, **retain** the previous write slot.
4. Copy `so_status` and `so_matcher` references; there is **no** claim that the matcher automaton has advanced.
5. Calculate `golden_token = seq.mtp.next_golden_tokens(output_len,nextn)`: if available, take `2*nextn+1` reference positions starting at `output_len`, padding beyond the reference end with `-1`. If `output_len >= size` or the golden reference is absent, the output is `null`.

`deepinfer_decode_metadata.js` implements this deterministic subset using pure data copies and returns `nextSequence` separately. Because the source mutates the original C++ Sequence whereas the teaching implementation returns a fresh object, callers can inspect the before/after transition explicitly. It is *not* an exact physical Python/Pybind object lifecycle.

## Connected cross-step experiment

The Worker timeline already supports explicit per-Step `metadataFixture` (Q/K and previous accepted tokens). The new optional `decodeFixture` names one Step, the **uncorrected scheduler Q/K** plus physical Block Table IDs and per-request MTP states. It is validated against that Step's sequence identities and uncorrected lengths, then packed and updated using that Step's *real corrected Q* from the previous-step sampler. All output is available under `metadataTrace[step].decode`.

The browser shows the pack's physical block IDs, block offsets, number of blocks, cumulative Q, sampler-side Q/K, MTP read/write slots and Golden Token window. The values are explicitly user-editable **synthetic fixtures**, not model-generated samples.

Example: `block_size=4, ratio=2, max_seqlen_q=3`; scheduler sequences B (Q=1, K=13, physical [7,4,8,9]) and C (Q=1, K=5, physical [2,3]). Result:
- `numBlocks=[4,2]`, `blockTables=[7,4,8,9,2,3]`, `blockTableOffsets=[0,4,6]`, `cuSeqlensQ=[1,2]`.
- Previous sampler's global confidence-budget correction sets Q of B/C to `[2,1]`; with explicitly provided cache token counts `[12,4]`, sampling K is `[14,5]`.
- B's draft information is consumed once from MTP slot 5; its Golden Token window with `outputLen=1, nextn=2` is `[11,12,13,14,15]`. C is fresh, therefore the refresh method is *not* invoked on C.

## Test evidence and boundaries

`tests/deepinfer_decode_metadata_contracts.js` asserts the exact packing, empty compressed case, placeholder and optional buffers, invalid resource bounds, immutable refresh semantics, read/write slot ownership, and Worker integration.

`tests/deepinfer_decode_metadata_py_parity.js` independently performs packing and sampling arithmetic with actual CPU PyTorch/Python and compares every produced array to JavaScript for four shapes, plus matched and fresh request paths.

`tests/browser_deepinfer_worker_events.py` reads the live packed arrays and state transitions in real Chromium; its screenshot covers the new panel, and invalid fixture input must not overwrite the prior valid computation. Existing CI includes all prior inference and training tests.

**Excluded:** real cache allocation or freeing, RDMA/EP, GPU numerical Forward, C++ mutable object pool, Scheduler preemption, compressed token content, sampling-matcher token advance and real elapsed time. Those need separate source reconstruction and hardware-backed evidence.
