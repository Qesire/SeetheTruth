# DeepInfer ZOS Metadata: exact CPU reference and cross-step integration (2026-10-20)

Source-verified upstream:
- [C++ `prepare_zos_inputs` in `csrc/engine/zos.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/zos.cpp)
- [`compute_seqlens_k_zos_ref` Python / CUDA TileLang](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/compute_seqlens_k_zos.py)
- [`Worker._correct_seqlens_qk`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py)
- [`MetadataZOS` fields](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/zos_status.py)

## Exact relevant source semantics

1. `prepare_zos_inputs`: compare current and previous `SequenceMetadata.seq_id`. Same-order fast path uses identity remap if batch lengths agree, indices match, and no preceding sequence is placeholder. Otherwise hash the non-placeholder previous batch and map each current `seq_id` to its previous batch index, or `-1` for a fresh sequence. If **no overlap**, return four empty/false outputs, so no GPU correction should be scheduled.
2. For overlapping batches, allocate pinned CPU `remap`, `seqlens_q`, `seqlens_k`; base `seqlens_k[i] = current[i].seqlen_k - (nextn>0 ? current[i].seqlen_q : 0)`. CUDA Graph padding rows have `remap=-1`, `q=1`, `k=1`.
3. Optional DSpark variable MTP length: `compute_seqlens_q_by_confidence_rates` corrects actual unpadded Q rows by remapping prior-sequence confidence rates and ranking cumulative confidence rates within a global `total_q` budget. The CPU teaching implementation delegates to the already-tested `deepinfer_runtime_ops.schedule`, which matches the upstream Python Reference.
4. For each **active** row, the upstream `compute_seqlens_k_zos_ref` computes
   `corrected_k = base_k + corrected_q - (remap>=0 ? nextn - prev_accepted[remap] : 0)`.
   The previous accepted count is indexed by the *previous batch index*, not current row index. A remap of `-1` has **no subtraction**. Padded rows remain exactly `q=k=1`.
5. With varlen enabled, only the active corrected Q is copied back to pinned CPU (`d2hSeqlensQ`); the ordinary fixed-Q ZOS path does not claim an equivalent D2H event.
6. For the *sampling* view only, the official `update_metadata_for_sampling_zos` sets `seqlen_q=correctedQ` and `seqlen_k=correctedQ+seq.cache.num_tokens`; it calls `refresh_for_sampling` for matched sequences. This microcontract produces a **read-only preview** and flags whether refresh is required, but does **not** recreate that side effect.

## Executable capability added to SeetheTruth

`deepinfer_zos_metadata.js` exports the pure functions `prepare`, `correct`, and `samplingView`. They use real, exact integer arrays with bounded size, reject malformed IDs/metadata/ranges, and do not require a GPU.

The existing `deepinfer_worker_events.js` can now accept an **optional explicit** `metadataFixture={nextn,variableMtp,records:[{sequences,accepted,confidenceRates,totalQ,cachedTokens,paddedLength},...]}`. A worker correction event carries actual `remap`, corrected Q and K, prior accepted tokens and varlen D2H status. The returned `metadataTrace` also exposes base K and the sampling-side read-only preview, one record per correction. Other event details, dependency edges, CPU/GPU resource restrictions and deferred sample lifecycle are preserved.

The browser's [Worker event laboratory](../deepinfer_worker_lab.html) now offers a **clearly identified synthetic sampler fixture** with editable JSON. The new table displays remap, corrected Q, base K, corrected K and sampling-side Q/K. Users can turn off numerical metadata: no fabricated numerical arrays are inserted when the fixture is disabled. Changing to another or custom sequence scenario automatically disables the incompatible numerical fixture until explicitly revalidated against that scenario.

## Independent validation

- `tests/deepinfer_zos_metadata_contracts.js`: exact remap/padding/no-overlap, past accepted-count sensitivity, stable varlen Q allocation, K correction, sampler preview, Worker event integration, invalid fixtures.
- `tests/deepinfer_zos_metadata_py_parity.js`: 6 independent test cases run **real CPU PyTorch** for stable cumprod confidence ranking and `seqlens_k_zos` reference arithmetic, and compare against the JavaScript arrays. This is not a GPU run.
- `tests/browser_deepinfer_worker_events.py`: verify real Chromium shows the corrected numbers, rejects malformed fixture values without corrupting the last successful result, and clears numerical traces when disabled.
- Existing training, source, browser and CPU export regression suites remain intact.

## Explicit limits

This change **does not execute** DeepInfer's C++ scheduler, CUDA/TileLang compiled kernel, true pinned host memory, H2D/D2H transfers, CUDA Graph placeholder padding selection, actual sampler/acceptance head, `refresh_for_sampling` side effects or distributed KV BlockManager. The small fixtures are authored for teaching and inspection, not recorded output from a DeepSeek checkpoint or a production GPU execution trace. No simulated times are performance measurements.

Next: reconstruct exact `prepare_decode_zos_inputs` / compressed block-table offsets and `refresh_for_sampling` memory-state effects, then align multi-stream CUDA events with live GPU provenance.
