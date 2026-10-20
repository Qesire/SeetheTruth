# DeepInfer KV Block Ownership & Lifecycle — source-led CPU reconstruction

Source revision verified on 2026-10-20:
- [BlockManager / Blocks free-list and refcounts](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/block_manager.hpp)
- [Scheduler fork/branch, _append_slot, _preempt and free_seq_nolock](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp)
- [Sequence::get_num_blocks / reset_for_recompute](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/sequence.hpp)

This **is a bounded executable CPU state machine of source-derived allocation semantics**, not CUDA memory allocation, a deployed PrefixCache, byte-copy kernel, or performance evidence.

## Source-derived invariants

1. **Physical Block 0 is reserved.** `Blocks(num_blocks)` constructs an ordered FIFO free list `1..num_blocks-1`; zero is never a returned user block. It supports `OK` (sufficient available), `LATER` (capacity possible but currently exhausted), and `NEVER` (`to_allocate_blocks >= num_blocks`, even if no blocks are occupied).
2. **RefCount is actual ownership.** `allocate` dequeues from the front and sets refcount 1. `pin_block` adds a reference to an already-owned block, never to a free block. `free_blocks_nolock` decrements the supplied vector in **reverse** order; a block is returned to the **tail** of the FIFO free-list iff refcount becomes zero. A parent request finishing cannot free a block still pinned by its forked child.
3. **Sequence required physical blocks:** `ceil((tokenLen + totalInflightTokens + draftTokensCount) / blockSize)`. This is `Sequence::get_num_blocks()`, distinct from the compressed block-table *read* layout of the earlier `prepare_decode_zos_inputs` lesson. `Scheduler::_append_slot` adds **one** new block when needed or releases surplus trailing blocks.
4. **Forked branch prefix sharing is conditional.** This source's branch handling starts with a pinned reference to another request's block table, not arbitrary hash-based Prefix Cache lookup. On activation, `num_reserve_blocks = floor(cachedTokens / blockSize)` complete blocks remain shared. The child gets `get_num_blocks()-num_reserve_blocks` newly allocated blocks. If its cached prefix ends in a partial block, the source records `clone_blocks = (oldTail, newFirstBlock)` and releases the child's old-tail pin. This is an **explicit copy action**, not an already-completed GPU memory copy. `_clone_state` additionally clones a distinct GPU state buffer when enabled; our simulator does not implement that state allocation.
5. **Preemption separates stop from recompute.** `Scheduler::_preempt` can stop a request immediately when `stop_preempted_request` or structured-output status forbids recompute. Otherwise it invokes `Sequence::reset_for_recompute()` (rolling KV/MTP/scheduling state reset), returns the request to WAITING at the front and frees owned blocks. Its original token stream is retained. Our bounded state resets cached token count, inflight and draft counters; it does not re-create the full real request queue, structured-output grammar automaton, sampler state or MTP info-slot pool.

### Atomicity and conservation

Each `advance(previous,op)` constructs a new immutable-by-convention snapshot, applies one operation and validates **every block**:

`refs[block] == number of request BlockTables currently containing that block`.

The free FIFO contains exactly the blocks with `refcount=0`, no duplicates, and all IDs except the sentinel are either free or owned. The earlier snapshot remains unchanged if an operation throws. This contract catches double frees, shared ownership leaks, invalid fork activation and unsafe allocation.

## Browser experiment

[KV lifecycle lab](../deepinfer_kv_lifecycle_lab.html) has physical block cells, live owners, refcounts, FIFO queue order, request status and block tables, fork copy pairs, step-by-step timeline, editable operations and downloadable state audit JSON. [Worker event lab](../deepinfer_worker_lab.html) links to it; the separation is deliberate:

- Worker event lab models **step-level causal scheduling and per-Step Q/K metadata**.
- KV lifecycle lab models **physical block ownership and sequence resource lifetime**.

A single fake tensor DAG would obscure the distinct semantics of event ordering and ownership. Both labs reuse official source file links and describe their evidence limits.

One independently verifiable fork scenario, total `numBlocks=7`:
- A receives blocks `[1,2]`, B receives `[3]`;
- C pins A's `[1,2]` with `cachedTokens=5`, both refs rise to 2;
- C activation reserves one complete shared block `[1]`, allocates `[4]`, and records partial-tail copy `2→4`, releasing its pin on A's block 2;
- finishing A, C and B preserves shared ownership and returns free blocks in the precise order `[5,6,2,4,1,3]`.

The browser screenshot and independent JS contracts verify the actual arrays, not merely illustrated labels.

## Validation

- `node tests/deepinfer_kv_lifecycle_contracts.js`: exact sentinel, FIFO, refcount, fork sharing/partial-tail copy, state conservation, overflow, duplicate/double free, LATER/NEVER and SO-preemption behavior.
- `python tests/browser_deepinfer_kv_lifecycle.py`: genuine headless Chromium, fork timeline, copy events, physical owner colors, export JSON, invalid input rollback, preempt vs recompute and screenshots.
- GitHub CI runs these alongside existing BlockTable packing, Worker ZOS Q/K and CPU Torch source math regressions.

No claim is made that the generic Prefix Cache subsystem, cross-device migration, multimodal chunk-specific prefix matching, physical GPU memory or real serving throughput has been reproduced. The fork path is a documented subset of DeepInfer's actual Scheduler behavior.
