# DeepInfer Scheduler → KV BlockManager → Worker: decode-only source-grounded integration

Date 2026-10-20. **This is a real executable bounded CPU control model, not an implementation of the production DeepInfer C++ Scheduler.** Source basis:

- [DeepInfer `csrc/engine/scheduler.hpp::_schedule_decode`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp) — required decode priority, resource guard, preemption and batch ordering.
- [`_preempt` and `free_seq_nolock`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp) — recompute vs terminate, release of blocks and MTP slot state.
- [`csrc/engine/block_manager.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/block_manager.hpp) — reserve block zero; FIFO physical free list and refcounts.
- [`deepinfer/worker/worker.py::execute_model_step`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py) — causally ordered Worker Forward, prior-step Post-process, Sample, metadata correction and Flush.

## Why integration matters

The existing SeetheTruth modules independently verify BlockManager KV ownership and a stateful Worker event graph. Their independent validity does **not** guarantee scheduling validity: a naive combination could forward a preempted request, use a freed block, automatically restart a recompute request in Decode without Prefill, or invent GPU overlap metrics. This increment explicitly generates **each Worker batch from live Scheduler resource decisions**.

## Source-derived bounded scheduling contract

For each decode cycle:

1. A caller supplies the current running requests' **output lengths** (the source uses `Sequence::get_output_len()`), optional new token/inflight/draft workload requirements, and any previously accepted WAITING request explicitly admitted by a prefill-shaped action. A preempted WAITING request **does not automatically become RUNNING**.
2. The upstream C++ sorts RUNNING requests by **descending generated output length**, then schedules from the **back**. Thus requests with **shorter generated outputs** get priority and the front of the ordered list is selected as preemption victim. This is the actual source mechanism; “LCFS” as a comment does not override the comparator. For equal output lengths, our demo uses a stable pedagogical ID tie-break; C++ `std::sort` does not promise that exact tie ordering.
3. If no physical block is free and there are multiple remaining candidates, preempt a victim from the opposite end until the guard succeeds or only the current candidate remains. **Source nuance:** `_schedule_decode` checks `can_allocate_single_nolock()` before calling `_append_slot` **even if a candidate’s current required block count would not grow**.
4. A successful request gets one `_append_slot` call using the actual bounded `get_num_blocks` arithmetic, and may release excess blocks. The Scheduler's candidate batch is then sorted by **descending total token length** before it is handed off to Worker.
5. Preemption releases the request’s blocks. If `stop_preempted_request` is enabled, or structured-output status is nonzero, the request is terminated (`FINISHED_PREEMPTED`). Otherwise its rolling cached/inflight/Draft state is reset and it enters `WAITING` for eventual **Prefill recomputation**, not the next decode batch.
6. Only still-RUNNING requests with owned KV blocks can enter the Worker. The Worker event graph uses exactly the Scheduler-approved `sequenceIds`, checks causal dependencies and emits synthetic CPU/GPU event ordering. Its durations are *assumed*, not device traces.

The current model validates block conservation and ownership after **every primitive allocation, release, append and preemption**, and validates its stored snapshots. A failed transaction or invalid editing input never replaces a previous valid run.

### Important constraints and unsupported claims

- **Decode-only subset.** The full DeepInfer NORMAL engine dynamically switches between prompt and decode using distributed ALL/ANY voting, DP synchronization and a GPU cache watermark. **Those branches are not modeled.** The simulator exposes later admission as an *explicit prerequisite* corresponding to a separate Prefill pass, not a claim to implement upstream `_schedule_prefill`.
- **No actual sampling.** Worker sample and GPU/CPU bars are simulated logical events, not generated model token IDs or wall-clock timings; outputs are never written back into token sequences. The `outputLens` per cycle are given as controlled small-example input.
- **No exact MTP-slot scheduling.** The existing standalone numerical DSpark Q/K and MTP slot modules are available elsewhere, but this new decode priority loop intentionally omits actual `max_num_seqs`, MTP slot allocation and distributed CP/EP constraints. It cannot be called a full DeepInfer Scheduler replica.
- **No GPU physical allocation.** The BlockManager's physical block *IDs, reference counters and free FIFO* are modeled as deterministic CPU arrays; no KV bytes, CUDA copies, TP/CP collectives, or production throughput exists.
- **No globally deduplicated prefix cache.** Fork sharing was modeled in the previous lifecycle lab; current integrated demo focuses on memory pressure and preemption rather than automatic prefix matching.

## Falsifiable cases and outputs

**Memory pressure:** seven physical block IDs including reserved zero. Three active requests initially occupy all six usable blocks. At the next Decode, request A has longest output length and is preempted into WAITING, while shorter B/C retain active blocks; only B/C are passed to Worker. Later, an explicit prefill admission puts A back into RUNNING, after sufficient blocks have been released. Without this admission, A would not have a valid Worker forward.

**Structured output:** a request holding structured-output state is preempted and terminated, not recomputed, matching the official restriction on structure-preserving replay.

**Sufficient resources:** no preemption, and valid batches progress until explicitly finished. This is a control example, not a GPU efficiency measurement.

## Validation

- `tests/deepinfer_scheduler_workflow_contracts.js`: full source-priority policy, memory conservation, preempt/WAITING transition, explicit re-admission, SO stop, Worker batch identity, CPU/GPU causal ordering, invalid state guard and atomic rollback.
- `tests/browser_deepinfer_scheduler.py`: real Chromium live KV owner/refs, runtime scheduling choices, generated CPU/GPU event Gantt, stop/recompute toggles, screenshot artifacts, editable workflow and JSON export.
- GitHub Actions also executes prior DeepInfer numerical Q/K and KV Decode BlockTable tests, broader numerical learning flows and existing CPU PyTorch training regressions.

Entry point: [DeepInfer integrated Scheduler lab](../deepinfer_scheduler_lab.html), linked to both the KV lifecycle and Worker labs.

## Next engineering direction

Source reconstruction should next cover the actual `_schedule_prefill` admission and chunked/multimodal prompt constraints, the NORMAL engine DP ALL/ANY vote, free `MTP` slot pool and request budget caps. Once these are present, integrate the existing real Q/K metadata, Draft sampling and repeated resource transaction. Keep simulated values separate from independently measured GPU events.
