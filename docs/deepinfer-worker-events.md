# DeepInfer Worker event semantics — source-grounded *logical* execution model

Date: 2026-10-20. Official upstream read: [DeepInfer Worker.execute_model_step](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py), [LLMEngine.step](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/engine/llm_engine.py).

**Scope:** SeetheTruth now implements a cross-step, two-resource, discrete logical-event model, not just a feed-forward tensor DAG. Every event has a start/end value, CPU or GPU resource, step identity, explicit dependencies, metadata and source owner. Durations are **user-supplied hypothetical ticks**, not GPU traces. A new browser interface is at [DeepInfer Worker timeline](../deepinfer_worker_lab.html). The simulator does not execute DeepInfer itself, load its checkpoint, or measure performance.

## Authoritative sequence in DeepInfer

The 2026-10-20 `worker.py` `execute_model_step` docstring explicitly states:

1. Forward current step (whether prompt/decode/placeholder).
2. Postprocess previous step.
3. Sample current step, after metadata correction when needed.
4. Prompt or non-ZOS: postprocess the current step immediately. ZOS decode: store `prev_step_state` and postprocess it in the next step (or `flush_inflight_reqs`).

The source calls `_post_process_step` on `prev_step_state` **after** launching the current forward. That method waits on the recorded `sample_end_evt` before invoking model postprocess and engine callbacks. `_correct_seqlens_qk` uses previous sampler GPU outputs (and `prepare_zos_inputs` remapping) to correct lagged `seqlens_q`/`seqlens_k`; when variable-length MTP is enabled, the worker also performs D2H and waits before sampling. Non-ZOS and prompt do not delay the current postprocess. The C++ `Scheduler.schedule` is invoked from [`LLMEngine.step`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/engine/llm_engine.py) before worker execution.

## Bounded logical realization

The `deepinfer_worker_events.js` state machine takes up to 12 `{mode:'prompt'|'decode'|'placeholder', sequenceIds:[...]}` steps and an explicit duration table.

It emits:
- CPU: schedule, launch-forward, post-prev (if pending), launch-sample, immediate current postprocess or explicit flush;
- GPU: optional ZOS metadata correction when current and prior batches overlap, forward, sample;
- cross-step dependencies: current correction depends on preceding sampled output, previous postprocessing depends on the prior sample completion event, and current sampling follows its own forward and CPU launch;
- user-facing result-ready states with `immediate`, `deferred` or `flush` provenance.

Both CPU and GPU are deliberately modeled as **one non-overlapping resource each**. The simulator rejects illegal negative timing assumptions, invalid step/sequence structures, and unsatisfied event dependencies. It guarantees no duplicated result delivery. A pending last ZOS decode step is not declared complete until explicitly flushed; `flush:false` leaves the state pending rather than inventing a result.

The source's asynchronous GPU launch, multiple hardware streams, CPU-GPU memory-transfer costs, CUDA event synchronization, C++ scheduler internals, EP, CP and advanced speculative metadata are **not literally executed**. The CPU launch and GPU forward are split to permit conceptual overlap; the correction event represents a dependency rather than bit-exact GPU `seqlens_k` behavior. This model has no source-measured duration parameters.

## Falsifiable counterfactual

With identical hypothetical stage durations, compare:

- **ZOS**: current GPU forward may overlap CPU postprocessing of the prior step; decode outputs are delayed until next step or explicit flush.
- **Non-ZOS**: current CPU postprocess waits for the current GPU sampling and is paid in the same step.

This is a scheduling dependency comparison, **not** evidence that DeepInfer is faster by the model's numerical makespan difference. The exact result depends on the chosen assumptions; no fixed `×` performance claim or GPU utilization follows.

The browser exposes editable step JSON, a time scale, separate CPU/GPU bands for each policy, inspectable event dependencies and a downloadable complete event graph. It connects from the Learning Studio as an **adjacent stateful systems laboratory**. We deliberately do not force the across-step recurrence into the current forward-only DAG of tensor blocks; this would silently misrepresent cyclic state as acyclic tensor computation.

## Acceptance tests

- `tests/deepinfer_worker_events_contracts.js`: dependency topological legality, per-resource occupancy, prompt/decode/placeholder paths, remap dependency, pending/flush semantics, exactly-once delivery, rejecting malformed timing/sequence inputs and showing a permitted overlap.
- `tests/browser_deepinfer_worker_events.py`: real Chromium control interactions, two rendered independent timelines, editable step list, detailed source/dependency view, valid exported JSON, rejected malformed editor input, and screenshot artifacts.
- Existing training and inference regression pipelines remain mandatory in CI.

## Next layer

Faithfully model `prepare_zos_inputs`, `compute_seqlens_k_zos` and the real source's `MetadataZOS` remap/accepted-token corrections, then align DSpark confidence allocation from `schedule_mtp_varlen.py` with worker event dependencies. Next implement multi-stream GPU event timelines using explicit CPU-D2H sync guards and show *only* algebraic cost or trace-grounded durations.
