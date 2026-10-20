# DeepInfer source-derived runtime microcontracts: DSpark / YoCo (2026-10-20)

## Exact source path: confidence-based variable MTP query allocation

Upstream source: [DeepInfer `schedule_mtp_varlen.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py), specifically **`compute_seqlens_q_by_confidence_rates_ref`**. Called by the ZOS branch of [`worker.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py), after current query-length metadata is staged and before the next model step.

This block **matches the reference ranking semantics**, not just its purpose:

1. `remap[i]` is the previous batch index for current sequence `i`, or `-1` for a fresh sequence.
2. For each current sequence `i` and draft index `j`, compute `cum[i,j]=prod(prev_confidence[remap[i],0:j+1])`. A fresh sequence has all zero cumulative rates.
3. Flatten all `N*nextn` rates into row-major order; rank descending by rate, **breaking ties by smaller original flattened position**.
4. Give every sequence one base query token. Award precisely `totalQ-N` additional query tokens across sequences according to this global ranking. Thus `sum(seqlens_q)=totalQ` and `1<=seqlens_q[i]<=nextn+1`.
5. Reject out-of-range remaps/confidence, invalid budgets and tensors.

The demo starts from `remap=[0,1,-1]`, confidence `[[.9,.8],[.5,.5]]`, and `totalQ=5`, giving `[3,1,1]`. Reference ordering here matters: for ties, replacing stable ordering with per-sequence sorting changes the result. The real DeepInfer implementation has a TileLang kernel for the same rank-budget semantics and handles H2D/D2H, C++ scheduler updates, acceptance counts and cross-step events. **No GPU timing or actual Draft forward inference is claimed**.

## Bounded YoCo SWA consumer replay: mask vs compact

Official paths: [`model_runner.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/model_runner.py), [`environ.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/environ.py), and [YoCo `yoco_equiv.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/yoco/yoco_equiv.py).

The official `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` modes are `off`, `mask`, `compact`:
- `mask` retains query slots and narrows the SWA visibility region;
- `compact` actually excludes prefix consumer queries and changes the active query layout;
- `off` disables this optimization.

Safety invariant: `keepTokens >= window + backtracking`. Real deployments must also correctly maintain chunk boundaries, index pivots, cache state and decoder backtracking.

**Teaching microcontract, explicitly not the full DeepInfer metadata:** inputs `Q[T,D]`, `K[T,D]`, `V[T,Dv]` and `window`, `keepTokens`, `backtracking`. For retained positions `t >= max(0,T-keepTokens)`, compute causal local Attention using `K[max(0,t-window+1):t+1]`, normalised by `sqrt(D)` and stable Softmax. `mask` returns `[T,Dv]` with zero sentinel prefix rows, whereas `compact` produces only the retained `[min(T,keepTokens),Dv]` rows.

`mask[-retained:]` must agree with `compact` on the **same Q/K/V inputs**, not by comparing unrelated synthetic streams. The zero prefix illustrates non-reusable consumer outputs; the actual upstream mask metadata need not physically output these zeros. The query-row count and valid QK pairs are explicit *logical* costs, **not observed physical FLOPs, bandwidth or latency**.

The workbench cost panel reports each YoCo block's scheduled query rows, retained rows and eligible QK pairs. For one T=4, keep=3, window=2 example, mask schedules 4 query slots and compact schedules 3; both have 6 eligible QK score pairs. That distinction is why compact may save work when the real consumer implementation cannot skip masked query rows, **not a guarantee of speedup**.

## Tests and reproducibility

- `node tests/deepinfer_runtime_contracts.js`: official MTP reference hand oracle, stable ties, fresh sequence handling, budget sum, remap guard, 4-token SWA suffix equivalence and invariant checks.
- `node tests/deepinfer_runtime_codegen_parity.js`: execute every output of both course DAGs with JavaScript, independently generated NumPy, and real CPU PyTorch; each output is compared.
- `python tests/browser_multimodal.py`: real Chromium course execution, source-card links, generated NumPy/PyTorch downloads and screenshots.
- CI also runs old symbolic graph, scheduler-free training math and model contracts.

This release **does not** claim to implement DeepInfer's GPU DSpark scheduler, YoCo chunk/pivot mechanism, confidence head, ZOS CPU/device overlap, full KV cache system, real source-compatible V4.1 checkpoint, or deployable Serving engine.

## Next source-driven work

Rebuild the real `prepare_prompt_v4` chunked SWA metadata semantics and the YoCo `mask-compact` official test matrix, including fully clipped chunks, pivot boundaries and original position recovery. After that, model the Worker ZOS event order and crossed-step sampler dependency without inventing GPU runtime, then multi-device EP/CP topology and communication/compute overlap.
