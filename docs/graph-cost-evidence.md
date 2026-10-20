# Live graph cost evidence: exact coverage and exclusions

The Learning Studio **GRAPH COST** panel profiles the *current validated user-wired DAG*, not a
model-family template. It calls `GraphReadiness.check` for the same port and tensor-shape
contracts used before running a learning graph. Missing ports, incompatible shapes and
cycles prevent a numeric cost claim.

## Counted arithmetic

For a standard 2D `matmul(A[m,k], B[k,n])`, the count is `m*k*n` multiply-accumulates
(MAC). For four-axis `nd_batched_matmul`, the count is
`max(B_A,B_B)*max(H_A,H_B)*m*k*n`, following the workbench's allowed
singleton broadcast on batch and head axes. `linear(X[t,k], out=n)` similarly
counts `t*k*n` MAC. The conventional `2*MAC` field is explicitly a rough
multiplication-plus-addition arithmetic convention, **not** a measured kernel FLOP count:
true addition count for a dot product is typically one less per dot, and fused
kernels may implement operations differently.

The analyzer **does not claim full model FLOPs**. Attention implementations, routers,
normalization, activation, layer/state control, transfers, optimizer updates,
MoE dispatch, and kernel launches that are not expanded into supported MatMul
nodes remain explicitly outside this count. The `unprofiledKinds` field enumerates
the applicable other node kinds.

## Physical parameters versus synthetic teaching weights

Each `parameter` or `nd_parameter` node contributes its actual shape element
count exactly once even if multiple edges consume it or a weight is reused
across depth. The `directConsumers` field exposes immediate edge fanout.
This does not automatically detect semantic parameter aliasing across *distinct*
Parameter nodes.

The fixed pedagogical `linear` kind internally constructs deterministic
pseudoweights. Those values are reported separately as
`implicitTeachingWeightElements`; they **do not count as persistent trainable
Parameter tensors**.

The sum of logical output tensor elements counts shapes of every graph node.
It is neither a peak-live allocation, a VRAM estimate, nor a PyTorch/CUDA
liveness-and-reuse analysis. Nothing here extrapolates to device latency or
throughput.

## Reproducibility and limitations

- `node tests/graph_cost_report_contracts.js`: independent 2D oracle,
  repeated consumers of one physical Parameter, 4D batched computation and
  missing-port rejection.
- `node tests/release_surface_contracts.js`: verify script, UI panel and
  local HTTP route are wired.
- `python tests/browser_catalog_screenshots.py`: real Chromium preset changes
  must update visible parameter and arithmetic numbers.
- Every value is synthetic shape-derived symbolic evidence. Real-source and
  RuntimeTrace evidence, actual FLOP utilization, model-weight provenance,
  gradient/optimizer work and deployment latency remain separate modules.

## Next expansion

Resolve hardware-backed evidence by graph node and immutable capture source,
then distinguish (a) algorithmic operation count, (b) materialization/bytes
and (c) actual profiler/kernel attribution. None should silently inherit a
more authoritative evidence tier from the others.

## Attention, expert projection and cache extensions (2026-10-20)

`attention` is the two unmasked dense 2D GEMMs, `QK^T` and `PV`:
`Tq*Tk*(Dk+Dv)` MAC. Fused `gqa` similarly multiplies by `Hq` using
actual `Hq,Hkv`, `Dk=Q_width/Hq` and `Dv=V_width/Hkv`, with
`Tq*Tk*Hq*(Dk+Dv)`. For these kinds the synthetic input Q/K/V projections
are already complete: **do not** add input-projection GEMMs or attribute this
count to the cache update. Neither includes Softmax, RoPE, masking, bandwidth
or latency. When the graph decomposes Attention into explicit MatMuls, only
those MatMuls are counted (no heuristic double-count from naming).

`expert_linear_bank` accounts for every dispatched token-row projected
through its selected expert matrix: `(T*K)*Din*Dout` MAC. Separate gate,
expert weight storage, routing, all-to-all, packing and communication are
not magically included. Under this bounded teaching graph all selected
rows execute: `Top-k` describes actual mathematical visits, not sparse
GPU kernel occupancy. The `expert_bank_mix` static vector lookup is **not**
a conditional Expert FFN and is not included in this count.

`nd_cache_append` and `cache_write_segment` expose logical elements in
their **resulting immutable snapshot**. The sum of successive snapshots is
not peak memory, cache residency, allocated HBM or network bytes; there is
no inferred dtype and the report never assumes K/V separately unless the
user builds both branches. `nd_repeat_kv` expands a conceptual output,
but may be only a logical view in deployed models; no physical duplication
is asserted. Cache-read slicing is likewise not a measured bandwidth event.

`GraphCost.compare(a,b)` rejects incomplete graphs, calculates both actual
DAG profiles, and exposes `baseline,candidate,difference,ratio` for each
metric. In the workbench, “固定当前图为基线” deep-copies the **current valid
connected graph**; subsequent edits or lesson selections recompute exact
shape-derived deltas. This is not an automatically controlled scientific
ablation: users must ensure semantic, parameter, data and quality comparability.

Run `node tests/graph_cost_evidence_extensions.js` and the Chromium cost
comparison acceptance before using these results as validated mathematical
claims. Still excluded: GPU kernels, actual time and memory, all optimizer
FLOPs, softmax and nonlinear costs, measured routing overhead.
