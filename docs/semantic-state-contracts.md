# Named-axis tensor and immutable stream contracts · 2026-10-21

## Implemented scope

`semantic_state.js` builds upon **existing `tensor_ir.js`**. It supports actual tiny Float64 numerical arrays for ranks 1–5: named-axis permutation (`reorder`), singleton-axis addition (`align`), explicit `broadcast`, element-wise `binary`, axis-targeted `slice` and `concat`.

For tensors `A` and `B` carrying named axes, binary operations require the **exact union of axis identities** and support only dimensions that are equal or one. Merely matching lengths does not permit a silent mapping from `T` to `Tk`. Element order is verified against an independent four-dimensional `nd_tensor.js` oracle. These operations *materialize independent numeric arrays*. They do not promise zero-copy views or GPU kernel semantics.

## Immutable stream formalization

Stream state `S=(tensor,axis,origin,capacity,version,nextPosition)` enforces:

```text
nextPosition = origin + shape(tensor)[axis]
append(S, chunk, position) is defined iff
  position == S.nextPosition
  axes(chunk) == axes(tensor)
  shape(chunk)[other axes] == shape(tensor)[other axes]
  shape(tensor)[axis] + shape(chunk)[axis] <= capacity
append(S,chunk,position) = S' with new materialized tensor,
  version' = version+1 and nextPosition' = nextPosition+len(chunk)
```

`read(S,start,length)` uses absolute logical positions, rejects unpopulated or overwritten ranges, and returns a new tensor with explicit `[start,start+length)`. `fork(S)` is an immutable **copy**: two concurrent continuation branches can diverge without altering the common initial state.

This models the **logical numerical/time semantics** of a cache or recurrent state. It does **not** model physical KV paging, reference counting, rollback, CUDA memory aliasing, GPU execution, throughput, attention cache ownership, or differentiability. Streaming a state does not imply that the numerical operator itself is recurrent; the caller must supply the correct chunk calculation.

## Worked invariants

- Four-dimensional broadcast and permutation checked against the existing independent `nd_tensor.js` numerical reference.
- Rank-five `[B,T,H,W,C]` video tensor sliced and re-concatenated along `T`: every scalar exactly equal to unsplit execution.
- Cached prefix `[B,T,D]` with absolute origin 100: append to 102 creates version 1, read window [101,103), independent forked continuation.
- Invalid axis identities, shape collisions, stale/gapped writes, capacity overflow and out-of-bound reads must fail closed.

No existing saved learning graph JSON schema or original 134-block palette is changed. The new `semantic_state.html` teaching laboratory is an **extra interactive demonstration**, not a falsely counted 60th preset.

## Follow-on integrations

1. Introduce typed semantic input/output ports into `learning_lab.js`, `concept_flow.js`, `graph_readiness.js` with explicit conversion nodes between old plain arrays and new named-axis tensors.
2. Add `time_step`, `state_read`, `state_write`, `fork` as versioned graph operators with shape and temporal semantics; test looped shared-parameter gradients and valid detach boundaries.
3. Make PyTorch and NumPy lowering opt-in only after a full independent numerical and VJP test suite. Do not add these classes to `supportedKinds` prematurely.
4. Handle real cache ref-count, block-table layouts, memory residency and physical ownership *separately* in the scheduler/KV labs.

Cross-reference: [PyTorch broadcasting rules](https://pytorch.org/docs/stable/notes/broadcasting.html), [PyTorch tensor views](https://pytorch.org/docs/stable/tensor_view.html).
