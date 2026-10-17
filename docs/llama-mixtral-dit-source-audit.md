# Real-model static concordance: Llama / Mixtral / DiT

This audit is intentionally **static**. It compares source callsite locations with
SeetheTruth graph nodes and tests the exact *identity* of output slots, not model
quality, actual runtime operator count, runtime tensor sizes or GPU kernels.

## Immutable upstream sources

| Source | Immutable upstream commit | Audited code |
|---|---|---|
| Hugging Face Transformers Llama | `174890280b340b89c5bfa092f6b4fb0e2dc2d7fc` (v4.44.2) | `src/transformers/models/llama/modeling_llama.py` |
| Hugging Face Transformers Mixtral | Same | `src/transformers/models/mixtral/modeling_mixtral.py` |
| Meta DiT | `ed81ce2229091fd4ecc9a223645f95cf379d582b` | `models.py` |

The CI downloads source bytes *only*, imposes a source file size cap, parses with
Python `ast`, and runs no upstream model code. It compares original AST callsite
line numbers directly against graph `node.source.line`, and verifies source file
SHA-256 in graph metadata. Existing GPT-2 pinned source audit stays enabled.

## Tested source facts

1. **Llama GQA:** `LlamaAttention.forward` includes distinct Q/K/V/O linear
   projections; its eager attention path has two `torch.matmul` callsites.
   `repeat_kv` occurs twice and the helper contains an `expand` and a
   `reshape`. This does **not** imply separate copies or kernels; runtime
   strides/memory materialization are not inferred. `apply_rotary_pos_emb`
   produces distinct query and key return slots.
2. **Mixtral MoE:** `MixtralSparseMoeBlock.forward` contains a router linear
   projection, `softmax`, `topk`, expert-index processing, and
   `index_add_` accumulation. Returned top-k weights and indices are not
   interchangeable. `self.experts[expert_idx]` is marked as **dynamic
   ModuleList dispatch**, not expanded by guessing an expert.
   Separately, `MixtralBlockSparseTop2MLP.forward` has three explicit
   Linear modules (`w1`, `w3`, `w2`).
3. **DiT AdaLN-Zero:** `DiTBlock.forward` yields six different parameters
   via `.chunk(6, dim=1)`, not six references to one output. Constructor
   inspection of the explicit `nn.Sequential(nn.SiLU(),nn.Linear(...))`
   yields a **logical constructor-derived** activation and linear stage,
   with provenance separate from source-observed runtime calls.
   The `self.attn` and `self.mlp` classes are external `timm` modules
   in that snapshot: **no claim is made to have opened their internal
   GEMMs**. The local `modulate` helper is source-audited.
4. **Return value tracking:** A tuple-returning inner function exposes
   `Return[0]`, `Return[1]`, etc. These slots map independently to
   caller destructuring projections when the callee is statically resolved.
   Unknown/dynamic calls retain a generic output slot projection instead.

## Defects repaired

- Original AST tuple unpacking assigned every target to the same final
  producer. Slots are now represented separately and independently linked.
- Calls to explicitly constructed `nn.Sequential` exposed only a
  `Sequential` node. Direct constructor arguments with known math rules are
  now represented as logically ordered child nodes. This does not infer
  runtime kernel fusion, configuration-driven replacement or shapes.
- In-place `index_add_` was shown as a call but subsequent tensor reads
  did not use the mutated result; name-level in-place updates now point to
  the mutation node (aliasing outside a simple named receiver remains
  unresolved).
- Dynamic `ModuleList` indexing is now marked unresolved rather than
  silently advertised as expanded expert internals.
- Nested function return tuples can now feed the right caller output
  slots through cross-scope data edges.

## Explicitly **not** proved

- All possible Python dynamic dispatch, aliasing, control-flow joins or
  in-place modifications are statically resolved.
- Training-time and inference-time tensor shapes, masks and layouts can be
  deduced for an arbitrary model from AST alone.
- Every external module (especially `timm.Attention` and `timm.Mlp`)
  is available to this parser: **external libraries must be added as actual
  pinned source inputs**, rather than guessed.
- Counted callsites are runtime FLOPs, GPU kernels, invocation frequencies,
  communication volume or total training costs.
- Sparse dispatch `topk` and per-expert work execute with a fixed number
  of tokens or complete static expert callgraph.

## Reproduce

Offline fixtures, no Torch dependency:

    python -m unittest discover -s tests -p 'test_tuple_sequential_semantics.py' -v

Pinned upstream (network for **source text**, never imports/executes it):

    SEETHTRUTH_AUDIT_UPSTREAM=1 python -m unittest discover -s tests -p 'test_pinned_llama_mixtral_dit.py' -v

These gates run in GitHub Actions with the existing GPT-2, browser, and
CPU tensor-contract checks. A red gate blocks merge.

**Next correctness gate:** canonical inter-module symbol resolution across
pinned dependencies and explicit alias / in-place mutation modeling, before
claiming complete source-to-matrix coverage.
