# Source-evidence operator drill-down (phase 1)

This feature is for **real, explicitly scanned Python ASTs**. It does not replace
the separate model teaching templates or the runtime trace.

## Local import binding

The AST index records local module paths and explicit absolute and relative
`from module import Class as Alias` / `from .module import Class` statements.
A bounded chain of explicit reexports can be followed, without executing imports. For a constructor
`self.attn = Alias(...)`, its `forward` implementation is traversed
**only when the exact imported module and class were included in the input
source directory**. The graph node exposes `resolved_scope` and
`resolved_source_file`, and retains file SHA-256 fingerprints. Classes
with identical names in other local modules are not substituted.

Star imports, dynamic reexports, arbitrary Python import hooks, closures, factory
functions, dynamic dispatch, `ModuleList` indices, alias mutations and
`torch.compile` graphs are **not generally resolved**. Missing external
modules remain explicitly unresolved. No weights or inspected Python modules
are imported or executed.

## Evidence views

The five-level toolbar becomes available for imported `static-python-ast`
graphs. Its **macro/block** options display immediate nodes within the
selected source scope, while **operator/matrix/tensor** show a capped, flattened
AST descendant projection. Projections retain original graph node IDs,
source locators and data edges; direct producers can appear as clearly marked
`CONTEXT` nodes.

- **Matrix**: only actual AST `@` or explicit known linear/matmul/bmm/
  addmm/einsum/baddbmm callsites. Unknown `Conv1D`/custom functions are
  not arbitrarily labeled GEMMs.
- **Tensor**: AST-declared reshape/transpose/broadcast, indexing, activation,
  elementwise arithmetic and matrix operations. No runtime shapes are inferred.
- **Operator**: AST-declared operations and functions, including unresolved
  calls, but not a claim that every path executes.
- Source view projections have an explicit 190-node presentation cap,
  separate from parser graph truncation; total matches and hidden matches
  are shown without modifying the original exported graph.

Evidence source is `static-python-ast`; examples of existing validated
views are GQA, DiT AdaLN, and the Mixtral MoE branch. A zero-match matrix
projection does **not** imply the program has no matrix operations; it may
mean dependencies were not loaded, omitted by the parser, or hidden behind
custom operations. Consult `meta.unresolved_calls_and_limits`,
`meta.truncated`, `meta.shape_verified`, and `meta.runtime_validated`.

## Regression gates

`test_source_import_resolution.py` creates multiple local packages,
including a deliberately conflicting class name, and compares exact file,
callsite, nested node, input binding and unresolved-dependency behavior.
`source_evidence_contracts.js` verifies category boundaries, nested
callsite reachability, no fabricated shapes and capacity limits.
`browser_acceptance.py` exercises real AST import and switching among
matrix and tensor views in Chromium.

This is the first stage of genuine source-oriented drill-down, **not**
complete symbolic tensor shape inference, external-library closure, or
same-run source-to-runtime attribution. Those require subsequent separately
tested milestones.
