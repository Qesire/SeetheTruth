# Dual-tier executable mathematics verification

This repository uses **two independent test gates** for the same real,
validated LearningLab graph. Registration, shape inference, forward execution,
backend export eligibility and independent numerical evidence remain separate
claims. The tests do not introduce a second operator registry or a second
learning/runtime execution path.

## Tier F: source-math constraints (fast)

Run:

```bash
node tests/source_math_invariants.js
node tests/nd_tensor_contracts.js
node tests/nd_vjp_local_contracts.js
node tests/portable_operator_rules_contracts.js
node tests/shared_lowering_plan_contracts.js
node tests/operator_certificate_registry_contracts.js
```

Dependencies: Node 22 only, no Python/PyTorch/NumPy/GPU. CI job:
`source-math-fast` in `.github/workflows/math-verification.yml`.

The new independent scalar reference evaluates the **original validated
4D graph** `x -> biased -> gated -> summed -> loss`. It retrieves the
fixture's original input tensors, checks the actual graph's node kinds and
edge/port lineage, and writes its own coordinate loops for each computed
tensor and whole-loss adjoint. It does not call `nd_tensor` or `nd_autodiff`
to calculate the reference quantities.

For `X\in\mathbb{R}^{B\times H\times T\times C}`,
`\beta,\gamma\in\mathbb{R}^{1\times1\times1\times C}` and
`Y\in\mathbb{R}^{B\times H\times1\times C}`:

```text
U[b,h,t,c] = X[b,h,t,c] + beta[c]
V[b,h,t,c] = U[b,h,t,c] * gamma[c]
S[b,h,0,c] = sum_t V[b,h,t,c]
L = sum_{b,h,c} (S[b,h,0,c] - Y[b,h,0,c])^2 / (B*H*C)
D[b,h,c] = 2*(S[b,h,0,c] - Y[b,h,0,c]) / (B*H*C)
dL/dX[b,h,t,c] = D[b,h,c] * gamma[c]
dL/dbeta[c] = sum_{b,h,t} D[b,h,c] * gamma[c]
dL/dgamma[c] = sum_{b,h,t} D[b,h,c] * U[b,h,t,c]
```

All intermediate derivatives (summed/gated/biased) are checked as well.
The suite also verifies graph-source immutability, rejects incompatible
parameter dimensions and deliberately injects a missing broadcast aggregation
to ensure the assertion actually detects it. This is a **source-math proof on
specific bounded graphs**, not independent hardware execution.

## Tier H: independent cross-backend numerical proof (heavy)

Run in a pinned CPU Python 3.11 environment with NumPy and PyTorch 2.6.0+cpu:

```bash
node tests/cross_backend_nd_oracle.js
node tests/portable_operator_runtime_parity.js
node tests/torch_nd_broadcast_reduction_parity.js
node tests/torch_operator_certificate_ledger.js --check
```

CI job: `cross-backend-independent`. It intentionally has no `needs`
dependency on the fast job; either test gate can fail independently.

The new test `cross_backend_nd_oracle.js` loads the **same exact source graph**
and runs the native JS forward and `GraphAutodiff.backward`. Independently,
it uses the existing `python_codegen.js` and `torch_codegen.js` to generate
standalone Python programs. `python_cross_backend_reference.py` runs each
program in its own namespace and computes a third, scalar-indexed formula.
This reference **does not use** the shared portable lowering emitter,
NumPy broadcasting, or Torch autograd to derive expected outputs.

For each fixed fixture it checks:

| Subject | Forward | Whole-loss VJP | Provenance |
| --- | --- | --- | --- |
| Original LearningLab/GraphAutodiff | Every named node | Every named differentiable node | Actual source graph and reverse pass |
| Emitted NumPy | Every named node | Not claimed | Standalone generated `run_graph` |
| Emitted PyTorch CPU Float64 | Every named node | Every named differentiable node | Standalone `GraphRuntime.forward` / autograd |
| Independent Python scalar equations | Reference | Reference | Separate loops, no exporter arithmetic |

The comparator checks **full nested shapes**, finite values and every tensor
coordinate; tolerance `atol=3e-9, rtol=1e-10`. A corrupt NumPy result is
intentionally injected after a successful run to prove the numerical judge
rejects a backend error. The output includes per-backend coordinate counts,
maximum absolute error and a digest of relevant source files. It does not
write or self-certify `operator_certificates_data.*`.

The pre-existing heavy tests extend coverage to the complete portable
operator inventory, two independent 4D finite-difference/aggregation
checks, and strict source-pinned certificate regeneration using PyTorch.
The final-head ledger's own `--check` requirement remains unchanged.

## Independence and evidence boundaries

- **A common lowering rule is not an independent oracle**: NumPy and PyTorch
  may agree while both are wrong. The third explicit scalar formula
  distinguishes that failure.
- **A static mathematical invariant is not execution parity**. Passing F
  does not imply H; passing H on fixed fixture shapes does not imply arbitrary
  tensor sizes, dtypes, graphs, model-source parity, GPU kernels or performance.
- **A compiled program is not a proof until executed**. H executes emitted
  Python and compares actual outputs, not merely source-code substrings.
- **Gradient claims are scoped**. NumPy is checked only for forward values;
  PyTorch provides autograd, while the independent scalar oracle fixes
  broadcast sums over B/H/T. Loss-only comparisons are insufficient.
- **Nondifferentiable or mixed-precision cases require declared semantics**:
  record kink/subgradient conventions, parameter coordinate exclusions,
  dtypes and tolerances before adding them to certificates.
- **A source change invalidates proof**. Include new mathematical source files
  in the relevant test's source digest; the existing checked-in operator
  certificate ledger already rejects changes to its pinned source files.
  The new heavy job emits an execution digest, not a signed universal badge.

## Adding a new operator/graph

1. Reconstruct the *actual* input ports, shapes, axes, numerical operation,
   conditioning/state semantics, whole-loss dependency and non-smooth points
   from source. Add a real validated fixture; never infer from a standard
   Transformer diagram.
2. In the fast suite add independent scalar/indexed forward and derivative
   constraints and at least one targeted failing mutation (wrong axis,
   dropped broadcast contributor, transposed/invalid shape, missing loss path).
3. In the heavy suite add the exact fixture to the standalone NumPy/Torch
   execution and a separate mathematical reference, testing every
   intermediate coordinate and all declared differentiable coordinates.
   Include finite differences for trainable coordinates when smooth.
4. Run both gates, then generate **fixture-scoped** source-pinned certificates
   if the operator is to be displayed as independently proven. Preserve the
   capability matrix's distinction between *eligible* and *executed*.
5. Review fail-closed unsupported backends and PC/Android teaching
   surfaces only when their runtime contract or display actually changes.

The two new graph cases are deliberately bounded examples:
`nd_token_reduce_head_broadcast` and
`nd_batch_reduce_channel_broadcast`. Extend to arbitrary-axis reductions,
non-contiguous tensor semantics, complex activation compositions and the full
Wan block through new **independently specified** fixtures, not by
extrapolating this two-fixture result.
