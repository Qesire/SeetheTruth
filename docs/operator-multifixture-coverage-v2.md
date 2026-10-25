# Independent operator-case coverage v2 — 2026-10-25

## Purpose

Turn the source-pinned independent PyTorch CPU certificate into an **extensible matrix of discriminating graph fixtures**, rather than accumulating a misleading count of globally certified operator kinds. A kind may have multiple independently executed source graphs. Every proof is attached to `(fixtureId, kind, nodeId, concrete shape, sourceDigest)`, not a universal rule for all instances of that kind.

## Computation/evidence flow

```text
tests/fixtures/{coordinate_training_graph,operator_coverage_graphs}.js
                       │
                 LearningLab.validateGraph
                       │
         ┌─────────────┴─────────────────┐
         │                               │
   JS LearningLab.evaluate      Torch.compile(same original graph)
   JS GraphAutodiff.backward     standalone Python PyTorch 2.6.0 CPU
         │                        native torch.autograd
         └─────────────┬─────────────────┘
                       │
          Compare each Forward and Loss-VJP coordinate
          + centered finite differences only where smooth
          + declared PyTorch subgradient at an exact kink
                       │
              tests/torch_operator_certificate_ledger.js
                       │
           source-bound v2 JSON / offline JS mirror
                       │
      operator_certificate_registry.js: per-fixture validation
                       │
      block_capabilities.matrix().rows[].certifiedCases[]
                       │
     Capabilities browser: display every tested fixture separately
```

## Specific independent cases

| Fixture | Shape and mechanism | Why this test discriminates |
| --- | --- | --- |
| `coordinate_training_graph` | Two 2×2 trainable matrices, frozen target, nonlinear square and two routes across a shared intermediate output | Original baseline; tests the full-loss adjoint and multiple paths through the same intermediate tensor |
| `rectangular_relu_fanout` | 2×3 × 3×2 matrix multiplication; mixed-sign ReLU and a second scaled bypass route; 2×2 target | Catches transpose/shape handling and wrong gradient on ReLU's negative branch; ensures a zero ReLU derivative does not hide the independent bypass |
| `shared_weight_two_matmuls` | Two distinct 2×3 inputs multiply a **single** 3×2 parameter; one branch scaled negative, then combined and squared | Forces reverse-mode accumulation into the same trainable parameter from multiple consumers; catches overwritten instead of summed gradients |
| `relu_exact_zero` | A parameter with a coordinate exactly at zero, ReLU, and a nonzero frozen target | Checks actual source and PyTorch `ReLU'(0)=0` convention. The two-sided derivative at the kink does not exist; finite difference for that coordinate is **explicitly excluded**, rather than recorded as a gradient proof |

All graphs retain the actual `LearningLab` port connections and source `GraphAutodiff` semantics. No substitute mathematical model runs in place of either execution engine. The independent Python script must execute `torch.autograd` on the actual emitted graph and retain the same intermediate gradients.

## Certificate schema, coverage, and meaning

Version 2 contains an ordered `fixtures[]` inventory with each source path, graph digest, target, mechanism tags, node count, counted parameter finite differences and explicit kink exclusions. It holds `cases[]` (one record per **fixture and kind**) and `nodes[]` (one record per **fixture and node**). A shared node name across different graphs cannot be mistaken for a shared execution identity. The registry validates coverage, shape × coordinate counts, per-fixture node counts, unique references and absence of orphan measurements.

The capability matrix keeps its general `not_attested` state and publishes *only fixture-scoped* `certifiedCases[]`. The UI enumerates all genuine examples for a kind, showing fixture identity, actual tested Forward and reverse coordinates, source digest and an explicit non-generalization boundary.

For numerical stability across runners the ledger records **validated error upper-bound buckets**; the producer independently checks the *actual floating-point error of every element* against the strict tolerance **before** emitting any certificate. The upper bound cannot be used to hide a failure, and a new certificate is rejected if sources or checked-in JSON/browser mirror drift. SHA-256 protects against accidental source changes, not certificate author impersonation.

## Acceptance and extension rules

1. `node tests/torch_operator_certificate_ledger.js --check`: independently re-executes each fixture on pinned PyTorch CPU, checks Forward and Loss adjoints by coordinate, checks differentiable parameter entries with independent central differences, confirms explicit zero-subgradient semantics at ReLU's kink and compares to the checked-in JSON and browser payload.
2. `node tests/operator_certificate_registry_contracts.js`: checks source SHA digests, multi-case references, non-duplicated node identities, shape/coordinate accounting, forged count/fixture rejection and that a successful example never promotes global VJP or backend eligibility to universal proof.
3. `python tests/browser_unified_specialists.py`: desktop and mobile verify the capability explorer filters to executed kinds, shows the distinct rectangular and shared-weight cases for matmul, and keeps unattested operations unverified.
4. Existing complete Chromium, CPU numerical, PyTorch training and Android offline package acceptance must all pass on the **final** PR commit.

To introduce a new case, define a bounded, real graph with a testable discriminating failure mode in the fixture file; add the fixture to the generator's declarative `specs`; explicitly identify nondifferentiable coordinates that cannot support central differences. Execute the independent backend and commit regenerated versioned artifacts. Do not label a kink coordinate `finite_difference_verified`, and do not use an unsupported intermediate derivative as numeric zero.

**Next technical expansion:** broader N-D broadcasting and reduction fixtures once their exact actual `GraphAutodiff` and generated PyTorch lowering share a verified axis contract; higher-dimensional tests should derive full tensor indices, not reuse the v2 rank-two finite-difference loop.
