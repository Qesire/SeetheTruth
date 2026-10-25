# Trusted installed block packs and explicit operator capabilities (v1)

**Goal.** A new research paper, model family or engineering primitive should extend the same executable graph and teaching system without creating a new graph engine, unsupported model claims, or a second Android entrypoint.

## One trusted build-time manifest

`installed_block_packs.js` is the ordered authoritative set of **approved first-party/reviewed packs**. Each entry declares:

```js
{id: "named_axis", module: "axis_block_pack.js",
 global: "SeetheTruthAxisBlockPack"}
```

The `module` name must follow `*_block_pack.js`; module names cannot contain slashes, URLs or traversal. Each referenced pack must export its declared `name`, `defs`, `validate`, `infer` and `execute`. Additional `trace` and lesson `recipes` hooks are optional. The existing `block_extension_registry.js` obtains its installed pack modules exclusively through `SeetheTruthInstalledBlockPacks.load()`, and the existing `LearningLab` continues using that registry for symbolic shape, actual bounded numerical execution and trace substeps.

**To install a trusted pack:**

1. Implement `some_family_block_pack.js` with a unique ID and complete metadata and source documentation. Use `block_extension_registry.createRegistry([pack])` for isolated numerical/shape/invalid-case tests before touching the installed set.
2. Add exactly one ordered entry to `installed_block_packs.js`. Confirm the source module exports the same ID and unique block-kind names.
3. Run `node scripts/sync_installed_block_packs.js --write`. This deterministically synchronizes script load order in `learn.html` and `capabilities.html`, the static server map `atlas_server.py`, and `android/teaching-assets.txt`. Run `--check` to confirm there is no drift.
4. Add a valid arbitrarily composed LearningLab graph using the new kind and prove its infer/forward/trace value and at least one decisive negative case. Check independent NumPy/PyTorch/codegen/gradient support **only where actually implemented**. CI checks browser execution and Android package consistency on the final commit.

This is a **build-time declared installation**. It neither executes untrusted external downloads nor installs arbitrary third-party code at runtime. All approved JavaScript modules still require source review. New pack files may have dependencies of their own; their prerequisite browser scripts must be installed ahead of their pack script and explicitly included in the Android asset manifest until the loader contract supports dependency metadata.

## What the generator guarantees

The `--write` operation is deterministic and idempotent: on the trusted pack entries, the same output files should remain unchanged after a second execution. The `--check` operation exits nonzero if any build-time installation source differs. The CI contract invokes `--check`, and separately probes the real Node loader and the generic pack registry using an otherwise uninstalled third-pack fixture. Browser and Android tests verify the actual installed entrypoints.

`block_extension_registry.createRegistry` continues to expose a testable isolated registry. Creating one does **not** silently install its kinds into the active LearningLab runtime; only the approved build-time manifest does.

## Operator capability matrix

`SeetheTruthCapabilities.audit()` keeps its existing backwards-compatible v1 eligibility report. New `SeetheTruthCapabilities.matrix()` and `describe(kind)` expose explicit **availability** and **verification** independently for every registered kind, with schema `seethetruth-operator-capability-matrix-v1`.

| Capability | Meaning of state | Does *not* mean |
| --- | --- | --- |
| `registered` | Operator metadata and source contract exist | Complete end-to-end runtime success |
| `staticShape` | Requires a well-typed graph | Every possible shape combination was tested |
| `boundedForward` | Available through a valid bounded graph case | Real model, GPU or production-kernel parity |
| `coordinateTrace` | Requires an actual completed numerical trace | All coordinate math is materialized at registration |
| `localVjp` | Requires an independent graph-level gradient oracle | Universal gradients for all 150 operators |
| `numpyExport`, `torchExport` | `eligible` or `unsupported`, derived from the real code generator allowlists | A code export that has passed an independent numerical test for arbitrary parameters |
| `modelParity`, `gpuKernel` | `not_claimed` | An implicit real model or hardware guarantee |

All *static* claims bear `not_executed` or `not_attested` where appropriate. Independent dynamic proofs belong to runtime and oracle tests and must not be inferred from the static registration alone. `tests/operator_capability_matrix_contracts.js` checks every operator against the existing exporter allowlists, forbids invented universal proof, and preserves negative outcomes for unsupported kinds.

## Extension invariants

The identity, execution and native microscope hooks of specialist domains remain owned by `learning_experience_core.js` `domainRunners`. Graph blocks remain owned by `LearningLab` and installed packs; there is no migration to a second runtime. `Core.describeActions(id)` exposes actual domain-operation binding and declared trace kind. Portable snapshots continue replaying original engines; no independent arithmetic is introduced by the installation manifest.

**Required regression:** `node tests/installed_pack_manifest_contracts.js`, `node tests/operator_capability_matrix_contracts.js`, `node tests/architecture_extension_spine_contracts.js`, and full Chromium/Android CI. A large registerable operator count alone is not a sufficient foundation acceptance criterion.

## Remaining work

- Add a trusted, typed **pack dependency declaration** so that dependent browser JS/Android resources can also be generated, not merely the pack's own entry script.
- Expand reproducible per-kind backend and local VJP *verified execution* certificates from actual independent oracles; do not set `verified` using an allowlist.
- Connect real graph-training node/parameter/VJP identities to the unified state checkpoint and observation cursor, completing bidirectional graph/training inspection.
