# Foundation strength assessment — 2026-10-26

## Assertion and boundary

**Decision:** SeetheTruth has a solid *bounded, deterministic teaching-runtime
foundation* for typed graphs, native small-tensor execution, source-linked
observation, guided lessons, and offline teaching assets **for its verified
subset**. It does **not yet** qualify as a fully general, industrially robust,
all-backend model development/runtime substrate.

"Solid" here means the central contract is coherent and adversarially
tested, newly installed trusted blocks do not require a new graph runtime, and
mathematical claims are scoped to executed evidence. It does not mean every
registered brick has independent numerical/gradient/backend proof.

## Live structural seams

| Layer | Source of truth | Independent guard | Judgment |
| --- | --- | --- | --- |
| Graph identity/shape/ports | LearningLab + ExecutionIR | Target slice, invalid-port, shape and trace adversaries | Solid for validated teaching graphs |
| Numeric teaching computation | LearningLab, GraphAutodiff, N-D VJP | Node-only scalar obligations and independent PyTorch CPU/NumPy oracles | Strong on tested graphs; incomplete coverage |
| Operator capability and evidence | Block capabilities + source-pinned certificate registry | Separates registered, eligible, executed and unverified | Sound reporting, limited proof coverage |
| Plugin-like extensibility | Trusted installed pack manifest and generic defs/validate/infer/execute | Synthetic third-pack test and active graph integration | Build-time extension solid; no arbitrary runtime loading |
| State and time | Native domain runners, shared observation/replay/snapshots | Replay, invalid state, causal lineage, checkpoint adversaries | Strong on existing supported engines |
| Delivery | Page dependency contracts, single static publication policy, pack-driven generation | Negative tests, HTTP/Android publication parity, Chromium and APK CI | Now structurally unified; deep dynamic imports still need scrutiny |

The connected graph path is:

```text
installed source bricks / native defs
          -> LearningLab.validateGraph + indexGraph
          -> ExecutionIR (node, port, shape, bounded trace integrity)
         /                                           \
GraphSession / UnifiedCore              GraphLoweringPlan
 source teaching/replay                 /            \
                                   NumPy emitter   PyTorch emitter
                                         \          /
                                 independent scalar/CPU proof
```

The installed pack list is `installed_block_packs.js` (not an arbitrary
network plugin loader). The static publication policy is
`static_asset_policy.json`; its module inventory is augmented from the same
installed manifest, and `scripts/sync_runtime_assets.js --write/--check`
materializes the HTTP route whitelist and Android asset list. The existing
`scripts/sync_installed_block_packs.js` owns only page script installation
and calls this shared publication generator. Both tools are deterministic.

## Important limits, not completeness claims

At the current audited revision the capability inventory has **150 registered
operators**, **122 NumPy eligible** and **116 PyTorch eligible**. Eligibility
counts do not establish runnable arbitrary graphs. The last independently
reexecuted source-pinned CPU certificate covers **6 exact fixtures**, **13
operator kinds**, **33 kind-by-fixture cases** and **75 differentiable
parameter finite-difference coordinates**. That is meaningful verification,
but remains a small fraction of all registered kinds. Evidence remains fixed
graph/shape/Float64 CPU scoped; nothing establishes GPU kernel parity,
mixed-precision fidelity, performance, full pretrained-model compatibility,
or arbitrary shape/state continuity.

The next real architecture risks are: (1) source shape/semantic metadata
drifting across N-D and newer 5D extensions; (2) modules relying on indirect
JavaScript imports that a direct HTML-dependency closure cannot discover;
(3) valid-looking exporter allowlists covering unproven operator
combinations; (4) source-revision provenance not being identical to an actual
downloaded industrial model execution. These are test/research targets, not
reasons to fragment the central abstraction again.

## Required acceptance for publication-policy changes

```bash
node scripts/sync_installed_block_packs.js --check
node scripts/sync_runtime_assets.js --check
node tests/runtime_asset_policy_contracts.js
node tests/android_asset_manifest_contracts.js
node tests/architecture_extension_spine_contracts.js
```

A new trusted pack is registered **once** in the pack manifest, then page
script references and server/Android outputs are regenerated. Other new
static files must be declared once in the reviewed policy as offline or
web-only. A malformed path, duplicate publication, missing import or attempt
to move a desktop-only dependency into Android fails the fast gate. CI also
checks unchanged real Chromium behavior and assembles the APK.

**Next bar for generality:** grow independent graph-level forward/VJP cases
across complete Wan and other multimodal block compositions, subject the
5D/temporal state interface to adversarial metamorphic and backend tests,
and obtain independent round-trip evidence for any additional execution
backend. Continue to distinguish mathematical, code-generation,
actual-model, and hardware performance evidence.
