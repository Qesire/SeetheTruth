# Contributing to SeetheTruth

We welcome reproducible improvements to the learning runtime, numerical correctness, model-source explanations, accessibility and teaching content.

## How to contribute

1. Begin with an observable learning goal or a reproducible bug. Include the relevant page, preset/operator ID, input, browser/OS and expected versus actual behavior.
2. Trace the existing source computation before changing mathematical claims. Specify port types, shapes, axis semantics, state transitions, relevant paper/source revision and the smallest counterexample.
3. Reuse the canonical `LearningLab / GraphSession / ExecutionIR` graph and shared `UnifiedCore` observation contracts. New trusted operator packs follow [the installation protocol](docs/trusted-block-pack-installation-v1.md); they are reviewed at build time, not loaded as arbitrary network plugins.
4. Add a fast source-math test and an independent CPU/NumPy/PyTorch oracle where numerical/gradient equivalence is claimed. Make registered, export-eligible, fixture-tested and real-model-backed capabilities explicit.
5. For visual changes, test desktop and phone behavior. For Android asset changes run the shared static publication-policy checks. In a PR include exact tests run and their results; label pending checks as pending.

## Useful local checks

Node.js 22 and Python 3.11 are the tested CI entrypoints. Depending on the changed domain, run:

~~~bash
node tests/release_surface_contracts.js
node tests/source_math_invariants.js
node tests/architecture_extension_spine_contracts.js
node tests/runtime_asset_policy_contracts.js
node scripts/sync_runtime_assets.js --check
python -m unittest discover -s tests -p 'test_*.py'
~~~

The full validation suite, independent numerical programs, Chromium and Android checks run through [GitHub Actions](https://github.com/Qesire/SeetheTruth/actions). See [AGENTS.md](AGENTS.md) for detailed existing extension boundaries.

## Bug reports and attribution

A strong bug report contains steps, minimal graph/sample input, numerical expectation, observed output, browser/runtime details, and where relevant a paper or code revision. If the report involves a sensitive vulnerability, avoid posting exploit details publicly; first check for a private reporting option under GitHub Security.

Keep credit, source citations and upstream licensing intact. The repository's own open-source license remains pending copyright-holder approval; a publicly readable repository alone does not grant reuse rights.