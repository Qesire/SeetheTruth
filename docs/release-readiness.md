# SeetheTruth · Public Alpha release gates

**Snapshot:** 2026-10-26, main `2db505c2df7e471d16c1f48a26f95c9617eed707`. This document is a current release checklist rather than a historical feature log; dated implementation summaries live in [CHANGELOG.md](../CHANGELOG.md).

## v0.1.0-alpha scope

A local learning-first web app with free typed graph composition, symbolic shape/semantic reasoning, bounded CPU numerical execution, explanatory papers and source references, and actual standalone NumPy/PyTorch CPU export where the source graph is supported. Real-model/profiler studies are optional research routes; Android is a separate offline classroom debug APK.

The scope is a **teaching runtime**. Real Wan checkpoint execution, multi-block end-to-end video generation, GPU performance and arbitrary-block gradient/backend equivalence require their own independent evidence.

## Release gates

| Gate | Current evidence and required completion |
| --- | --- |
| Source/runtime contracts | `LearningLab`, `GraphSession`, `ExecutionIR` and source-derived [capability page](../capabilities.html); rerun mathematical and graph tests on final tag SHA |
| Independent math | Fast source constraints and external NumPy/PyTorch CPU oracle checks from [dual-tier proof](dual-tier-mathematics-verification.md); maintain fixture-scoped conclusions |
| Real browser | Chromium desktop/phone and [Wan complete-block browser test](../tests/browser_wan_full_block.py); independently follow [15-minute first-use guide](first-15-minutes.md) in a clean browser |
| Android | [APK CI](../.github/workflows/android-teaching-apk.yml) builds reviewed offline teaching assets; artifact is a debug build, not app-store distribution |
| Distribution rights | There is currently no repository-root `LICENSE`; copyright owner must select a compatible license after reviewing bundled source/material rights |
| Release publication | No GitHub Release exists; publish an explicitly marked `v0.1.0-alpha` prerelease on a tested commit, with notes, compatibility, known scope and evidence links |

All eight workflow runs on main commit `2db505c` succeeded on 2026-10-26, including static analysis, heavy mathematical validation, Chromium and APK; inspect [Actions](https://github.com/Qesire/SeetheTruth/actions) for exact run evidence. **The next PR/Release requires its own final-head pass.**

## Publication sequence

1. Rights holder approves the project license and checks third-party components.
2. Run `node tests/release_surface_contracts.js`, the independent mathematics workflows, real-browser suite and Android packaging verification on the intended release commit.
3. Follow the [first-use guide](first-15-minutes.md) with no internal project knowledge. Record missing controls, console errors, phone overflow and shape/numerical mismatches.
4. Execute one supported exported program with a real input override and compare its output against the same teaching graph.
5. Publish versioned release notes and a pinned GitHub prerelease, linking current [capabilities](../capabilities.html), [contribution process](../CONTRIBUTING.md) and [changelog](../CHANGELOG.md).

## Next maturity levels

- **Beta:** external-user test tasks, accessible keyboard interaction, error recovery, compatibility tests for `seethetruth-learning-lab-v1` saved graphs and repeatable feedback.
- **Technical scale:** hierarchical multi-block graph composition, cross-state VJP and independent multi-block oracle evidence.
- **Sustained open source:** reproducible third-party extensions, release cadence, ownership/security reporting and contribution review.

Avoid embedding hardcoded operator, lesson or backend-eligible counts in this checklist; derive current inventory from `node scripts/report_capabilities.js --json` and source UI registrations.