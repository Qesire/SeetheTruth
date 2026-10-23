# Unified block packs — incremental main promotion, 2026-10-23

Two independent packs were extracted from the divergent draft PR #49, without merging its remaining large changes. The **named-axis pack** implements five rank-2 [T,D] operators; the **semantic-video pack** implements three actual rank-5 B,T,H,W,C source/slice/concat operators. The registry now installs eight operators into the original 138 kinds, for **146 kinds**.

Every pack exposes metadata, parameter validation, shape inference, bounded numeric execution, optional trace, optional named input/output axes, source references and optionally a cloned recipe. The core interpreter calls a **single registry dispatch**. The named-axis recipe demonstrates original native `transpose2d` and `subtract` in the same graph as new source/broadcast/transpose/mean operations, with exact zero difference. Another adversarial graph composes both independently installed packs with native operators.

Three independent gates (conceptual inference, incremental graph readiness, numeric execution) reject a [D,T] → [T,D] semantic mismatch despite equal 3×3 physical dimensions. Original upstream metadata is carried rather than inferred by array shape alone. The existing graph-derived learning path uses the same operator and knowledge contracts; Android packages identical JS modules. NumPy/PyTorch backends remain fail-closed for these new kinds and should not be misreported as supported.

## Run
- `node tests/unified_pack_registry_contracts.js`
- `python tests/browser_block_packs.py` with Playwright Chromium installed.
- Existing static, training, codegen, source and Android workflows are mandatory merge gates.

Next phase: extract compute microscope and paper arithmetic packs by the same incremental method, including source-anchored arithmetic validation and native compiler support only where independently established.
