# Real small-model sampling contract

Training demonstrations should be grounded in **real small-model execution**, not arbitrary seeded matrices. Synthetic examples remain available only if visibly tagged `synthetic_teaching_example`.

## Source tiers

1. **REAL_CAPTURE** — tokenized public prompt; pinned model repo and immutable revision; actual forward/backward; module path, tensor statistics and parameter gradients captured.
2. **REAL_DERIVED** — numerically computed transformations of a REAL_CAPTURE, with explicit source pointer and derivation. Truncation invalidates claims of full-layer equivalence.
3. **SYNTHETIC** — controlled mathematical counterexamples for branch convergence, gradient checking, MoE capacity edges or PPO clipping. Never label these real-model observations.
4. **SYMBOLIC** — shape/operation contracts only, without empirical values.

Run `python scripts/capture_model_trace.py --model roneneldan/TinyStories-1M --revision <40-hex-commit> --out samples/tinystories-1m.json` in a CPU environment with PyTorch, Transformers and the checkpoint available. This command is **not evidence** that a capture has already run. Do not commit prompt text itself if private; the exporter retains a SHA-256 hash and input token IDs.

## Trace versus teaching execution

Use actual sample values and descriptive statistics for evidence panels; use independently constructed **small matrices** for reversible exercises unless the necessary **complete operands** have also been captured. Slicing a true weight matrix to K columns generally does not reproduce the original full-model activation: omitted contractions contribute. Track all captured original shapes separately from flattened numeric sample arrays.

For MoE samples, capture per-token router logits, expert identities, capacity/drop statistics, gate probabilities, selected expert input/output and expert-weight gradients; baseline decoder-only TinyStories cannot be presented as MoE evidence. For RL samples, capture actual policy logits, action IDs, old-policy logprobs, reward/value/advantage and clipping statistics from a small policy training rollout; a standard next-token LM forward is not an RL trajectory.

## Acceptance

- Source commit SHA and exact module path must be present.
- Repeated CPU runs under the same controlled configuration should satisfy specified numerical tolerance.
- Assert recorded activation and weight statistics against independently computed values.
- Compare backwards gradient against finite differences for selected tiny local operations.
- Screenshots should expose source tier and provenance beside values.
- Reject provenance ambiguity and do not fill missing gradients with zeros.

**Current verified CI state (2026-10-19):** actual pretrained `roneneldan/TinyStories-1M` model CPU Forward + Backward was executed **twice independently**, both at full immutable SHA `77f1b168e219585646439073245fe87e56b3023e`. Captured module: `transformer.h.0.attn.attention.k_proj`, weight shape `[64,64]`, causal LM loss `2.174146890640259`, 8 token IDs. The independent replay test found **maximum numerical difference 0** among checked statistics/samples. See [CI replay and actual evidence artifacts](https://github.com/Qesire/SeetheTruth/actions/runs/39217748548). This is a reproducible real *small-model local module trace*, not a complete Transformer activation tensor or an independent security audit of upstream checkpoint loading.

`real_model_lab.html` now imports captured JSON and displays provenance, parameter weight/gradient statistics and bounded flattened input/output samples with honest evidence status. Real Chromium rendering was exercised with the **actual** captured trace, distinct from marked synthetic structural-validation tests; both PNGs and numerical artifact are archived on the same workflow run.

The immutable trace is stored as a CI artifact rather than disguised as constant demo data. Future capture jobs resolve the upstream commit and include it inside exported provenance; repeated SHA/samples are checked by `tests/verify_real_model_replay.py`. Model ownership and external download remain opt-in, never required to run the Scratch learning app.
