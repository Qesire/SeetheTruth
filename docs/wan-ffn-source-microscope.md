# Wan2.2 I2V A14B · Pinned FFN source reconstruction

## Reference and scope

Official upstream: `Wan-Video/Wan2.2`, commit `1ea34ff48f87168174e12956e200b1d908b1c5ff`, `wan/modules/model.py` (WanAttentionBlock), `wan/configs/wan_i2v_A14B.py` (A14B configuration). Verified against the fixed upstream files directly.

At that revision: **dim 5120**, **ffn_dim 13824**, **num_layers 40**, **heads 40**, **eps 1e-6**. The block's `norm2` is `WanLayerNorm` with no learned affine parameters, **not RMSNorm**. The FFN is `nn.Linear(dim, ffn_dim) → nn.GELU(approximate='tanh') → nn.Linear(ffn_dim, dim)`, not SwiGLU. The conditioning `e` has six chunks: `e[3]` shift, `e[4]` scale, `e[5]` post-FFN gate; FFN runs after the cross-attention residual update.

## Math reconstruction

For a single token and the small Float64 **synthetic** demo tensors:

1. `e = (self.modulation.unsqueeze(0) + e0).chunk(6, dim=2)`
2. `x = x_after_self + externally_supplied_cross_attention_delta`
3. `z = LayerNorm_C(x; eps=1e-6)`
4. `a = z * (1 + e4) + e3` (teaching scale and shift counterfactual multipliers available)
5. `u_h = b1_h + Σ_c W1[h,c] a_c`
6. `v_h = 0.5 u_h (1+tanh(√(2/π)(u_h + 0.044715u_h³)))`
7. `d_c = b2_c + Σ_h W2[c,h] v_h`
8. `delta_c = d_c * e5_c` (a separate teacher-only multiplier can set the gate to zero)
9. `out_c = x_c + delta_c`

The source page `wan_ffn_lab.html` reuses **exactly** `compute_microscope.js` and `compute_microscope_ui.js` from the general editor, not a new bespoke operator inspector. Each stage includes a pinned source link, actual bounded numerical tensor and explanatory formula. Two GEMMs are independently reconstructed per coordinate. A separate nine-stage oracle inside the common microscope recalculates all intermediate outputs and rejects inconsistent input modulation, cross output, norm, affine transform, GELU(tanh), down projection, output gate or residual. The synthetic weights and upstream cross-attention delta are explicitly **not** the official model's checkpoint data or a GPU profile.

## Experimental implications

Deleting one `v_h` coordinate before `W2` changes all output channels by `W2[c,h] * v_h * e5[c]`. Compare the exact affected output vector and distinguish sparsification effects from e5 gating. A zero e5 multiplier makes the model slice return its incoming cross-attention residual exactly. This provides a source-correct teaching counterpart for dynamic Wan FFN sparsity research, **not** experimental proof that sparsification preserves real video quality.

## Gates

- `node tests/wan_ffn_microscope_contracts.js`: official source identifier/config and mathematical oracle, source-adapter consistency, forged evidence and corrupted-stage rejection.
- `python tests/browser_wan_ffn.py`: real desktop/mobile Chromium, GEMM coordinate inspection, gate interventions, source links and shared microscope.
- Existing universal microscope, full browser CI and offline Android asset acceptance must all pass; backend export and VJP coverage remain separate from source reconstruction.
