# Wan2.2 I2V — fully connected single Transformer Block

## Native source and precise claim

Built from `Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff`,
`wan/modules/model.py` (WanAttentionBlock and WanRMSNorm) and
`wan/configs/wan_i2v_A14B.py` (cross_attn_norm enabled). The new
`LearningLab.preset('nd_wan_i2v_full_block')` is an actual **115-node,
editable and executable** graph extending the prior 89-node Self+Cross graph;
no second Wan-specific arithmetic interpreter was introduced.

This fixture uses **synthetic Float64 tiny tensors**: video tokens `T=8`,
feature width `C=6`, FFN hidden `H=8` and text keys `S=4`. The separate
standalone `wan_ffn_trace.js` still explains source-aligned nine FFN stages;
this new lesson binds them to the actual full Self/Cross graph and a final MSE
loss. Source structure is pinned, but tensors are **not** official checkpoint
weights, no GPU timing is measured, and no full diffusion-video generation is
claimed.

## Source-ordered computation graph

```text
X video
  -> norm1(X)*(1+e1)+e0
  -> video Self-Attention QKV -> Q/K learned fullwidth RMSNorm
  -> Wan F/H/W 3D RoPE on video Q/K
  -> Self QK^T / sqrt(d) -> Softmax -> PV -> LinearOut
  -> e2 gated residual = after_self
  -> norm3(after_self) with learned affine gamma, beta
  -> Text Cross-Attention (video queries, text keys/values, no video RoPE)
  -> Cross output projection, residual = after_cross
  -> norm2(after_cross) = N
  -> (1+e4) * N + e3 = A
  -> Linear1: U[t,h] = sum_c A[t,c] W1[c,h] + b1[h]
  -> GELU(tanh): V[t,h] = .5 U (1+tanh(sqrt(2/pi)(U+.044715 U^3)))
  -> Linear2: F[t,c] = sum_h V[t,h] W2[h,c] + b2[c]
  -> after_ffn[t,c] = after_cross[t,c] + e5[t,c] * F[t,c]
  -> MSE(after_ffn, target)
```

The six `e_i` values are explicitly created from the respective per-token
conditions and a shared across-token modulation Parameter for each group.
`e3` is Shift, `e4` is Scale and `e5` is the output Gate. The `1+e4`
operation is represented as `N + N*e4`, preserving the original function.
Existing 2D `gelu_tanh` is reused via exact 4D→2D Flatten and 2D→4D
Expand; no approximate activation substitution is made.

## Executed proof hierarchy

**Fast, native complete graph:** `tests/wan_full_block_contracts.js`
verifies the 89 original nodes are preserved and the final loss is rewired to
`after_ffn`, reconstructs 7 FFN stages via independent scalar loops, checks
the 115-node graph identity and tests whole-loss gradients by central
differences spanning Self `mod0`, Cross `cw_q/cw_v/norm3 gamma`, and FFN
`mod3/mod4/mod5/W1/W2` — **9 actual complete-loss coordinate comparisons**.
This includes the effect of the FFN residual on the upstream Self/Cross
path; a local-FFN-only loss would miss it. A wrong e3/e4 operation order is an
explicit counterexample.

**Heavy, independent FFN closure:** `tests/python_wan_full_block_ffn_oracle.py`
executes NumPy Float64 and a *separate actual PyTorch CPU Float64 program*
from the native graph's `after_cross`, e3/e4/e5, weights/bias and target.
It checks all 8 stages including final loss, and compares every
`after_cross`, e3/e4/e5, W1/W2 and b1/b2 **whole-loss gradient coordinate**
against the native graph. One coordinate from each independently varied
source is checked by additional NumPy finite differences. This proves the
**FFN closure and its upstream boundary VJP**, while the native whole-graph
finite differences verify that derivatives continue to reach Self/Cross.
No claim is made that independently executing *only* this FFN closure
establishes full Torch reexecution of Wan Self/Cross, or real model parity.

**Display:** `tests/browser_wan_full_block.py` opens the new preset through
the real Learning Studio on desktop and phone, confirms 115 rendered nodes,
complete graph readiness, numeric execution, staged tensor shapes and final
loss. The graph automatically uses the same source-informed layout, the
shared per-coordinate microscope, ExecutionIR and VJP tools as other
free-form lessons.

The six historical source-pinned operator certificates were reexecuted
against the modified `learning_lab.js` and regenerated in
`operator_certificates_data.json/js`; the new Wan full-block case is
a **separate** executed case, not falsely appended as an already certified
kind/fixture in the old ledger.

## Next independent research tasks

Expand from the current single-block fixture to *multiple complete*
sequential blocks with an explicit state/conditioning interface. The native
free-DAG teaching budget currently caps each graph at 128 nodes, so joining
two full 115-node blocks in the same editing canvas will require a principled
hierarchical/subgraph representation instead of silently increasing every
resource guard. When extending, verify error accumulation across every block
and multiple denoising steps with a separate source-pinned CPU oracle,
explicit real-model inputs/checkpoint provenance, and finally benchmark
actual GPU kernels only when real hardware measurements exist.
