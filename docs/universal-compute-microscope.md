# Universal computation microscope — main-compatible extraction

## Scope

This patch extracts the **core** from draft PR #49 into the current production free-form graph. The microscope is mounted in `learn.html`, with no new isolated execution page. It reuses the exact existing `Learning.evaluate` numerical result or `ConceptFlow.evaluate` static result. The inspector displays the same source graph's nodes in topological order.

### Contracts

- All numeric outputs are **bounded rank 1–5** tensors with explicit row-major shape and finite values. Static runs contain shapes but **no numeric values**.
- Reconstructs original `linear` and `matmul` with actual input values and deterministic parameter formula; original `attention` exposes the actual QK scaled scores, stable softmax and PV result, including GEMM dot-sum explanations.
- Registered `pack.trace` observations are carried through a shared nested substep contract. `axis_named_transpose`, `semantic_video_slice` and `semantic_video_concat` expose exact input flat-coordinate mappings that must reproduce their displayed output. Each output coordinate is selectable on desktop and mobile, including rank-5 leading axis selectors.
- Structural dependencies may refer only to already-computed upstream nodes. Revisions that alter a GEMM output, index mapping, source indices, recorded shape, tier, or dependencies fail validation. Provenance may not be escalated to `REAL_MODEL_VERIFIED`.
- Other operators still expose the real input/output tensors, declared formulas and dependency chain. They do **not** claim a valid GEMM reconstruction, automatic VJP, or an unimplemented operator-specific microscope.
- Supported explanation tiers: symbolic/no-execution and bounded synthetic numerical execution. Existing `Wan` adapter in core is dormant: full pinned-source teaching UI promotion requires a separate source verification. This PR does not claim actual checkpoint activations or CUDA timing.
- Only a read-only UI component is added; no GPU runtime or unsafe user-authored executable script imports.

## Acceptance and coverage

`node tests/universal_microscope_contracts.js`: native FFN, attention, residual, SwiGLU, named-axis transpose, 5D video slicing and rejoin; dot and exact index reconstruction; rejected tampered coordinates, untrusted source tier, non-causal dependency, and wrong numerical evidence.

`python tests/browser_universal_microscope.py`: Chromium pointer navigation, real arithmetic breakdown, attention substeps, 5D axis selection, static fallback and mobile named-axis lessons. Desktop server and offline Android load the same JS/CSS.

All existing graph, source, training, codegen and Android CI remain required. The wider issue #61 remains open for complete per-kind numerical microscope and independent Wan source-adapter promotion.

## 2026-10-23 · Typed coordinates and native arithmetic upgrade

Independent follow-up: registered pack `outputAxes` and `portAxes` now propagate as authenticated mathematical metadata into rank-2 to rank-5 tensor previews, output-coordinate readouts and source-index trace labels. For example, a time-concatenated video coordinate is displayed as `[B=0,T=2,H=1,W=0,C=1]`, with the corresponding **source** `T=0` inside the second slice rather than assuming the same index. Unregistered generic outputs remain explicitly positional.

Original native `add` and `mul` now have validated per-coordinate arithmetic `y[i]=a[i]±b[i]` / `y[i]=a[i]×b[i]` for the actual executor's two-port same-shape semantics (the add contract uses addition, not subtraction). These are standalone pointwise proofs, never incorrectly presented as GEMM or source index copies. Altered outputs or claimed math operations fail the proof validation. Pack-provided final substep inputs must match the parent graph's shape, numeric values and named axes, and their output must match the parent output.

The additional native proof and axis annotation do not imply unsupported operator gradients, backend export or real-checkpoint data. The execution is bounded Float64 teaching numerics. See `tests/microscope_named_axes_pointwise_contracts.js` and the real Chromium acceptance suite.

## 2026-10-23 · Native nonlinear and row-normalization proofs

This incremental extension gives the **same** numerical microscope full verification contracts for:

- `RMSNorm`: per-token `mean_k x[k]^2`, the configured positive `epsilon`, `1/sqrt(mean+epsilon)` and every output coordinate `x[c]*inverse`. It is not LayerNorm.
- `SiLU`: exact `x/(1+exp(-x))` at each coordinate, with a visible denominator. This is not the GELU(tanh) activation in the pinned Wan I2V adapter.
- `Softmax`: stable row-wise `max(x)`, `exp(x[k]-max)`, their sum and normalized per-coordinate output. The existing Attention module's **actual** intermediate score matrix is independently checked using this same rowwise proof.
- Distinguish these proofs from per-coordinate GEMM dot summands and copy/index maps. Each kind retains an explicit operator identity and input port, rather than accepting unverified claims attached by arbitrary metadata.

Corrupting a numerical output, changing the operator name, swapping an input port, changing epsilon or editing an internal Attention probability fails a real Node proof. Static mode still only provides shapes, not invented numeric proofs.

This is forward arithmetic reconstruction, **not** proof of a VJP, autograd graph, real model activations or GPU timing.

## 2026-10-24 · Portable native coordinate proofs

The common arithmetic microscope now independently reconstructs bounded forward results of **Subtract/Divide**; **Scale/Bias/Square/ReLU/ReLU²/Sigmoid/Tanh/GELU(tanh)/Softplus**; and **Transpose2D/SliceRows/SliceCols/ConcatCols/RepeatRows**. These are three shared binary/unary/indexed proof families, not isolated educational pages. Native indexed proofs regenerate every source port and flat offset from operator-specific shape/params. Wrong mappings are rejected even when the source has duplicate values. The UI displays these individual operations without misidentifying them as GEMM terms.

An adversarial mixed-graph regression, `node tests/native_portable_coordinate_proofs.js`, verifies numerical closure, falsified parameters/indices/operator identity and concept-only separation. This is synthetic bounded Float64 forward evidence, **not** gradient/VJP validation, codegen eligibility, real checkpoint capture or CUDA performance. Existing saved graphs and specialist adapters are unchanged.

## 2026-10-24 · One selected coordinate, authenticated forward proof and seeded VJP

`coordinate_vjp_bridge.js` is a thin cross-platform adapter over the existing
`compute_microscope.js`, `element_lineage.js`, `learning_lab.js`, and source-driven
`block_capabilities.js`. It authenticates the **whole graph's** microscope frame
against its actual forward execution, then verifies the exact selected coordinate
matches the seed of the scalar-lineage engine. On demand the same microscope shows
forward point terms, first-hop partial derivatives, accumulated source sensitivities,
boundary status, and independently classified per-operator NumPy/PyTorch eligibility.
It never claims graph-level backend lowering or parameter-gradient eligibility.

The Lineage core now supports native fixed-weight `linear` as
`∂Y[t,h]/∂X[t,k]=W[h,k]`, using the deterministic teaching weight identity. It
**does not claim trainable W**: learnable weight VJPs must travel through an
explicit Parameter feeding MatMul. The shared-path source derivative is checked by
independent central differences of the original small forward interpreter;
stale trace, invalid coordinate, reordered graph, forged microscope and structural
mode are refused. All numeric evidence remains bounded synthetic Float64.

## Full loss adjoint · selected coordinate / full-graph gradient

A valid loss must be a scalar MSE/MAE/CE-derived objective approved by
`training_objective.js` (with optional explicit `lossId` for compound objectives).
The shared coordinate bridge now calls the existing `graph_autodiff.js` on the
**same authenticated numerical forward run**. It keeps the following distinct:

- `dY[i]/dX[j]`: a **unit-seeded** scalar dependency, from ElementLineage
- `dL/dY[i]`: the selected coordinate of the global reverse-pass adjoint
- `(dL/dY[i])·(dY[i]/dX[j])`: contribution **through this one output**, only
  when its complete scalar path is supported and no relevant stop-gradient is crossed
- `dL/dX[j]`: independent whole-graph backward tensor, including all other
  downstream output coordinates and graph branches

For a loss itself, the scalar microscope may show an explicit **boundary**
even though the dedicated training objective VJP is supported; this is
intentional: fixed targets must not be assigned imaginary gradients by
implicitly treating MSE as an ordinary mathematical scalar function.

Absent objectives expose `NO_LOSS_OBJECTIVE`. Unsupported reachable
operators expose `UNAVAILABLE_GRAPH_LOSS_VJP` and their specific failure,
without fabricating zero; cut edges remain visible. A source with zero
whole-loss gradient can still have a nonzero seeded derivative of another
output (or vice versa). The source and output views do not imply full
backend-generated VJP support or GPU execution.

**Independent oracle:** `tests/coordinate_loss_adjoint_contracts.js`
recomputes actual bounded graph forwards under per-element central differences,
verifies `dL/dX` for both shared MatMul sources, and checks the exact chain-rule
conservation by summing all 2×2 output-coordinate contributions across shared
fan-out routes. Compound weighted loss selection, explicit fixed target,
Detach and unsupported reachable division are negative/semantic cases. Real
Chromium checks desktop and mobile teaching screenshots of the full loss view.

## Exact composite scalar Jacobians (2026-10-24)

The **same free-graph** `element_lineage.js` now exposes exact coordinate
dependencies and bounded internal mathematical stages for these implemented
2-D teaching primitives, available without a separate specialist engine:

- **Attention:** for `Y[t,c]=Σ_j P[t,j]V[j,c]`,
  `P=softmax(QKᵀ/√D)`. A selected `Y[t,c]` includes
  `∂Y/∂V[j,c]=P_j`,
  `∂Y/∂Q[t,d]=Σ_j P_j(V[j,c]−Y[t,c])K[j,d]/√D`, and
  `∂Y/∂K[j,d]=P_j(V[j,c]−Y[t,c])Q[t,d]/√D`.
  Each key's stable score, probability, weighted value and local
  softmax-score derivative are inspectable; using one node as Q/K/V
  sums all three roles without silently removing duplicate edges.
- **SwiGLU / GEGLU:** `Y=SiLU(Gate)·Up` or
  `Y=GELU_tanh(Gate)·Up`; gate derivative is
  `Up·activation'(Gate)`, distinct from the up derivative
  `activation(Gate)`. A shared source receives both contributions.
- **Routed Expert:** a dispatch index maps route `r=tK+k` back
  to token `t`; an expert linear output uses the actual expert ID and
  the flattened weight bank coordinate `[expert·D+d,h]`.
  `routed_weighted_reduce` emits one product per chosen route;
  `router_gate_weights` differentiates the selected continuous
  affinities, with the correct identity / sigmoid / softmax and optional
  normalization Jacobians. Repeated selected expert IDs keep their
  contributions, never overwrite. **TopK indices remain fixed discrete
  choices** and are surfaced as conditional cut metadata, not assigned
  fabricated real-valued gradients.

The common microscope shows the local stage ledger and selected discrete
route IDs, with a bounded number of rows on the screen; full bounded
Jacobian calculation is not truncated by UI display limits.
A structural trace lacking the selected expert index stops at an explicit
frontier. These are *coordinate-level teaching proofs*, not claims of
PyTorch/Numpy exported VJP parity, unbounded tensor scalability or
GPU kernel performance.

Independent oracle `tests/composite_coordinate_jacobians.js`: perturb
real Parameter tensors and rerun the original numerical graph interpreter,
checking multiple Q/K/V keys, tied QKV, shared gate/up branches, routed
MoE expert banks, selected router affinities, and a no-numeric selection
counterexample. Chromium desktop/mobile checks the shared microscope.

## Four-axis GQA / MQA, RoPE and immutable cache scalar VJP (2026-10-24)

The generic `element_lineage.js` graph interpreter now continues through the
**already registered four-axis operators** `nd_rope`, `nd_cache_read`,
`nd_cache_append`, and `nd_causal_mask`. These are local analytic Jacobians,
not new copies of a specialized decoder or a claim to reconstruct actual Wan
runtime weights.

- **RoPE:** adjacent-pair convention (the actual `nd_tensor.rope4` CPU
  kernel); for `θ=(offset+token)/base^(2·pair/rotaryDim)`, the selected
  channel depends on its own input by `cosθ` and the partner by
  `±sinθ`. Channels outside the rotary window have derivative 1.
- **Cache read:** output Token `t` pulls from absolute input Token
  `start+t`. **Cache append:** output Token `t` pulls from either
  prefix cache `t` or newly appended segment `t-prefixLength`, with
  exact source port metadata. This is immutable tensor concatenation.
- **Causal visibility:** absolute `queryStart+q` and `keyStart+k`
  decide if an attention score is kept. Visible scores have local derivative
  1; masked scores are assigned the constant `-1e9` and have derivative
  **zero**. This does not give a real-valued gradient to discrete mask
  topology or replace the teaching kernel with symbolic infinities.

Together with existing `nd_repeat_kv`, `nd_permute`,
`nd_batched_matmul`, `nd_scale` and `nd_softmax_last`, the same
general-purpose element lineage now traverses a complete numerical
`nd_cached_gqa_decode` pipeline from one selected context coordinate
to original Q/K/V values. The UI displays local RoPE rotation terms,
absolute cache positions and causal visibility through the common
PC/Android graph microscope. Source shapes, batch/head grouping and
realized stage values come from this graph, not from guessed model
architecture.

`tests/nd_incremental_coordinate_jacobians.js` independently perturbs
four-axis Q/K/V parameters and reruns the **original forward interpreter**.
It covers GQA and MQA head sharing, causal exclusion and whole-loss VJP,
plus cached decode with RoPE offsets and prefix/segment source boundaries.
Structural-only source graphs and stale traces are separate negative tests.
Chromium desktop/mobile checks the visible read-only microscope.

## Wan2.2 I2V · F/H/W complex-frequency rotary (2026-10-24)

Pinned reference: [Wan-Video/Wan2.2 `1ea34ff48f87168174e12956e200b1d908b1c5ff`](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py#L27-L66),
`rope_params()` and `rope_apply()`. The actual source accepts
`[B,Token,Heads,D]`, represents adjacent real pairs as complex values,
splits the **D/2 complex frequency channels** into
`[D/2-2*floor((D/2)/3), floor((D/2)/3), floor((D/2)/3)]` for
**frames, height, width**, and multiplies each complex channel by the
corresponding video-grid frequency. Frequency exponents preserve the
**global pair index**, including across F/H/W group boundaries.

The general N-D teaching graph uses explicit `[B,Head,Token,D]` and
offers `nd_wan_rope3d` on that view. For a grid `[F,H,W]`,
`t=f·H·W+h·W+w`, global complex pair `p=floor(d/2)` and axis
`a(p)`, `θ=position_{a(p)}/base^(2p/D)`.
For even/odd real channels, the complex multiplication is:

`out[...,2p]=x[...,2p]cosθ-x[...,2p+1]sinθ`;
`out[...,2p+1]=x[...,2p]sinθ+x[...,2p+1]cosθ`.

This is **not equivalent** to the previous one-dimensional
`nd_rope(offset+token)`. It is a full-dimensional complex rotation,
not a split-half RoPE. The bounded lesson requires an exact
`T=F·H·W` (no padded tokens or varying grids per sample).

Runnable lesson: `nd_wan_video_rope_attention`. It wires the rotary
operator into the existing `nd_batched_matmul`, `nd_permute`,
`nd_scale`, `nd_softmax_last`, and the final context projection
(`PV`), *without a causal mask* because the pinned WanSelfAttention
uses a FlashAttention call with explicit `window_size` rather than a
causal decoder. This first lesson treats Q/K/V as **already-projected,
already-normalized** synthetic micro-tensors; the real `norm_q/norm_k`
WanRMSNorm, Q/K/V learned projections, output projection, batch-specific
grid padding and GPU FlashAttention remain separate source tasks.
It does **not** claim a complete Wan block reproduction.

Exact analytic scalar coordinate gradients come from
`element_lineage.js` and the local full-tensor adjoint from
`nd_autodiff.js`. `tests/wan_video_rope3d_contracts.js` independently
constructs the complex phase, verifies norm preservation, all input
coordinate central differences for three grid shapes/base values,
full QK→Softmax→PV parity and sampled end-to-end context derivatives.
Negative contracts reject mismatched F/H/W, odd/short D, wrong one-axis
formula and stale graph observations.

## Wan2.2 pinned weighted QK norm and fully wired QKV projection (2026-10-24)

Pinned official source: [WanSelfAttention, WanRMSNorm, WanAttentionBlock](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py#L69-L259).

In WanSelfAttention, `self.norm_q(self.q(x)).view(b,s,n,d)` and the
corresponding K path mean the learned gamma and RMS denominator belong to
the **entire channel dimension C=n*d**, not to individual attention heads.
WanRMSNorm applies `x.float() * rsqrt(mean(x.float()**2)+eps)` and then
per-feature `self.weight`. A headwise RMSNorm before view is in general
**not equivalent**, and has a different cross-channel Jacobian.

The portable native `nd_wan_qk_rmsnorm` accepts `[B,1,T,C]` and a
learned `[1,1,1,C]` tensor (the singleton head dimension deliberately
models the original unsplit `[B,T,C]`). For `r=sqrt(eps+Σ_j x_j²/C)`:

`y_i=gamma_i*x_i/r`;
`∂y_i/∂x_j=gamma_i*(delta_ij/r-x_i*x_j/(C*r³))`;
`∂y_i/∂gamma_j=delta_ij*x_i/r`.

Full-tensor VJP sums loss contributions to gamma across batch/tokens and
uses the complete cross-channel Jacobian for input. Its coordinate lineage
exposes all C source coordinates and the gamma leaf. The full rewirable
`nd_wan_qkv_fullwidth_attention` graph uses existing `nd_parameter`,
`nd_batched_matmul`, `nd_broadcast_add`, `nd_reshape`, `nd_permute`,
`nd_wan_rope3d`, `nd_softmax_last` and `nd_mse` primitives.
That trace is source-ordered: input already represents the upstream norm1
and e0/e1 AdaLN-modulated attention input; independent Wq/Wk/Wv + bias;
full-width learned RMSNorm on Q/K only; .view(B,T,H,D) followed by an
**explicit** [B,H,T,D] permutation; full F/H/W RoPE on Q/K only; scaled
QK→Softmax→PV; Wo+b_o, and an illustrative training MSE target.

This lesson uses **one attention head** with C=D=6 so that the full
F/H/W RoPE complex-frequency groups are all present while the bounded
N-D CPU tensor limit is respected. The native primitives support freely
rewired graphs with other compatible dimensions; the lesson makes no claim
of real Wan 40-head parameter magnitudes or FlashAttention implementation.

`tests/wan_qk_weighted_rmsnorm_contracts.js` independently reconstructs
Wan normalization, verifies all X and gamma coordinates via central
finite differences and the wrong per-head alternative, and checks full
loss gradients through learned gamma and Q/K/V/O projection weights.

## Wan2.2 norm1 · e0/e1 pre-Self-Attention · e2 residual (2026-10-25)

Pinned source [WanAttentionBlock.forward `model.py:237-258`](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py#L237-L258).
In the original forward code, `e=(modulation.unsqueeze(0)+e).chunk(6,dim=2)`, then `y=self.self_attn(self.norm1(x).float()*(1+e[1].squeeze(2))+e[0].squeeze(2), ...)`, then `x=x+y*e[2].squeeze(2)`. The source later applies Cross-Attention after the e2 residual, followed by e3/e4 FFN conditioning and e5 gated residual. The demonstration in this PR covers **only the first e0/e1/e2 self-attention envelope**, reusing the already verified Q/K/V learned projectors and F/H/W rotary. It does not claim the full Transformer Block.

To avoid a second execution engine or a monolithic AdaLN-only operator, two genuinely reusable tensor bricks are introduced:
`nd_layernorm_last(x:[B,H,T,C])` calculates a non-affine last-channel LayerNorm, with cross-channel analytic VJP; and `nd_broadcast_mul(a,b)` multiplies broadcast-compatible four-axis tensors and sums gradients over repeated singleton coordinates. The existing `nd_broadcast_add` builds the rest. E0/E1/E2 in the lesson are **explicit three bounded slices** from the original six-chunk conditioning; each combines a synthetic timestep input with an independent per-channel trainable modulation tensor. No hidden six-chunk value is invented.

The exact computation of the bounded lesson is
`e_i=condition_i+modulation_i` for i=0,1,2,
`u=LayerNorm(x)`,
`z=u + u*e1 + e0`,
`a=SelfAttention(z)`,
`y=x+a*e2`.
The original `SelfAttention(z)` contains learned Wq/Wk/Wv, Q/K full-width weighted RMSNorm, Head view, Wan 3D RoPE and output Wo/b. MSE is an explicitly *illustrative* training target to verify gradients, not a Wan diffusion loss.

`tests/wan_adaln_self_gate_contracts.js` independently reimplements LN arithmetic and verifies all input and gate gradients through centered differences, checks the e0/e1/e2 wiring, tests selected whole-graph parameter FD, enforces zero-e2 cut of Attention parameter gradients while residual identity remains, checks that gate parameters remain sensitive at zero, and rejects stale numeric traces. Desktop/mobile Chromium screenshots follow the native `norm1` and broadcast gate stages in the shared microscope.
