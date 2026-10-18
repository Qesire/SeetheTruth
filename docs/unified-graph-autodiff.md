# Unified Scratch-block VJP and AdamW tutorial

This capability extends the same editable `learn.html` graph. A browser executes only deterministic tiny Float64 matrices, **never a pretrained checkpoint**, CUDA kernel, network optimizer or distributed training job. The interactive inspector displays adjoints for the selected block after the numeric trace and an explicit reverse-pass action.

## Graph contract

- `graph_autodiff.js` accepts `L.validateGraph(raw)`, topologically ordered nodes, and an exact numeric `L.evaluate(graph)` forward trace. The graph must have a scalar loss (`mse` by default). The reverse engine checks node identity and every returned shape before adding any VJP.
- It starts `dL/dL = 1` and processes the DAG **in reverse topological order**, adding contributions from every outgoing consumer. Identical input-node identities therefore accumulate gradient correctly even when reused in multiple MatMul branches **and** the Q/K/V ports of a single Attention block.
- Supported differentiable primitives include MatMul, linear-to-input (deterministic teaching weight), add/subtract, elementwise multiply, scale/bias/square, transpose, ReLU/SiLU/Sigmoid/Tanh, Softmax/LogSoftmax, RMSNorm/QK Norm, LayerNorm, SwiGLU, Attention, MSE, row sums/means, feature concat/slice, whole-matrix Repeat Rows, and explicit Detach.
- The target data port of MSE is stopped. Unsupported *reachable* operations throw an error instead of propagating invented zeros. Unreachable nodes carry zero adjoints because they have no dependency on the selected loss.
- The `linear` block does not have explicit trainable weight inputs; its VJP only propagates to its input. To inspect actual parameter-matrix gradients, use a MatMul with an ordinary explicit matrix input.

### Numerically verified 11-block lesson: `autodiff_core`

```
x[T=2,D=2] ----- MatMul -------- P ----- RMSNorm ------+
                  ^                                        |
                  |                               +--------+--------+
      shared W[2,2]-------------------------->  MatMul gate     MatMul up
                  ^                               |               |
                  |                               +----- SwiGLU ---+
                  |                                        |
                  |                               Q = K = V (shared)
                  |                                        |
                  |                                   Attention
                  |                                        |
                  +--------------------- P ----- Add (residual)
                                                           |
                                             MSE(prediction, target)
                                                           |
                                                     scalar loss
```

All 2×2 elements of `x` and shared `W` are perturbed separately by central finite differences. This checks all eight analytic gradient coordinates. Parameter `W` is consumed three times, and the SwiGLU activation is consumed via three Attention ports; these routes must be aggregated, not overwritten.

**Attention mathematics** for `Y = P V`, `P = softmax(S)`, `S = Q Kᵀ / √D`:

```
dV = Pᵀ dY
dP = dY Vᵀ
dS = rowSoftmaxVJP(P,dP) / √D
dQ = dS K
dK = dSᵀ Q
```

When Q/K/V come from one node, its final gradient is `dQ+dK+dV`. The `RMSNorm` backward uses the exact row mean square; the SwiGLU backward uses the true SiLU derivative and product rule. Tests verify the entire graph rather than only these subexpressions in isolation.

## Composed MoE reverse flow: `moe_backward` (24 nodes)

This lesson reuses the **same 22 editable routed-Expert-FFN blocks** as lesson 16 and appends an explicit target matrix plus MSE. The unified `graph_autodiff.js` imports the already-tested `expert_ffn_vjp.js` local rules, instead of duplicating a fake Expert backward demonstration.

- `dispatch_routed_tokens(X,indices)`: adjoints of the K token copies add back into one `dX[t,:]`.
- `expert_linear_bank(Z,indices,W)`: expert IDs are fixed, while each selected token route computes the derivative with respect to its data and that expert's own W1/W3/W2. Multiple routes selecting the same expert **sum into the same weight rows**.
- `routed_weighted_reduce(V,gates)`: gradients flow to selected expert outputs and their corresponding continuous weights.
- `router_gate_weights(affinity,indices,mode=identity,normalize=true)`: each selected positive affinity receives the exact normalized-gate derivative; non-selected experts get zero on this path.
- `indices` routing edges are **detached**, not approximated by straight-through gradients. Bias/group-selection branches affect the forward index choices, but the discrete selection event has no derivative in this teaching rule.
- The independent shared Expert path also receives gradients through `Linear → SwiGLU → Linear`. Its parameter-less teaching `linear` nodes only propagate gradients to input; they are not presented as trained weights.

**Numerical acceptance:** The test perturbs every element of the three routed expert weight banks (3×8×2 = 48 scalar coordinates) and every entry of the 3×4 affinity tensor (12 more), then recomputes the actual selected route/expert calculations. All 60 scalar derivatives must agree with the full reverse graph gradient under a central finite-difference oracle that **holds discrete expert indices fixed**. The final output of the combined lesson remains a scalar MSE.

The optimizer preview in the same inspector can be invoked on an explicit expert weight bank, treating its accumulated VJP tensor as a parameter gradient. Its values are not committed back into the graph.

## AdamW optimizer **preview**, not training commit

On a *selected explicit matrix input that feeds the right-hand input of a MatMul* (after numeric forward and backward), the UI offers `预览 AdamW 单步更新`. This calls the standalone pure `adamw_micro.js` on the selected matrix and its accumulated gradient.

For step `t` with elementwise gradient `g`:

```
m_t = β₁ m_(t−1) + (1−β₁) g
v_t = β₂ v_(t−1) + (1−β₂) g²
m̂_t = m_t/(1−β₁^t); v̂_t = v_t/(1−β₂^t)
θ_t = θ_(t−1) − α [ m̂_t/(√v̂_t+ε) + λ θ_(t−1) ]
```

The `λ θ` term is **decoupled weight decay**, not adding `λ θ` to the raw gradient before moment updates. Source: Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019.

The preview shows `Δθ` and updated parameter *without changing the graph*, its seed, its original value matrix or the forward trace. The pure optimizer helper separately supports second and later steps when its immutable state is passed back in; unit tests verify two-step bias correction and the zero-gradient decay case. To actually train a parameter, the system would need an explicit persistent parameter-block identity and a new forward computation after the update. This is **not yet implemented**.

## Research and implementation boundaries

- The approach computes local VJPs of the graph's current executable operators, not automatic differentiation through arbitrary future blocks.
- The dedicated `grpo_backward.js` still handles PPO/GRPO policy-specific stop-gradient and clipping semantics. Both modes are available through the same main inspector.
- MatMul, gated FFN, Attention and gradient flow concepts are grounded in [Attention Is All You Need](https://arxiv.org/abs/1706.03762), source framework documentation, and the block's paper knowledge entries. GPU throughput numbers are not measured or simulated.
- Browser acceptance must check input and weight gradients, one-step optimizer preview, screenshot visibility, and that modifying/reloading a graph invalidates stale gradients. Unit tests check finite-difference oracle, detachment, shape validity, and immutable optimizer state.

## Trainable open-model building blocks: Q/K/V, SwiGLU, MoE

The same `parameter`, `matmul`, `swiglu`, `attention`, `expert_linear_bank`, `mse`, graph VJP, and AdamW primitives provide **three additional reusable recipes**, not three new opaque executable model types.

| Lesson | Explicit trainable tensors | Forward path | Gradient semantics |
|---|---|---|---|
| 22 · `trainable_attention` | `Wq`, `Wk`, `Wv` (each 2×2) | `XWq,XWk,XWv → SDP Attention → MSE` | independent per-matrix VJP through Q/K/V, including row Softmax |
| 23 · `trainable_swiglu` | `W1`, `W3`, `W2` (each 2×2) | `SiLU(XW1)⊙(XW3) → W2 → MSE` | SiLU product rule and three explicit matrix derivatives |
| 24 · `trainable_moe` | expert banks `W1`, `W3`, `W2` (each 8×2 for four 2×2 experts) | top-k indices → token dispatch → expert SwiGLU → weighted combine → shared expert + MSE | same expert receives gradients summed over every selected token; discrete index selection is detached |

These are generic mathematical structures used by attention and FFN architectures such as [Qwen3](https://arxiv.org/abs/2505.09388), [DeepSeek-V3](https://arxiv.org/abs/2412.19437), and [Mixtral](https://arxiv.org/abs/2401.04088), **not full architecture or checkpoint reproductions**. Real models have additional dimensions, normalization, masking, positional treatment, capacity handling, quantization, expert parallelism and other details. In lesson 24, the shared expert's deterministic `linear` blocks remain fixed; only routed expert banks are explicit trainable Parameter nodes.

### Configurable bounded optimizer transactions

The workbench exposes `trainLearningRate`, `trainWeightDecay`, and `trainCount`. A training transaction can run 1, 2, 4 or 8 sequential CPU teaching steps; the entire selection is a single undoable operation. On **every** step it recomputes the actual forward trace and full reverse graph gradients before updating all active parameter nodes, so the second step cannot accidentally reuse the first step's gradient.

The in-project JSON stores the exact `values` and `optimizer = {t,m,v}` for *each stable node ID*, and all three architecture recipes use the same state layout. The UI shows per-step before/after loss, allows the student to compare hyperparameters, and does not assert monotonic improvement or claim actual large-model learning.

**Verification:** 33 whole-model finite-difference parameter coordinates across the Q/K/V, SwiGLU and routed MoE configurations, 2-step update for every parameter, JSON save/restore equivalence of the subsequent update, invalid count and LR rejection, and Chromium clicks through all three architecture lessons. The teaching input seeds are deterministic and no GPU or real model inference is required.

## Trainable Parameter block: closed teaching optimization loop

The `parameter` block is a real, separately addressable source tensor within the small graph. Its **node ID is the stable parameter identity**: moving the node never changes its identity, and any number of consumer MatMul nodes can connect to one parameter output. The graph JSON includes the full bounded Float64 `params.values` and the AdamW state `params.optimizer = {t,m,v}`. These are **not hidden browser-global mutable values**.

One click on `↻ 训练一步` in numerical mode runs an atomic transaction (`train_state.js`):

1. Validate the entire graph and calculate its numeric forward trace.
2. Compute the scalar MSE gradient using the same graph-autodiff VJP library already validated by finite differences.
3. Select only `parameter` nodes reachable from the loss. Sum all incoming adjoints for each stable node ID.
4. Compute a complete AdamW step (first/second moments, bias correction and decoupled weight decay), without mutating the original graph snapshot.
5. Validate updated tensors and moments, evaluate the next forward pass, and return a **new** graph plus `beforeLoss`, `afterLoss`, gradient/update norms and optimizer step counters.
6. The workbench only commits a fully successful result to the undo stack, clears stale backward gradients, and displays the new forward values.

### Why this is different from AdamW preview

The earlier preview was intentionally read-only. The new explicit training action **does update** the small project's parameter values, and the new values become the next iteration's forward weights. Undo restores the exact previous parameter **and optimizer moment tensors**; redo restores the newer version. `Save` exports the values and optimizer state into the project JSON. Loading that JSON then continuing training must produce the same next graph as training without the round-trip.

The separate `Matrix values (JSON)` editor allows controlled changes to a selected parameter. Manual replacement resets its moment tensors and step counter; blindly reusing moments from a different tensor would be semantically invalid.

Lesson **21**, `trainable_core`, retains the shared MatMul/RMSNorm/SwiGLU/Attention/MSE graph from lesson 19, but changes the shared matrix source to `parameter`. The initial weights are synthetic small tensors. The loss before/after each step is computed from the actual changed weights, not hardcoded to decrease. An optimizer step may legitimately increase a nonconvex loss; this is a teaching trace, not proof of convergence.

**Verified boundaries:** no stochastic data loader, no real checkpoint, no cross-epoch generalization or GPU, no arbitrary-weight `linear` model training. The system optimizes only explicit `parameter` sources in supported numeric DAGs, using one small batch and a scalar MSE. Source references: [PyTorch Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html), [AdamW paper](https://arxiv.org/abs/1711.05101).

