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
