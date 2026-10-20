# PyTorch CPU: genuine independent trainable graph export

The Scratch-like workbench can export **NumPy forward computations** or an independent **PyTorch CPU float64** program. Select *PyTorch · 自动微分与训练* before using `导出 .py`.

```bash
python -m pip install torch
python seethetruth-training-graph-torch.py
```

The exported `GraphRuntime` owns stable named `torch.nn.Parameter` objects; `GraphRuntime.forward()` calculates each supported tensor op in topological order and preserves PyTorch's real autograd graph. It does not replay precomputed JavaScript outputs. Supported operations include conventional 2D Linear/MatMul, LayerNorm/RMSNorm, SiLU/GELU/GLU, SDP Attention, exact 4D batched MatMul, GQA repeated KV heads, RoPE, causal mask, cache tensor concat/slices, finite scalar MSE/MAE loss, 2D reductions, softmax masks, stable log-softmax and low-rank matmul compositions. **Discrete expert routing is now executable**: group Top-k, masked routing, router gate weights, token dispatch, routed expert Linear, weighted reduction and shared parameter gradients. The original MoE teaching recipes run as independent Python programs. GRPO and other unimplemented primitives continue to refuse export explicitly.

## API

```python
graph = GraphRuntime()
tensor_outputs = graph.forward()
print(tensor_outputs["loss"])
history = graph.train(steps=2, lr=0.004, weight_decay=0.005)
graph.save_checkpoint("example.pt")

restored = GraphRuntime()
restored.load_checkpoint("example.pt")  # only load a checkpoint you trust
restored.train(steps=1, lr=0.004, weight_decay=0.005)
```

Using `GraphRuntime(overrides={"tokens": values})` supplies complete input tensors with the expected shape; overrides are checked for finiteness. Train uses PyTorch `loss.backward()` followed by genuine `torch.optim.AdamW.step()` with *persistent optimizer moments* on a single graph instance. Both parameter values and AdamW state survive a trusted local checkpoint round-trip. Changing optimizer hyperparameters without explicitly reconstructing the runtime is rejected to avoid hidden state resets.

A `Parameter` with validated `params.frozen=true` is a value-preserving **non-trainable Torch tensor**, omitted from `torch.optim.AdamW`. This permits a paper-driven LoRA rewrite to keep the original pretrained/base `W₀` unchanged while training the low-rank A/B adapters; no hidden copies or resetting to synthetic weights occurs.

**Backpropagation is verified on supported graph recipes, not all 116 blocks.** A standalone generated PyTorch file has no SeetheTruth imports. CPU/no GPU is the default. It does **not** capture or train actual pretrained model weights, execute arbitrary Python supplied through saved project JSON, implement distributed expert dispatch, or export a production checkpoint of an LLM.

### Evidence

`tests/torch_codegen_contracts.js` compiles actual user-visible graph presets and launches the generated source with Python/PyTorch. It checks every forward intermediate against the existing JavaScript teaching engine, compares gradients to independently computed reverse VJPs and two optimizer steps against `train_state.js`, verifies checkpointed third-step equality, and tests incomplete branches and unsupported operators. `tests/torch_codegen_extensions_contracts.js` additionally checks 19 operations and real MAE training. `tests/moe_executable_codegen_contracts.js` runs three full MoE graphs in NumPy and Torch, checks all intermediate tensors, and compares expert gradients, two optimizer steps and checkpoint resumed training against JavaScript. `.github/workflows/torch-export.yml` executes all three suites with CPU-only PyTorch and NumPy.

The NumPy exporter remains the smaller-dependency forward-only option; choose it for quick runnable calculations that do not need gradients or checkpoints.
