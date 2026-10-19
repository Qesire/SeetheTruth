# Executable Python export — verified calculation, not decorative blocks

The Scratch-like workbench supports **Download .py**. It compiles the validated,
topologically sorted **user-edited graph** into Python 3.10+ NumPy source that
runs with no SeetheTruth runtime dependency.

    python -m pip install numpy
    python seethetruth-full-graph.py

The generated program loads initial source tensors, executes every supported
matrix/tensor operation, and prints a JSON object containing *all intermediate
outputs*, keyed by stable block ID. RoPE/GQA export includes actual
transposes, KV-head replication, sinusoidal rotation, causal masking,
Softmax, and batched MatMul—not a table of memorized final results.

To change sources after export:

    from my_exported_graph import run_graph
    import numpy as np
    tensors = run_graph(overrides={"query": np.ones((1, 4, 3, 4))})
    print(tensors["context"].shape)

Only source nodes (Input, Parameter and N-D equivalents) can be overridden.
The exported runner rejects a wrong source shape or nonfinite source values.

## Freedom-first export scopes

- **Full graph**: every declared node must have the required connections.
  Operations are lowered by topological dependency, without matching any
  known Transformer architecture template.
- **Selected branch**: export the selected node's upstream dependency
  closure. Incomplete parallel experiments remain in the editor.
- **Unsupported operations**: compilation **fails** with the node IDs and
  unsupported kinds. There are no output-embedding or placeholder fallbacks.
- Sources can be modified on the Python side; the compiler never assumes
  that arbitrary Parameter sources are already trained.

## Tested NumPy support

2D source tensors, MatMul, deterministic Linear, attention, MSE, elementwise
arithmetic, Softmax, RMSNorm, LayerNorm, SiLU/GELU and other activations.
4D operations cover GQA/MQA KV reuse, causal attention, RoPE, cache read/append,
broadcasting, reshaping, axis permutations/reductions, batched GEMM and losses.

The six functional optimizer comparison blocks (AdamW, Adam, SGD momentum,
RMSprop, Adagrad, Lion) export as **one-step pure updates from zero state**.
Their inputs are not silently modified.

The **NumPy backend remains functional and forward-only**; for actual parameter gradients, persistent AdamW, and resume-from-checkpoint use the separate [CPU PyTorch export](pytorch-executable-training.md). Both backends now cover ordinary reductions, normalization, 2D attention masks, low-rank projections and the fully composed teaching MoE routed expert path. The PyTorch backend implements differentiable weight sharing and explicit non-differentiable expert-index selection. GRPO and other unsupported operator kinds are rejected. This is not GPU kernel generation or a full pretrained model training system.

## Real cross-language acceptance

CI installs NumPy and invokes generated Python scripts as subprocesses. The extended suite `tests/python_codegen_extensions_contracts.js` checks 19 more operator families; `tests/moe_executable_codegen_contracts.js` executes entire MoE routing and three expert weight banks through standalone Python.
It compares **every intermediate tensor** with the JavaScript graph executor
for cached GQA decoding, 4D attention, trainable-parameter-source recipes,
FFN/Attention compositions, activations, and functional optimizer blocks.

A second acceptance case changes an exported Query source tensor and proves
that downstream Attention results change; the source is not a precomputed
result lookup. Another rejects incomplete full graphs and exports only a
complete selected branch of the same project.

GitHub Actions also packages independently runnable generated scripts as the
artifact named executable-numpy-graph-examples.
