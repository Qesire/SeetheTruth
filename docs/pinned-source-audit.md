# Real-source operator fidelity audit (GPT-2)

This is an **AST-only** audit against the official Hugging Face Transformers v4.44.2 GPT-2 implementation at immutable Git commit:
`174890280b340b89c5bfa092f6b4fb0e2dc2d7fc`, file `src/transformers/models/gpt2/modeling_gpt2.py`.

We fetch the source text from `raw.githubusercontent.com` **only in a separate CI test** and parse it as AST. We never import or execute it. The exact SHA-256 of each inspected file is attached to a static graph in `meta.source_sha256` to detect source changes.

## Independent comparison oracle

| Actual source method | Expected static calculation sites | Meaning and caveats |
|---|---|---|
| `GPT2Attention._attn` | 2 distinct `torch.matmul` calls | QK score and A·V aggregation; source occurrences, not measured GEMM launches |
| `GPT2Attention._upcast_and_reordered_attn` | 1 `torch.baddbmm` + 1 `torch.matmul` | Alternative upcast path; must not disappear from graph |
| `GPT2Attention.forward` | Transitive call scopes include both above methods with conditional guards | Both are possible branches; never sum their call sites as if executed together |
| `GPT2SdpaAttention.forward` | 1 `scaled_dot_product_attention` source call | Fused logical attention, not proof of separate physical QK/Softmax/AV kernels |
| `GPT2MLP.forward` | Calls to `self.c_fc` and `self.c_proj`, with custom constructor `Conv1D` | Unknown custom implementation must not be labeled generic torch.conv1d without inspecting its definition |

Each expected occurrence is independently read from the official AST with its **exact source line** and then compared with the graph nodes' source locator. The graph must not be truncated. No dynamic Tensor shape is claimed verified.

## Parsing errors fixed during audit

- Calls of the form `self._attn(...)` and `self._upcast_and_reordered_attn(...)` previously stopped at the call node. They now recurse into same-class methods within the depth budget and preserve the conditional context.
- `torch.baddbmm` previously lacked a named matrix-operation semantic rule.
- `+=` and inline `a if cond else b` were not first-class source graph operators / joins.
- Variables defined on only one side of a branch were previously incorrectly marked definitely available. They now carry a `φ?` analysis-only construct with `conditional_defined=true`.
- Recursive parsing now maps actual positional/keyword arguments into the nested function's parameter nodes with explicit cross-scope edges.
- File SHA-256 and truncation state now make the provenance and graph completeness independently inspectable.

## Limitations

- Custom imports, aliasing, closures, dynamic dispatch, `torch.compile` graph rewriting, mutation/alias analysis, and path feasibility are **not yet fully resolved**.
- Parsing both `if` branches is **not** evidence that both ran.
- The import name `Conv1D` does not by itself establish a math equation or physical GEMM, and official GPT-2's class implementation must be inspected for that claim.
- Symbolic graph operator counts are **source callsite counts**, not GPU kernel launch counts. Complete tensor shape inference, exact FLOPs and true execution time require runtime evidence.
- CI pinned-upstream fetch depends on external GitHub availability and may fail on network errors; local deterministic tests work offline.

## How to run

Offline tests:

    python -m unittest discover -s tests -p 'test_operator_integrity.py' -v

Pinned official-source audit (source is downloaded but never imported/executed):

    SEETHTRUTH_AUDIT_UPSTREAM=1 python -m unittest discover -s tests -p 'test_pinned_gpt2_source.py' -v
