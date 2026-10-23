# Graph-derived learning route

A common graph-derived projection closes the gap between shape readiness, numeric execution and literature-grounded explanations. The route is built from the current arbitrary DAG and its real port sources using `GraphSession`, `OperatorContracts` and `BlockKnowledge` — never a predefined Transformer template.

On `learn.html` expand **LEARNING PATH**. Inspect each node in topological order: incoming ports and source shapes, output shape, formula, calculation flow, design rationale, industrial usage, pitfalls and a falsifiable exercise. Clicking its node opens the existing unified operator inspector.

The default route is **static only**; pressing **验证本图的微型数值** executes the bounded graph once through `GraphSession.inspect({numeric:true})` and shows a sampled subset of actual numeric output values. Failed or incomplete graphs preserve their explanatory route but do not invent numeric observations. SYNTHETIC_NUMERIC is not REAL_CAPTURE. NumPy/PyTorch candidate labels come from the source inventory, not a guarantee that the complete graph compiles.

Checks: `node tests/graph_learning_path_contracts.js` and `python tests/browser_learning_path.py` (requires Playwright Chromium). Both desktop and offline Android package the exact same graph projection source.
