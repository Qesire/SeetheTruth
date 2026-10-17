/* Evidence-only source projection for imported AST graphs.
 * These projections NEVER infer runtime execution, shapes, kernels or FLOPs.
 * Every rendered operation keeps the original node ID and source locator.
 */
(function(root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.SeetheTruthSourceView = api;
})(typeof window !== "undefined" ? window : {}, function() {
  "use strict";
  const LEVELS = Object.freeze(["macro", "block", "operator", "matrix", "tensor"]);
  const GEMM = new Set(["matmul", "mm", "bmm", "einsum", "addmm", "baddbmm", "linear"]);
  const SHAPE = new Set(["view", "reshape", "flatten", "unflatten", "transpose",
    "permute", "movedim", "squeeze", "unsqueeze", "contiguous", "expand",
    "broadcast_to", "repeat", "repeat_interleave", "split", "chunk",
    "cat", "concat", "stack", "gather", "scatter", "index_add",
    "masked_fill", "where", "softmax", "log_softmax", "topk"]);
  const ELEMENTWISE = new Set(["add", "sub", "mult", "div", "mul", "pow", "matmult",
    "silu", "gelu", "exp", "clamp", "minimum", "mean", "sum"]);
  function classify(n) {
    const callee = String(n.callee || n.aten_op || "").split(".").pop().toLowerCase().replace(/_$/, "");
    const label = String(n.label || "").toLowerCase().replace(/\(\)$/, "");
    const ctor = String(n.module_constructor || "").toLowerCase();
    if (GEMM.has(callee) || GEMM.has(label) || label === "matmult" || ctor === "linear") return "matrix";
    if (SHAPE.has(callee) || SHAPE.has(label) || ELEMENTWISE.has(label) || ELEMENTWISE.has(callee)) return "tensor";
    if (n.certainty === "source_operator" || n.callee || n.aten_op || n.constructor_op) return "operator";
    return "other";
  }
  function project(graph, scope, level, capacity = 190) {
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) throw Error("Invalid graph");
    if (!LEVELS.includes(level)) throw Error("Invalid source level");
    if (!Number.isSafeInteger(capacity) || capacity < 1) throw Error("Invalid projection capacity");
    const byId = new Map(graph.nodes.map(n => [n.id, n]));
    if (scope != null && !byId.has(scope)) throw Error("Unknown source scope");
    const children = new Map();
    for (const n of graph.nodes) {
      const parent = n.parent || null;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(n);
    }
    const direct = children.get(scope || null) || [];
    const descendants = [];
    const queue = [...direct];
    // Nodes are already validated for parent cycles by the graph importer.
    for (let i = 0; i < queue.length; i++) {
      const n = queue[i];
      descendants.push(n);
      queue.push(...(children.get(n.id) || []));
    }
    if (level === "macro" || level === "block") {
      const nodes = direct.slice(0, capacity);
      const ids = new Set(nodes.map(n => n.id));
      return {nodes, edges: graph.edges.filter(e => ids.has(e.from) && ids.has(e.to)),
        matches: ids, matchCount: direct.length, total: direct.length,
        omitted: Math.max(0, direct.length - nodes.length), level, scope};
    }
    const matching = descendants.filter(n => {
      const kind = classify(n);
      return level === "operator" ? kind !== "other" :
        level === "matrix" ? kind === "matrix" :
        kind === "tensor" || kind === "matrix";
    });
    const chosen = new Set(matching.slice(0, capacity).map(n => n.id));
    const matches = new Set(chosen);
    if (chosen.size < capacity) {
      // Include bounded direct operands for interpretability; never assert that
      // an absent dependency was observed. Nonmatching nodes are CONTEXT.
      const available = new Set(descendants.map(n => n.id));
      for (const edge of graph.edges) {
        if (chosen.size >= capacity) break;
        if (edge.kind === "hierarchy" || edge.kind === "feedback") continue;
        if (matches.has(edge.to) && available.has(edge.from)) chosen.add(edge.from);
      }
    }
    const nodes = descendants.filter(n => chosen.has(n.id));
    const edges = graph.edges.filter(e => chosen.has(e.from) && chosen.has(e.to));
    return {nodes, edges, matches, matchCount: matching.length,
      total: descendants.length, omitted: Math.max(0, matching.length - matches.size),
      level, scope};
  }
  return Object.freeze({LEVELS, classify, project});
});
