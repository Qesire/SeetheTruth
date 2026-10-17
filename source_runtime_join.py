"""Conservative, opt-in source/ATen attribution.

No source modules are imported or executed. Runtime side supplies an explicit
source_root and opt-in frame capture; an exact file hash and unique AST callsite
are mandatory. An unproved event is left unmatched, never guessed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path


class SourceLocator:
    """Capture a source position only inside a declared, trusted Python root.

    This is opt-in stack inspection, not a reconstruction of fused kernels.
    Files are SHA-256 hashed with a finite size cap, with no source code output.
    """

    def __init__(self, source_root, *, max_file_bytes=2_000_000):
        root = Path(source_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("source_root must be a directory")
        self.root = root
        self.max_file_bytes = max_file_bytes
        self.digests = {}
        self.omitted = 0

    def capture(self):
        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            for _ in range(80):
                if frame is None:
                    break
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                frame = frame.f_back
                if not filename.endswith(".py"):
                    continue
                try:
                    path = Path(filename).resolve(strict=True)
                    relative = path.relative_to(self.root)
                except (ValueError, OSError):
                    continue
                if str(relative) in ("runtime_trace.py", "source_runtime_join.py"):
                    continue
                try:
                    if relative.as_posix() not in self.digests:
                        if path.stat().st_size > self.max_file_bytes:
                            self.omitted += 1
                            continue
                        self.digests[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
                    return {"file": relative.as_posix(), "line": lineno,
                            "sha256": self.digests[relative.as_posix()]}
                except OSError:
                    self.omitted += 1
            self.omitted += 1
            return None
        finally:
            del frame


# Static operator identity, not inferred architecture role. Only forward
# events with a single exact expression at the source line are eligible.
MATMUL_ATEN = {"mm", "bmm", "matmul"}
ATEN_ALIASES = {
    "_softmax": "softmax",
    "mul_": "mul",
    "add_": "add",
    "div_": "div",
    "sub_": "sub",
    "transpose_": "transpose",
}
BINOP = {"Add": "add", "Sub": "sub", "Mult": "mul", "Div": "div", "MatMult": "matmul"}
CALL_NAMES = {"mm", "bmm", "matmul", "addmm", "baddbmm", "einsum",
              "transpose", "permute", "reshape", "view", "softmax",
              "silu", "gelu", "mul", "add", "sub", "div", "mean", "sum",
              "cat", "contiguous", "masked_fill", "linear"}


def _op_matches(ast_node, aten_name):
    op = aten_name.removeprefix("aten.").split(".")[0].lower()
    op = ATEN_ALIASES.get(op, op)
    if ast_node.get("certainty") == "source_operator":
        static = BINOP.get(ast_node.get("label"))
        return bool(static and (op in MATMUL_ATEN if static == "matmul" else op == static))
    callee = str(ast_node.get("callee", "")).split(".")[-1]
    if callee not in CALL_NAMES:
        return False
    if callee in MATMUL_ATEN:
        return op in MATMUL_ATEN
    return op == callee


def join_graphs(static_graph, runtime_graph):
    """Return an annotated *copy* of runtime graph with conservatively proved matches.

    The method does not match events by order, tensor shape or class name alone.
    Dynamic branches only contribute observed callsites. Multiple same-line
    static expressions are marked ambiguous; unmatched status is explicit.
    """
    if static_graph.get("schema") != "model-atlas-graph-v1" or runtime_graph.get("schema") != "model-atlas-graph-v1":
        raise ValueError("Both inputs must use model-atlas-graph-v1")
    if static_graph.get("meta", {}).get("analysis") != "static-python-ast":
        raise ValueError("Expected an AST static graph")
    if runtime_graph.get("meta", {}).get("analysis") != "runtime-torch-dispatch":
        raise ValueError("Expected a runtime-dispatch graph")
    digests = static_graph["meta"].get("source_sha256", {})
    result = copy.deepcopy(runtime_graph)
    stats = {"matched_module": 0, "matched_operator": 0,
             "ambiguous": 0, "unmatched": 0, "missing_location": 0,
             "source_mismatch": 0, "ineligible": 0}
    if not runtime_graph["meta"].get("source_capture", {}).get("enabled"):
        result["meta"]["source_attribution"] = {
            "status": "blocked_capture_disabled",
            "reason": "Runtime callsite collection was not opted in; no source claims permitted.",
            "stats": stats,
        }
        return result
    if static_graph["meta"].get("truncated") or runtime_graph["meta"].get("truncated"):
        result["meta"]["source_attribution"] = {
            "status": "blocked_truncated_graph",
            "reason": "Cannot prove unique callsites when either graph is incomplete.",
            "stats": stats,
        }
        return result

    by_location = {}
    for n in static_graph.get("nodes", []):
        loc = n.get("source") or {}
        if isinstance(loc.get("line"), int) and isinstance(loc.get("file"), str):
            by_location.setdefault((loc["file"], loc["line"]), []).append(n)
    for n in result["nodes"]:
        if n.get("phase") != "forward":
            stats["ineligible"] += 1
            continue
        is_module = bool(n.get("module")) and not n.get("aten_op")
        is_operator = bool(n.get("aten_op"))
        if not is_module and not is_operator:
            stats["ineligible"] += 1
            continue
        loc = n.get("source") or {}
        if not (isinstance(loc.get("file"), str) and isinstance(loc.get("line"), int) and isinstance(loc.get("sha256"), str)):
            status = "missing_location"
        elif digests.get(loc["file"]) != loc["sha256"]:
            status = "source_mismatch"
        else:
            matching = []
            for cand in by_location.get((loc["file"], loc["line"]), []):
                if is_module:
                    callee = cand.get("callee", "")
                    receiver = callee.split(".")[-1]
                    class_name = cand.get("module_constructor")
                    # Require observed module instance suffix AND exact type.
                    if (callee.startswith("self.") and
                        receiver == n["module"].split(".")[-1] and
                        class_name == n.get("module_type")):
                        matching.append(cand)
                elif _op_matches(cand, n["aten_op"]):
                    matching.append(cand)
            if len(matching) == 1:
                c = matching[0]
                n["source_attribution"] = {
                    "status": "matched",
                    "static_node_id": c["id"],
                    "static_source": {"file": loc["file"], "line": loc["line"],
                                      "sha256": loc["sha256"]},
                    "evidence": "exact_file_sha256_line_and_" + ("module_type" if is_module else "operator"),
                    "not_evidence_of": ["kernel launch", "measured latency", "all source branches"],
                }
                stats["matched_module" if is_module else "matched_operator"] += 1
                continue
            status = "ambiguous" if len(matching) > 1 else "unmatched"
        n["source_attribution"] = {"status": status}
        stats[status] += 1
    result["meta"]["source_attribution"] = {
        "status": "partial_verified",
        "method": "sha256 + source callsite line + exact module type / operator identity",
        "stats": stats,
        "source_graph_name": str(static_graph.get("name", ""))[:100],
        "runtime_capture_scope": runtime_graph["meta"].get("source_capture", {}),
        "no_heuristic_fallback": True,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="Annotate runtime events using exact static AST source evidence")
    ap.add_argument("--static", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    static = json.loads(Path(args.static).read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    joined = join_graphs(static, runtime)
    Path(args.out).write_text(json.dumps(joined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(joined["meta"]["source_attribution"], ensure_ascii=False))


if __name__ == "__main__":
    main()
