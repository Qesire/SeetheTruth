"""One opt-in CPU/CUDA profiler + ATen trace over the exact SAME model pass.

The caller executes their own trusted model exactly once. This context does
not automatically call forward/backward/optimizer or read tensor values.
Per-module observations require unique invocation identity and shape metadata.
"""
from __future__ import annotations

from contextlib import ExitStack
from uuid import uuid4
from pathlib import Path
import json

from runtime_trace import RuntimeTrace
from telemetry_capture import TelemetrySession
from source_runtime_join import join_graphs


def bind_same_pass(runtime_graph, telemetry, *, run_id, model_name):
    """Evidence-only join for *co-captured* sessions, never independent files.

    The random run ID is generated in VerifiedPass and cannot be supplied here
    through public CLI, which prevents accidental cross-run cost joins.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("verified run ID required")
    if runtime_graph.get("meta", {}).get("truncated"):
        return {"status": "blocked_truncated_runtime", "stats": {"joined": 0}, "graph": runtime_graph}
    graph = json.loads(json.dumps(runtime_graph))
    per_call = {(row["name"], row["invocation_index"]): row
                for row in telemetry.get("module_calls", [])}
    stats = {"joined": 0, "unmatched": 0, "tensor_signature_mismatch": 0,
             "type_mismatch": 0, "unattributed_operators": telemetry["meta"].get("unattributed_operator_events", 0)}
    for n in graph["nodes"]:
        if not n.get("module") or n.get("phase") != "forward":
            continue
        path = n["module"]
        if n.get("parent") is None and path == model_name:
            path = "<root>"
        candidate = per_call.get((path, n.get("invocation_index")))
        if candidate is None:
            n["same_pass_profile"] = {"status": "unmatched"}
            stats["unmatched"] += 1
            continue
        if candidate["type"] != n.get("module_type"):
            n["same_pass_profile"] = {"status": "type_mismatch"}
            stats["type_mismatch"] += 1
            continue
        r_inputs = (n.get("tensor") or {}).get("inputs", [])
        r_outputs = (n.get("tensor") or {}).get("outputs", [])
        def signature(xs):
            return [(v.get("shape"), v.get("stride"), v.get("dtype")) for v in xs]
        if (signature(r_inputs) != signature(candidate.get("last_inputs", []))
                or signature(r_outputs) != signature(candidate.get("last_outputs", []))):
            n["same_pass_profile"] = {"status": "tensor_signature_mismatch"}
            stats["tensor_signature_mismatch"] += 1
            continue
        n["same_pass_profile"] = {
            "status": "matched", "measurement": "same_instrumented_pass",
            "run_id": run_id, "invocation_key": candidate["invocation_key"],
            "host_wall_ms": candidate["host_wall_ms"],
            "exclusive_self_cpu_ms": candidate["exclusive_self_cpu_ms"],
            "inclusive_attributed_cpu_ms": candidate["inclusive_attributed_cpu_ms"],
            "exclusive_self_device_ms": candidate["exclusive_self_device_ms"],
            "inclusive_attributed_device_ms": candidate["inclusive_attributed_device_ms"],
            "exclusive_operator_events": candidate["exclusive_operator_events"],
            "inclusive_operator_events": candidate["inclusive_operator_events"],
            "attribution_scope": candidate["profiler_scope"],
            "not_evidence_of": ["GPU device critical path", "kernel boundaries",
                                "uninstrumented latency", "complete autograd trace"],
        }
        stats["joined"] += 1
    graph["meta"]["same_pass_profile"] = {
        "status": "partial_verified", "run_id": run_id,
        "method": "one_live_pass_with_module_path_invocation_index_type_and_tensor_signature",
        "stats": stats, "source_join_not_required": True,
        "profiler_is_instrumented": True,
    }
    return {"status": "partial_verified", "stats": stats, "graph": graph}


class VerifiedPass:
    """Explicit same-pass instrumentation. Caller selects workload.

    with VerifiedPass(model, source_root=source_root) as capture:
        output = model(inputs)
    result = capture.report(static_graph=ast_graph_or_None)

    If static_graph is omitted the same-pass profiler attribution is valid,
    but SOURCE->RUNTIME identification is not asserted.
    """

    def __init__(self, model, *, source_root=None, max_events=700,
                 capture_values=False, record_gradients=False):
        self.model = model
        self.run_id = uuid4().hex
        self.trace = RuntimeTrace(model, source_root=source_root,
                                  max_events=max_events,
                                  record_gradients=record_gradients)
        self.profile = TelemetrySession(model, capture_values=capture_values)
        self._stack = ExitStack()
        self._closed = False
        self._entered = False

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("VerifiedPass is single-use")
        self._entered = True
        try:
            self._stack.enter_context(self.trace)
            self._stack.enter_context(self.profile)
        except BaseException:
            self._stack.close()
            self._closed = True
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._stack.__exit__(exc_type, exc, tb)
        finally:
            self._closed = True

    def report(self, *, static_graph=None):
        if not self._closed:
            raise RuntimeError("VerifiedPass must exit before report")
        rt = self.trace.graph()
        telemetry = self.profile.report()
        evidence = bind_same_pass(rt, telemetry, run_id=self.run_id,
                                  model_name=type(self.model).__name__)
        joined = evidence["graph"]
        if static_graph is not None:
            # Source matching never borrows profiler numbers from separate passes.
            joined = join_graphs(static_graph, joined)
        joined["meta"]["same_pass_profile"] = {
            **joined["meta"].get("same_pass_profile", {}),
            "status": evidence["status"], "run_id": self.run_id,
            "stats": evidence["stats"],
            "co_captured": True,
        }
        telemetry["meta"]["same_pass_run_id"] = self.run_id
        return {"schema": "seethetruth-correlated-pass-v1", "run_id": self.run_id,
                "graph": joined, "telemetry": telemetry}

    def save(self, *, graph_path, telemetry_path, static_graph=None):
        report = self.report(static_graph=static_graph)
        Path(graph_path).write_text(json.dumps(report["graph"], ensure_ascii=False, indent=2))
        Path(telemetry_path).write_text(json.dumps(report["telemetry"], ensure_ascii=False, indent=2))
        return report
