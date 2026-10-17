"""Opt-in PyTorch profiler and bounded tensor sampling for SeetheTruth.

This module NEVER runs a model by itself. Caller chooses the workload to execute.
Profiler figures are observations of the instrumented pass, not uninstrumented
production latency. Tensor values are OFF by default; opt-in samples may expose data.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import suppress
from pathlib import Path

try:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
except ImportError as exc:
    raise ImportError("Telemetry requires PyTorch; static SeetheTruth does not.") from exc


def _walk_tensors(value, limit=6):
    found = []

    def walk(v, depth=0):
        if depth > 4 or len(found) >= limit:
            return
        if isinstance(v, torch.Tensor):
            found.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v[:limit]:
                walk(x, depth + 1)
        elif isinstance(v, dict):
            for x in list(v.values())[:limit]:
                walk(x, depth + 1)
    walk(value)
    return found


def _tensor_info(t):
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).removeprefix("torch."),
        "device": str(t.device),
        "numel": int(t.numel()),
        "element_size": int(t.element_size()),
        "logical_bytes": int(t.numel() * t.element_size()),
        "contiguous": bool(t.is_contiguous()),
        "requires_grad": bool(t.requires_grad),
    }


def _sample(t, cap=1024, matrix_side=20):
    """Representative bounded *sample*, never a full-tensor statistic.

    Explicit opt-in; copying to CPU synchronizes CUDA and changes profiling.
    Only module activations, never parameters, are visited.
    """
    if not t.is_floating_point() or not t.numel():
        return None
    with torch.no_grad():
        v = t.detach()
        step = max(1, math.ceil(v.numel() / cap))
        chosen = v.reshape(-1)[::step][:cap].to(device="cpu", dtype=torch.float32).tolist()
        finite = [float(x) for x in chosen if math.isfinite(float(x))]
        zeros = sum(float(x) == 0.0 for x in chosen)
        if not finite:
            stats = {"min": None, "max": None, "mean": None, "std": None}
        else:
            mu = sum(finite) / len(finite)
            stats = {"min": min(finite), "max": max(finite), "mean": mu,
                     "std": math.sqrt(sum((x-mu)**2 for x in finite)/len(finite))}
        bins = [0] * 24
        if finite:
            lo, hi = stats["min"], stats["max"]
            for x in finite:
                ix = min(23, int((x-lo)/(hi-lo)*24)) if hi > lo else 12
                bins[ix] += 1
        # Show one representative last-two-axis matrix slice. For higher rank
        # choose first indices of leading dimensions, explicitly label them.
        plane = v
        while plane.ndim > 2:
            plane = plane[0]
        if plane.ndim == 0:
            plane = plane.reshape(1, 1)
        elif plane.ndim == 1:
            plane = plane.unsqueeze(0)
        rows, cols = map(int, plane.shape)
        rstep = max(1, math.ceil(rows / matrix_side))
        cstep = max(1, math.ceil(cols / matrix_side))
        tile = plane[::rstep, ::cstep][:matrix_side, :matrix_side]
        matrix = tile.to(device="cpu", dtype=torch.float32).tolist()
    # JSON cannot encode NaN/Inf reliably; represent nonfinite cells as null.
    matrix = [[float(x) if math.isfinite(float(x)) else None for x in row] for row in matrix]
    return {
        "sampling": "strided_linear_sample_and_first_leading_indices_matrix",
        "sampled": len(chosen),
        "population": int(t.numel()),
        "finite_count": len(finite),
        "nonfinite_count": len(chosen) - len(finite),
        "sample_zero_fraction": zeros / len(chosen),
        "statistics": stats,
        "histogram": {"bins": bins, "domain": [stats["min"], stats["max"]]},
        "matrix": matrix,
        "matrix_original_shape": [rows, cols],
        "matrix_stride": [rstep, cstep],
        "leading_indices": [0] * max(0, t.ndim-2),
    }


def _event_number(evt, names):
    for key in names:
        try:
            value = getattr(evt, key)
            if value is not None:
                return float(value)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return 0.0


class TelemetrySession:
    """Profile one explicitly executed forward/backward/optimizer workload.

    >>> with TelemetrySession(model, capture_values=False) as stats:
    ...     output = model(inputs)  # caller executes actual work
    >>> stats.save("telemetry.json")

    CUDA device time derives from PyTorch profiler; no GPU utilization claim.
    Module wall time is HOST/enqueue time (CUDA launches are asynchronous).
    """

    def __init__(self, model, *, capture_values=False, max_snapshots=12,
                 max_sample_elements=1024, max_operator_rows=120):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a PyTorch Module")
        if not 1 <= max_snapshots <= 64 or not 32 <= max_sample_elements <= 4096:
            raise ValueError("invalid sampling bounds")
        if not 10 <= max_operator_rows <= 500:
            raise ValueError("invalid operator bound")
        self.model = model
        self.capture_values = bool(capture_values)
        self.max_snapshots = max_snapshots
        self.max_sample_elements = max_sample_elements
        self.max_operator_rows = max_operator_rows
        self._names = {id(m): (n or "<root>") for n, m in model.named_modules()}
        self._handles = []
        self._stacks = {}
        self._active_calls = []
        self._invocation_counts = {}
        self._module_rows = []
        self._snapshots = []
        self._profiler = None
        self._entered = False
        self._result = None
        self._started = None
        self._gpu = torch.cuda.is_available() and any(
            p.device.type == "cuda" for p in model.parameters()
        )
        self._memory_before = None

    def _before(self, module, args):
        name = self._names[id(module)]
        index = self._invocation_counts.get(name, 0) + 1
        self._invocation_counts[name] = index
        # The profiler tag identifies an individual invocation, not only an
        # aggregate module path. It is emitted by the hook inside this pass.
        tag = "SeetheTruth.module/" + name + "/invocation/" + str(index)
        key = name + "#" + str(index)
        parent = self._active_calls[-1] if self._active_calls else None
        start = time.perf_counter_ns()
        annotation = record_function(tag)
        annotation.__enter__()
        self._stacks.setdefault(id(module), []).append(
            (start, annotation, args, index, key, parent, tag))
        self._active_calls.append(key)

    def _after(self, module, args, output):
        stack = self._stacks.get(id(module), [])
        if not stack:
            return
        start, annotation, original_args, index, key, parent, tag = stack.pop()
        # Only well-nested, synchronous module hooks are supported. A later
        # exceptional/unbalanced forward cannot silently claim verified costs.
        if not self._active_calls or self._active_calls[-1] != key:
            raise RuntimeError("Unbalanced module profiling invocation stack")
        self._active_calls.pop()
        annotation.__exit__(None, None, None)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        name = self._names[id(module)]
        inputs = [_tensor_info(t) for t in _walk_tensors(original_args)]
        outs = [_tensor_info(t) for t in _walk_tensors(output)]
        self._module_rows.append({
            "name": name, "type": type(module).__name__, "host_wall_ms": elapsed,
            "host_timing_scope": "python_hook_and_enqueue_not_cuda_execution",
            "invocation_index": index, "invocation_key": key,
            "parent_invocation_key": parent, "annotation_label": tag,
            "inputs": inputs, "outputs": outs,
        })
        if self.capture_values and len(self._snapshots) < self.max_snapshots:
            for t in _walk_tensors(output, 2):
                if len(self._snapshots) >= self.max_snapshots:
                    break
                # Do not serialize trainable parameters (only module output).
                if any(t is p for p in module.parameters(recurse=False)):
                    continue
                if not t.is_floating_point() or not t.numel():
                    continue
                self._snapshots.append({
                    "module": name,
                    "tensor": _tensor_info(t),
                    "summary": _sample(t, self.max_sample_elements),
                })

    def __enter__(self):
        if self._entered or self._result is not None:
            raise RuntimeError("TelemetrySession is single-use")
        self._entered = True
        self._started = time.perf_counter()
        activities = [ProfilerActivity.CPU]
        if self._gpu:
            activities.append(ProfilerActivity.CUDA)
            self._memory_before = int(torch.cuda.memory_allocated())
        self._profiler = profile(activities=activities, record_shapes=True,
                                 profile_memory=True, with_stack=False)
        self._profiler.__enter__()
        try:
            for module in self.model.modules():
                self._handles.append(module.register_forward_pre_hook(self._before))
                self._handles.append(module.register_forward_hook(self._after))
        except BaseException:
            self._profiler.__exit__(None, None, None)
            for h in reversed(self._handles):
                h.remove()
            self._handles.clear()
            self._entered = False
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            for h in reversed(self._handles):
                h.remove()
            self._handles.clear()
            self._profiler.__exit__(exc_type, exc, tb)
            self._result = self._assemble()
        finally:
            self._entered = False
        return False

    def _attributed_module_events(self):
        """Use profiler CPU parent tree to assign each ATen *self* cost once.

        A child module's operators belong exclusively to that child; parent
        inclusive totals are derived separately. Device time is profiler
        attribution, not a device critical path or kernel timeline.
        """
        rows = {r["invocation_key"]: r for r in self._module_rows}
        markers = {r["annotation_label"]: r["invocation_key"] for r in self._module_rows}
        cost = {key: {"operator_events": 0, "self_cpu_us": 0.0,
                      "self_device_us": 0.0} for key in rows}
        unmatched = 0
        for event in self._profiler.events():
            name = str(getattr(event, "name", getattr(event, "key", "")))
            if not name.startswith("aten::"):
                continue
            parent = getattr(event, "cpu_parent", None)
            key = None
            hops = 0
            while parent is not None and hops < 150:
                candidate = str(getattr(parent, "name", getattr(parent, "key", "")))
                if candidate in markers:
                    key = markers[candidate]
                    break
                parent = getattr(parent, "cpu_parent", None)
                hops += 1
            if key is None:
                unmatched += 1
                continue
            record = cost[key]
            record["operator_events"] += 1
            record["self_cpu_us"] += _event_number(event, ("self_cpu_time_total",))
            if self._gpu:
                record["self_device_us"] += _event_number(
                    event, ("self_device_time_total", "self_cuda_time_total"))
        # Forward post-hooks append children before parents. Accumulate
        # inclusive work bottom-up in one pass, rather than O(n^2) searching.
        inclusive = {key: dict(value) for key, value in cost.items()}
        for key, row in rows.items():
            parent = row["parent_invocation_key"]
            if parent in inclusive:
                for metric in inclusive[key]:
                    inclusive[parent][metric] += inclusive[key][metric]
        result = {}
        for key, row in rows.items():
            incl = inclusive[key]
            result[key] = {
                "exclusive_operator_events": cost[key]["operator_events"],
                "inclusive_operator_events": incl["operator_events"],
                "exclusive_self_cpu_ms": round(cost[key]["self_cpu_us"] / 1000, 5),
                "inclusive_attributed_cpu_ms": round(incl["self_cpu_us"] / 1000, 5),
                "exclusive_self_device_ms": round(cost[key]["self_device_us"] / 1000, 5) if self._gpu else None,
                "inclusive_attributed_device_ms": round(incl["self_device_us"] / 1000, 5) if self._gpu else None,
                "profiler_scope": "aten_self_time_assigned_to_nearest_module_profiler_annotation",
            }
        return result, unmatched

    def _assemble(self):
        per_call_cost, unassigned = self._attributed_module_events()
        module_calls = []
        for row in self._module_rows:
            key = row["invocation_key"]
            module_calls.append({
                "name": row["name"], "type": row["type"],
                "invocation_index": row["invocation_index"],
                "invocation_key": key, "parent_invocation_key": row["parent_invocation_key"],
                "host_wall_ms": round(row["host_wall_ms"], 5),
                "last_inputs": row["inputs"], "last_outputs": row["outputs"],
                **per_call_cost[key],
            })
        operators = []
        for item in self._profiler.key_averages(group_by_input_shape=False):
            key = str(item.key)
            if key.startswith("SeetheTruth.module/"):
                continue
            count = int(getattr(item, "count", 0) or 0)
            cpu_us = _event_number(item, ("self_cpu_time_total",))
            dev_us = _event_number(item, ("self_device_time_total", "self_cuda_time_total"))
            operators.append({
                "name": key, "count": count,
                "self_cpu_ms": round(cpu_us / 1000, 5),
                "self_device_ms": round(dev_us / 1000, 5) if self._gpu else None,
                "self_cpu_memory_bytes": int(_event_number(item, ("self_cpu_memory_usage",))),
                "self_device_memory_bytes": int(_event_number(item, ("self_device_memory_usage", "self_cuda_memory_usage"))) if self._gpu else None,
            })
        operators.sort(key=lambda x: (
            x["self_device_ms"] if self._gpu else x["self_cpu_ms"]), reverse=True)
        module_agg = {}
        for row in self._module_rows:
            a = module_agg.setdefault(row["name"], {
                "name": row["name"], "type": row["type"], "calls": 0,
                "host_wall_ms": 0.0, "last_inputs": [], "last_outputs": []})
            a["calls"] += 1
            a["host_wall_ms"] += row["host_wall_ms"]
            a["last_inputs"] = row["inputs"]
            a["last_outputs"] = row["outputs"]
        for item in module_agg.values():
            item["host_wall_ms"] = round(item["host_wall_ms"], 5)
            rows = [r for r in module_calls if r["name"] == item["name"]]
            for metric in ("exclusive_self_cpu_ms", "inclusive_attributed_cpu_ms",
                           "exclusive_operator_events", "inclusive_operator_events"):
                item[metric] = round(sum(r[metric] for r in rows), 5)
            for metric in ("exclusive_self_device_ms", "inclusive_attributed_device_ms"):
                item[metric] = round(sum(r[metric] or 0 for r in rows), 5) if self._gpu else None
            item["profiler_scope"] = "per_invocation_aten_self_time_not_wall_latency"
        gpu = None
        if self._gpu:
            gpu = {"allocated_before_bytes": self._memory_before,
                   "allocated_after_bytes": int(torch.cuda.memory_allocated()),
                   "peak_allocated_bytes_since_last_reset": int(torch.cuda.max_memory_allocated())}
        return {
            "schema": "seethetruth-telemetry-v1",
            "name": type(self.model).__name__,
            "meta": {
                "measurement": "single_instrumented_pass",
                "operator_timing_source": "torch.profiler.key_averages",
                "module_timing_source": "host_python_wall_including_nested_calls",
                "host_timing_is_not_gpu_latency": True,
                "gpu_activity": self._gpu,
                "tensor_values_opt_in": self.capture_values,
                "sampling_may_synchronize_cuda_and_perturb_performance": self.capture_values,
                "bounded_samples_only": True,
                "small_activations_may_be_fully_sampled": self.capture_values,
                "parameter_values_not_intentionally_sampled": True,
                "operator_rows_truncated": len(operators) > self.max_operator_rows,
                "module_observations": len(self._module_rows),
                "module_cost_attribution": "nearest_module_profiler_annotation_exclusive_aten_self",
                "unattributed_operator_events": unassigned,
                "module_device_cost_is_not_kernel_critical_path": True,
                "module_inclusive_costs_overlap_and_must_not_be_summed": True,
                "wall_session_ms": round((time.perf_counter() - self._started)*1000, 4),
                "notes": [
                    "GPU device time is profiler attribution, not device utilization.",
                    "GPU peak memory is since last external reset, not necessarily this session.",
                    "Module host wall times include children; do not sum them.",
                    "All histogram and matrix statistics are bounded samples, not tensor-wide statistics."
                ],
            },
            "operators": operators[:self.max_operator_rows],
            "modules": sorted(module_agg.values(), key=lambda x:x["host_wall_ms"], reverse=True),
            "module_calls": module_calls,
            "snapshots": self._snapshots,
            "gpu_memory": gpu,
        }

    def report(self):
        if self._result is None:
            raise RuntimeError("Finish the profiling context before reading results")
        return self._result

    def save(self, path):
        p = Path(path)
        p.write_text(json.dumps(self.report(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return p
