"""Opt-in PyTorch runtime observation for SeetheTruth.

This module NEVER executes a model by itself. Import it in a script you control.
No tensor values, gradients, checkpoint bytes, environment variables, or pointers
are serialized. Captures ATen dispatcher calls (not GPU kernel launches).
"""
from __future__ import annotations

import json
import weakref
from pathlib import Path
from source_runtime_join import SourceLocator

try:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError as exc:
    raise ImportError("Runtime tracing requires PyTorch; static analysis does not.") from exc


def _tensors(value, limit=12):
    """Bounded traversal; never materializes a tensor."""
    result = []
    def walk(obj, depth=0):
        if len(result) >= limit or depth > 5:
            return
        if isinstance(obj, torch.Tensor):
            result.append(obj)
        elif isinstance(obj, (tuple, list)):
            for item in obj[:limit]:
                walk(item, depth + 1)
        elif isinstance(obj, dict):
            for item in list(obj.values())[:limit]:
                walk(item, depth + 1)
    walk(value)
    return result


def _meta(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": tensor.device.type + (":" + str(tensor.device.index) if tensor.device.index is not None else ""),
        "requires_grad": bool(tensor.requires_grad),
        "contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
    }


def _shape_text(items):
    return " / ".join(
        "[" + ",".join(map(str, item["shape"])) + "] " + item["dtype"]
        for item in items[:4]
    ) or "∅"


# Operator semantics are established library math, not recovered model design intent.
# We never assign a Q/K/V or FFN role merely because an ATen matrix multiply appeared.
_MATH = {
    "mm": ("Y = A @ B", "2-D matrix multiplication"),
    "bmm": ("Y[b] = A[b] @ B[b]", "batched matrix multiplication"),
    "matmul": ("Y = matmul(A,B)", "broadcast-aware matrix multiplication"),
    "addmm": ("Y = beta*C + alpha*(A @ B)", "affine matrix multiplication"),
    "linear": ("Y = X @ W.T + b (optional)", "learned affine transform"),
    "view": ("Y = view(X,new_shape)", "view with layout constraints"),
    "reshape": ("Y = reshape(X,new_shape)", "logical shape change, possible copy"),
    "permute": ("Y = permute(X,axes)", "reorders logical tensor axes"),
    "transpose": ("Y = transpose(X,axis1,axis2)", "swaps two logical axes"),
    "contiguous": ("Y = contiguous(X)", "layout/materialization boundary"),
    "cat": ("Y = concat(X_i,dim)", "concatenates tensors on an axis"),
    "_softmax": ("softmax(X,dim)", "normalized exponential weights"),
    "softmax": ("softmax(X,dim)", "normalized exponential weights"),
    "gelu": ("Y = GELU(X)", "elementwise GELU activation"),
    "silu": ("Y = X*sigmoid(X)", "elementwise SiLU activation"),
    "mul": ("Y = A * B", "elementwise multiplication (broadcasting possible)"),
    "add": ("Y = A + B", "elementwise addition (broadcasting possible)"),
    "div": ("Y = A / B", "elementwise division"),
    "sum": ("Y = sum(X, axes)", "tensor reduction"),
    "mean": ("Y = mean(X, axes)", "tensor reduction"),
    "masked_fill": ("Y = where(mask, value, X)", "mask application"),
    "embedding": ("Y = table[index]", "indexed lookup, not dense GEMM"),
    "layer_norm": ("Y = gamma*(X-mean)/sqrt(var+eps)+beta", "channel normalization"),
    "native_layer_norm": ("Y = normalized X (plus statistics)", "layer normalization and stats"),
    "detach": ("Y = stop_gradient(X)", "autograd history boundary"),
}


class _DispatchRecorder(TorchDispatchMode):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Run exactly once; do not clone/inspect values or force device sync.
        output = func(*args, **kwargs)
        self.trace._operation(str(func), (args, kwargs), output)
        return output


class RuntimeTrace:
    """Bounded experimental recorder.

    Usage:
        with RuntimeTrace(model, max_events=500) as tr:
            loss = model(batch).square().mean()
            tr.backward(loss)
            tr.step(optimizer)  # optional; calls optimizer.step() exactly once
        tr.save("trace.json")

    Dispatch observation is opt-in and can change performance/graph behavior.
    Use a small representative step before applying it to a large model.
    """

    def __init__(self, model, *, max_events=500, record_gradients=True,
                 source_root=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be torch.nn.Module")
        if not 16 <= max_events <= 10000:
            raise ValueError("max_events must be between 16 and 10000")
        self.model = model
        self.max_events = max_events
        self.record_gradients = bool(record_gradients)
        self._locator = SourceLocator(source_root) if source_root is not None else None
        self._module_invocations = {}
        self.nodes = []
        self.edges = []
        self._stack = []
        self._handles = []
        self._producer = {}
        self._sequence = 0
        self._dispatch = None
        self._phase = "forward"
        self._entered = False
        self._dropped = 0
        self._model_name = type(model).__name__
        self._module_names = {id(module): (name or self._model_name)
                              for name, module in model.named_modules()}
        self._parameter_names = {id(p): name for name, p in model.named_parameters()}
        self._root_id = None
        self._parameter_grad_ids = {}

    def _add(self, label, category, *, parent=None, inputs=(), outputs=(), **fields):
        if len(self.nodes) >= self.max_events:
            self._dropped += 1
            return None
        self._sequence += 1
        ident = "rt%05d" % self._sequence
        node = {
            "id": ident,
            "label": label[:90],
            "type": category,
            "parent": parent,
            "input": _shape_text(inputs),
            "output": _shape_text(outputs),
            "purpose": fields.pop("purpose", "Observed operation; mathematical interpretation requires operator semantics."),
            "formula": fields.pop("formula", label),
            "certainty": "runtime_observed",
            "phase": self._phase,
            "tensor": {"inputs": list(inputs), "outputs": list(outputs)},
            **fields,
        }
        self.nodes.append(node)
        return ident

    def _producer_for(self, tensor):
        pair = self._producer.get(id(tensor))
        return pair[1] if pair and pair[0]() is tensor else None

    def _bind(self, tensor, ident):
        if ident is not None:
            self._producer[id(tensor)] = (weakref.ref(tensor), ident)
        if len(self._producer) > self.max_events * 6:
            self._producer = {k: v for k, v in self._producer.items() if v[0]() is not None}

    def _link(self, src, target, kind="data"):
        if src and target and src != target:
            self.edges.append({"from": src, "to": target, "kind": kind})

    def _track_grad(self, tensor, forward_id, parent):
        if not (self.record_gradients and tensor.requires_grad):
            return
        # Only selected matrix outputs: avoids hooks on every temporary.
        def grad_hook(grad):
            meta = [_meta(grad)]
            ident = self._add("∇ " + str(forward_id), "gate", parent=parent,
                              inputs=meta, outputs=meta, formula="∂L/∂output",
                              purpose="Observed gradient for this forward matrix output.",
                              linked_forward=forward_id, phase="backward")
            self._link(forward_id, ident, "feedback")
            return grad
        self._handles.append(tensor.register_hook(grad_hook))

    def _operation(self, op, args, result):
        inputs = _tensors(args)
        outputs = _tensors(result)
        in_meta = [_meta(t) for t in inputs]
        out_meta = [_meta(t) for t in outputs]
        parent = self._stack[-1] if self._stack else self._root_id
        label = op.replace("aten.", "").split(".")[0]
        key = label.lower()
        is_matmul = key in {"mm", "addmm", "bmm", "matmul", "linear"}
        category = "attn" if is_matmul else ("norm" if key in {"view", "reshape", "permute", "transpose", "contiguous"} else "ffn")
        formula, intent = _MATH.get(key, (op, "ATen operator executed; semantics not annotated"))
        eid = self._add(label + " · " + self._phase, category, parent=parent,
                        inputs=in_meta, outputs=out_meta, formula=formula,
                        purpose=intent,
                        explanation_certainty=("operator_semantic_rule" if key in _MATH else "unannotated"),
                        aten_op=op,
                        **({"source": self._locator.capture()} if self._locator and
                           self._phase == "forward" else {}))
        if eid is None:
            return
        for t in inputs:
            self._link(self._producer_for(t), eid)
        for t in outputs:
            self._bind(t, eid)
            if is_matmul and self._phase == "forward":
                self._track_grad(t, eid, parent)

    def _pre(self, module, inputs):
        parent = self._stack[-1] if self._stack else None
        name = self._module_names.get(id(module), type(module).__name__)
        self._module_invocations[name] = self._module_invocations.get(name, 0) + 1
        ident = self._add(name, "head", parent=parent,
                          inputs=[_meta(t) for t in _tensors(inputs)],
                          formula=type(module).__name__ + ".forward",
                          purpose="Module forward invocation observed at runtime.",
                          module=name, module_type=type(module).__name__,
                          invocation_index=self._module_invocations[name],
                          **({"source": self._locator.capture()} if self._locator else {}))
        if not self._stack and self._root_id is None:
            self._root_id = ident
        self._stack.append(ident)

    def _post(self, module, inputs, output):
        ident = self._stack.pop() if self._stack else None
        if ident:
            node = next((n for n in reversed(self.nodes) if n["id"] == ident), None)
            if node:
                metas = [_meta(t) for t in _tensors(output)]
                node["output"] = _shape_text(metas)
                node["tensor"]["outputs"] = metas

    def __enter__(self):
        if self._entered:
            raise RuntimeError("trace cannot be re-entered")
        self._entered = True
        for module in self.model.modules():
            self._handles.append(module.register_forward_pre_hook(self._pre))
            self._handles.append(module.register_forward_hook(self._post))
        if self.record_gradients:
            for p in self.model.parameters():
                if not p.requires_grad:
                    continue
                pname = self._parameter_names[id(p)]
                def record_parameter_grad(g, name=pname):
                    mid = self._add("∇ " + name, "gate", parent=self._root_id,
                                    inputs=[_meta(g)], outputs=[_meta(g)],
                                    purpose="Observed parameter gradient hook.",
                                    formula="∂L/∂" + name,
                                    parameter=name, phase="backward")
                    if mid is not None:
                        self._parameter_grad_ids[name] = mid
                    return g
                self._handles.append(p.register_hook(record_parameter_grad))
        self._dispatch = _DispatchRecorder(self)
        self._dispatch.__enter__()
        return self

    def backward(self, loss, *args, **kwargs):
        if not self._entered:
            raise RuntimeError("use backward inside a trace session")
        prev = self._phase
        self._phase = "backward"
        try:
            return loss.backward(*args, **kwargs)
        finally:
            self._phase = prev

    def step(self, optimizer, *args, **kwargs):
        """Execute precisely one caller-requested optimizer step and note version changes."""
        if not self._entered:
            raise RuntimeError("use step inside a trace session")
        versions = {id(p): p._version for group in optimizer.param_groups for p in group["params"]}
        prev = self._phase
        self._phase = "optimizer"
        try:
            result = optimizer.step(*args, **kwargs)
            changed = [self._parameter_names.get(id(p), "<unregistered>")
                       for group in optimizer.param_groups for p in group["params"]
                       if p._version != versions[id(p)]]
            step_id = self._add("Optimizer " + type(optimizer).__name__, "head",
                      parent=self._root_id, formula=type(optimizer).__name__ + ".step",
                      purpose="Explicit optimizer.step() completed; modified parameter versions observed.",
                      optimizer=type(optimizer).__name__,
                      updated_parameters=changed[:64], updated_count=len(changed))
            for name in changed:
                self._link(self._parameter_grad_ids.get(name), step_id, "data")
            return result
        finally:
            self._phase = prev

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._dispatch:
                self._dispatch.__exit__(exc_type, exc, tb)
        finally:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            self._entered = False
        return False

    def graph(self):
        return {
            "schema": "model-atlas-graph-v1",
            "name": "RUNTIME · " + self._model_name,
            "meta": {
                "analysis": "runtime-torch-dispatch",
                "trace_schema": "seethetruth-runtime-v1",
                "runtime_validated": True,
                "scope": "one instrumented Python process / opted-in execution",
                "captures": ["ATen dispatched operations", "module hooks",
                             "selected matrix output gradients", "parameter gradients",
                             "explicit optimizer step"],
                "not_captured": ["GPU kernel launches", "GPU timings", "tensor values",
                                 "unexecuted branches", "all autograd graph edges",
                                 "untraced optimizers or external processes"],
                "truncated": bool(self._dropped),
                "source_capture": {
                    "enabled": self._locator is not None,
                    "mode": "opt_in_source_root_exact_line_and_sha256",
                    "files_hashed": len(self._locator.digests) if self._locator else 0,
                    "unlocated_frames": self._locator.omitted if self._locator else 0,
                    "source_root_serialized": False,
                    "capture_has_overhead": self._locator is not None,
                },
                "dropped_events": self._dropped,
                "recorded_nodes": len(self.nodes),
            },
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def save(self, path):
        data = self.graph()
        p = Path(path)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
