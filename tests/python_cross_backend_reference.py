#!/usr/bin/env python3
"""Third-oracle comparison of the actual SeetheTruth JS graph and emitted NumPy/Torch.

The reference is explicit scalar indexing, NOT nd_tensor/nd_autodiff,
the shared portable lowering emitter, NumPy broadcasting, or Torch autograd.
The two generated programs execute in separate Python namespaces.
"""
import copy
import json
import math
import sys


def shape_of(value):
    if not isinstance(value, list):
        return ()
    if not value:
        raise AssertionError("empty tensor axis")
    first = shape_of(value[0])
    if any(shape_of(v) != first for v in value[1:]):
        raise AssertionError("ragged observed tensor")
    return (len(value),) + first


def compare(expected, observed, label, atol=3e-9, rtol=1e-10):
    if shape_of(expected) != shape_of(observed):
        raise AssertionError(f"{label} shape {shape_of(observed)} != {shape_of(expected)}")
    checked = 0
    maximum = 0.0

    def walk(a, b, index):
        nonlocal checked, maximum
        if isinstance(a, list):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, index + (i,))
            return
        if not isinstance(b, (int, float)) or not math.isfinite(b):
            raise AssertionError(f"{label}{index} nonfinite or nonnumeric")
        diff = abs(float(a) - float(b))
        maximum = max(maximum, diff)
        if diff > atol + rtol * abs(float(a)):
            raise AssertionError(
                f"{label}{index} mismatch: expected={a}, observed={b}, error={diff}"
            )
        checked += 1

    walk(expected, observed, ())
    return checked, maximum


def scalar_reference(data):
    B, H, T, C = data["shape"]
    x, bias, gain, target = (data["sources"][key] for key in
                            ("x", "bias", "gain", "target"))
    assert shape_of(x) == (B, H, T, C)
    assert shape_of(bias) == shape_of(gain) == (1, 1, 1, C)
    assert shape_of(target) == (B, H, 1, C)
    u = [[[[x[b][h][t][c] + bias[0][0][0][c]
             for c in range(C)] for t in range(T)]
           for h in range(H)] for b in range(B)]
    v = [[[[u[b][h][t][c] * gain[0][0][0][c]
             for c in range(C)] for t in range(T)]
           for h in range(H)] for b in range(B)]
    s = [[[[sum(v[b][h][t][c] for t in range(T))
            for c in range(C)]]
          for h in range(H)] for b in range(B)]
    loss = [[sum((s[b][h][0][c] - target[b][h][0][c]) ** 2
                 for b in range(B) for h in range(H) for c in range(C))
             / (B * H * C)]]
    ds = [[[[2 * (s[b][h][0][c] - target[b][h][0][c]) / (B * H * C)
             for c in range(C)]]
           for h in range(H)] for b in range(B)]
    dv = [[[[ds[b][h][0][c] for c in range(C)]
            for t in range(T)] for h in range(H)] for b in range(B)]
    du = [[[[dv[b][h][t][c] * gain[0][0][0][c]
            for c in range(C)] for t in range(T)]
           for h in range(H)] for b in range(B)]
    dbias = [[[[sum(du[b][h][t][c]
                     for b in range(B) for h in range(H) for t in range(T))
                for c in range(C)]]]]
    dgain = [[[[sum(dv[b][h][t][c] * u[b][h][t][c]
                     for b in range(B) for h in range(H) for t in range(T))
                for c in range(C)]]]]
    return {
        "forward": {"x": x, "bias": bias, "gain": gain, "target": target,
                    "biased": u, "gated": v, "summed": s, "loss": loss},
        "gradients": {"x": du, "bias": dbias, "gain": dgain,
                      "biased": du, "gated": dv, "summed": ds},
    }


def numpy_run(source):
    scope = {"__name__": "standalone_numpy_graph"}
    exec(compile(source, "<generated-numpy>", "exec"), scope)
    output = scope["run_graph"]()
    import numpy as np
    assert all(v.dtype == np.float64 for v in output.values())
    return {k: v.tolist() for k, v in output.items()}


def torch_run(source):
    scope = {"__name__": "standalone_torch_graph"}
    exec(compile(source, "<generated-torch>", "exec"), scope)
    import torch
    assert torch.__version__ == "2.6.0+cpu", "pin the independent CPU oracle"
    torch.set_num_threads(1)
    model = scope["GraphRuntime"]()
    output = model.forward()
    for node in output.values():
        assert node.dtype == torch.float64
        if node.requires_grad:
            node.retain_grad()
    output["loss"].backward()
    return (
        {k: v.detach().tolist() for k, v in output.items()},
        {k: v.grad.detach().tolist() for k, v in output.items()
         if v.grad is not None},
    )


def run_case(data):
    name = data["id"]
    expected = scalar_reference(data)
    np_output = numpy_run(data["code"]["numpy"])
    torch_output, torch_grad = torch_run(data["code"]["torch"])
    observed = {
        "js": (data["js"]["forward"], data["js"]["gradients"]),
        "numpy": (np_output, None),
        "torch": (torch_output, torch_grad),
    }
    counts = {}
    worst = {}
    for backend, (forward, gradients) in observed.items():
        count = 0
        maximum = 0.0
        for phase, want, got in (
            ("forward", expected["forward"], forward),
            ("gradient", expected["gradients"], gradients),
        ):
            if got is None:
                continue  # NumPy emitter is forward-only; not a zero gradient.
            for node, values in want.items():
                if node not in got:
                    raise AssertionError(f"{name}/{backend}/{phase}/{node} missing")
                n, err = compare(values, got[node],
                                 f"{name}/{backend}/{phase}/{node}")
                count += n
                maximum = max(maximum, err)
        counts[backend] = count
        worst[backend] = maximum
    # Mutation of the comparison input must be rejected, proving the heavy
    # harness cannot pass just because both generated backends execute.
    corrupted = copy.deepcopy(np_output["gated"])
    corrupted[0][0][0][0] += 1e-2
    try:
        compare(expected["forward"]["gated"], corrupted,
                name + "/injected_backend_error")
    except AssertionError as error:
        if "mismatch" not in str(error):
            raise
    else:
        raise AssertionError("cross-backend oracle failed its mutation sentinel")
    return {"id": name, "coordinates": counts, "maxAbsError": worst}


def main():
    payload = json.load(sys.stdin)
    results = [run_case(data) for data in payload]
    if len(results) != 2 or len({x["id"] for x in results}) != 2:
        raise AssertionError("missing independently specified 4D fixture")
    print("CROSS_BACKEND_SCALAR_ORACLE_PASS " +
          json.dumps({"cases": results, "oracle": "separate explicit scalar Python",
                      "backends": ["seethetruth_js", "numpy_float64", "torch_2.6_cpu_float64"],
                      "scope": "two exact 4D source graphs"}, sort_keys=True))


if __name__ == "__main__":
    main()
