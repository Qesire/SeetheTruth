#!/usr/bin/env python3
"""Independent scalar 5D witness plus actual NumPy/PyTorch Float64 programs.

Native JS SemanticState is exercised by semantic_5d_source_oracle.js.
This program uses separate scalar loops, then backend-native operations and
Torch autograd. It does not represent emitted graph codegen for named-axis
pack kinds, or industrial Wan/video kernel certification.
"""
import json
import math
import sys

import numpy as np
import torch

SHAPE = (1, 3, 2, 2, 2)
AXES = ["B", "T", "H", "W", "C"]


def close(label, actual, expected, eps=2e-11):
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(expected, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{label}: nonfinite coordinate")
    err = float(np.max(np.abs(a - b)))
    if err > eps:
        raise AssertionError(f"{label}: maximum abs error {err} > {eps}")
    return err


def pure_scalar(video, bias, gain):
    """Independent flattened (t,h,w,c) equations without numpy operations."""
    bvals, yvals = [], []
    for t in range(3):
        for h in range(2):
            for w in range(2):
                for c in range(2):
                    i = ((t * 2 + h) * 2 + w) * 2 + c
                    v = video[i] + bias[c]
                    bvals.append(v)
                    yvals.append(v * gain[t])
    loss = sum(v * v for v in yvals) / len(yvals)
    dv = [2 * yvals[i] * gain[i // 8] / len(yvals) for i in range(len(yvals))]
    db = [sum(dv[((t * 2 + h) * 2 + w) * 2 + c]
              for t in range(3) for h in range(2) for w in range(2))
          for c in range(2)]
    dg = [sum(2 * yvals[((t * 2 + h) * 2 + w) * 2 + c]
              * bvals[((t * 2 + h) * 2 + w) * 2 + c] / len(yvals)
              for h in range(2) for w in range(2) for c in range(2))
          for t in range(3)]
    return {"biased": bvals, "gated": yvals, "loss": loss,
            "gradients": {"video": dv, "bias": db, "gain": dg}}


def finite_difference(video, bias, gain, name, index, step=1e-6):
    data = {"video": video[:], "bias": bias[:], "gain": gain[:]}
    data[name][index] += step
    upper = pure_scalar(**data)["loss"]
    data[name][index] -= 2 * step
    lower = pure_scalar(**data)["loss"]
    return (upper - lower) / (2 * step)


def main():
    doc = json.load(sys.stdin)
    if doc["schema"] != "seethetruth-5d-semantic-numerical-v1":
        raise AssertionError("unexpected payload schema")
    if tuple(doc["shape"]) != SHAPE or doc["axes"] != AXES:
        raise AssertionError("5D case identity mismatch")
    x, b, g = doc["video"], doc["bias"], doc["gain"]
    if len(x) != 24 or len(b) != 2 or len(g) != 3:
        raise AssertionError("shape or parameter arity mismatch")
    reference = pure_scalar(x, b, g)
    for key in ("biased", "gated"):
        close("native " + key, doc["native"][key], reference[key])
        close("JS scalar " + key, doc["independent"][key], reference[key])
    close("native slice/concat", doc["native"]["rejoined"], reference["gated"])
    close("native reorder round trip", doc["native"]["restored"], reference["gated"])
    close("JS scalar loss", [doc["independent"]["loss"]], [reference["loss"]])
    for key, expected in reference["gradients"].items():
        close("JS analytic " + key, doc["independent"]["gradients"][key], expected)

    # NumPy: executed independently from SemanticState's index iteration.
    arr = np.asarray(x, dtype=np.float64).reshape(SHAPE)
    nb = np.asarray(b, dtype=np.float64).reshape(1, 1, 1, 1, 2)
    ng = np.asarray(g, dtype=np.float64).reshape(1, 3, 1, 1, 1)
    np_biased = arr + nb
    np_gated = np_biased * ng
    np_left, np_right = np_gated[:, :2], np_gated[:, 2:]
    np_joined = np.concatenate((np_left, np_right), axis=1)
    np_permuted = np.transpose(np_joined, (0, 4, 2, 3, 1))
    np_restored = np.transpose(np_permuted, (0, 4, 2, 3, 1))
    for key, actual, expected in (
        ("biased", np_biased, reference["biased"]),
        ("gated", np_gated, reference["gated"]),
        ("concat", np_joined, reference["gated"]),
        ("permuted", np_permuted, doc["native"]["permuted"]),
        ("restore", np_restored, reference["gated"])):
        close("NumPy " + key, actual, expected)
    if tuple(np_permuted.shape) != (1, 2, 2, 2, 3):
        raise AssertionError("NumPy permutation shape drift")

    # PyTorch: separate real Float64 execution and autograd through EVERY
    # operator, not through JS gradient formulas or shared NumPy functions.
    tx = torch.tensor(x, dtype=torch.float64).reshape(SHAPE).requires_grad_()
    tb = torch.tensor(b, dtype=torch.float64).requires_grad_()
    tg = torch.tensor(g, dtype=torch.float64).requires_grad_()
    tbias = tx + tb.reshape(1, 1, 1, 1, 2)
    ty = tbias * tg.reshape(1, 3, 1, 1, 1)
    pieces = torch.split(ty, [2, 1], dim=1)
    joined = torch.cat(pieces, dim=1)
    permuted = joined.permute(0, 4, 2, 3, 1)
    restored = permuted.permute(0, 4, 2, 3, 1)
    loss = (restored * restored).mean()
    loss.backward()
    for key, actual, expected in (
        ("biased", tbias.detach().numpy(), reference["biased"]),
        ("gated", ty.detach().numpy(), reference["gated"]),
        ("concat", joined.detach().numpy(), reference["gated"]),
        ("permuted", permuted.detach().numpy(), doc["native"]["permuted"]),
        ("restore", restored.detach().numpy(), reference["gated"])):
        close("Torch " + key, actual, expected)
    close("Torch loss", [loss.item()], [reference["loss"]])
    coords = 0
    max_fd_error = 0.
    for key, actual in (("video", tx.grad), ("bias", tb.grad), ("gain", tg.grad)):
        close("Torch complete VJP " + key,
              actual.detach().numpy(), reference["gradients"][key])
        for index, analytical in enumerate(reference["gradients"][key]):
            fd = finite_difference(x, b, g, key, index)
            err = abs(fd - analytical)
            max_fd_error = max(max_fd_error, err)
            if err > 4e-8:
                raise AssertionError(f"FD {key}[{index}] {fd} != {analytical}")
            coords += 1
    if torch.__version__ != "2.6.0+cpu" or str(tx.device) != "cpu":
        raise AssertionError("backend version/device are not the pinned witness")
    print("SEMANTIC_5D_BACKENDS_EXECUTED_PASS " + json.dumps({
        "shape": list(SHAPE), "numpy": np.__version__, "torch": torch.__version__,
        "device": str(tx.device), "dtype": str(tx.dtype),
        "forwardChecks": 10, "wholeLossGradients": coords,
        "independentFiniteDifferences": coords, "maxFiniteDifferenceError": max_fd_error
    }, allow_nan=False))


if __name__ == "__main__":
    main()
