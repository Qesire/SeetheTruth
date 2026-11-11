#!/usr/bin/env python3
"""Independent NumPy and PyTorch Float64 oracle for an actual VLM graph."""
import json
import math
import sys
import numpy as np
import torch

def check(name,a,b,tol=2e-11):
    x=np.asarray(a,dtype=np.float64)
    y=np.asarray(b,dtype=np.float64)
    if x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise AssertionError(f"{name}: invalid shape or nonfinite values")
    diff=float(np.max(np.abs(x-y)))
    if diff>tol:
        raise AssertionError(f"{name}: max_abs_error={diff}")
    return diff

def evaluate(case):
    q=np.asarray(case["q"],dtype=np.float64)
    k=np.asarray(case["k"],dtype=np.float64)
    v=np.asarray(case["v"],dtype=np.float64)
    topk=case["topk"]
    gridw=case["gridWidth"]
    # Independent NumPy math.
    score=(q@k.T)/math.sqrt(q.shape[1])
    ids=np.argsort(-score,axis=1,kind="stable")[:,:topk]
    def sm(a):
        shifted=a-np.max(a,axis=1,keepdims=True)
        ex=np.exp(shifted)
        return ex/np.sum(ex,axis=1,keepdims=True)
    dense=sm(score)@v
    sparse=np.einsum("tk,tkd->td",sm(np.take_along_axis(score,ids,axis=1)),v[ids])
    coords=np.stack((ids//gridw,ids%gridw),axis=-1).reshape(-1,2)
    delta=dense-sparse
    mse=float(np.mean(delta**2))
    # Independently programmed PyTorch CPU math.
    tq=torch.tensor(case["q"],dtype=torch.float64)
    tk=torch.tensor(case["k"],dtype=torch.float64)
    tv=torch.tensor(case["v"],dtype=torch.float64)
    ts=tq@tk.T/math.sqrt(tq.shape[1])
    ti=torch.argsort(ts,dim=1,descending=True,stable=True)[:,:topk]
    td=torch.softmax(ts,dim=-1)@tv
    tw=torch.softmax(torch.gather(ts,1,ti),dim=1)
    tx=torch.einsum("tk,tkd->td",tw,tv[ti])
    tc=torch.stack((torch.div(ti,gridw,rounding_mode="floor"),torch.remainder(ti,gridw)),dim=-1).reshape(-1,2)
    if ids.tolist()!=ti.tolist() or ids.tolist()!=case["js"]["indices"]:
        raise AssertionError("TopK index mismatch across three engines")
    if coords.tolist()!=tc.tolist() or coords.tolist()!=case["js"]["coordinates"]:
        raise AssertionError("Patch-grid location mismatch across three engines")
    errors=[]
    for label,a,b,js in [
        ("scores",score,ts.numpy(),case["js"]["scores"]),
        ("dense",dense,td.numpy(),case["js"]["dense"]),
        ("sparse",sparse,tx.numpy(),case["js"]["sparse"]),
        ("delta",delta,(td-tx).numpy(),case["js"]["delta"]),
        ("mse",[mse],[float(torch.mean((td-tx)**2).item())],[case["js"]["mse"]])
    ]:
        errors.extend([check(label+"/torch",a,b),check(label+"/js",a,js)])
    R=float(max(np.linalg.norm(row) for row in v))
    dropped=[]
    all_weights=sm(score)
    for i in range(q.shape[0]):
        mass=max(0,1-float(sum(all_weights[i,j] for j in ids[i])))
        if float(np.linalg.norm(delta[i]))>2*mass*R+2e-11:
            raise AssertionError("discarded mass error bound rejected")
        dropped.append(mass)
    if topk==k.shape[0] and not np.allclose(dense,sparse,atol=2e-11,rtol=0):
        raise AssertionError("TopK=all not dense-equivalent")
    return {"topk":topk,"droppedMass":dropped,"mse":mse,"maxAbsError":max(errors)}

if __name__=="__main__":
    cases=json.load(sys.stdin)
    if [c["topk"] for c in cases]!=[2,4]:
        raise AssertionError("expected TopK=2 and TopK=4 arms")
    result=[evaluate(c) for c in cases]
    print("VLM_CROSS_BACKEND_PASS "+json.dumps({"cases":result,"scope":"independent NumPy/PyTorch CPU Float64, actual JS graph values"}))
