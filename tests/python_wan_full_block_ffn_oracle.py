#!/usr/bin/env python3
"""Independent CPU numerical witness for source-ordered Wan FFN closure.

The native full graph covers Self + Cross + FFN. This reference independently
executes the FFN and compares full-loss gradients at its **after_cross**
boundary and all FFN operands. It does not claim an independently reexecuted
whole Self/Cross attention graph or official Wan weight equality.
"""
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

def close(label, observed, expected, tolerance=2e-10):
    a=np.asarray(observed,dtype=np.float64)
    b=np.asarray(expected,dtype=np.float64)
    if a.shape!=b.shape:
        raise AssertionError(f"{label} shape: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{label} has nonfinite values")
    delta=float(np.max(np.abs(a-b)))
    if delta>tolerance:
        raise AssertionError(f"{label}: maximum error {delta} > {tolerance}")
    return delta

def numpy_forward(operands):
    x,e3,e4,e5=(np.asarray(operands[k],dtype=np.float64)
                for k in ("after_cross","e3_sum","e4_sum","e5_sum"))
    w1=np.asarray(operands["ffn_w1"],dtype=np.float64)
    w2=np.asarray(operands["ffn_w2"],dtype=np.float64)
    b1=np.asarray(operands["ffn_b1"],dtype=np.float64)
    b2=np.asarray(operands["ffn_b2"],dtype=np.float64)
    target=np.asarray(operands["target"],dtype=np.float64)
    centered=x-np.mean(x,axis=-1,keepdims=True)
    normalized=centered/np.sqrt(np.mean(centered**2,axis=-1,keepdims=True)+1e-6)
    conditioned=normalized*(1+e4)+e3
    up=np.matmul(conditioned,w1)+b1
    active=.5*up*(1+np.tanh(np.sqrt(2/np.pi)*(up+.044715*up**3)))
    down=np.matmul(active,w2)+b2
    gated=down*e5
    output=x+gated
    return {
      "norm2_ffn":normalized,"ffn_conditioned":conditioned,
      "ffn_up_bias":up,"ffn_gelu_tanh":active.reshape(8,8),
      "ffn_down_bias":down,"ffn_gated":gated,"after_ffn":output,
      "loss":np.asarray([[np.mean((output-target)**2)]])
    }

def main():
    doc=json.load(sys.stdin)
    if doc.get("schema")!="seethetruth-wan-full-block-boundary-oracle-v1":
        raise AssertionError("unexpected witness schema")
    o=doc["operands"]
    expected_shapes={"after_cross":(1,1,8,6),"e3_sum":(1,1,8,6),
      "e4_sum":(1,1,8,6),"e5_sum":(1,1,8,6),"ffn_w1":(1,1,6,8),
      "ffn_w2":(1,1,8,6),"ffn_b1":(1,1,1,8),
      "ffn_b2":(1,1,1,6),"target":(1,1,8,6)}
    for k,shape in expected_shapes.items():
        if np.asarray(o[k]).shape!=shape:
            raise AssertionError(f"bad source shape {k}")
    n=numpy_forward(o)
    for key,actual in doc["native"].items():
        close("NumPy stage "+key,n[key],actual)
    tensors={k:torch.tensor(v,dtype=torch.float64,requires_grad=k!="target")
             for k,v in o.items()}
    t=tensors
    x=t["after_cross"]
    normalized=(x-x.mean(dim=-1,keepdim=True)) * torch.rsqrt(
        ((x-x.mean(dim=-1,keepdim=True))**2).mean(dim=-1,keepdim=True)+1e-6)
    conditioned=normalized*(1+t["e4_sum"])+t["e3_sum"]
    up=torch.matmul(conditioned,t["ffn_w1"])+t["ffn_b1"]
    activation=F.gelu(up,approximate="tanh")
    down=torch.matmul(activation,t["ffn_w2"])+t["ffn_b2"]
    gated=down*t["e5_sum"]
    output=x+gated
    loss=((output-t["target"])**2).mean()
    torch_stages={"norm2_ffn":normalized,"ffn_conditioned":conditioned,
       "ffn_up_bias":up,"ffn_gelu_tanh":activation.reshape(8,8),
       "ffn_down_bias":down,"ffn_gated":gated,
       "after_ffn":output,"loss":loss.reshape(1,1)}
    for key,actual in doc["native"].items():
        close("Torch stage "+key,torch_stages[key].detach().numpy(),actual)
    loss.backward()
    coords=0
    for k,reference in doc["nativeGradients"].items():
        if t[k].grad is None:
            raise AssertionError("missing PyTorch gradient "+k)
        close("cross-block gradient "+k,t[k].grad.detach().numpy(),reference)
        coords+=t[k].numel()
    # Independent central differences via the NumPy path, for one
    # representative element of every distinct gradient tensor.
    maximum_fd=0.
    for k in doc["nativeGradients"]:
        trial={field:np.asarray(value,dtype=np.float64).copy() for field,value in o.items()}
        shape=trial[k].shape
        index=tuple(dim//2 for dim in shape)
        step=1e-6
        trial[k][index]+=step
        plus=float(numpy_forward(trial)["loss"][0,0])
        trial[k][index]-=2*step
        minus=float(numpy_forward(trial)["loss"][0,0])
        finite=(plus-minus)/(2*step)
        analytical=float(np.asarray(doc["nativeGradients"][k])[index])
        err=abs(finite-analytical)
        maximum_fd=max(maximum_fd,err)
        if err>2e-7:
            raise AssertionError(f"independent finite difference {k}: {finite} != {analytical}")
    if torch.__version__!="2.6.0+cpu" or str(x.device)!="cpu":
        raise AssertionError("unexpected Torch runtime version/device")
    print("WAN_FULL_BLOCK_BOUNDARY_CPU_PASS "+json.dumps({
      "stageCount":len(doc["native"]), "wholeLossGradientCoordinates":coords,
      "independentNumpyFiniteDifferences":len(doc["nativeGradients"]),
      "maxFiniteDifferenceError":maximum_fd,
      "dtype":"torch.float64","torch":torch.__version__,
      "claim":"native full connected block forward/VJP; independently executed FFN closure, not full Self/Cross external backend"
    },allow_nan=False))

if __name__=="__main__":
    main()
