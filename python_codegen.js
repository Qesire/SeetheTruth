/* Compile a validated, executable SeetheTruth tensor DAG to standalone NumPy Python.
 * Unlike source dumping or a screenshot, the output computes each node at runtime.
 * Unsupported operators fail CLOSED. Source tensors are overrideable by node ID.
 */
(function(root, factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthPythonExport=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const node=typeof module==="object"&&module.exports;
const L=node?require("./learning_lab.js"):root.SeetheTruthLearning;
const Branch=node?require("./branch_execution.js"):root.SeetheTruthBranchExecution;
const SOURCES=new Set(["input","parameter","nd_input","nd_parameter"]);
const SUPPORTED=new Set([
 ...SOURCES,"linear","matmul","add","mul","subtract","divide","transpose2d",
 "rmsnorm","layernorm","softmax","attention","mse","silu","relu","gelu_tanh","gelu_erf",
 "sigmoid","tanh","softplus","scale","bias","square","swiglu","geglu",
 "relu_squared","mish","leaky_relu","elu","hardswish",
 "nd_permute","nd_reshape","nd_flatten","nd_expand4d","nd_broadcast_add",
 "nd_reduce_sum","nd_batched_matmul","nd_scale","nd_softmax_last","nd_causal_mask",
 "nd_repeat_kv","nd_mse","nd_rope","nd_cache_read","nd_cache_append",
 "nd_silu","nd_relu_squared","nd_mish","nd_gelu_erf","nd_elu","nd_hardswish",
 "opt_adamw","opt_adam","opt_adagrad","opt_sgd_momentum","opt_rmsprop","opt_lion"
]);
const encoded=x=>JSON.stringify(JSON.stringify(x));
const py=x=>typeof x==="number"?(Number.isFinite(x)?String(x):(()=>{throw Error("nonfinite literal")})()):JSON.stringify(x);
function weight(id,r,c){
 let h=2166136261;
 for(const char of id+":"+r+":"+c)h=Math.imul(h^char.charCodeAt(0),16777619);
 return ((h>>>0)%2001-1000)/2600;
}
const PRELUDE=String.raw`# Auto-generated from a validated SeetheTruth tensor DAG.
# Python >= 3.10; dependency: pip install numpy
# CPU Float64 teaching calculation; not a full pretrained model or hardware benchmark.
# No imports from SeetheTruth are required to execute this file.
import json
import math
import numpy as np

def _source(name, snapshot, expected, overrides):
    data = json.loads(snapshot) if name not in overrides else overrides[name]
    value = np.asarray(data, dtype=np.float64)
    if tuple(value.shape) != tuple(expected):
        raise ValueError(f"source {name} expected {expected}, received {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"source {name} contains NaN or Infinity")
    return value.copy()

def _softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)

def _act(x, name, **kw):
    if name == "silu":
        return x / (1 + np.exp(-x))
    if name == "relu":
        return np.maximum(x, 0)
    if name == "relu_squared":
        return np.maximum(x, 0)**2
    if name == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if name == "tanh":
        return np.tanh(x)
    if name == "softplus":
        return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    if name == "gelu_tanh":
        return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))
    if name == "gelu_erf":
        return .5*x*(1+np.vectorize(math.erf,otypes=[float])(x/np.sqrt(2)))
    if name == "mish":
        return x*np.tanh(_act(x, "softplus"))
    if name == "leaky_relu":
        return np.where(x >= 0, x, kw.get("slope", .01)*x)
    if name == "elu":
        return np.where(x > 0, x, kw.get("alpha", 1)*np.expm1(x))
    if name == "hardswish":
        return x*np.clip(x+3, 0, 6)/6
    raise ValueError("unsupported activation " + name)

def _rope(x, offset, r, base):
    out=x.copy()
    steps=np.arange(x.shape[2],dtype=np.float64)[None,None,:,None]+offset
    frequencies=base**(-2*np.arange(r//2,dtype=np.float64)/r)
    angles=steps*frequencies[None,None,None,:]
    even=x[...,:r:2]
    odd=x[...,1:r:2]
    out[...,:r:2]=even*np.cos(angles)-odd*np.sin(angles)
    out[...,1:r:2]=even*np.sin(angles)+odd*np.cos(angles)
    return out

def _causal(x, qstart, kstart):
    queries=qstart+np.arange(x.shape[2])[:,None]
    keys=kstart+np.arange(x.shape[3])[None,:]
    if np.any((queries >= keys).sum(axis=1)==0):
        raise ValueError("causal query has no legal key")
    return np.where(keys<=queries, x, -1e9)

def _opt(kind, weights, gradient, lr, wd, momentum=.9, alpha=.99, lr_decay=0):
    # One-step functional optimizer block; zero prior state by definition.
    b1,b2,eps=.9,.999,1e-8
    eff=gradient+wd*weights
    if kind=="opt_adamw":
        return weights-lr*(gradient/(np.abs(gradient)+eps)+wd*weights)
    if kind=="opt_adam":
        return weights-lr*eff/(np.abs(eff)+eps)
    if kind=="opt_sgd_momentum":
        return weights-lr*eff
    if kind=="opt_rmsprop":
        return weights-lr*eff/(np.sqrt((1-alpha)*eff**2)+eps)
    if kind=="opt_adagrad":
        return weights-lr*eff/(np.sqrt(eff**2)+eps)
    if kind=="opt_lion":
        return weights-lr*(np.sign((1-b1)*gradient)+wd*weights)
    raise ValueError("optimizer not implemented")

def run_graph(overrides=None):
    """Execute actual node operations in topological order; return all outputs."""
    overrides = {} if overrides is None else dict(overrides)
    v = {}
`;
function compile(raw,options={}){
 const graph=options.targetId?Branch.slice(raw,options.targetId):L.validateGraph(raw);
 const sorted=L.order(graph);
 const unsupported=sorted.filter(n=>!SUPPORTED.has(n.kind));
 if(unsupported.length)throw Error("无法导出可执行 Python：不支持 "+unsupported.map(n=>n.id+"("+n.kind+")").join(", ")+"；不会将其伪装为可执行代码");
 // The normal micro-interpreter rejects missing sockets, invalid parameter-dependent values etc.
 const forward=L.evaluate(graph),steps=new Map(forward.steps.map(x=>[x.id,x]));
 const lines=[PRELUDE],provenance=[];
 const edges=new Map(graph.edges.map(e=>[e.to+":"+e.port,e.from]));
 for(const n of sorted){
   const kind=n.kind,p=n.params,id=n.id,variable='v['+py(id)+']';
   const port=k=>{
     const from=edges.get(id+":"+k);
     if(!from)throw Error("缺少输入连接: "+id+"."+k);
     return 'v['+py(from)+']';
   };
   let expression;
   if(SOURCES.has(kind)){
     expression='_source('+py(id)+','+encoded(steps.get(id).values)+','+
       JSON.stringify(steps.get(id).shape)+',overrides)';
   }else if(kind==="linear"){
     const sh=steps.get(edges.get(id+":x")).shape, dim=sh[1],cols=p.out;
     const W=Array.from({length:dim},(_,k)=>Array.from({length:cols},(_,j)=>weight(id,j,k)));
     const bias=Array.from({length:cols},(_,j)=>weight(id,97,j));
     expression='('+port("x")+' @ np.asarray(json.loads('+encoded(W)+'),dtype=np.float64) + np.asarray(json.loads('+encoded(bias)+'),dtype=np.float64))';
   }else if(kind==="matmul"||kind==="nd_batched_matmul")expression='('+port("a")+' @ '+port("b")+')';
   else if(["add","mul","subtract","divide"].includes(kind))expression='('+port("a")+' '+({add:"+",mul:"*",subtract:"-",divide:"/"}[kind])+' '+port("b")+')';
   else if(kind==="nd_broadcast_add")expression='('+port("a")+' + '+port("b")+')';
   else if(kind==="transpose2d")expression='('+port("x")+'.T)';
   else if(kind==="nd_permute")expression='np.transpose('+port("x")+','+JSON.stringify([p.axis0,p.axis1,p.axis2,p.axis3])+')';
   else if(kind==="nd_reshape")expression=port("x")+'.reshape('+JSON.stringify([p.b,p.h,p.t,p.d])+')';
   else if(kind==="nd_expand4d")expression=port("x")+'[None,None,:,:]';
   else if(kind==="nd_flatten")expression=port("x")+'.reshape(-1,'+port("x")+'.shape[-1])';
   else if(kind==="nd_reduce_sum")expression='np.sum('+port("x")+',axis='+p.axis+',keepdims=True)';
   else if(kind==="nd_repeat_kv")expression='np.repeat('+port("x")+','+p.queryHeads+'//'+port("x")+'.shape[1],axis=1)';
   else if(kind==="nd_cache_read")expression=port("x")+'[:,:, '+p.start+':'+(p.start+p.length)+',:]';
   else if(kind==="nd_cache_append")expression='np.concatenate(('+port("cache")+','+port("segment")+'),axis=2)';
   else if(kind==="nd_rope")expression='_rope('+port("x")+','+p.offset+','+p.rotaryDim+','+p.base+')';
   else if(kind==="nd_causal_mask")expression='_causal('+port("x")+','+p.queryStart+','+p.keyStart+')';
   else if(kind==="nd_scale"||kind==="scale")expression='('+port("x")+' * '+p.factor+')';
   else if(kind==="bias")expression='('+port("x")+' + '+p.bias+')';
   else if(kind==="square")expression='('+port("x")+' ** 2)';
   else if(kind==="nd_softmax_last"||kind==="softmax")expression='_softmax('+port("x")+')';
   else if(kind==="rmsnorm")expression='('+port("x")+'/np.sqrt(np.mean('+port("x")+'**2,axis=-1,keepdims=True)+'+p.epsilon+'))';
   else if(kind==="layernorm")expression='(('+port("x")+'-'+port("x")+'.mean(axis=-1,keepdims=True))/np.sqrt(np.mean(('+port("x")+'-'+port("x")+'.mean(axis=-1,keepdims=True))**2,axis=-1,keepdims=True)+'+p.epsilon+'))';
   else if(kind==="attention")expression='(_softmax(('+port("q")+' @ '+port("k")+'.T)/np.sqrt('+port("q")+'.shape[-1])) @ '+port("v")+')';
   else if(kind==="mse"||kind==="nd_mse")expression='np.asarray([[np.mean(('+port("prediction")+'-'+port("target")+')**2)]],dtype=np.float64)';
   else if(kind==="swiglu")expression='(_act('+port("gate")+',"silu") * '+port("up")+')';
   else if(kind==="geglu")expression='(_act('+port("gate")+',"gelu_tanh") * '+port("up")+')';
   else if(kind.startsWith("opt_"))expression='_opt('+py(kind)+','+port("parameter")+','+port("gradient")+','+p.lr+','+p.weightDecay+','+(p.momentum??.9)+','+(p.alpha??.99)+','+(p.lrDecay??0)+')';
   else{
     const activation=kind.startsWith("nd_")?kind.slice(3):kind;
     if(["silu","relu","gelu_tanh","gelu_erf","sigmoid","tanh","softplus","relu_squared","mish","leaky_relu","elu","hardswish"].includes(activation)){
       expression='_act('+port("x")+','+py(activation)+(activation==="elu"?",alpha="+p.alpha:activation==="leaky_relu"?",slope="+p.slope:"")+')';
     }else throw Error("no Python lowering for "+kind);
   }
   lines.push("    # "+kind+" · "+id);
   lines.push("    "+variable+" = "+expression);
   lines.push("    if not np.isfinite("+variable+").all():");
   lines.push("        raise ValueError("+py(id+": non-finite operation output")+")");
   provenance.push({id,kind,shape:steps.get(id).shape});
 }
 lines.push("    return v");
 lines.push("");
 lines.push("if __name__ == '__main__':");
 lines.push("    outputs = run_graph()");
 lines.push("    print(json.dumps({k: x.tolist() for k,x in outputs.items()},allow_nan=False))");
 const source=lines.join("\n")+"\n";
 return {source,filename:"seethetruth-"+(options.targetId||"full-graph")+".py",nodes:provenance,
    dependency:"numpy",outputId:options.targetId||sorted.at(-1).id,
    statement:"NumPy program computes every supported node; unsupported kinds fail without export"};
}
return Object.freeze({compile,supportedKinds:Array.from(SUPPORTED).sort()});
});
