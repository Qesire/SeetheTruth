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
const Plan=node?require("./graph_lowering_plan.js"):root.SeetheTruthLoweringPlan;
const StreamNumpy=node?require("./numpy_stream_lowering.js"):root.SeetheTruthStreamNumpy;
const Portable=node?require("./portable_operator_rules.js"):root.SeetheTruthPortableOperatorRules;
const Norm=node?require("./normalization_rules.js"):root.SeetheTruthNormalizationRules;
const SOURCES=new Set(["input","parameter","nd_input","nd_parameter","token_ids"]);
const SUPPORTED=new Set([
 ...SOURCES,...StreamNumpy.kinds,"deepinfer_mtp_confidence_schedule","deepinfer_yoco_swa_mask","deepinfer_yoco_swa_compact","inference_topk","inference_sparse_attention","inference_paged_kv_read","inference_greedy_verify","video_causal_conv3d","vision_conv2d","audio_mel_filterbank","diffusion_ddim_step","vision_patchify","video_tubelet","audio_stft_power","audio_log_power","linear","matmul","add","mul","subtract","divide","transpose2d",
 "rmsnorm","layernorm","softmax","attention","mse","silu","relu","gelu_tanh","gelu_erf",
 "sigmoid","tanh","softplus","scale","bias","square","swiglu","geglu",
 "relu_squared","mish","leaky_relu","elu","hardswish",
 "nd_permute","nd_reshape","nd_flatten","nd_expand4d","nd_broadcast_add",
 "nd_reduce_sum","nd_batched_matmul","nd_scale","nd_softmax_last","nd_causal_mask",
 "nd_repeat_kv","nd_mse","nd_rope","nd_cache_read","nd_cache_append",
 "nd_silu","nd_relu_squared","nd_mish","nd_gelu_erf","nd_elu","nd_hardswish",
 "opt_adamw","opt_adam","opt_adagrad","opt_sgd_momentum","opt_rmsprop","opt_lion",
 "row_mean","row_sum","row_var","row_logsumexp","log_softmax","l2norm","qk_norm","logit_softcap",
 "low_rank_project","row_bias_add","weighted_sum","masked_softmax","token_depth_mask","state_mask_blend","sliding_window_mask",
 "temperature","residual_scale","mae_loss","topk","argmax","detach_demo","causal_mask",
 "group_topk_mask","masked_router_topk","router_gate_weights","dispatch_routed_tokens",
 "expert_linear_bank","routed_weighted_reduce","expert_bank_mix","gather_routed_bank",
 "router_topk","gather_rows","scatter_add","embedding","cross_entropy","pos_encoding","slice_rows","slice_cols","concat_cols","rope2d","repeat_rows"
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

def _depth_mask(depths, step, width):
    if depths.ndim!=2 or depths.shape[1]!=1 or not np.isfinite(depths).all() or \
       not np.all(depths==np.floor(depths)) or np.any(depths<0) or np.any(depths>3):
        raise ValueError("Token depth budgets must be integers in [0,3]")
    return np.broadcast_to((depths>=step).astype(np.float64),(depths.shape[0],width)).copy()

def _blend_state(previous, candidate, mask):
    if previous.shape!=candidate.shape or previous.shape!=mask.shape or \
       not np.all((mask==0)|(mask==1)):
        raise ValueError("state blend needs matching tensors and a binary mask")
    return np.where(mask==1,candidate,previous)

def _masked_softmax(logits, mask):
    if logits.shape != mask.shape or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("masked_softmax requires same-shape boolean 0/1 mask")
    allowed = mask.astype(bool)
    if not np.all(allowed.any(axis=-1)):
        raise ValueError("masked_softmax has a fully masked row")
    scores = np.where(allowed, logits, -np.inf)
    maximum = np.max(scores, axis=-1, keepdims=True)
    exp = np.where(allowed, np.exp(scores-maximum), 0)
    return exp / np.sum(exp,axis=-1,keepdims=True)

def _sliding_window(x, width):
    if x.shape[0] != x.shape[1]:
        raise ValueError("sliding_window_mask requires a square token matrix")
    rows = np.arange(x.shape[0])[:,None]
    cols = np.arange(x.shape[1])[None,:]
    return ((cols <= rows) & (rows-cols < width)).astype(np.float64)

def _rope2d(x, offset):
    if x.ndim!=2 or x.shape[1]%2:
        raise ValueError("RoPE pair layout requires even 2D head features")
    t=np.arange(x.shape[0],dtype=np.float64)[:,None]+offset
    j=np.arange(x.shape[1]//2,dtype=np.float64)[None,:]
    angle=t/np.power(10000.,2*j/x.shape[1])
    a=x[:,0::2];b=x[:,1::2]
    out=np.empty_like(x)
    out[:,0::2]=a*np.cos(angle)-b*np.sin(angle)
    out[:,1::2]=a*np.sin(angle)+b*np.cos(angle)
    return out

def _pos_encoding(x):
    if x.ndim!=2 or x.shape[1]%2:
        raise ValueError("sinusoidal absolute PE requires even feature width")
    rows=np.arange(x.shape[0],dtype=np.float64)[:,None]
    dims=np.arange(x.shape[1],dtype=np.int64)[None,:]
    angle=rows/np.power(10000.,2*(dims//2)/x.shape[1])
    offset=np.where(dims%2==0,np.sin(angle),np.cos(angle))
    return x+offset

def _fixed_embedding(ids, vocab, width):
    if ids.ndim!=2 or ids.shape[1]!=1 or not np.all(np.isfinite(ids)) or \
       not np.all(ids==np.floor(ids)) or np.any(ids<0) or np.any(ids>=vocab):
        raise ValueError("Embedding token IDs must be integral [T,1] within vocab")
    rows=np.arange(vocab,dtype=np.float64)[:,None]+1
    cols=np.arange(width,dtype=np.float64)[None,:]+1
    table=np.sin(rows*.331+cols*.73+11*.11)*.52+np.cos(rows*cols*.17)*.1
    return table[ids[:,0].astype(np.int64)]

def _cross_entropy(logits,labels):
    if logits.ndim!=2 or labels.shape!=(logits.shape[0],1) or not np.all(np.isfinite(labels)) or \
       not np.all(labels==np.floor(labels)) or np.any(labels<0) or np.any(labels>=logits.shape[1]):
        raise ValueError("cross entropy requires integral labels [T,1] within class count")
    picked=logits[np.arange(logits.shape[0]),labels[:,0].astype(np.int64)]
    return np.asarray([[np.mean(np.logaddexp.reduce(logits,axis=-1)-picked)]],dtype=np.float64)

def _route_ids(values, upper):
    ids=np.asarray(values)
    if ids.ndim != 2 or not np.isfinite(ids).all() or not np.all(ids == np.floor(ids)) or np.any(ids < 0) or np.any(ids >= upper):
        raise ValueError("invalid discrete router indices")
    return ids.astype(np.int64)

def _group_topk(scores, groups, keep):
    rows, experts = scores.shape
    if groups < 1 or experts % groups or keep < 1 or keep > groups:
        raise ValueError("invalid expert groups")
    group_size=experts//groups
    answer=np.zeros_like(scores,dtype=np.float64)
    for t in range(rows):
        ranks=[]
        for g in range(groups):
            block=scores[t,g*group_size:(g+1)*group_size]
            score=float(np.sort(block)[::-1][:2].sum())
            ranks.append((-score,g))
        kept=sorted(ranks)[:keep]
        for _,g in kept:answer[t,g*group_size:(g+1)*group_size]=1.
    return answer

def _masked_topk(scores, mask, k):
    if scores.shape != mask.shape or not np.all((mask == 0)|(mask == 1)):
        raise ValueError("invalid routing mask")
    picks=[]
    for row,allowed in zip(scores,mask):
        ranked=sorted(((-float(v),i) for i,v in enumerate(row) if allowed[i] == 1))
        if len(ranked)<k:raise ValueError("insufficient unmasked experts")
        picks.append([i for _,i in ranked[:k]])
    return np.asarray(picks,dtype=np.float64)

def _router_gate_weights(scores, ids, mode, normalize):
    index=_route_ids(ids,scores.shape[1])
    selected=np.take_along_axis(scores,index,axis=1)
    if mode == "identity":
        if np.any(selected < 0):raise ValueError("negative identity gate affinity")
        result=selected
    elif mode == "sigmoid":result=1./(1.+np.exp(-selected))
    elif mode == "softmax":result=_softmax(selected)
    else:raise ValueError("invalid gate mode")
    if normalize:
        denom=np.sum(result,axis=1,keepdims=True)
        if np.any(denom <= 0):raise ValueError("zero router gate sum")
        return result/denom
    return result

def _dispatch_tokens(x,ids):
    _route_ids(ids,2**31)
    if x.shape[0] != ids.shape[0] or ids.size > 8:raise ValueError("invalid routed-token rows")
    return np.repeat(x,ids.shape[1],axis=0)

def _expert_linear_bank(x,ids,weights):
    if x.ndim!=2 or weights.ndim!=2 or weights.shape[0]%x.shape[1]:
        raise ValueError("invalid expert weight layout")
    token,k=ids.shape
    if token*k!=x.shape[0]:raise ValueError("expert tokens != T*K")
    dims=x.shape[1];experts=weights.shape[0]//dims
    selected=weights.reshape(experts,dims,weights.shape[1])[_route_ids(ids,experts)]
    return np.einsum("tkd,tkdh->tkh",x.reshape(token,k,dims),selected).reshape(token*k,weights.shape[1])

def _routed_reduce(values,gates):
    token,k=gates.shape
    if values.shape[0]!=token*k:raise ValueError("routed reduction requires T*K rows")
    return (values.reshape(token,k,values.shape[1])*gates[:,:,None]).sum(axis=1)

def _expert_bank_mix(ids,gates,experts):
    if ids.shape!=gates.shape:raise ValueError("expert mix shape mismatch")
    selected=experts[_route_ids(ids,experts.shape[0])]
    return np.sum(selected*gates[:,:,None],axis=1)

def _gather_routed_bank(x,ids):
    if ids.size>8:raise ValueError("routed gather exceeds eight rows")
    return x[_route_ids(ids,x.shape[0]).reshape(-1)]

def _gather_rows(x,ids):
    if ids.shape[1]!=1:raise ValueError("gather index must be [T,1]")
    return x[_route_ids(ids,x.shape[0])[:,0]]

def _scatter_add(ids,values,rows):
    if ids.shape!=(values.shape[0],1):raise ValueError("scatter shape mismatch")
    out=np.zeros((rows,values.shape[1]),dtype=np.float64)
    np.add.at(out,_route_ids(ids,rows)[:,0],values)
    return out

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
 const {graph,sorted,edges,steps}=Plan.prepare(raw,{
   targetId:options.targetId,supportedKinds:SUPPORTED,
   unsupportedMessage:bad=>"无法导出可执行 Python：不支持 "+bad.map(n=>n.id+"("+n.kind+")").join(", ")+"；不会将其伪装为可执行代码"
 });
 const hasStream=sorted.some(n=>StreamNumpy.kinds.includes(n.kind));
 const lines=[PRELUDE.replace("def run_graph(overrides=None):","\ndef _deepinfer_mtp(remap, rates, budget):\n    mapping = remap[:,0].astype(np.int64)\n    if np.any((mapping < -1) | (mapping >= rates.shape[0])):\n        raise ValueError(\"invalid DeepInfer remap\")\n    if np.any((rates < 0) | (rates > 1)):\n        raise ValueError(\"confidence must be in [0,1]\")\n    selected = np.where(mapping[:,None] >= 0, rates[np.maximum(mapping,0)], 0.0)\n    cum = np.cumprod(selected, axis=1).reshape(-1)\n    top = np.argsort(-cum,kind=\"stable\")[:budget-remap.shape[0]]\n    return (1 + np.bincount(top//rates.shape[1],minlength=remap.shape[0])).astype(np.float64).reshape(-1,1)\n\ndef _deepinfer_yoco(q,k,v,window,keep,backtracking,mode):\n    if keep < window+backtracking: raise ValueError(\"invalid YoCo keepTokens bound\")\n    n = q.shape[0]; cut=max(0,n-keep); out=[]\n    for t in range(cut if mode==\"compact\" else 0,n):\n        if t<cut:\n            out.append(np.zeros(v.shape[1],dtype=np.float64)); continue\n        idx=np.arange(max(0,t-window+1),t+1)\n        scores=k[idx]@q[t]/np.sqrt(q.shape[1])\n        logits=np.exp(scores-np.max(scores)); probs=logits/np.sum(logits)\n        out.append(probs@v[idx])\n    return np.stack(out,axis=0)\n\n"+"\ndef run_graph(overrides=None):")],provenance=[];
 if(hasStream){
   lines[0]=lines[0].replace("def run_graph(overrides=None):",
     StreamNumpy.prelude+"\ndef run_graph(overrides=None):");
   lines[0]=lines[0].replace("    v = {}\n","    v = {}\n    s = {}\n");
 }
 for(const n of sorted){
   const kind=n.kind,p=n.params,id=n.id,variable='v['+py(id)+']';
   const port=k=>{
     const from=edges.get(id+":"+k);
     if(!from)throw Error("缺少输入连接: "+id+"."+k);
     return 'v['+py(from)+']';
   };
   const portable=Portable.emit(n,port);
   const normalized=Norm.emit(n,"numpy",port);
   let expression;
   if(portable!==null)expression=portable;
   else if(normalized!==null)expression=normalized;
   else if(SOURCES.has(kind)){
     expression='_source('+py(id)+','+encoded(steps.get(id).values)+','+
       JSON.stringify(steps.get(id).shape)+',overrides)';
   }else if(StreamNumpy.kinds.includes(kind)){
     const lowered=StreamNumpy.lower(n,edges);
     lines.push(...lowered.before);
     expression=lowered.expression;
   }else if(kind==="linear"){
     const sh=steps.get(edges.get(id+":x")).shape, dim=sh[1],cols=p.out;
     const W=Array.from({length:dim},(_,k)=>Array.from({length:cols},(_,j)=>weight(id,j,k)));
     const bias=Array.from({length:cols},(_,j)=>weight(id,97,j));
     expression='('+port("x")+' @ np.asarray(json.loads('+encoded(W)+'),dtype=np.float64) + np.asarray(json.loads('+encoded(bias)+'),dtype=np.float64))';
   }else if(kind==="deepinfer_mtp_confidence_schedule"){
     expression="_deepinfer_mtp("+port("remap")+","+port("prev_confidence")+","+p.totalQ+")";
   }else if(kind==="deepinfer_yoco_swa_mask"||kind==="deepinfer_yoco_swa_compact"){
     expression="_deepinfer_yoco("+port("q")+","+port("k")+","+port("v")+","+
      p.window+","+p.keepTokens+","+p.backtracking+","+
      JSON.stringify(kind==="deepinfer_yoco_swa_mask"?"mask":"compact")+")";
   }else if(kind==="inference_topk"){
     expression='np.argsort(-'+port("scores")+',axis=1,kind="stable")[:,:'+p.topk+'].astype(np.float64)';
   }else if(kind==="inference_sparse_attention"){
     const q=port("q"),k=port("k"),v=port("v"),idx=port("indices");
     expression='np.stack([(_softmax(('+k+'[ind.astype(np.int64)] @ '+q+'[t])/np.sqrt('+q+'.shape[1])) @ '+v+'[ind.astype(np.int64)]) for t,ind in enumerate('+idx+')],axis=0)';
   }else if(kind==="inference_paged_kv_read"){
     const cache=port("cache"),table=port("blocks"),pos=port("positions"),size=p.pageSize;
     expression='np.stack(['+cache+'[int('+table+'[int(loc)//'+size+',0])*'+size+'+int(loc)%'+size+'] for loc in '+pos+'[:,0]],axis=0)';
   }else if(kind==="inference_greedy_verify"){
     const d=port("draft"),t=port("target");
     expression='np.asarray([[float(np.all('+d+'[:i+1,0]=='+t+'[:i+1,0]))] for i in range('+d+'.shape[0])],dtype=np.float64)';
   }else if(kind==="video_causal_conv3d"){
     const x=port("x"),w=port("kernel"),T=p.frames,H=p.height,kt=p.kernelFrames,
       kh=p.kernelHeight,kw=p.kernelWidth,C=p.channels,oc=p.outChannels,S=p.stride;
     const oh=Math.floor((H-kh)/S)+1,ow=Math.floor((p.width-kw)/S)+1;
     expression='np.asarray([[sum('+x+'[(t+dt-'+(kt-1)+')*'+H+'+y*'+S+'+dy,(col*'+S+'+dx)*'+C+'+ci]*'+
       w+'[outch,((dt*'+kh+'+dy)*'+kw+'+dx)*'+C+'+ci] for dt in range('+kt+') '+
       'for dy in range('+kh+') for dx in range('+kw+') for ci in range('+C+') '+
       'if t+dt>='+ (kt-1)+') for col in range('+ow+') for outch in range('+oc+
       ')] for t in range('+T+') for y in range('+oh+')],dtype=np.float64)';
   }else if(kind==="vision_conv2d"){
     const x=port("x"),w=port("kernel"),C=p.channels,kh=p.kernelHeight,kw=p.kernelWidth,S=p.stride,O=p.outChannels;
     const oh=Math.floor((p.height-kh)/S)+1,ow=Math.floor((p.width-kw)/S)+1;
     expression='np.asarray([[sum('+x+'[y*'+S+'+dy,(col*'+S+'+dx)*'+C+'+ci]*'+w+'[co,(dy*'+kw+'+dx)*'+C+'+ci] for dy in range('+kh+') for dx in range('+kw+') for ci in range('+C+')) for col in range('+ow+') for co in range('+O+')] for y in range('+oh+')],dtype=np.float64)';
   }else if(kind==="audio_mel_filterbank"){
     const Adv=typeof module==="object"&&module.exports?require("./multimodal_advanced.js"):globalThis.SeetheTruthMultimodalAdvanced;
     expression='('+port("x")+' @ np.asarray('+JSON.stringify(Adv.melWeights(p))+',dtype=np.float64))';
   }else if(kind==="vision_patchify"){
     const x=port("x"),H=p.height,W=p.width,C=p.channels,ph=p.patchHeight,pw=p.patchWidth;
     expression=x+'.reshape('+[H,W,C].join(',')+').reshape('+[H/ph,ph,W/pw,pw,C].join(',')+').transpose(0,2,1,3,4).reshape(-1,'+(ph*pw*C)+')';
   }else if(kind==="video_tubelet"){
     const x=port("x"),T=p.frames,H=p.height,W=p.width,C=p.channels,pt=p.patchFrames,ph=p.patchHeight,pw=p.patchWidth;
     expression=x+'.reshape('+[T,H,W,C].join(',')+').reshape('+[T/pt,pt,H/ph,ph,W/pw,pw,C].join(',')+').transpose(0,2,4,1,3,5,6).reshape(-1,'+(pt*ph*pw*C)+')';
   }else if(kind==="audio_stft_power"){
     const x=port("x"),N=p.fftSize,h=p.hop;
     expression='np.stack([np.abs(np.fft.rfft('+x+'[0,i:i+'+N+']*np.hanning('+(N+1)+')[:-1]))**2 for i in range(0,'+x+'.shape[1]-'+N+'+1,'+h+')],axis=0)';
   }else if(kind==="audio_log_power"){
     expression='np.log10(np.maximum('+port("x")+','+p.floor+'))';
   }
   else if(kind==="transpose2d")expression='('+port("x")+'.T)';
   else if(kind==="nd_permute")expression='np.transpose('+port("x")+','+JSON.stringify([p.axis0,p.axis1,p.axis2,p.axis3])+')';
   else if(kind==="nd_reshape")expression=port("x")+'.reshape('+JSON.stringify([p.b,p.h,p.t,p.d])+')';
   else if(kind==="nd_expand4d")expression=port("x")+'[None,None,:,:]';
   else if(kind==="nd_flatten")expression=port("x")+'.reshape(-1,'+port("x")+'.shape[-1])';
   else if(kind==="nd_reduce_sum")expression='np.sum('+port("x")+',axis='+p.axis+',keepdims=True)';
   else if(kind==="nd_repeat_kv")expression='np.repeat('+port("x")+','+p.queryHeads+'//'+port("x")+'.shape[1],axis=1)';
   else if(kind==="nd_cache_append")expression='np.concatenate(('+port("cache")+','+port("segment")+'),axis=2)';
   else if(kind==="nd_rope")expression='_rope('+port("x")+','+p.offset+','+p.rotaryDim+','+p.base+')';
   else if(kind==="nd_causal_mask")expression='_causal('+port("x")+','+p.queryStart+','+p.keyStart+')';
   else if(kind==="mse"||kind==="nd_mse")expression='np.asarray([[np.mean(('+port("prediction")+'-'+port("target")+')**2)]],dtype=np.float64)';
   else if(kind==="row_mean")expression='np.mean('+port("x")+',axis=-1,keepdims=True)';
   else if(kind==="row_sum")expression='np.sum('+port("x")+',axis=-1,keepdims=True)';
   else if(kind==="row_var")expression='np.var('+port("x")+',axis=-1,keepdims=True)';
   else if(kind==="l2norm")expression='('+port("x")+'/np.maximum(np.linalg.norm('+port("x")+',axis=-1,keepdims=True),'+p.epsilon+'))';
   else if(kind==="logit_softcap")expression='('+p.cap+'*np.tanh('+port("x")+'/'+p.cap+'))';
   else if(kind==="token_depth_mask")expression='_depth_mask('+port("depths")+','+p.step+','+p.width+')';
   else if(kind==="state_mask_blend")expression='_blend_state('+port("previous")+','+port("candidate")+','+port("mask")+')';
   else if(kind==="masked_softmax")expression='_masked_softmax('+port("logits")+','+port("mask")+')';
   else if(kind==="sliding_window_mask")expression='_sliding_window('+port("x")+','+p.window+')';
   else if(kind==="mae_loss")expression='np.asarray([[np.mean(np.abs('+port("prediction")+'-'+port("target")+'))]],dtype=np.float64)';
   else if(kind==="topk")expression='np.take_along_axis('+port("x")+',np.argsort(-'+port("x")+',axis=-1,kind="stable")[:,:'+p.k+'],axis=-1)';
   else if(kind==="argmax")expression='np.argmax('+port("x")+',axis=-1).reshape(-1,1).astype(np.float64)';
   else if(kind==="detach_demo")expression=port("x")+'.copy()';
   else if(kind==="repeat_rows")expression='np.tile('+port("x")+',('+p.times+',1))';
   else if(kind==="concat_cols")expression='np.concatenate(('+port("a")+','+port("b")+'),axis=1)';
   else if(kind==="rope2d")expression='_rope2d('+port("x")+','+p.offset+')';
   else if(kind==="pos_encoding")expression='_pos_encoding('+port("x")+')';
   else if(kind==="causal_mask")expression='np.tri('+port("x")+'.shape[0],'+port("x")+'.shape[0],dtype=np.float64)';
   else if(kind==="embedding")expression='_fixed_embedding('+port("ids")+','+p.vocab+','+p.width+')';
   else if(kind==="cross_entropy")expression='_cross_entropy('+port("logits")+','+port("labels")+')';
   else if(kind==="group_topk_mask")expression='_group_topk('+port("scores")+','+p.groups+','+p.keepGroups+')';
   else if(kind==="masked_router_topk")expression='_masked_topk('+port("scores")+','+port("mask")+','+p.k+')';
   else if(kind==="router_topk")expression='_masked_topk('+port("x")+',np.ones_like('+port("x")+'),'+p.k+')';
   else if(kind==="router_gate_weights")expression='_router_gate_weights('+port("scores")+','+port("indices")+','+py(p.mode)+','+(p.normalize?'True':'False')+')';
   else if(kind==="dispatch_routed_tokens")expression='_dispatch_tokens('+port("x")+','+port("indices")+')';
   else if(kind==="expert_linear_bank")expression='_expert_linear_bank('+port("x")+','+port("indices")+','+port("weights")+')';
   else if(kind==="routed_weighted_reduce")expression='_routed_reduce('+port("values")+','+port("gates")+')';
   else if(kind==="expert_bank_mix")expression='_expert_bank_mix('+port("indices")+','+port("gates")+','+port("experts")+')';
   else if(kind==="gather_routed_bank")expression='_gather_routed_bank('+port("x")+','+port("indices")+')';
   else if(kind==="gather_rows")expression='_gather_rows('+port("x")+','+port("indices")+')';
   else if(kind==="scatter_add")expression='_scatter_add('+port("indices")+','+port("values")+','+p.rows+')';
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
 lines.push(hasStream?"    return _StreamOutputs(v,s)":"    return v");
 lines.push("");
 lines.push("if __name__ == '__main__':");
 lines.push("    outputs = run_graph()");
 lines.push("    print(json.dumps({k: x.tolist() for k,x in outputs.items()},allow_nan=False))");
 const source=lines.join("\n")+"\n";
 return {source,filename:"seethetruth-"+(options.targetId||"full-graph")+".py",nodes:provenance,
    dependency:"numpy",outputId:options.targetId||sorted.at(-1).id,
    statement:hasStream?"Independent NumPy logical-state forward with runtime time checks; no VJP or physical KV cache":"NumPy program computes every supported node; unsupported kinds fail without export"};
}
return Object.freeze({compile,supportedKinds:Array.from(SUPPORTED).sort()});
});
