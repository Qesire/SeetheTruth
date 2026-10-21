/* Autograd-preserving PyTorch logical state graph lowering. */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthStreamTorch=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const kinds=Object.freeze(["stream_state_init","stream_state_append","stream_state_read","stream_state_fork"]);
 const prelude=String.raw`# Typed logical-stream state for bounded CPU PyTorch autograd.
# No detach/no in-place updates: torch.cat, clone and indexed reads preserve
# differentiability and correctly sum shared-prefix gradients across branches.
def _stream_tensor(x):
    if not torch.is_tensor(x) or x.dtype != DTYPE or x.device.type != "cpu":
        raise ValueError("stream requires Torch CPU float64 tensor")
    if x.ndim != 2 or any(d < 1 or d > 32 for d in x.shape) or x.numel() > 4096:
        raise ValueError("logical stream requires bounded [T,D]")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("nonfinite logical stream data")
    return x.clone()

def _stream_validate(s):
    if not isinstance(s,dict) or s.get("type") != "logical-stream-state" or s.get("axes") != ["T","D"]:
        raise ValueError("expected logical-stream-state")
    data = _stream_tensor(s["tensor"])
    for key,low,high in (("origin",0,4096),("capacity",1,32),
                           ("version",0,4096),("nextPosition",0,4128)):
        v=s.get(key)
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError("invalid stream "+key)
    if data.shape[0] > s["capacity"] or s["nextPosition"] != s["origin"] + data.shape[0]:
        raise ValueError("corrupt stream time or capacity")
    return data

def _stream_init(x,origin,capacity):
    data = _stream_tensor(x)
    if type(origin) is not int or not 0 <= origin <= 4096:
        raise ValueError("invalid origin")
    if type(capacity) is not int or not 1 <= capacity <= 32 or data.shape[0] > capacity:
        raise ValueError("initial stream exceeds capacity")
    return {"type":"logical-stream-state","axes":["T","D"],"tensor":data,
            "origin":origin,"capacity":capacity,"nextPosition":origin+data.shape[0],
            "version":0}

def _stream_append(state,chunk,position):
    previous = _stream_validate(state)
    new_chunk = _stream_tensor(chunk)
    if type(position) is not int or position != state["nextPosition"]:
        raise ValueError("out-of-order/stale append position")
    if previous.shape[1] != new_chunk.shape[1]:
        raise ValueError("stream append feature axis D mismatch")
    if previous.shape[0]+new_chunk.shape[0] > state["capacity"]:
        raise ValueError("append exceeds stream capacity")
    return {**state,"tensor":torch.cat((previous,new_chunk),dim=0),
            "nextPosition":state["nextPosition"]+new_chunk.shape[0],
            "version":state["version"]+1}

def _stream_fork(state):
    return {**state,"tensor":_stream_validate(state).clone()}

def _stream_read(state,start,length):
    data = _stream_validate(state)
    if type(start) is not int or type(length) is not int or length < 1:
        raise ValueError("invalid stream read position/length")
    if start < state["origin"] or start+length > state["nextPosition"]:
        raise ValueError("stream read exceeds known absolute positions")
    i=start-state["origin"]
    return data[i:i+length,:].clone()

def _stream_public(states):
    # Scalar provenance only; no mutable tensor or autograd graph escapes here.
    return {key:{k:value for k,value in state.items() if k!="tensor"}
            for key,state in states.items()}

class _StreamOutputs(dict):
    def __init__(self,values,metadata):
        super().__init__(values)
        self.state_metadata = metadata

`;
 function lower(n,edges){
  const id=JSON.stringify(n.id),p=n.params;
  const from=port=>{const prior=edges.get(n.id+":"+port);if(!prior)throw Error("missing stream port "+n.id+"."+port);return JSON.stringify(prior);};
  const v=port=>"v["+from(port)+"]",state=port=>"s["+from(port)+"]";
  let before=[],expression;
  if(n.kind==="stream_state_init"){
   before=["        s["+id+"] = _stream_init("+v("x")+","+p.origin+","+p.capacity+")"];
   expression="s["+id+"]['tensor'].clone()";
  }else if(n.kind==="stream_state_append"){
   before=["        s["+id+"] = _stream_append("+state("state")+","+v("chunk")+","+p.position+")"];
   expression="s["+id+"]['tensor'].clone()";
  }else if(n.kind==="stream_state_fork"){
   before=["        s["+id+"] = _stream_fork("+state("state")+")"];
   expression="s["+id+"]['tensor'].clone()";
  }else if(n.kind==="stream_state_read"){
   expression="_stream_read("+state("state")+","+p.start+","+p.length+")";
  }else throw Error("unsupported Torch stream kind "+n.kind);
  return Object.freeze({before,expression});
 }
 return Object.freeze({kinds,prelude,lower});
});
