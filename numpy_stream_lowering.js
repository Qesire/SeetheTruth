/* Standalone Python NumPy logical-stream lowering. No native KV allocation,
 * implicit training gradient or hard-coded intermediate execution data.
 */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthStreamNumpy=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const kinds=Object.freeze(["stream_state_init","stream_state_append","stream_state_read","stream_state_fork"]);
 const PY=String.raw`# NumPy immutable logical stream state; CPU-only, not a GPU KV cache.
def _state_tensor(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or any(d < 1 or d > 32 for d in a.shape) or a.size > 4096 or not np.isfinite(a).all():
        raise ValueError("logical stream requires finite bounded [T,D]")
    return a.copy()

def _state_check(s):
    if not isinstance(s,dict) or s.get("type") != "logical-stream-state" or s.get("axes") != ["T","D"]:
        raise ValueError("expected logical-stream-state")
    data = _state_tensor(s["tensor"])
    for key,low,high in (("origin",0,4096),("capacity",1,32),("version",0,4096),("nextPosition",0,4128)):
        x = s.get(key)
        if not isinstance(x,int) or isinstance(x,bool) or not low <= x <= high:
            raise ValueError("invalid stream " + key)
    if data.shape[0] > s["capacity"] or s["nextPosition"] != s["origin"]+data.shape[0]:
        raise ValueError("corrupt stream state time or capacity")
    return data

def _state_init(x,origin,capacity):
    data = _state_tensor(x)
    if not isinstance(origin,int) or isinstance(origin,bool) or not 0 <= origin <= 4096:
        raise ValueError("invalid origin")
    if not isinstance(capacity,int) or isinstance(capacity,bool) or not 1 <= capacity <= 32 or data.shape[0] > capacity:
        raise ValueError("initial state exceeds capacity")
    return dict(type="logical-stream-state",axes=["T","D"],tensor=data,origin=origin,capacity=capacity,
                nextPosition=origin+data.shape[0],version=0)

def _state_append(s,chunk,position):
    data = _state_check(s)
    part = _state_tensor(chunk)
    if not isinstance(position,int) or isinstance(position,bool) or position != s["nextPosition"]:
        raise ValueError("out-of-order/stale append position")
    if data.shape[1] != part.shape[1]:
        raise ValueError("stream append feature axis D mismatch")
    if data.shape[0]+part.shape[0] > s["capacity"]:
        raise ValueError("append exceeds stream capacity")
    return {**s,"tensor":_state_tensor(np.concatenate((data,part),axis=0)),
            "version":s["version"]+1,"nextPosition":s["nextPosition"]+part.shape[0]}

def _state_fork(s):
    return {**s,"tensor":_state_check(s).copy()}

def _state_read(s,start,length):
    data = _state_check(s)
    if not isinstance(start,int) or isinstance(start,bool) or not isinstance(length,int) or isinstance(length,bool) or length < 1:
        raise ValueError("invalid read position/length")
    if start < s["origin"] or start+length > s["nextPosition"]:
        raise ValueError("stream read exceeds known absolute positions")
    start -= s["origin"]
    return data[start:start+length,:].copy()

class _StreamOutputs(dict):
    def __init__(self,values,states):
        super().__init__(values)
        # Intentionally omit private numerical storage from public state metadata.
        self.state_metadata = {key:{name:value for name,value in state.items() if name!="tensor"}
                               for key,state in states.items()}
`;
 // The lowering is a list of complete Python lines; never interpolates a
 // user-supplied Python expression. Caller supplies escaped ID and bounded params.
 function lower(n,edges){
  const id=JSON.stringify(n.id),p=n.params,from=port=>{
   const s=edges.get(n.id+":"+port);
   if(!s)throw Error("missing stream edge "+n.id+"."+port);
   return JSON.stringify(s);
  };
  const ref=port=>"v["+from(port)+"]",state=port=>"s["+from(port)+"]";
  let op,expression;
  if(n.kind==="stream_state_init"){
   op="s["+id+"] = _state_init("+ref("x")+","+p.origin+","+p.capacity+")";
   expression="s["+id+"]['tensor'].copy()";
  }else if(n.kind==="stream_state_append"){
   op="s["+id+"] = _state_append("+state("state")+","+ref("chunk")+","+p.position+")";
   expression="s["+id+"]['tensor'].copy()";
  }else if(n.kind==="stream_state_fork"){
   op="s["+id+"] = _state_fork("+state("state")+")";
   expression="s["+id+"]['tensor'].copy()";
  }else if(n.kind==="stream_state_read"){
   expression="_state_read("+state("state")+","+p.start+","+p.length+")";
  }else throw Error("unsupported stream kind "+n.kind);
  return Object.freeze({before:op?["    "+op]:[],expression});
 }
 return Object.freeze({kinds,prelude:PY,lower});
});
