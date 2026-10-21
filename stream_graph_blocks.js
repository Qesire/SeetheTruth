/* Executable, tagged logical stream blocks for the free-form Learning Studio DAG.
 * Transport values are Float64 [T,D] matrices; tagged state lives in graph provenance.
 * JS micro-execution is forward-only; exported Torch CPU preserves autograd.
 * No physical GPU KV paging or cross-invocation BPTT.
 */
(function(root,factory){"use strict";
 const api=factory(typeof module==="object"&&module.exports?require("./semantic_state.js"):root.SeetheTruthSemanticState,
                   typeof module==="object"&&module.exports?require("./tensor_ir.js"):root.SeetheTruthTensorIR);
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthStreamGraph=api;
})(typeof window!=="undefined"?window:null,function(S,IR){"use strict";
 if(!S||!IR)throw Error("StreamGraph requires semantic_state.js and tensor_ir.js");
 const block=(title,en,ports,params,formula,why,lesson,caution)=>Object.freeze({
  title,en,category:"memory",ports,params,formula,why,lesson,caution});
 const defs=Object.freeze({
  stream_state_init:block("流状态初始化","Logical State Init",["x"],{origin:0,capacity:8},
   "S₀=(X,[T,D],origin,capacity,v=0,next=origin+T)",
   "把已有二维 [T,D] 数值显式变为可检查的状态载体：形状相同，但新增绝对 Token 位置、容量和状态版本。",
   "输入一个 2×3 张量，设置 origin=10，检查 next=12 与容量约束。","仅逻辑状态；不分配物理 KV 页、也不自动拥有 Attention 缓存。"),
  stream_state_append:block("按绝对位置追加状态","State Append",["state","chunk"],{position:2},
   "S' = append(S,ΔX,position), position==S.next",
   "上一状态和新 token chunk 都是必须的计算依赖；禁止偷偷按形状推断下一位置或覆盖历史。",
   "从 next=12 的状态写入 position=12，然后故意设成 11，比较版本和拒绝机制。",
   "内存物化、不可变逻辑追加；不执行 GPU KV 复用和跨请求显存抢占。"),
  stream_state_read:block("读取状态时间窗口","State Read",["state"],{start:0,length:1},
   "Y = S.tensor[start-origin:start-origin+length,:]",
   "从有版本的上游状态读取绝对位置区间，严格区分相对数组索引和逻辑 Token 序号。",
   "选择 [11,13) 窗口并检查输出 2×D；尝试读不存在的未来时间。",
   "输出是普通二维张量的复制，不会返回原地 View，也不具备真实缓存带宽语义。"),
  stream_state_fork:block("不可变状态分叉","State Fork",["state"],{},
   "S_branch = fork(S), no mutation to parent",
   "一条上下文前缀可以延伸为多个候选分支，基于明确依赖图保持相同历史且允许未来状态分叉。",
   "将相同 prefix 连到两个 Fork，再追加不同 chunk，对比相同前缀和不同后缀。",
   "演示的是不可变数值拷贝，不等于分页 KV COW 引用计数或实际内存节省。")
 });
 const kinds=Object.freeze(Object.keys(defs));
 const isKind=k=>Object.hasOwn(defs,k);
 const isStatePort=(kind,port)=>["stream_state_append","stream_state_read","stream_state_fork"].includes(kind)&&port==="state";
 const isStateProducer=kind=>["stream_state_init","stream_state_append","stream_state_fork"].includes(kind);
 const limit=(n,low,high,name)=>{if(!Number.isSafeInteger(n)||n<low||n>high)throw Error(name+" must be integer in ["+low+","+high+"]");};
 function validate(kind,p){
  if(!isKind(kind))throw Error("unknown stream operator "+kind);
  if(kind==="stream_state_init"){limit(p.origin,0,4096,"origin");limit(p.capacity,1,32,"capacity");}
  if(kind==="stream_state_append")limit(p.position,0,4096,"position");
  if(kind==="stream_state_read"){limit(p.start,0,4096,"start");limit(p.length,1,32,"length");}
 }
 function require2D(shape,name){
  if(!Array.isArray(shape)||shape.length!==2||shape.some(x=>!Number.isSafeInteger(x)||x<1||x>32)||
     shape[0]*shape[1]>IR.MAX_ELEMENTS)throw Error(name+" requires bounded [T,D] shape");
  return shape;
 }
 function infer(n,shapes,states={}){
  const kind=n.kind;
  // Typed edges: a state cannot feed a plain tensor operator even if the carrier's
  // numerical dimensions happen to match.
  const ports=(isKind(kind)?defs[kind].ports:Object.keys(shapes));
  for(const port of ports){
   if(!Object.hasOwn(shapes,port))continue;
   const expectedState=isStatePort(kind,port),provided=!!states[port];
   if(expectedState!==provided)throw Error(kind+"."+port+" expected "+(expectedState?"stream-state":"numeric-tensor")+" input; got "+(provided?"stream-state":"numeric-tensor"));
  }
  if(!isKind(kind))return null;
  validate(kind,n.params);
  let shape,meta=null;
  if(kind==="stream_state_init"){
   shape=require2D(shapes.x,"init").slice();
   if(shape[0]>n.params.capacity)throw Error("initial state exceeds capacity");
   meta={type:"logical-stream-state",axes:["T","D"],origin:n.params.origin,
    capacity:n.params.capacity,nextPosition:n.params.origin+shape[0],version:0};
  }else{
   const state=states.state;
   if(!state||state.type!=="logical-stream-state")throw Error("stream-state upstream provenance is required");
   shape=require2D(shapes.state,"state").slice();
   if(state.nextPosition!==state.origin+shape[0])throw Error("invalid stream state provenance");
   if(kind==="stream_state_append"){
    const chunk=require2D(shapes.chunk,"chunk");
    if(shape[1]!==chunk[1])throw Error("stream append feature axis D mismatch");
    if(n.params.position!==state.nextPosition)throw Error("out-of-order/stale append position");
    shape[0]+=chunk[0];
    if(shape[0]>state.capacity||shape[0]>32)throw Error("append exceeds stream capacity");
    meta={...state,nextPosition:state.nextPosition+chunk[0],version:state.version+1};
   }else if(kind==="stream_state_read"){
    if(n.params.start<state.origin||n.params.start+n.params.length>state.nextPosition)
     throw Error("stream read exceeds known absolute positions");
    shape=[n.params.length,shape[1]];
   }else{
    meta={...state};
   }
  }
  return {shape,logicalState:meta,rule:kind==="stream_state_init"?"数值矩阵→具名轴 [T,D] 状态与绝对位置":
    kind==="stream_state_append"?"状态 S 与 chunk 依赖、校验绝对位置后创建新版本":
    kind==="stream_state_fork"?"不可变状态复制：上游版本不变":
    "读取有界绝对时间窗口，返回普通 Tensor",
   conditions:["JS 画布仅前向；导出 Torch CPU 可验证分叉梯度，未实现跨轮 BPTT 或物理 GPU KV"]};
 }
 function run(n,records){
  if(!isKind(n.kind))throw Error("run supports only state operators");
  const shapes=Object.fromEntries(Object.entries(records).map(([p,r])=>[p,r.shape]));
  const states=Object.fromEntries(Object.entries(records).map(([p,r])=>[p,r.logicalState||null]));
  const shapeProof=infer(n,shapes,states);
  const value=r=>IR.fromMatrix(r.values,["T","D"]);
  let streamState=null,output;
  if(n.kind==="stream_state_init")streamState=S.init(value(records.x),{axis:"T",origin:n.params.origin,capacity:n.params.capacity});
  if(n.kind==="stream_state_append")streamState=S.append(records.state.streamState,value(records.chunk),{position:n.params.position});
  if(n.kind==="stream_state_fork")streamState=S.fork(records.state.streamState);
  if(n.kind==="stream_state_read")output=S.read(records.state.streamState,n.params.start,n.params.length).tensor;
  if(streamState)output=streamState.tensor;
  if(!output)throw Error("stream operation produced no tensor");
  const vals=IR.plane(output).view,shape=shapeProof.shape;
  if(vals.length!==shape[0]||vals[0].length!==shape[1])throw Error("stream runtime/inference shape diverged");
  const st=streamState||records.state.streamState;
  return {values:vals,shape,flops:null,logicalState:shapeProof.logicalState,
   streamState,
   internals:{logical_clock:{shape:[1,4],values:[[st.origin,st.nextPosition,st.version,st.capacity]],
    note:"[origin,nextPosition,version,capacity]; state is an immutable materialized CPU tensor, not GPU KV"}}
  };
 }
 return Object.freeze({defs,kinds,isKind,isStateProducer,isStatePort,validate,infer,run});
});
