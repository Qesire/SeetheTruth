/* Exact cross-node scalar dependency graph for bounded, freely wired teaching DAGs.
 * Local VJP weights are evaluated only where the mathematical rule is implemented.
 * Unknown operators end at an explicit frontier rather than creating fictitious links.
 */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?{
  L:require("./learning_lab.js"),Ready:require("./graph_readiness.js"),
  Shape:require("./shape_index_rules.js"),ND:require("./nd_tensor.js"),
  Activation:require("./activation_math.js")
 }:{L:root.SeetheTruthLearning,Ready:root.SeetheTruthGraphReadiness,
    Shape:root.SeetheTruthShapeIndexRules,ND:root.SeetheTruthND,
    Activation:root.SeetheTruthActivations});
 if(node)module.exports=api;
 if(root)root.SeetheTruthElementLineage=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const at=(x,i)=>d.ND.at(x,i);
// Matches the deterministic fixed teaching Linear weights in learning_lab.calculate.
// This derivative does not create a trainable W parameter: only the input x is differentiated.
const fixedLinearWeight=(id,r,c)=>{let h=2166136261;
 for(const ch of id+":"+r+":"+c)h=Math.imul(h^ch.charCodeAt(0),16777619);
 return ((h>>>0)%2001-1000)/2600;
};
const leafKinds=new Set(["input","parameter","nd_input","nd_parameter","token_ids"]);
const unary=new Set(["scale","nd_scale","residual_scale","bias","square","temperature"]);
const binary=new Set(["add","mul","subtract","divide"]);
function inspect(raw,options={}){
 const graph=d.L.validateGraph(raw),ordered=d.L.indexGraph(graph);
 const id=options.targetId,node=graph.nodes.find(n=>n.id===id);
 if(!node)throw Error("lineage target node is missing");
 const readiness=d.Ready.check(graph),shapes=new Map(readiness.nodes.filter(n=>n.shape).map(n=>[n.id,n.shape]));
 const targetShape=shapes.get(id);
 if(!targetShape)throw Error("lineage target must have valid upstream tensor shapes");
 const index=options.outputIndex??targetShape.map(n=>Math.min(1,n-1));
 if(!Array.isArray(index)||index.length!==targetShape.length||index.some((v,i)=>!Number.isSafeInteger(v)||v<0||v>=targetShape[i]))
  throw Error("lineage output coordinate out of bounds");
 const limit=options.maxVertices??512,maxEdges=options.maxEdges??2048;
 if(!Number.isSafeInteger(limit)||limit<2||limit>8192||!Number.isSafeInteger(maxEdges)||maxEdges<2||maxEdges>32768)
  throw Error("invalid lineage exploration limit");
 const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
 const incoming=ordered.incomingByPort,orderIds=new Map(ordered.ordered.map((n,i)=>[n.id,i]));
 let numeric=null;
 if(options.trace){
  const steps=options.trace.steps;
  if(!Array.isArray(steps)||steps.length!==ordered.ordered.length||
     steps.some((step,i)=>step.id!==ordered.ordered[i].id||!same(step.shape,shapes.get(step.id))))
   throw Error("lineage numerical trace does not match current graph");
  if(steps.some(step=>step.values===undefined))
   throw Error("lineage numerical trace does not match current graph: incomplete values");
  // Shapes and node IDs alone cannot bind a trace to current parameters or wiring.
  // Re-evaluate the bounded deterministic teaching graph and compare every output.
  // This is a consistency check, not an independent numerical oracle.
  const reference=d.L.evaluate(graph);
  if(steps.some((step,i)=>step.kind!==reference.steps[i].kind||
      !same(step.values,reference.steps[i].values)))
   throw Error("lineage numerical trace does not match current graph: stale or altered values");
  numeric=new Map(steps.map(step=>[step.id,step.values]));
 }
 const sampleValue=(nodeId,coordinate)=>{
  if(!numeric?.has(nodeId))return null;
  const result=at(numeric.get(nodeId),coordinate);
  if(typeof result!=="number"||!Number.isFinite(result))throw Error("nonfinite lineage value");
  return result;
 };
 function inputsFor(n){
  return Object.fromEntries(d.L.defs[n.kind].ports.map(port=>{
   const from=incoming.get(n.id+":"+port);
   if(!from||!shapes.has(from))throw Error("lineage upstream shape missing "+n.id+":"+port);
   return [port,shapes.get(from)];
  }));
 }
 const cutEdges=[];
 function dependencies(n,coordinate){
  const kind=n.kind,shape=shapes.get(n.id),p=n.params||{};
  const inputs=inputsFor(n),read=(port,i)=>sampleValue(incoming.get(n.id+":"+port),i);
  const result=[],stages=[];
  const cut=(port,reason,selection=null)=>cutEdges.push(Object.freeze({
   nodeId:n.id,port,outputIndex:Object.freeze([...coordinate]),reason,selection
  }));
  const add=(port,coords,rule,factor,term="")=>{
   const sh=inputs[port];
   if(!sh||coords.length!==sh.length||coords.some((v,i)=>v<0||v>=sh[i]))
    throw Error("lineage source index mismatch "+n.id+":"+port);
   result.push({port,index:coords,rule,factor,term});
  };
  if(d.Shape.kinds.includes(kind)){
   const projection=d.Shape.analyze(n,inputs,{outputIndex:coordinate});
   for(const source of projection.sourceIndices)
    add(source.port,source.index,projection.gradientRule,1);
   return {rule:projection.rule,links:result,gradientRule:projection.gradientRule};
  }
  if(unary.has(kind)){
   const value=read("x",coordinate);
   let factor,rule;
   if(kind==="scale"||kind==="nd_scale"||kind==="residual_scale"){factor=p.factor;rule="逐元素乘常数";}
   else if(kind==="bias"){factor=1;rule="逐元素加偏置";}
   else if(kind==="square"){factor=value===null?null:2*value;rule="平方，局部导数为 2x";}
   else {factor=1/p.temperature;rule="除以温度参数";}
   add("x",[...coordinate],rule,factor);
   return {rule,links:result,gradientRule:"局部导数乘上游梯度"};
  }
  if(["sigmoid","tanh","relu","softplus"].includes(kind)){
   const x=read("x",coordinate),y=sampleValue(n.id,coordinate);
   const factor=x===null||y===null?null:
    kind==="sigmoid"?y*(1-y):
    kind==="tanh"?1-y*y:
    kind==="relu"?(x>0?1:0):
    x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
   add("x",[...coordinate],"pointwise "+kind+"′(x)",factor);
   return {rule:kind+" 逐坐标数学变换",links:result,
    gradientRule:"从真实前向计算值取得导数，折点按教学引擎分段约定处理"};
  }
  const activationKind=kind.startsWith("nd_")?kind.slice(3):kind;
  if(d.Activation.KINDS.includes(activationKind)){
   const x=read("x",coordinate);
   const factor=x===null?null:d.Activation.derivative(activationKind,x,p);
   add("x",[...coordinate],"f′(x) / 约定分段梯度："+activationKind,factor);
   return {rule:"逐元素激活 "+activationKind,links:result,
    gradientRule:"复用教学引擎的局部梯度；Leaky ReLU、ELU 和 HardSwish 的不可微折点采用实现中明确规定的分段值"};
  }
  if(kind==="rmsnorm"||kind==="qk_norm"||kind==="layernorm"){
   const D=shape[shape.length-1],axis=shape.length-1;
   const row=Array.from({length:D},(_,j)=>{
    const source=[...coordinate];source[axis]=j;return read("x",source);
   });
   const hasNumbers=row.every(v=>v!==null);
   const mean=hasNumbers?row.reduce((s,v)=>s+v,0)/D:null;
   const centered=hasNumbers?row.map(v=>v-mean):null;
   const squares=hasNumbers?(kind==="layernorm"?centered:row).reduce((s,v)=>s+v*v,0)/D:null;
   const den=hasNumbers?Math.sqrt(squares+p.epsilon):null;
   const xi=hasNumbers?(kind==="layernorm"?centered:row)[coordinate[axis]]:null;
   for(let j=0;j<D;j++){
    const source=[...coordinate];source[axis]=j;
    const delta=j===coordinate[axis]?1:0;
    const factor=!hasNumbers?null:kind==="layernorm"?
     (delta-1/D)/den-xi*centered[j]/(D*den*den*den):
     delta/den-xi*row[j]/(D*den*den*den);
    add("x",source,kind==="layernorm"?
     "(δij−1/D)/σ − (xi−μ)(xj−μ)/(Dσ³)":
     "δij/r − xi*xj/(D*r³)",factor);
   }
   return {rule:kind+" 每个输出坐标依赖同一行全部 "+D+" 个输入特征",
    links:result,gradientRule:"显式展开最后一维的完整 Jacobian，而非错误的逐元素缩放"};
  }
  if(binary.has(kind)){
   const a=read("a",coordinate),b=read("b",coordinate);
   if(!same(inputs.a,shape)||!same(inputs.b,shape))throw Error("lineage binary tensor dimensions differ");
   const fa=kind==="add"||kind==="subtract"?1:kind==="mul"?b:b===null?null:1/b;
   const fb=kind==="add"?1:kind==="subtract"?-1:kind==="mul"?a:(a===null||b===null)?null:-a/(b*b);
   add("a",[...coordinate],"∂输出/∂a",fa);add("b",[...coordinate],"∂输出/∂b",fb);
   return {rule:"同一坐标的逐元素运算",links:result,gradientRule:"按各输入的局部导数分流梯度"};
  }
  if(kind==="linear"){
   const x=inputs.x;
   if(x.length!==2||shape.length!==2||x[0]!==shape[0]||shape[1]!==p.out)
    throw Error("lineage Linear input/output shape mismatch");
   for(let k=0;k<x[1];k++)
    add("x",[coordinate[0],k],"W[out,in] · 固定教学线性权重",
     fixedLinearWeight(n.id,coordinate[1],k),"k="+k);
   return {rule:"Y=XWᵀ+b，按输入通道求和；W 为固定的确定性教学参数",
    links:result,gradientRule:"dY[t,h]/dX[t,k]=W[h,k]；无可训练权重端口，不声称 dW"};
  }
  if(kind==="matmul"||kind==="nd_batched_matmul"){
   const a=inputs.a,b=inputs.b,four=kind==="nd_batched_matmul";
   const kdim=four?a[3]:a[1];
   for(let k=0;k<kdim;k++){
    if(four){
     const ac=[a[0]===1?0:coordinate[0],a[1]===1?0:coordinate[1],coordinate[2],k];
     const bc=[b[0]===1?0:coordinate[0],b[1]===1?0:coordinate[1],k,coordinate[3]];
     add("a",ac,"dC/dA = B",read("b",bc),"k="+k);
     add("b",bc,"dC/dB = A",read("a",ac),"k="+k);
    }else{
     const ac=[coordinate[0],k],bc=[k,coordinate[1]];
     add("a",ac,"dC/dA = B",read("b",bc),"k="+k);
     add("b",bc,"dC/dB = A",read("a",ac),"k="+k);
    }
   }
   return {rule:"矩阵乘法按 K 维求和；每项使用一对 A/B 元素",
    links:result,gradientRule:"乘以相应另一输入元素，并对所有 K 维贡献求和"};
  }
  if(kind==="nd_wan_rope3d"){
   const x=inputs.x;
   if(x.length!==4||!same(x,shape))
    throw Error("Wan three-axis rotary requires fixed four-axis input");
   const m=d.ND.wanVideoRoPE3DCoordinate(shape,coordinate,
    p.frames,p.height,p.width,p.base);
   const other=[...coordinate.slice(0,3),m.partner],
    v=read("x",coordinate),paired=read("x",other),odd=coordinate[3]%2===1;
   add("x",[...coordinate],"Wan F/H/W complex-pair cosθ",m.cos,"self");
   add("x",other,odd?"+sinθ partner":"−sinθ partner",
    odd?m.sin:-m.sin,"pair");
   stages.push({stage:"wan_video_3d_rope",axis:m.axis,frame:m.frame,
    height:m.height,width:m.width,frequencyPair:m.pair,angle:m.angle,
    cos:m.cos,sin:m.sin,temporalPairs:m.temporalPairs,
    heightPairs:m.heightPairs,widthPairs:m.widthPairs,
    output:v===null||paired===null?null:v*m.cos+(odd?1:-1)*paired*m.sin});
   return {rule:"固定 Wan2.2 video rope_apply：复数通道切分 F/H/W；输入明确为 [B,N,T,D] 而原代码为 [B,T,N,D]",
    links:result,stages,gradientRule:"对两相邻实通道，局部 VJP 是旋转矩阵的转置"};
  }
  if(kind==="nd_rope"){
   const x=inputs.x,dim=coordinate[3],rotary=p.rotaryDim,base=p.base,position=p.offset+coordinate[2];
   if(x.length!==4||!same(x,shape)||!Number.isSafeInteger(rotary)||
      rotary<2||rotary>x[3]||rotary%2!==0||!Number.isFinite(base)||base<2)
    throw Error("lineage four-axis adjacent-pair RoPE shape/parameters mismatch");
   if(dim>=rotary){
    add("x",[...coordinate],"outside rotary window: identity",1);
    stages.push({stage:"rope_passthrough",channel:dim,rotaryDim:rotary,absolutePosition:position});
   }else{
    const first=dim-dim%2,pair=[...coordinate];pair[3]=dim%2?first:first+1;
    const theta=position/Math.pow(base,first/rotary),cos=Math.cos(theta),sin=Math.sin(theta);
    const self=read("x",coordinate),other=read("x",pair);
    add("x",[...coordinate],"RoPE adjacent pair diagonal cos(theta)",cos,"self");
    add("x",pair,dim%2?"odd channel +sin(theta)":"even channel -sin(theta)",
     dim%2?sin:-sin,"paired");
    stages.push({stage:"adjacent_rope_pair",absolutePosition:position,channel:dim,
     partner:pair[3],angle:theta,cos,sin,input:self,pairedInput:other,
     output:self===null||other===null?null:self*cos+(dim%2?1:-1)*other*sin});
   }
   return {rule:"RoPE 四轴 [B,H,T,D]：同一 Token 的邻接通道二维旋转；位置 offset+t",
    links:result,stages,gradientRule:"转置旋转 Jacobian：对本通道 cosθ，对配对通道 ±sinθ"};
  }
  if(kind==="nd_cache_read"){
   const x=inputs.x;
   if(x.length!==4||shape.length!==4||shape[0]!==x[0]||shape[1]!==x[1]||
      shape[2]!==p.length||shape[3]!==x[3]||p.start+shape[2]>x[2])
    throw Error("lineage cache window shape mismatch");
   const atToken=p.start+coordinate[2],source=[coordinate[0],coordinate[1],atToken,coordinate[3]];
   add("x",source,"pure cache slice, gradient scatters to absolute token",1);
   stages.push({stage:"cache_window",start:p.start,length:p.length,
    windowToken:coordinate[2],absoluteToken:atToken});
   return {rule:"CacheRead4 输出 Token t 读取输入 Token start+t；无状态变更",
    links:result,stages,gradientRule:"对窗口内来源坐标恒等映射，对窗口外来源梯度零"};
  }
  if(kind==="nd_cache_append"){
   const cache=inputs.cache,segment=inputs.segment;
   if(cache.length!==4||segment.length!==4||shape.length!==4||
      cache[0]!==segment[0]||cache[1]!==segment[1]||cache[3]!==segment[3]||
      shape[0]!==cache[0]||shape[1]!==cache[1]||shape[2]!==cache[2]+segment[2]||
      shape[3]!==cache[3])throw Error("lineage cache append geometry mismatch");
   const token=coordinate[2],prefix=cache[2],isPrefix=token<prefix;
   const port=isPrefix?"cache":"segment",source=[...coordinate];
   if(!isPrefix)source[2]=token-prefix;
   add(port,source,"immutable KV append selects "+port+" token",1);
   stages.push({stage:"immutable_cache_append",prefixLength:prefix,
    selectedPort:port,outputToken:token,sourceToken:source[2]});
   return {rule:"CacheAppend4 沿 Token 轴拼接，不修改输入历史帧",
    links:result,stages,gradientRule:"按绝对 Token 来源拆分 VJP，prefix 和新增 segment 各自收集梯度"};
  }
  if(kind==="nd_causal_mask"){
   const x=inputs.x;
   if(x.length!==4||shape.length!==4||!same(x,shape))
    throw Error("lineage causal scores require [B,H,Tq,Tk]");
   const tq=p.queryStart+coordinate[2],tk=p.keyStart+coordinate[3],allowed=tk<=tq;
   // The teaching forward sets masked logits to the constant -1e9:
   // their derivative is zero, not the identity. Mask topology has no input port.
   add("x",[...coordinate],allowed?
    "visible score: ∂maskedScore/∂score=1":
    "future key: fixed -1e9 ∂maskedScore/∂score=0",allowed?1:0);
   stages.push({stage:"absolute_causal_visibility",queryPosition:tq,keyPosition:tk,
    allowed,localDerivative:allowed?1:0,maskedConstant:allowed?null:-1e9});
   return {rule:"Causal Mask 使用绝对 queryStart/keyStart；未来 Key 数值替换为固定 -1e9",
    links:result,stages,gradientRule:"合法分数保留梯度，遮蔽分数 Jacobian 为零（不是对 Mask 求导）"};
  }
  if(kind==="swiglu"||kind==="geglu"){
   if(!same(inputs.gate,shape)||!same(inputs.up,shape))
    throw Error("lineage gated FFN shape mismatch");
   const gate=read("gate",coordinate),up=read("up",coordinate);
   const gelu=kind==="geglu",name=gelu?"gelu_tanh":"silu";
   const activated=gate===null?null:d.Activation.value(name,gate,p);
   const slope=gate===null?null:d.Activation.derivative(name,gate,p);
   add("gate",[...coordinate],"Up · "+name+"′(Gate)",
    slope===null||up===null?null:up*slope);
   add("up",[...coordinate],name+"(Gate)",activated);
   stages.push({stage:"gate_activation",value:activated,derivative:slope},
    {stage:"elementwise_product",gate,up});
   return {rule:"Y="+name+"(Gate) ⊙ Up；门控非线性与上投影独立分流",
    links:result,stages,gradientRule:"dY/dGate=Up·"+name+"′(Gate), dY/dUp="+name+"(Gate)"};
  }
  if(kind==="attention"){
   const q=inputs.q,k=inputs.k,v=inputs.v;
   const [t,c]=coordinate,D=q[1],N=k[0];
   if(q.length!==2||k.length!==2||v.length!==2||
     k[1]!==D||v[0]!==N||shape[0]!==q[0]||shape[1]!==v[1])
    throw Error("lineage attention Q/K/V shape mismatch");
   const scores=Array.from({length:N},(_,j)=>{
    const products=Array.from({length:D},(_,d)=>{
     const a=read("q",[t,d]),b=read("k",[j,d]);
     return a===null||b===null?null:a*b;
    });
    return products.every(x=>x!==null)?
      products.reduce((a,b)=>a+b,0)/Math.sqrt(D):null;
   });
   const ready=scores.every(x=>x!==null);
   const shifted=ready?Math.max(...scores):null;
   const ex=ready?scores.map(z=>Math.exp(z-shifted)):null;
   const den=ready?ex.reduce((a,b)=>a+b,0):null;
   const P=ready?ex.map(x=>x/den):null;
   const Y=sampleValue(n.id,coordinate);
   for(let j=0;j<N;j++){
    const val=read("v",[j,c]),prob=P===null?null:P[j];
    const scoreDelta=prob===null||val===null||Y===null?null:
     prob*(val-Y)/Math.sqrt(D);
    add("v",[j,c],"P[t,j] · V[j,c]",prob,"value-key="+j);
    for(let d=0;d<D;d++){
     const kVal=read("k",[j,d]),qVal=read("q",[t,d]);
     add("q",[t,d],"P[t,j](V[j,c]−Y[t,c])K[j,d]/√D",
      scoreDelta===null||kVal===null?null:scoreDelta*kVal,
      "query-key="+j+",feature="+d);
     add("k",[j,d],"P[t,j](V[j,c]−Y[t,c])Q[t,d]/√D",
      scoreDelta===null||qVal===null?null:scoreDelta*qVal,
      "key="+j+",feature="+d);
    }
    stages.push({stage:"key_softmax_value",key:j,scaledScore:scores[j],
     probability:prob,value:val,weightedContribution:
      val===null||prob===null?null:prob*val,
     softmaxScoreDerivative:scoreDelta===null?null:scoreDelta*Math.sqrt(D)});
   }
   return {rule:"Scores=QKᵀ/√D → row Softmax → Context=P·V；逐 Key 展开数值项",
    links:result,stages,gradientRule:"dV[j,c]=P_j; dQ[t,d]=Σ_j P_j(V_jc−Y_tc)K_jd/√D; dK[j,d]=P_j(V_jc−Y_tc)Q_td/√D"};
  }
  if(kind==="dispatch_routed_tokens"){
   const [row,d]=coordinate,K=inputs.indices[1],t=Math.floor(row/K),slot=row%K;
   if(inputs.indices[0]!==inputs.x[0]||shape[0]!==inputs.x[0]*K||shape[1]!==inputs.x[1])
    throw Error("lineage dispatch route shape mismatch");
   const expert=read("indices",[t,slot]);
   cut("indices","离散专家下标决定复制位置，反向不穿过专家选择",expert);
   add("x",[t,d],"Z[tK+k,d]=X[t,d]",1,"slot="+slot);
   stages.push({stage:"selected_route",token:t,slot,expert});
   return {rule:"Token × TopK 路由复制，k 是离散选择位置",
    links:result,stages,gradientRule:"dX[t,d]=Σ_k dZ[tK+k,d]；相同 Token 多路贡献相加"};
  }
  if(kind==="expert_linear_bank"){
   const [row,h]=coordinate,K=inputs.indices[1],t=Math.floor(row/K),slot=row%K,D=inputs.x[1];
   if(shape[0]!==inputs.x[0]||shape[1]!==inputs.weights[1]||
     inputs.x[0]!==inputs.indices[0]*K||inputs.weights[0]%D)
    throw Error("lineage routed expert weight shape mismatch");
   const expert=read("indices",[t,slot]);
   if(expert!==null&&(!Number.isSafeInteger(expert)||expert<0||expert>=inputs.weights[0]/D))
    throw Error("lineage expert index mismatch");
   cut("indices","专家 ID 是离散路由，权重只对选定专家求导",expert);
   for(let d=0;d<D;d++){
    const input=read("x",[row,d]);
    if(expert!==null){
     const w=read("weights",[expert*D+d,h]);
     add("x",[row,d],"W[eD+d,h]",w,"route="+row+",d="+d);
     add("weights",[expert*D+d,h],"X[route,d]",input,"expert="+expert+",d="+d);
     stages.push({stage:"expert_channel_product",expert,route:row,feature:d,
       activation:input,weight:w,product:input===null||w===null?null:input*w});
    }
   }
   if(expert===null)return null; // structural mode cannot bind expert weight rows
   return {rule:"Y[route,h]=Σ_d X[route,d]·W[e·D+d,h]，e=Indices[t,k]",
    links:result,stages,gradientRule:"对 X 是相应 W；对所选专家 W 是相应 X；复用专家行时梯度相加"};
  }
  if(kind==="routed_weighted_reduce"){
   const [t,d]=coordinate,K=inputs.gates[1];
   if(shape[0]!==inputs.gates[0]||inputs.values[0]!==shape[0]*K||
      inputs.values[1]!==shape[1])throw Error("lineage weighted expert reduce shape mismatch");
   for(let slot=0;slot<K;slot++){
    const value=read("values",[t*K+slot,d]),gate=read("gates",[t,slot]);
    add("values",[t*K+slot,d],"Gate[t,k]",gate,"slot="+slot);
    add("gates",[t,slot],"Value[tK+k,d]",value,"slot="+slot);
    stages.push({stage:"route_weighted_contribution",slot,gate,value,
      product:value===null||gate===null?null:value*gate});
   }
   return {rule:"Y[t,d]=Σ_k Gate[t,k]·Value[tK+k,d]",
    links:result,stages,gradientRule:"值路径由 Gate 加权，门控梯度对通道求和"};
  }
  if(kind==="router_gate_weights"){
   const [t,i]=coordinate,K=inputs.indices[1];
   if(inputs.scores[0]!==shape[0]||inputs.indices[0]!==shape[0]||shape[1]!==K)
    throw Error("lineage router weights shape mismatch");
   const selected=Array.from({length:K},(_,j)=>read("indices",[t,j]));
   const values=selected.map(e=>e===null?null:read("scores",[t,e]));
   if(selected.some(e=>e===null))return null; // cannot locate score dependencies without an executed TopK choice
   const ready=values.every(v=>v!==null);
   const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
   const modes=["identity","sigmoid","softmax"];
   if(!modes.includes(p.mode))throw Error("lineage gate mode not supported");
   let raw=null;
   if(ready){
    raw=p.mode==="identity"?values:p.mode==="sigmoid"?values.map(sigmoid):
     (()=>{const max=Math.max(...values),ex=values.map(z=>Math.exp(z-max)),
      sum=ex.reduce((a,b)=>a+b,0);return ex.map(z=>z/sum);})();
   }
   const den=raw===null?null:raw.reduce((a,b)=>a+b,0);
   const gate=raw===null?null:p.normalize?raw.map(z=>z/den):raw;
   for(let j=0;j<K;j++){
    const expert=selected[j];
    cut("indices","TopK 专家选择 ID 不可微；仅对选中的连续分数反传",expert);
    if(expert===null)continue;
    let factor=null;
    if(ready){
     if(p.mode==="softmax"){
      factor=gate[i]*((i===j?1:0)-gate[j]);
     }else{
      const prime=p.mode==="identity"?1:raw[j]*(1-raw[j]);
      factor=(p.normalize?
       ((i===j?den:0)-raw[i])/(den*den):i===j?1:0)*prime;
     }
    }
    add("scores",[t,expert],"∂Gate_i/∂Score_expert_j",factor,"selection_slot="+j);
    stages.push({stage:"selected_gate",slot:j,expert,affinity:values[j],
     gate:gate===null?null:gate[j],localDerivative:factor});
   }
   return {rule:"先按 Indices 选出 logits，再进行 "+p.mode+
    (p.normalize?" 与归一化":"；未额外归一化"),
    links:result,stages,gradientRule:"每个选定 slot 对 scores 求导；重复 ID 的路径归并；未选专家无连续分数梯度"};
  }
  if(kind==="softmax"||kind==="nd_softmax_last"){
   const last=shape.length-1;
   const yi=sampleValue(n.id,coordinate);
   for(let j=0;j<shape[last];j++){
    const coords=[...coordinate];coords[last]=j;
    const yj=sampleValue(n.id,coords);
    const factor=yi===null||yj===null?null:yi*((j===coordinate[last]?1:0)-yj);
    add("x",coords,"y_i(δ_ij−y_j)",factor);
   }
   return {rule:"Softmax 的一个输出依赖所在行的全部 logits",
    links:result,gradientRule:"Jacobian：y_i(δ_ij−y_j)，不是仅依赖一个输入坐标"};
  }
  return null;
 }
 const vertices=new Map(),edges=[],frontiers=[],seenEdges=new Set();
 let truncated=false;
 function visit(nodeId,coordinate,depth){
  const key=nodeId+"@"+coordinate.join(",");
  if(vertices.has(key))return key;
  if(vertices.size>=limit){truncated=true;return null;}
  const n=nodes.get(nodeId),shape=shapes.get(nodeId);
  const v={key,id:nodeId,kind:n.kind,index:Object.freeze([...coordinate]),shape:Object.freeze([...shape]),
   depth,value:sampleValue(nodeId,coordinate),status:"computed",rule:null,gradientRule:null,stages:null};
  vertices.set(key,v);
  if(leafKinds.has(n.kind)||d.L.defs[n.kind].ports.length===0){v.status="source";return key;}
  const deps=dependencies(n,coordinate);
  if(!deps){v.status="boundary";v.rule="该算子尚未提供精确的逐元素依赖规则";frontiers.push(key);return key;}
  v.rule=deps.rule;v.gradientRule=deps.gradientRule;v.stages=deps.stages||null;
  for(const link of deps.links){
   if(edges.length>=maxEdges){truncated=true;break;}
   const parent=incoming.get(nodeId+":"+link.port);
   const source=visit(parent,link.index,depth+1);
   if(!source)continue;
   const id=key+"|"+link.port+"|"+source+"|"+link.term;
   if(seenEdges.has(id))continue;
   seenEdges.add(id);
   edges.push(Object.freeze({from:key,to:source,port:link.port,term:link.term,
    localDerivative:link.factor,rule:link.rule}));
  }
  return key;
 }
 const rootKey=visit(id,index,0);
 const accumulated=new Map([[rootKey,1]]);
 // Reverse topological propagation sums contributions at shared coordinates.
 const byNode=new Map();
 for(const v of vertices.values()){const group=byNode.get(v.id)||[];group.push(v.key);byNode.set(v.id,group);}
 const outgoing=new Map();
 for(const e of edges){const list=outgoing.get(e.from)||[];list.push(e);outgoing.set(e.from,list);}
 for(const n of [...ordered.ordered].reverse()){
  for(const key of byNode.get(n.id)||[]){
   if(!accumulated.has(key))continue;
   const upstream=accumulated.get(key);
   for(const edge of outgoing.get(key)||[]){
    const contribution=upstream===null||edge.localDerivative===null?null:upstream*edge.localDerivative;
    const old=accumulated.get(edge.to);
    accumulated.set(edge.to,old===undefined?contribution:
     old===null||contribution===null?null:old+contribution);
   }
  }
 }
 const report=Object.freeze({
  schema:"seethetruth-element-lineage-v1",targetId:id,outputIndex:Object.freeze([...index]),
  mode:numeric?"bounded-numeric":"structural",complete:!truncated&&frontiers.length===0,
  truncated,frontiers:Object.freeze(frontiers),cutEdges:Object.freeze(cutEdges),
  discreteConditional:cutEdges.length>0,
  vertices:Object.freeze([...vertices.values()].map(v=>Object.freeze({...v,
    sensitivity:accumulated.has(v.key)?accumulated.get(v.key):null}))),
  edges:Object.freeze(edges),
  note:"单个输出元素对上游元素的依赖和局部 VJP；离散路由按已执行的专家 ID 条件化，不求导；未知算子保留边界。空缺数值导数不等于零。"
 });
 return report;
}
return Object.freeze({inspect});
});
