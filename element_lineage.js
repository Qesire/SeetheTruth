/* Exact cross-node scalar dependency graph for bounded, freely wired teaching DAGs.
 * Local VJP weights are evaluated only where the mathematical rule is implemented.
 * Unknown operators end at an explicit frontier rather than creating fictitious links.
 */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?{
  L:require("./learning_lab.js"),Ready:require("./graph_readiness.js"),
  Shape:require("./shape_index_rules.js"),ND:require("./nd_tensor.js")
 }:{L:root.SeetheTruthLearning,Ready:root.SeetheTruthGraphReadiness,
    Shape:root.SeetheTruthShapeIndexRules,ND:root.SeetheTruthND});
 if(node)module.exports=api;
 if(root)root.SeetheTruthElementLineage=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const at=(x,i)=>d.ND.at(x,i);
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
  if(steps.every(step=>step.values!==undefined))numeric=new Map(steps.map(step=>[step.id,step.values]));
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
 function dependencies(n,coordinate){
  const kind=n.kind,shape=shapes.get(n.id),p=n.params||{};
  const inputs=inputsFor(n),read=(port,i)=>sampleValue(incoming.get(n.id+":"+port),i);
  const result=[];
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
  if(binary.has(kind)){
   const a=read("a",coordinate),b=read("b",coordinate);
   if(!same(inputs.a,shape)||!same(inputs.b,shape))throw Error("lineage binary tensor dimensions differ");
   const fa=kind==="add"||kind==="subtract"?1:kind==="mul"?b:b===null?null:1/b;
   const fb=kind==="add"?1:kind==="subtract"?-1:kind==="mul"?a:(a===null||b===null)?null:-a/(b*b);
   add("a",[...coordinate],"∂输出/∂a",fa);add("b",[...coordinate],"∂输出/∂b",fb);
   return {rule:"同一坐标的逐元素运算",links:result,gradientRule:"按各输入的局部导数分流梯度"};
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
   depth,value:sampleValue(nodeId,coordinate),status:"computed",rule:null,gradientRule:null};
  vertices.set(key,v);
  if(leafKinds.has(n.kind)||d.L.defs[n.kind].ports.length===0){v.status="source";return key;}
  const deps=dependencies(n,coordinate);
  if(!deps){v.status="boundary";v.rule="该算子尚未提供精确的逐元素依赖规则";frontiers.push(key);return key;}
  v.rule=deps.rule;v.gradientRule=deps.gradientRule;
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
  truncated,frontiers:Object.freeze(frontiers),
  vertices:Object.freeze([...vertices.values()].map(v=>Object.freeze({...v,
    sensitivity:accumulated.has(v.key)?accumulated.get(v.key):null}))),
  edges:Object.freeze(edges),
  note:"单个输出元素对上游元素的依赖和局部 VJP；仅已验证算子会继续展开。空缺的数值导数不等于零。"
 });
 return report;
}
return Object.freeze({inspect});
});
