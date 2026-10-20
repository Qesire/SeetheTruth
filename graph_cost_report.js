/* Exact symbolic counts over an actual validated Learning Studio DAG.
 * Deliberately partial: no GPU-time, memory residency, attention kernel or MoE
 * communication model is inferred from an operator count. */
(function(root,factory){
 "use strict";
 const api=factory(
  typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning,
  typeof module==="object"&&module.exports?require("./graph_readiness.js"):root.SeetheTruthGraphReadiness
 );
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthGraphCost=api;
})(typeof window!=="undefined"?window:{},function(L,R){"use strict";
 function safeProduct(shape,label){
  if(!Array.isArray(shape)||!shape.length||shape.some(v=>!Number.isSafeInteger(v)||v<1))
   throw Error(label+" has invalid tensor shape");
  return shape.reduce((a,b)=>{const v=a*b;if(!Number.isSafeInteger(v))throw Error(label+" exceeds integer precision");return v;},1);
 }
 function safeSum(a,b){
  const v=a+b;if(!Number.isSafeInteger(v))throw Error("graph count exceeds integer precision");return v;
 }
 function profile(raw){
  const graph=L.validateGraph(raw),ready=R.check(graph);
  if(!ready.executable)throw Error("Incomplete or invalid graph: "+(ready.firstError?.reason||ready.structuralError||"unknown"));
  const nodeShapes=new Map(ready.nodes.map(n=>[n.id,n.shape]));

  const edgeByPort=new Map(graph.edges.map(e=>[e.to+":"+e.port,e.from]));
  function inbound(n,port){
   const id=edgeByPort.get(n.id+":"+port),shape=nodeShapes.get(id);
   if(!id||!shape)throw Error("missing valid shape for "+n.id+"."+port);
   return shape;
  }
  let parameterElements=0,implicitTeachingWeights=0,coveredMAC=0,logicalOutputElements=0;
  let attentionMAC=0,expertMAC=0,attentionNodes=0,expertNodes=0,cacheSnapshotElements=0;
  const cacheSnapshots=[];
  let parameterTensors=0,coveredGemmNodes=0,otherOperatorNodes=0;
  const rows=[],unprofiledKinds=new Set();
  for(const n of L.order(graph)){
   const shape=nodeShapes.get(n.id);
   const elements=safeProduct(shape,n.id);
   logicalOutputElements=safeSum(logicalOutputElements,elements);
   let mac=null,attention=null,expert=null;
   if(n.kind==="parameter"||n.kind==="nd_parameter"){
    parameterElements=safeSum(parameterElements,elements);
    parameterTensors++;
   }else if(n.kind==="matmul"){
    const a=inbound(n,"a"),b=inbound(n,"b");
    if(a.length!==2||b.length!==2||a[1]!==b[0])throw Error("MatMul dimensions inconsistent at "+n.id);
    mac=safeProduct([a[0],a[1],b[1]],n.id+" MAC");
   }else if(n.kind==="nd_batched_matmul"){
    const a=inbound(n,"a"),b=inbound(n,"b");
    if(a.length!==4||b.length!==4||a[3]!==b[2]||
       (a[0]!==b[0]&&a[0]!==1&&b[0]!==1)||
       (a[1]!==b[1]&&a[1]!==1&&b[1]!==1))throw Error("Batched MatMul dimensions inconsistent at "+n.id);
    mac=safeProduct([Math.max(a[0],b[0]),Math.max(a[1],b[1]),a[2],a[3],b[3]],n.id+" MAC");
   }else if(n.kind==="attention"){
    const q=inbound(n,"q"),k=inbound(n,"k"),v=inbound(n,"v");
    if(q.length!==2||k.length!==2||v.length!==2||q[1]!==k[1]||k[0]!==v[0])
     throw Error("attention dimensions inconsistent at "+n.id);
    attention=safeProduct([q[0],k[0],q[1]+v[1]],n.id+" attention MAC");
   }else if(n.kind==="gqa"){
    const q=inbound(n,"q"),k=inbound(n,"k"),v=inbound(n,"v");
    const p={...L.defs[n.kind].params,...n.params};
    if(q.length!==2||k.length!==2||v.length!==2||k[0]!==v[0]||
       p.heads%p.kvHeads!==0||q[1]%p.heads!==0||k[1]%p.kvHeads!==0||
       v[1]%p.kvHeads!==0||q[1]/p.heads!==k[1]/p.kvHeads)
     throw Error("GQA attention shape or head divisibility invalid at "+n.id);
    attention=safeProduct([p.heads,q[0],k[0],q[1]/p.heads+v[1]/p.kvHeads],n.id+" GQA MAC");
   }else if(n.kind==="inference_sparse_attention"){
    const q=inbound(n,"q"),k=inbound(n,"k"),v=inbound(n,"v"),ix=inbound(n,"indices");
    if(q.length!==2||k.length!==2||v.length!==2||ix.length!==2||
      q[0]!==ix[0]||q[1]!==k[1]||k[0]!==v[0]||ix[1]>k[0])
     throw Error("indexed sparse attention dimensions inconsistent at "+n.id);
    // Exact QK+PV GEMM terms on selected indices only. TopK indexer, gather and
    // Softmax are excluded; selection-quality and kernel latency are unknown.
    attention=safeProduct([q[0],ix[1],q[1]+v[1]],n.id+" sparse attention MAC");
   }else if(n.kind==="expert_linear_bank"){
    const x=inbound(n,"x"),indices=inbound(n,"indices"),weights=inbound(n,"weights");
    if(x.length!==2||indices.length!==2||weights.length!==2||
       x[0]!==indices[0]*indices[1]||weights[0]%x[1]!==0)
     throw Error("Expert Linear dimensions inconsistent at "+n.id);
    expert=safeProduct([x[0],x[1],weights[1]],n.id+" routed expert MAC");
   }else if(n.kind==="nd_cache_append"||n.kind==="cache_write_segment"){
    cacheSnapshotElements=safeSum(cacheSnapshotElements,elements);
    cacheSnapshots.push({id:n.id,kind:n.kind,shape:shape.slice(),logicalElements:elements,unit:"elements, not bytes",keyOrValue:"unspecified"});
   }else if(n.kind==="linear"){
    const x=inbound(n,"x");
    if(x.length!==2)throw Error("Linear input must be 2D at "+n.id);
    const out=n.params.out;
    mac=safeProduct([x[0],x[1],out],n.id+" MAC");
    implicitTeachingWeights=safeSum(implicitTeachingWeights,safeSum(safeProduct([x[1],out],n.id+" weights"),out));
   }else if(L.defs[n.kind].ports.length){
    // These are real graph operations but outside the limited GEMM scope.
    otherOperatorNodes++;
    unprofiledKinds.add(n.kind);
   }
   if(mac!==null){coveredMAC=safeSum(coveredMAC,mac);coveredGemmNodes++;}
   if(attention!==null){attentionMAC=safeSum(attentionMAC,attention);attentionNodes++;}
   if(expert!==null){expertMAC=safeSum(expertMAC,expert);expertNodes++;}
   if(attention!==null||expert!==null||n.kind==="nd_cache_append"||n.kind==="cache_write_segment"){
     otherOperatorNodes++;
     unprofiledKinds.add(n.kind+" (partial accounting)");
   }
   rows.push({id:n.id,kind:n.kind,shape:shape.slice(),elements,mac,attentionMAC:attention,expertMAC:expert});
  }
  const usage=graph.nodes.filter(n=>n.kind==="parameter"||n.kind==="nd_parameter").map(n=>({
   id:n.id,elements:safeProduct(nodeShapes.get(n.id),n.id),directConsumers:graph.edges.filter(e=>e.from===n.id).length
  }));
  const flops=safeProduct([2,Math.max(1,coveredMAC)],"FLOP")*(coveredMAC===0?0:1);
  const knownArithmeticMAC=safeSum(safeSum(coveredMAC,attentionMAC),expertMAC);
  const knownArithmeticFlopTerms=knownArithmeticMAC===0?0:safeProduct([2,knownArithmeticMAC],"known FLOP terms");
  return Object.freeze({
   nodeCount:graph.nodes.length,edgeCount:graph.edges.length,parameterTensors,
   physicalParameterElements:parameterElements,implicitTeachingWeightElements:implicitTeachingWeights,
   countedMatmulNodes:coveredGemmNodes,countedMatmulMAC:coveredMAC,countedMatmulFlopTerms:flops,
   logicalOutputElements,unprofiledOperatorNodes:otherOperatorNodes,
   attentionNodes,attentionMAC,expertLinearNodes:expertNodes,expertLinearMAC:expertMAC,
   knownArithmeticMAC,knownArithmeticFlopTerms,cacheSnapshotElements,cacheSnapshots,
   unprofiledKinds:[...unprofiledKinds].sort(),parameterUsage:usage,rows,
   evidence:"shape-derived MAC for explicit GEMM and two internal attention GEMMs and routed expert projection; excludes attention softmax/mask/rope, router, kernel IO, cache traffic, hardware time; cacheSnapshotElements counts sequential logical snapshots, NOT peak allocated memory or bytes"
  });
 }
 function compare(left,right){
  const a=profile(left),b=profile(right);
  const metrics=["physicalParameterElements","implicitTeachingWeightElements","countedMatmulMAC",
    "attentionMAC","expertLinearMAC","knownArithmeticMAC","cacheSnapshotElements"];
  const deltas=Object.fromEntries(metrics.map(key=>[key,Object.freeze({
   baseline:a[key],candidate:b[key],difference:b[key]-a[key],
   ratio:a[key]===0?null:b[key]/a[key]
  })]));
  return Object.freeze({baseline:a,candidate:b,deltas,
   evidence:"only graph-based arithmetic and snapshot elements are comparable, not quality, hardware cost or semantically equivalent computation"});
 }
 return Object.freeze({profile,compare});
});
