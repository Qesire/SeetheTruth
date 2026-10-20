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
  const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
  const edgeByPort=new Map(graph.edges.map(e=>[e.to+":"+e.port,e.from]));
  function inbound(n,port){
   const id=edgeByPort.get(n.id+":"+port),shape=nodeShapes.get(id);
   if(!id||!shape)throw Error("missing valid shape for "+n.id+"."+port);
   return shape;
  }
  let parameterElements=0,implicitTeachingWeights=0,coveredMAC=0,logicalOutputElements=0;
  let parameterTensors=0,coveredGemmNodes=0,otherOperatorNodes=0;
  const rows=[],unprofiledKinds=new Set();
  for(const n of L.order(graph)){
   const shape=nodeShapes.get(n.id);
   const elements=safeProduct(shape,n.id);
   logicalOutputElements=safeSum(logicalOutputElements,elements);
   let mac=null;
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
   rows.push({id:n.id,kind:n.kind,shape:shape.slice(),elements,mac});
  }
  const usage=graph.nodes.filter(n=>n.kind==="parameter"||n.kind==="nd_parameter").map(n=>({
   id:n.id,elements:safeProduct(nodeShapes.get(n.id),n.id),directConsumers:graph.edges.filter(e=>e.from===n.id).length
  }));
  const flops=safeProduct([2,Math.max(1,coveredMAC)],"FLOP")*(coveredMAC===0?0:1);
  return Object.freeze({
   nodeCount:graph.nodes.length,edgeCount:graph.edges.length,parameterTensors,
   physicalParameterElements:parameterElements,implicitTeachingWeightElements:implicitTeachingWeights,
   countedMatmulNodes:coveredGemmNodes,countedMatmulMAC:coveredMAC,countedMatmulFlopTerms:flops,
   logicalOutputElements,unprofiledOperatorNodes:otherOperatorNodes,
   unprofiledKinds:[...unprofiledKinds].sort(),parameterUsage:usage,rows,
   evidence:"exact shape-derived counts for MatMul, batched MatMul, and synthetic Linear only; sums of logical outputs do not measure peak allocated memory; uncounted ops mean total FLOPs are unknown"
  });
 }
 return Object.freeze({profile});
});
