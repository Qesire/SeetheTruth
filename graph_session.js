/* Shared graph inspection: one identity space for readiness and numeric trace. */
(function(root,factory){
"use strict";
const isNode=typeof module==="object"&&module.exports;
const api=factory(isNode?{
 learning:require("./learning_lab.js"),
 readiness:require("./graph_readiness.js"),
 execution:require("./execution_ir.js"),
 branch:require("./branch_execution.js"),
 contracts:require("./operator_contracts.js")
}:{
 learning:root.SeetheTruthLearning,readiness:root.SeetheTruthGraphReadiness,
 execution:root.SeetheTruthExecutionIR,branch:root.SeetheTruthBranchExecution,
 contracts:root.SeetheTruthOperatorContracts
});
if(isNode)module.exports=api;
if(root)root.SeetheTruthGraphSession=api;
})(typeof window!=="undefined"?window:null,function(d){
"use strict";
function inspect(raw,options={}){
 const numeric=options.numeric===true,targetId=options.targetId||null;
 const ready=d.readiness.check(raw,{targetId});
 let ir=null,trace=null;
 if(ready.executable&&numeric){
  const graph=targetId?d.branch.slice(raw,targetId):d.learning.validateGraph(raw);
  try{
   const result=d.execution.evaluate(graph);
   ir=result.ir;trace=result.trace;
  }catch(error){
   return Object.freeze({schema:"seethetruth-graph-session-v1",executable:false,
    readiness:ready,ir:null,trace:null,
    error:Object.freeze({nodeId:error.nodeId||null,reason:error.message})});
  }
 }
 const observed=new Map(ir?.nodes.map(n=>[n.id,n])||[]);
 const nodes=ready.nodes.map(n=>{
  const contract=d.contracts.resolve(n.kind),runtime=observed.get(n.id)||null;
  if(runtime&&n.shape&&JSON.stringify(runtime.shape)!==JSON.stringify(n.shape))
   throw Error("static/numeric shape disagreement "+n.id);
  return Object.freeze({id:n.id,kind:n.kind,status:n.status,
   shape:n.shape?Object.freeze([...n.shape]):null,
   ports:contract.ports,inputSources:runtime?.inputSources||null,
   numericObserved:!!runtime,reason:n.reason||null});
 });
 return Object.freeze({schema:"seethetruth-graph-session-v1",
  executable:ready.executable,scope:ready.scope||"full_graph",
  targetId,readiness:ready,ir,trace,error:null,nodes:Object.freeze(nodes)});
}
return Object.freeze({inspect});
});
