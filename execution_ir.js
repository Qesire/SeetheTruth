/* Immutable node-identity trace envelope for deterministic graph execution. */
(function(root,factory){
"use strict";
const isNode=typeof module==="object"&&module.exports;
const api=factory(isNode?require("./learning_lab.js"):root.SeetheTruthLearning);
if(isNode)module.exports=api;
if(root)root.SeetheTruthExecutionIR=api;
})(typeof window!=="undefined"?window:null,function(L){
"use strict";
function snapshot(raw,options={}){
 if(!L)throw Error("learning runtime unavailable");
 const graph=L.validateGraph(raw),index=L.indexGraph(graph);
 const steps=options.trace?.steps;
 if(!Array.isArray(steps))throw Error("trace.steps must be an array");
 if(steps.length!==index.ordered.length)throw Error("trace step count does not match graph");
 const nodes=index.ordered.map((node,i)=>{
  const step=steps[i];
  if(step.id!==node.id||step.kind!==node.kind)throw Error("trace node identity/order mismatch "+node.id);
  const inputSources=Object.fromEntries(L.defs[node.kind].ports.map(p=>[p,index.incomingByPort.get(node.id+":"+p)||null]));
  const shape=Array.isArray(step.shape)?[...step.shape]:null;
  return Object.freeze({id:node.id,kind:node.kind,sequence:i,
   inputSources:Object.freeze(inputSources),
   shape:shape?Object.freeze(shape):null,
   logicalState:step.logicalState?Object.freeze({...step.logicalState}):null,
   valueAvailable:step.values!==undefined,
   flops:typeof step.flops==="number"?step.flops:null});
 });
 return Object.freeze({schema:"seethetruth-execution-ir-v1",
  graphSchema:graph.schema,
  evidence:"SYNTHETIC",
  producer:"deterministic_browser_teaching_simulation",
  nodes:Object.freeze(nodes)});
}
function evaluate(raw){const trace=L.evaluate(raw);return Object.freeze({trace,ir:snapshot(raw,{trace})});}
return Object.freeze({snapshot,evaluate});
});
