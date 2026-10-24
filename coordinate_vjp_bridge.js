/* Selected output coordinate → exact forward witness → seeded scalar VJP.
 * Reuses the single free-graph interpreter, mathematical microscope, and
 * cross-node element lineage; never implies parameter training or full export.
 */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?{
  L:require("./learning_lab.js"),M:require("./compute_microscope.js"),
  E:require("./element_lineage.js"),Cap:require("./block_capabilities.js")
 }:{
  L:root.SeetheTruthLearning,M:root.SeetheTruthComputeMicroscope,
  E:root.SeetheTruthElementLineage,Cap:root.SeetheTruthCapabilities
 });
 if(node)module.exports=api;
 if(root)root.SeetheTruthCoordinateVJP=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
const SCHEMA="seethetruth-coordinate-vjp-bridge-v1";
function must(ok,message){if(!ok)throw Error("coordinate VJP: "+message);}
function inspect(graph,run,frame,options={}){
 must(d.L&&d.M&&d.E&&d.Cap,"missing shared runtime");
 must(run&&run.mode!=="concept"&&frame?.evidenceTier===d.M.TIERS.MICRO,
  "only current bounded numerical execution supports VJP");
 const canonical=d.M.learning(graph,run,d.L);
 // Authentication is about the graph's *entire* trace, not just the selected
 // scalar. Never derive an attractive derivative from an edited display frame.
 must(JSON.stringify(frame)===JSON.stringify(canonical),"microscope frame differs from executed graph");
 const step=canonical.steps.find(s=>s.id===options.stepId);
 must(step&&step.out.values!==null,"selected graph node unavailable");
 const shape=step.out.shape,idx=options.coordinate;
 must(Array.isArray(idx)&&idx.length===shape.length&&idx.every((v,i)=>
  Number.isSafeInteger(v)&&v>=0&&v<shape[i]),"selected coordinate out of bounds");
 const rank=shape.length,leading=rank>2?idx.slice(0,-2):[];
 const row=rank===1?0:idx[rank-2],col=idx[rank-1];
 const forward=d.M.dot(canonical,step.id,row,col,leading);
 const scalar=d.E.inspect(graph,{targetId:step.id,outputIndex:idx,
  trace:run,maxVertices:512,maxEdges:2048});
 must(scalar.mode==="bounded-numeric"&&scalar.targetId===step.id&&
  scalar.outputIndex.join(",")===idx.join(","),"backward observation is unbound");
 const root=scalar.vertices.find(v=>v.key===step.id+"@"+idx.join(","));
 must(root&&root.value===forward.value&&root.sensitivity===1,
  "selected output forward/backward witness mismatch");
 const capability=d.Cap.audit().rows.find(row=>row.kind===step.kind);
 must(capability,"selected operator absent from capability registry");
 const sources=scalar.vertices.filter(v=>v.status==="source").map(v=>({
  nodeId:v.id,kind:v.kind,index:v.index,value:v.value,sensitivity:v.sensitivity
 }));
 const localEdges=scalar.edges.filter(e=>e.from===root.key).map(e=>({
  port:e.port,source:e.to,localDerivative:e.localDerivative,rule:e.rule,term:e.term
 }));
 return Object.freeze({
  schema:SCHEMA,evidence:"BOUNDED_SYNTHETIC_COORDINATE_VJP_NOT_FULL_GRAPH_LOSS",
  selected:Object.freeze({nodeId:step.id,kind:step.kind,index:[...idx],
   value:forward.value,forwardReconstructable:forward.reconstructable,
   forwardError:forward.reconstructable?forward.error:null}),
  backward:Object.freeze({complete:scalar.complete,truncated:scalar.truncated,
   boundaryNodes:scalar.frontiers.slice(),nodeCount:scalar.vertices.length,
   edgeCount:scalar.edges.length,localEdges,sources,
   note:"Unit seed at one selected output coordinate; accumulated derivatives are not necessarily a loss gradient."}),
  codegen:Object.freeze({numpyOperatorEligible:capability.numpyEligible,
   torchOperatorEligible:capability.torchEligible,wholeGraphExport:"NOT_VERIFIED",
   vjpExport:"NOT_CLAIMED"})
 });
}
return Object.freeze({SCHEMA,inspect});
});
