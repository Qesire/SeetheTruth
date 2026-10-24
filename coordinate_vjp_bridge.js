/* Selected output coordinate → exact forward witness → seeded scalar VJP.
 * Reuses the single free-graph interpreter, mathematical microscope, and
 * cross-node element lineage; never implies parameter training or full export.
 */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?{
  L:require("./learning_lab.js"),M:require("./compute_microscope.js"),
  E:require("./element_lineage.js"),Cap:require("./block_capabilities.js"),
  Grad:require("./graph_autodiff.js"),Objective:require("./training_objective.js"),ND:require("./nd_tensor.js")
 }:{
  L:root.SeetheTruthLearning,M:root.SeetheTruthComputeMicroscope,
  E:root.SeetheTruthElementLineage,Cap:root.SeetheTruthCapabilities,
  Grad:root.SeetheTruthGraphAutodiff,Objective:root.SeetheTruthTrainingObjective,ND:root.SeetheTruthND
 });
 if(node)module.exports=api;
 if(root)root.SeetheTruthCoordinateVJP=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
const SCHEMA="seethetruth-coordinate-vjp-bridge-v1";
function must(ok,message){if(!ok)throw Error("coordinate VJP: "+message);}
function lossAdjoint(graph,run,selected,scalar,options={}){
 must(d.Grad&&d.Objective&&d.ND,"loss-gradient dependency missing");
 const analyzed=d.Objective.analyze(graph);
 const targetId=options.lossId??analyzed.preferred;
 if(!targetId)return Object.freeze({status:"NO_LOSS_OBJECTIVE",
  message:"This graph has no eligible, explicitly derived scalar training loss."});
 must(analyzed.isLoss(targetId),"requested target is not a verified scalar training loss");
 try{
  const backward=d.Grad.backward(graph,run,{targetId});
  const grad=backward.gradients[selected.nodeId];
  must(grad!==undefined,"selected node has no reverse record");
  const adjoint=d.ND.at(grad,selected.index);
  must(Number.isFinite(adjoint),"nonfinite selected loss adjoint");
  const active=backward.pathIds.includes(selected.nodeId);
  const relevantCuts=backward.cutEdges.filter(c=>scalar.vertices.some(v=>v.id===c.nodeId)&&
    scalar.edges.some(e=>e.from.startsWith(c.nodeId+"@")&&e.port===c.port));
  const safe=scalar.complete&&active&&relevantCuts.length===0;
  const sourceContributions=scalar.vertices.filter(v=>v.status==="source").map(v=>{
   const overall=d.ND.at(backward.gradients[v.id],v.index);
   must(Number.isFinite(overall),"nonfinite source loss gradient");
   const throughSelected=safe&&v.sensitivity!==null?adjoint*v.sensitivity:null;
   return Object.freeze({nodeId:v.id,kind:v.kind,index:[...v.index],
    throughSelected,wholeLossGradient:overall,onLossPath:backward.pathIds.includes(v.id),
    stopped:backward.cutEdges.some(c=>c.from===v.id)});
  });
  return Object.freeze({status:"VERIFIED_GRAPH_LOSS_VJP",
   targetId,loss:backward.loss,selectedAdjoint:adjoint,
   selectedOnLossPath:active,completeSelectedPath:scalar.complete,
   throughSelectedEligible:safe,sourceContributions,
   cutEdges:backward.cutEdges.map(c=>({...c})),
   note:"dL/dy_i is the exact whole-loss adjoint at the selected output. "+
    "dL/dy_i × dy_i/dx_j is only the contribution THROUGH that one selected coordinate; "+
    "the whole-loss dL/dx_j includes all other routes. Fixed targets and Detach remain cut."});
 }catch(error){
  // A valid objective can include future operators whose exact VJP has not
  // been implemented. Report the blocker without faking a zero gradient.
  return Object.freeze({status:"UNAVAILABLE_GRAPH_LOSS_VJP",targetId,
   reason:String(error.message||error)});
 }
}
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
   internalStages:root.stages||[],discreteConditional:scalar.discreteConditional,
   conditionalCuts:scalar.cutEdges||[],
   note:"Unit seed at one selected output coordinate; accumulated derivatives are not necessarily a loss gradient."}),
  loss:lossAdjoint(graph,run,{nodeId:step.id,index:[...idx]},scalar,options),
  codegen:Object.freeze({numpyOperatorEligible:capability.numpyEligible,
   torchOperatorEligible:capability.torchEligible,wholeGraphExport:"NOT_VERIFIED",
   vjpExport:"NOT_CLAIMED"})
 });
}
return Object.freeze({SCHEMA,inspect});
});
