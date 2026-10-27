/* Real bounded graph parameter update -> whole-loss gradient coordinate ->
 * optional selected intermediate coordinate path -> native local VJP terms.
 * Original GraphAutodiff, CoordinateVJP and TrainingSession keep arithmetic authority.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const api=factory(isNode?{
  L:require("./learning_lab.js"),Branch:require("./branch_execution.js"),
  Session:require("./training_session.js"),Grad:require("./graph_autodiff.js"),
  Coord:require("./coordinate_vjp_bridge.js"),M:require("./compute_microscope.js"),
  ND:require("./nd_tensor.js")
 }:{
  L:root.SeetheTruthLearning,Branch:root.SeetheTruthBranchExecution,
  Session:root.SeetheTruthTrainingSession,Grad:root.SeetheTruthGraphAutodiff,
  Coord:root.SeetheTruthCoordinateVJP,M:root.SeetheTruthComputeMicroscope,
  ND:root.SeetheTruthND
 });
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthTrainingCoordinateProof=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
 const SCHEMA="seethetruth-training-coordinate-proof-v1";
 const fail=why=>{throw Error("training coordinate proof: "+why);};
 const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 const close=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&
  Math.abs(a-b)<=2e-11+1e-10*Math.max(Math.abs(a),Math.abs(b));
 function inspect(raw,options={},selection={}){
  if(!raw||!options||typeof options!=="object"||!selection||typeof selection!=="object")
   fail("requires original graph, training options and a selected parameter coordinate");
  const graph=d.L.validateGraph(raw),id=selection.parameterId,idx=selection.index;
  if(typeof id!=="string"||!Array.isArray(idx)||!idx.length||idx.length>4||
     idx.some(i=>!Number.isSafeInteger(i)||i<0))
   fail("invalid selected parameter coordinate");
  const scope=options.targetId?d.Branch.slice(graph,options.targetId):graph;
  const param=scope.nodes.find(n=>n.id===id);
  if(!param||!["parameter","nd_parameter"].includes(param.kind)||param.params.frozen)
   fail("requested coordinate must be a trainable parameter on the chosen loss path");
  const dims=d.ND.shape(param.params.values);
  if(dims.length!==idx.length||
     idx.some((v,i)=>v>=dims[i]))fail("parameter coordinate out of bounds");
  const session=d.Session.step(graph,options),previous=session.transaction.previous;
  const record=session.updates.find(u=>u.id===id);
  if(!record)fail("parameter has no actual optimizer update on selected loss path");
  const back=d.Grad.backward(scope,previous,{targetId:session.targetId});
  if(!back.pathIds.includes(id))fail("parameter has no loss-path VJP");
  const grads=back.gradients[id],g=d.ND.at(grads,idx);
  const oldValue=d.ND.at(param.params.values,idx);
  const updated=session.graph.nodes.find(n=>n.id===id);
  if(!updated)fail("updated parameter identity unavailable");
  const newValue=d.ND.at(updated.params.values,idx);
  const gradL2=Math.hypot(...d.ND.flatten(grads));
  const displacement=d.ND.flatten(param.params.values).map((v,i)=>
   v-d.ND.flatten(updated.params.values)[i]);
  if(!close(gradL2,record.gradientNorm)||
     !close(Math.hypot(...displacement),record.updateNorm)||
     !equal(record.shape,dims)||record.optimizerStep!==updated.params.optimizer.t)
   fail("native gradient or optimizer metadata mismatches source transaction");
  const originalState=param.params.optimizer,newState=updated.params.optimizer;
  const moment=(slot)=>({before:d.ND.at(originalState[slot],idx),
    after:d.ND.at(newState[slot],idx)});
  const selectedPath=selection.viaNodeId;
  let path={status:"NOT_REQUESTED"};
  if(selectedPath!==undefined){
   if(typeof selectedPath!=="string"||!Array.isArray(selection.viaIndex))
    fail("intermediate coordinate requires node identity and explicit indices");
   if(!scope.nodes.some(n=>n.id===selectedPath))
    fail("selected intermediate is not in the source training scope");
   const canonical=d.M.learning(scope,previous,d.L);
   const detail=d.Coord.inspect(scope,previous,canonical,{
    stepId:selectedPath,coordinate:selection.viaIndex,lossId:session.targetId});
   const match=detail.loss.sourceContributions?.find(c=>
     c.nodeId===id&&equal(c.index,idx));
   const unitSource=detail.backward.sources?.find(c=>
     c.nodeId===id&&equal(c.index,idx));
   const eligible=detail.backward.complete&&
    detail.loss.status==="VERIFIED_GRAPH_LOSS_VJP"&&
    detail.loss.throughSelectedEligible;
   if(eligible&&match&&unitSource&&Number.isFinite(unitSource.sensitivity)&&
      !close(detail.loss.selectedAdjoint*unitSource.sensitivity,match.throughSelected))
    fail("selected loss adjoint multiplied by unit VJP differs from native path contribution");
   if(eligible&&match&&!close(match.wholeLossGradient,g))
    fail("selected path and full-loss gradient disagree with source autodiff");
   path={status:eligible
    ?match&&Number.isFinite(match.throughSelected)
      ?"VERIFIED_SELECTED_PATH_CONTRIBUTION":"VERIFIED_ZERO_SELECTED_PATH"
    :"PARTIAL_OR_UNAVAILABLE_PATH",
    selected:detail.selected,localTerms:detail.backward.localEdges,
    internalStages:detail.backward.internalStages,
    complete:detail.backward.complete,frontiers:detail.backward.boundaryNodes,
    loss:detail.loss.status==="VERIFIED_GRAPH_LOSS_VJP"
     ?{targetId:detail.loss.targetId,selectedAdjoint:detail.loss.selectedAdjoint,
       throughSelectedEligible:detail.loss.throughSelectedEligible}: {status:detail.loss.status},
    parameterContribution:eligible?(match?match.throughSelected:0):null,
    parameterUnitVJP:eligible?(unitSource?unitSource.sensitivity:0):null,
    otherPathsMayContribute:true,
    sourceProof:detail.schema};
  }
  return Object.freeze({schema:SCHEMA,evidence:"EXECUTED_BOUNDED_CPU_TRAINING_COORDINATE",
   targetId:session.targetId,parameterId:id,index:[...idx],shape:dims,
   gradient:g,wholeLossGradient:g,gradientNorm:record.gradientNorm,
   updateNorm:record.updateNorm,
   parameter:{before:oldValue,after:newValue,delta:newValue-oldValue},
   optimizer:{kind:record.optimizer,stepBefore:originalState.t,
    stepAfter:newState.t,firstMoment:moment("m"),secondMoment:moment("v")},
   loss:{before:session.beforeLoss,after:session.afterLoss},
   path,source:{sessionSchema:session.schema,vjp:"graph_autodiff.js",
    optimizer:"train_state.js",coordinate:"coordinate_vjp_bridge.js"},
   note:"The whole-loss gradient and the selected-output path contribution are different quantities; only the original source engines execute optimizer and VJP."});
 }
 return Object.freeze({SCHEMA,inspect});
});
