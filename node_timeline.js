/* Read-only node timeline projection for teaching and visualization. */
(function(root,factory){"use strict";
const common=typeof module==="object"&&module.exports;
const api=factory(common?{
 Training:require("./training_session.js"),Gradient:require("./gradient_inspection.js"),
 L:require("./learning_lab.js"),Branch:require("./branch_execution.js")
}:{Training:root.SeetheTruthTrainingSession,Gradient:root.SeetheTruthGradientInspection,
 L:root.SeetheTruthLearning,Branch:root.SeetheTruthBranchExecution});
if(common)module.exports=api;if(root)root.SeetheTruthNodeTimeline=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
function train(raw,options={}){
 const transaction=d.Training.step(raw,options),targetId=options.targetId||null;
 const scope=targetId?d.Branch.slice(raw,targetId):d.L.validateGraph(raw);
 const gradient=d.Gradient.inspect(scope,{forward:transaction.transaction.previous,targetId:transaction.targetId});
 const grades=new Map(gradient.nodes.map(n=>[n.id,n]));
 const updates=new Map(transaction.updates.map(n=>[n.id,n]));
 const after=new Map(transaction.afterIR.nodes.map(n=>[n.id,n]));
 const nodes=transaction.beforeIR.nodes.map(n=>{
  const g=grades.get(n.id),next=after.get(n.id),update=updates.get(n.id);
  if(!g||!next)throw Error("node timeline identity mismatch "+n.id);
  return Object.freeze({id:n.id,kind:n.kind,sequence:n.sequence,
   forwardShape:n.shape,updatedShape:next.shape,
   gradient:g.gradient,onLossPath:g.onLossPath,
   localRule:g.localRule,optimizer:update||null});
 });
 return Object.freeze({schema:"seethetruth-node-timeline-v1",evidence:"SYNTHETIC",
  targetId:transaction.targetId,beforeLoss:transaction.beforeLoss,
  afterLoss:transaction.afterLoss,delta:transaction.delta,
  nodes:Object.freeze(nodes),cutEdges:gradient.cutEdges,
  graph:transaction.graph,transaction});
}
return Object.freeze({train});
});
