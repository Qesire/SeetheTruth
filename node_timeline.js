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
function fromSession(raw,transaction,options={}){
 const targetId=options.targetId||null;
 const scope=targetId?d.Branch.slice(raw,targetId):d.L.validateGraph(raw);
 const gradient=d.Gradient.inspect(scope,{forward:transaction.transaction.previous,targetId:transaction.targetId});
 const grades=new Map(gradient.nodes.map(n=>[n.id,n]));
 const updates=new Map(transaction.updates.map(n=>[n.id,n]));
 const after=new Map(transaction.afterIR.nodes.map(n=>[n.id,n]));
 const beforeValues=new Map(transaction.transaction.previous.steps.map(step=>[step.id,step.values]));
 const afterValues=new Map(transaction.transaction.next.steps.map(step=>[step.id,step.values]));
 const valueStats=values=>{if(!Array.isArray(values))return null;
  let elements=0,sum=0,minimum=Infinity,maximum=-Infinity,squares=0;
  const scan=x=>{if(Array.isArray(x)){for(const v of x)scan(v);}
   else{if(typeof x!=="number"||!Number.isFinite(x))throw Error("invalid numeric node output");
    elements++;sum+=x;minimum=Math.min(minimum,x);maximum=Math.max(maximum,x);squares+=x*x;}};
  scan(values);return elements?Object.freeze({elements,mean:sum/elements,min:minimum,max:maximum,l2:Math.sqrt(squares)}):null;
 };
 const nodes=transaction.beforeIR.nodes.map(n=>{
  const g=grades.get(n.id),next=after.get(n.id),update=updates.get(n.id);
  if(!g||!next)throw Error("node timeline identity mismatch "+n.id);
  return Object.freeze({id:n.id,kind:n.kind,sequence:n.sequence,
   forwardShape:n.shape,updatedShape:next.shape,
   forwardValue:valueStats(beforeValues.get(n.id)),updatedValue:valueStats(afterValues.get(n.id)),
   gradient:g.gradient,onLossPath:g.onLossPath,
   localRule:g.localRule,optimizer:update||null});
 });
 return Object.freeze({schema:"seethetruth-node-timeline-v1",evidence:"SYNTHETIC",
  targetId:transaction.targetId,beforeLoss:transaction.beforeLoss,
  afterLoss:transaction.afterLoss,delta:transaction.delta,
  nodes:Object.freeze(nodes),cutEdges:gradient.cutEdges,
  graph:transaction.graph,transaction});
}
function train(raw,options={}){return fromSession(raw,d.Training.step(raw,options),options);}
return Object.freeze({train,fromSession});
});
