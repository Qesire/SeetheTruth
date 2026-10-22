/* Training transaction bridge: preserves existing optimizer arithmetic and graph snapshots. */
(function(root,factory){"use strict";
const node=typeof module==="object"&&module.exports;
const api=factory(node?{
 Train:require("./train_state.js"),IR:require("./execution_ir.js"),
 Branch:require("./branch_execution.js"),L:require("./learning_lab.js")
}:{Train:root.SeetheTruthTrainStep,IR:root.SeetheTruthExecutionIR,
 Branch:root.SeetheTruthBranchExecution,L:root.SeetheTruthLearning});
if(node)module.exports=api;if(root)root.SeetheTruthTrainingSession=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
function step(raw,options={}){
 const result=d.Train.step(raw,options),targetId=options.targetId||null;
 const beforeGraph=targetId?d.Branch.slice(raw,targetId):d.L.validateGraph(raw);
 const afterGraph=targetId?d.Branch.slice(result.graph,targetId):result.graph;
 const before=d.IR.snapshot(beforeGraph,{trace:result.previous});
 const after=d.IR.snapshot(afterGraph,{trace:result.next});
 const beforeIds=new Set(before.nodes.map(x=>x.id)),afterIds=new Set(after.nodes.map(x=>x.id));
 const records=result.records.map(r=>{
  if(!beforeIds.has(r.id)||!afterIds.has(r.id))throw Error("training record has no matching graph node "+r.id);
  return Object.freeze({id:r.id,shape:Object.freeze([...r.shape]),gradientNorm:r.gradientNorm,
   updateNorm:r.updateNorm,optimizer:r.optimizer,optimizerStep:r.optimizerStep});
 });
 return Object.freeze({schema:"seethetruth-training-session-v1",
  evidence:"SYNTHETIC",targetId:result.targetId,
  beforeLoss:result.beforeLoss,afterLoss:result.afterLoss,delta:result.delta,
  beforeIR:before,afterIR:after,updates:Object.freeze(records),
  graph:result.graph,transaction:result});
}
function steps(raw,options={}){
 const count=options.count??1;
 if(!Number.isSafeInteger(count)||count<1||count>8)throw Error("training step count must be an integer from 1 to 8");
 const params={...options};delete params.count;
 let graph=d.L.validateGraph(raw);const history=[],sessions=[];
 for(let i=0;i<count;i++){
  const result=step(graph,params);
  history.push(Object.freeze({iteration:i+1,beforeLoss:result.beforeLoss,afterLoss:result.afterLoss,
   delta:result.delta,records:result.transaction.records}));
  sessions.push(result);
  graph=result.graph;
 }
 const targetId=options.targetId||null;
 const finalGraph=targetId?d.Branch.slice(graph,targetId):graph;
 return Object.freeze({graph,history,firstLoss:history[0].beforeLoss,finalLoss:history.at(-1).afterLoss,
  next:d.L.evaluate(finalGraph),sessions:Object.freeze(sessions),
  finalIR:sessions.at(-1).afterIR,
  evidence:"atomic bounded multi-step training, each iteration replays actual forward and VJP"});
}
return Object.freeze({step,steps});
});
