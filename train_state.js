/* CPU-only training transaction for an editable small graph.
 * The immutable graph JSON is the source of truth for both parameter values
 * and AdamW first/second moments. No hidden storage, GPU, or simulation claim.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthTrainStep=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const isNode=typeof module==="object"&&module.exports;
const L=isNode?require("./learning_lab.js"):root.SeetheTruthLearning;
const Grad=isNode?require("./graph_autodiff.js"):root.SeetheTruthGraphAutodiff;
const AdamW=isNode?require("./adamw_micro.js"):root.SeetheTruthAdamW;
function step(raw,options={}){
 const graph=L.validateGraph(raw);
 const parameters=graph.nodes.filter(n=>n.kind==="parameter");
 if(!parameters.length)throw Error("training requires a connected trainable Parameter block");
 const oldForward=L.evaluate(graph);
 const reverse=Grad.backward(graph,oldForward,{targetId:options.targetId});
 const active=parameters.filter(n=>reverse.pathIds.includes(n.id));
 if(!active.length)throw Error("no trainable parameter is reachable from the selected loss");
 if(active.length>12)throw Error("too many parameter nodes for bounded optimizer");
 const next=JSON.parse(JSON.stringify(graph));
 const records=[];
 for(const node of next.nodes.filter(n=>n.kind==="parameter"&&active.some(a=>a.id===n.id))){
   const current=node.params.values,grad=reverse.gradients[node.id],previous=node.params.optimizer;
   const update=AdamW.step(current,grad,previous,{
     lr:options.lr??0.01,weightDecay:options.weightDecay??0.01,
     beta1:options.beta1??0.9,beta2:options.beta2??0.999,eps:options.eps??1e-8
   });
   if(update.state.t>10000)throw Error("educational optimizer iteration limit");
   node.params.values=update.parameter;
   node.params.optimizer=update.state;
   records.push({id:node.id,shape:[current.length,current[0].length],
      gradientNorm:Math.hypot(...grad.flat()),updateNorm:Math.hypot(...update.update.flat()),optimizerStep:update.state.t});
 }
 const validated=L.validateGraph(next);
 const nextForward=L.evaluate(validated);
 const before=oldForward.steps.find(s=>s.id===reverse.targetId).values[0][0];
 const after=nextForward.steps.find(s=>s.id===reverse.targetId).values[0][0];
 if(!Number.isFinite(before)||!Number.isFinite(after))throw Error("non-finite loss in training step");
 return {graph:validated,beforeLoss:before,afterLoss:after,delta:after-before,
   targetId:reverse.targetId,records,previous:oldForward,next:nextForward,
   evidence:"exact bounded graph VJP + persisted AdamW matrix state; no real pretrained model or GPU execution"};
}
function steps(raw,options={}){
 const count=options.count??1;
 if(!Number.isSafeInteger(count)||count<1||count>8)throw Error("training step count must be an integer from 1 to 8");
 const kwargs={...options};delete kwargs.count;
 let graph=L.validateGraph(raw);const history=[];
 for(let i=0;i<count;i++){
   const result=step(graph,kwargs);
   history.push({iteration:i+1,beforeLoss:result.beforeLoss,afterLoss:result.afterLoss,
     delta:result.delta,records:result.records});
   graph=result.graph;
 }
 return {graph,history,firstLoss:history[0].beforeLoss,finalLoss:history.at(-1).afterLoss,
   next:L.evaluate(graph),evidence:"atomic bounded multi-step training, each AdamW iteration recomputes forward and VJP"};
}
function inspect(raw){
 const graph=L.validateGraph(raw);
 return graph.nodes.filter(n=>n.kind==="parameter").map(n=>({id:n.id,shape:[n.params.values.length,n.params.values[0].length],optimizerStep:n.params.optimizer.t}));
}
return Object.freeze({step,steps,inspect});
});
