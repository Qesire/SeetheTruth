'use strict';
const assert=require('node:assert/strict');
const ND=require('../nd_tensor.js'),AdamW=require('../nd_adamw_micro.js'),L=require('../learning_lab.js');
const Train=require('../train_state.js'),Auto=require('../graph_autodiff.js'),Readiness=require('../graph_readiness.js');
const near=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
const original=ND.seedTensor([1,2,3,2],5),g=ND.seedTensor([1,2,3,2],9);
const snapshot=JSON.stringify({original,g}),options={lr:.01,beta1:.9,beta2:.999,eps:1e-8,weightDecay:.04};
const first=AdamW.step(original,g,null,options);
assert.deepEqual(ND.shape(first.parameter),[1,2,3,2]);assert.equal(first.state.t,1);
ND.indices([1,2,3,2],idx=>{
 const w=ND.at(original,idx),v=ND.at(g,idx);
 near(ND.at(first.state.m,idx),.1*v);near(ND.at(first.state.v,idx),.001*v*v);
 near(ND.at(first.parameter,idx),w-.01*v/(Math.abs(v)+1e-8)-.01*.04*w,1e-10);
});
const second=AdamW.step(first.parameter,g,first.state,options);
assert.equal(second.state.t,2);
ND.indices([1,2,3,2],idx=>{
 const m=ND.at(second.state.m,idx)/(1-options.beta1**2),
       v=ND.at(second.state.v,idx)/(1-options.beta2**2),
       w=ND.at(first.parameter,idx);
 near(ND.at(second.parameter,idx),w-options.lr*(m/(Math.sqrt(v)+options.eps)+options.weightDecay*w));
});
assert.equal(JSON.stringify({original,g}),snapshot,'optimizer must be pure');
const zero=AdamW.step(ND.seedTensor([1,1,2,2],1),ND.build([1,1,2,2],()=>0),null,{lr:.1,weightDecay:.1});
ND.indices([1,1,2,2],idx=>near(ND.at(zero.parameter,idx),.99*ND.at(ND.seedTensor([1,1,2,2],1),idx)));
assert.throws(()=>AdamW.step(original,ND.seedTensor([1,1,3,2],2)),/shape mismatch/);
assert.throws(()=>AdamW.step(original,g,{t:0,m:ND.build([1,2,3,2],()=>0),v:ND.build([1,2,2,2],()=>0)}),/moment shape mismatch/);
assert.throws(()=>AdamW.step(original,g,null,{lr:-1}),/hyperparameters/);
let allGrads=0;
for(const name of ['nd_trainable_gqa','nd_trainable_mqa']){
 const graph=L.preset(name),before=JSON.stringify(graph);
 assert.equal(graph.nodes.length,13);
 assert.equal(Readiness.check(graph,{numeric:true}).executable,true);
 const source=graph.nodes.filter(n=>n.kind==='nd_parameter');
 assert.deepEqual(source.map(n=>n.id),['query','key','value']);
 const forward=L.evaluate(graph),back=Auto.backward(graph,forward);
 assert.equal(back.targetId,'loss');
 for(const n of source){
   assert.deepEqual(ND.shape(back.gradients[n.id]),ND.shape(n.params.values));
   assert.ok(ND.flatten(back.gradients[n.id]).some(x=>Math.abs(x)>1e-12));
   allGrads+=ND.product(ND.shape(n.params.values));
 }
 const step1=Train.step(graph,{targetId:'loss',lr:.005,weightDecay:.01});
 assert.equal(JSON.stringify(graph),before,'first step mutated original graph');
 assert.equal(step1.records.length,3);
 assert.equal(step1.targetId,'loss');
 assert.equal(step1.next.steps.at(-1).id,'loss');
 assert.ok(Math.abs(step1.afterLoss-step1.beforeLoss)>1e-9);
 const two=Train.steps(graph,{targetId:'loss',count:2,lr:.005,weightDecay:.01});
 assert.equal(two.history.length,2);
 for(const n of two.graph.nodes.filter(n=>n.kind==='nd_parameter')){
   assert.equal(n.params.optimizer.t,2);
   assert.deepEqual(ND.shape(n.params.optimizer.m),ND.shape(n.params.values));
   assert.deepEqual(ND.shape(n.params.optimizer.v),ND.shape(n.params.values));
 }
 assert.deepEqual(two.graph.nodes.map(n=>n.id),graph.nodes.map(n=>n.id));
 assert.deepEqual(two.graph.edges,graph.edges);
 assert.equal(two.history[0].records.length,3);
 assert.ok(two.history[0].records.every(n=>n.updateNorm>0));
 const roundTrip=L.validateGraph(JSON.parse(JSON.stringify(two.graph)));
 const a=Train.step(roundTrip,{targetId:'loss',lr:.005,weightDecay:.01}),
       b=Train.step(two.graph,{targetId:'loss',lr:.005,weightDecay:.01});
 assert.deepEqual(a.graph,b.graph,'persisted 4D AdamW must reproduce next update');
 assert.equal(Train.inspect(roundTrip).length,3);
 const broken=JSON.parse(JSON.stringify(roundTrip));
 broken.nodes.find(n=>n.id==='key').params.optimizer.v[0][0][0][0]=-1;
 assert.throws(()=>L.validateGraph(broken),/nonnegative/);
 const bad=JSON.parse(JSON.stringify(roundTrip));
 bad.nodes.find(n=>n.id==='key').params.optimizer.m=[[[[0]]]];
 assert.throws(()=>L.validateGraph(bad),/shape mismatch/);
 const partial=JSON.parse(JSON.stringify(graph));
 partial.nodes.push({id:'unfinished',kind:'nd_scale',x:200,y:220,params:{factor:1}});
 assert.equal(Readiness.check(partial).executable,false);
 const branch=Train.step(partial,{targetId:'loss',lr:.005});
 assert.equal(branch.graph.nodes.length,14);
 assert.deepEqual(branch.graph.nodes.find(n=>n.id==='unfinished'),partial.nodes.at(-1));
 console.log(name+' PASS: 3 parameter banks, 2 AdamW iterations; loss '+two.firstLoss+' -> '+two.finalLoss);
}
assert.equal(allGrads,84,'all 84 GQA/MQA source coordinates must receive shape-preserved gradients');
console.log('Four-axis AdamW training PASS: 84 graph gradient values, immutable tensors, JSON resume, shared KV updates, partial canvas');
