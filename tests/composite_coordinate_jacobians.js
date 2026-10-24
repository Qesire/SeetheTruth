'use strict';
/* Independent parameter perturbations of the actual LearningLab interpreter.
 * Do not call GraphAutodiff as the oracle: this independently checks scalar
 * coordinate Jacobians, shared-node aggregation and fixed-route MoE behavior. */
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),E=require('../element_lineage.js'),
 M=require('../compute_microscope.js'),Bridge=require('../coordinate_vjp_bridge.js'),
 ND=require('../nd_tensor.js'),Grad=require('../graph_autodiff.js');
const node=(id,kind,params={})=>({id,kind,params,x:0,y:0});
const edge=(from,to,port)=>({from,to,port});
const params=v=>({values:v,optimizer:{t:0,m:v.map(row=>row.map(()=>0)),
 v:v.map(row=>row.map(()=>0))}});
const perturb=(g,id,index,delta)=>{
 const x=structuredClone(g),n=x.nodes.find(n=>n.id===id);
 assert.ok(n&&n.kind==='parameter','must perturb real trainable source '+id);
 let val=n.params.values;for(const a of index.slice(0,-1))val=val[a];
 val[index.at(-1)]+=delta;return x;
};
const h=1e-6;
const fd=(g,id,index,outId,outIndex)=>{
 const sample=delta=>ND.at(L.evaluate(perturb(g,id,index,delta)).steps.find(s=>s.id===outId).values,outIndex);
 return (sample(h)-sample(-h))/(2*h);
};
function audit(g,target,index,tol=2e-5,minimum=1){
 const trace=L.evaluate(g),frame=M.learning(g,trace,L);
 const report=E.inspect(g,{targetId:target,outputIndex:index,trace,maxVertices:2048,maxEdges:8192});
 const connected=report.vertices.filter(v=>v.status==='source'&&
  g.nodes.find(n=>n.id===v.id)?.kind==='parameter'&&v.sensitivity!==null);
 assert.ok(report.complete,target+' dependency traversal incomplete: '+report.frontiers);
 assert.ok(connected.length>=minimum,'too few parameter coordinate dependencies for '+target);
 for(const v of connected){
  const independent=fd(g,v.id,v.index,target,index);
  assert.ok(Math.abs(independent-v.sensitivity)<tol,
    target+' dY/d'+v.id+v.index+' expected '+independent+' got '+v.sensitivity);
 }
 const bridge=Bridge.inspect(g,trace,frame,{stepId:target,coordinate:index});
 assert.equal(bridge.backward.complete,true,target+' bridge completeness');
 assert.equal(bridge.backward.sources.filter(x=>x.kind==='parameter').length,connected.length);
 return {report,bridge,trace};
}
// Scaled dot-product attention: one scalar output depends on every V key,
// all Q channels of its query, and every K key/channel.
const qkv=L.validateGraph({schema:L.SCHEMA,title:'scaled QKV Jacobian',nodes:[
 node('Q','parameter',params([[.22,-.3],[.41,.67]])),
 node('K','parameter',params([[.42,.16],[-.29,.52],[.71,-.18]])),
 node('V','parameter',params([[.2,.9],[-.4,.55],[.37,-.21]])),
 node('att','attention')
],edges:[edge('Q','att','q'),edge('K','att','k'),edge('V','att','v')]});
const att=audit(qkv,'att',[1,0],3e-5,12);
assert.equal(att.bridge.backward.internalStages.length,3);
assert.equal(att.bridge.backward.internalStages.filter(s=>s.probability>0).length,3);
assert.equal(att.bridge.backward.discreteConditional,false);
assert.equal(att.report.edges.filter(e=>e.from==='att@1,0').length,3*(1+2*2));
const qgrad=att.report.vertices.find(v=>v.key==='Q@1,0').sensitivity;
assert.ok(Math.abs(qgrad-fd(qkv,'Q',[1,0],'att',[1,0]))<3e-5);
// Shared Q=K=V identity: three ports must sum, not overwrite gradients.
const shared=L.validateGraph({schema:L.SCHEMA,title:'tied QKV attention',nodes:[
 node('X','parameter',params([[.31,.44],[-.52,.7],[.11,-.18]])),
 node('att','attention')
],edges:[edge('X','att','q'),edge('X','att','k'),edge('X','att','v')]});
const tied=audit(shared,'att',[0,1],4e-5,6);
assert.equal(tied.report.vertices.filter(v=>v.id==='X').length,6);
assert.equal(tied.report.edges.filter(e=>e.port==='q'||e.port==='k'||e.port==='v').length,15);
// Gate and Up VJP are intentionally different: derivative of SiLU and
// product rule, with both branches sharing the same upstream trainable input.
const gated=L.validateGraph({schema:L.SCHEMA,title:'shared FFN gate/up VJP',nodes:[
 node('X','parameter',params([[.13,-.72],[.4,.83]])),
 node('W1','parameter',params([[.43,-.1],[.7,.22]])),
 node('W3','parameter',params([[.17,.3],[-.4,.52]])),
 node('gate','matmul'),node('up','matmul'),
 node('swi','swiglu')
],edges:[edge('X','gate','a'),edge('W1','gate','b'),
 edge('X','up','a'),edge('W3','up','b'),
 edge('gate','swi','gate'),edge('up','swi','up')]});
const swi=audit(gated,'swi',[0,1],4e-5,12);
assert.equal(swi.bridge.backward.internalStages.length,2);
assert.equal(swi.report.edges.filter(e=>e.from==='swi@0,1').length,2);
assert.ok(swi.report.vertices.find(v=>v.id==='X'));
// Actual routed MoE recipe; perturb explicit W banks but hold the discrete
// route IDs invariant. This is forward-value independent of the VJP code.
const moe=L.preset('trainable_moe'),mTrace=L.evaluate(moe);
const expert=audit(moe,'expert_gate',[0,1],4e-5,4);
assert.equal(expert.bridge.backward.discreteConditional,true);
assert.equal(expert.bridge.backward.conditionalCuts.some(e=>e.port==='indices'),true);
assert.equal(expert.bridge.backward.internalStages.length,2);
assert.equal(expert.bridge.backward.internalStages[0].stage,'expert_channel_product');
const reduction=audit(moe,'routed',[0,1],8e-5,8);
assert.equal(reduction.bridge.backward.discreteConditional,true);
assert.equal(reduction.bridge.backward.internalStages[0].stage,'route_weighted_contribution');
const all=audit(moe,'out',[0,0],1e-4,10);
assert.equal(all.bridge.backward.discreteConditional,true);
assert.equal(all.bridge.loss.status,'VERIFIED_GRAPH_LOSS_VJP');
assert.equal(all.bridge.loss.targetId,'loss');
assert.ok(all.bridge.loss.cutEdges.some(x=>x.port==='indices'));
const forward=mTrace.steps.find(x=>x.id==='select').values;
const conditional=moe.nodes.find(x=>x.id==='select');
assert.ok(conditional&&forward.every(row=>row.every(Number.isSafeInteger)));
// Gate-only independent score perturbation: replace the upstream affinity
// operator by a trainable tensor at precisely the executed score values.
// Routing indices can remain stable for tiny perturbations.
const route=L.validateGraph({schema:L.SCHEMA,title:'selected router gates',nodes:[
 node('scores','parameter',params([[.29,.38,.42],[.17,.57,.26]])),
 node('indices','parameter',params([[1,2],[1,0]])),
 node('gates','router_gate_weights',{mode:'softmax',normalize:true})
],edges:[edge('scores','gates','scores'),edge('indices','gates','indices')]});
const gate=audit(route,'gates',[0,0],4e-5,2);
assert.equal(gate.bridge.backward.discreteConditional,true);
assert.equal(gate.report.vertices.filter(v=>v.id==='indices').length,0);
const raw=L.evaluate(route).steps.find(s=>s.id==='gates').values;
assert.ok(raw.every(row=>Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-10));
// Structural expert weight rows cannot be identified without a numerical
// selected expert: explicit incomplete boundary, not an invented complete path.
const noTrace=E.inspect(moe,{targetId:'expert_gate',outputIndex:[0,1]});
assert.equal(noTrace.complete,false);
assert.ok(noTrace.frontiers.some(s=>s.startsWith('expert_gate@')));
assert.equal(noTrace.vertices.find(v=>v.id==='expert_gate').status,'boundary');
console.log('COMPOSITE_COORDINATE_PASS: 3-key attention, tied QKV, gate/up SwiGLU, routed experts, fixed TopK, conditional structural frontier and independent FD');
