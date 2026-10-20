'use strict';
const assert=require('node:assert/strict');
const B=require('../block_catalog.js'),L=require('../learning_lab.js'),C=require('../concept_flow.js'),K=require('../block_knowledge.js');
const calc=(kind,inputs,params={})=>B.execute({id:'oracle',kind,params},inputs);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,String(a)+' != '+String(b));
const x=[[1,2],[3,4]],I=[[0,1],[1,0]],W=[[1,0],[0,1],[-1,0],[0,2]];
const Z=calc('dispatch_routed_tokens',{x,indices:I});
assert.deepEqual(Z.values,[[1,2],[1,2],[3,4],[3,4]]);
const P=calc('expert_linear_bank',{x:Z.values,indices:I,weights:W});
assert.deepEqual(P.values,[[1,2],[-1,4],[-3,8],[3,4]]);
assert.deepEqual(P.internals.expert_load.values,[[2],[2]]);
const gate=[[.25,.75],[.4,.6]];
const Y=calc('routed_weighted_reduce',{values:P.values,gates:gate});
assert.deepEqual(Y.values.map(row=>row.map(v=>+v.toFixed(12))),[[-.5,3.5],[.6,5.6]]);
assert.throws(()=>calc('expert_linear_bank',{x:Z.values,indices:I,weights:[[1],[2],[3]]}),/Experts\*D/);
assert.throws(()=>calc('expert_linear_bank',{x:Z.values,indices:[[0,2],[1,0]],weights:W}),/out of range/);
assert.throws(()=>calc('routed_weighted_reduce',{values:[[1,2]],gates:gate}),/T\*K/);
const g=L.preset('moe_expert_ffn'),numeric=L.evaluate(g),abstract=C.evaluate(g);
assert.equal(g.nodes.length,22);
assert.equal(numeric.steps.length,g.nodes.length);
assert.deepEqual(numeric.steps.map(t=>t.shape),abstract.steps.map(t=>t.shape));
assert.equal(numeric.steps.at(-1).id,'out');
assert.deepEqual(numeric.steps.at(-1).shape,[3,2]);
const steps=Object.fromEntries(numeric.steps.map(t=>[t.id,t.values]));
steps.dispatch.forEach((row,i)=>assert.deepEqual(row,steps.x[Math.floor(i/2)]));
for(let n=0;n<steps.dispatch.length;n++){
 const e=steps.select[Math.floor(n/2)][n%2],row=steps.dispatch[n];
 for(const [kind,weight] of [['expert_gate','w1'],['expert_up','w3']]){
  steps[kind][n].forEach((v,h)=>near(v,row.reduce((s,x,d)=>s+x*steps[weight][e*2+d][h],0)));
 }
}
for(let t=0;t<3;t++)for(let d=0;d<2;d++){
 const routed=steps.gates[t].reduce((acc,w,k)=>acc+w*steps.expert_down[t*2+k][d],0);
 near(routed,steps.routed[t][d]);
 near(steps.out[t][d],routed+steps.shared_down[t][d]);
}
for(const kind of ['dispatch_routed_tokens','expert_linear_bank','routed_weighted_reduce'])assert.ok(K.get(kind)?.refs?.length>0);
console.log('Composed 22-node expert SwiGLU forward, parameter sharing and shared-expert merge PASS');
