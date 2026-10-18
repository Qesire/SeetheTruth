'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),A=require('../graph_autodiff.js'),B=require('../block_catalog.js');
const g=L.preset('moe_backward'),f=L.evaluate(g),b=A.backward(g,f);
const fwd=Object.fromEntries(f.steps.map(s=>[s.id,s.values]));
const near=(x,y,tol=4e-6)=>assert.ok(Math.abs(x-y)<tol,'expected '+x+' ~ '+y);
assert.equal(g.nodes.length,24);
assert.deepEqual(f.steps.at(-1).shape,[1,1]);
assert.equal(b.targetId,'loss');
assert.deepEqual(b.gradients.teacher_target,Array.from({length:3},()=>[0,0]));
assert.ok(b.cutEdges.some(e=>e.nodeId==='expert_gate'&&e.port==='indices'));
assert.ok(b.cutEdges.some(e=>e.nodeId==='gates'&&e.port==='indices'));
assert.ok(b.pathIds.includes('routed')&&b.pathIds.includes('expert_down')&&b.pathIds.includes('gates'));
assert.deepEqual(b.gradients.select,[[0,0],[0,0],[0,0]]);
assert.deepEqual(b.gradients.biased,Array.from({length:3},()=>[0,0,0,0]));
const deps=new Map(g.nodes.map(n=>[n.id,g.edges.filter(e=>e.to===n.id)]));
function lossWithOverrides(overrides){
 const computed={};const dirty=new Set(Object.keys(overrides));
 for(const n of L.order(g)){
  if(Object.hasOwn(overrides,n.id)){computed[n.id]=overrides[n.id];continue;}
  const depsN=deps.get(n.id);
  if(!depsN.some(e=>dirty.has(e.from))){computed[n.id]=fwd[n.id];continue;}
  dirty.add(n.id);
  const ins=Object.fromEntries(depsN.map(e=>[e.port,computed[e.from]]));
  if(['expert_linear_bank','swiglu','routed_weighted_reduce'].includes(n.kind))computed[n.id]=B.execute(n,ins).values;
  else if(n.kind==='add')computed[n.id]=ins.a.map((row,i)=>row.map((v,j)=>v+ins.b[i][j]));
  else if(n.kind==='mse'){
   const x=ins.prediction,y=ins.target;
   computed[n.id]=[[x.reduce((s,row,i)=>s+row.reduce((a,v,j)=>a+(v-y[i][j])**2,0),0)/(x.length*x[0].length)]];
  }else throw Error('unexpected changed MoE node '+n.id+':'+n.kind);
 }
 return computed.loss[0][0];
}
near(lossWithOverrides({}),fwd.loss[0][0],1e-12);
let examined=0;
for(const name of ['w1','w3','w2']){
 const tensor=fwd[name];assert.deepEqual([tensor.length,tensor[0].length],[8,2]);
 for(let i=0;i<tensor.length;i++)for(let j=0;j<tensor[0].length;j++){
   const h=1e-6,a=tensor.map(row=>row.slice()),d=tensor.map(row=>row.slice());
   a[i][j]+=h;d[i][j]-=h;
   const numerical=(lossWithOverrides({[name]:a})-lossWithOverrides({[name]:d}))/(2*h);
   near(b.gradients[name][i][j],numerical);
   examined++;
 }
}
assert.equal(examined,48);
const selected=fwd.select,experts=fwd.expert_down,shared=fwd.shared_down,target=fwd.teacher_target;
function gateLoss(affinity){
 const gates=B.execute({kind:'router_gate_weights',params:{mode:'identity',normalize:true}},{scores:affinity,indices:selected}).values;
 const routed=B.execute({kind:'routed_weighted_reduce',params:{}},{values:experts,gates}).values;
 const pred=routed.map((row,i)=>row.map((v,j)=>v+shared[i][j]));
 return pred.reduce((s,row,i)=>s+row.reduce((a,v,j)=>a+(v-target[i][j])**2,0),0)/(pred.length*pred[0].length);
}
near(gateLoss(fwd.affinity),fwd.loss[0][0]);
let gateChecks=0;
for(let t=0;t<fwd.affinity.length;t++)for(let k=0;k<fwd.affinity[0].length;k++){
 const plus=fwd.affinity.map(row=>row.slice()),minus=fwd.affinity.map(row=>row.slice()),h=1e-6;
 plus[t][k]+=h;minus[t][k]-=h;
 near(b.gradients.affinity[t][k],(gateLoss(plus)-gateLoss(minus))/(2*h));
 gateChecks++;
}
assert.equal(gateChecks,12);
console.log('Unified MoE reverse PASS: 48 expert-weight and 12 gate-affinity finite differences; Top-k index cut; shared FFN gradients');
