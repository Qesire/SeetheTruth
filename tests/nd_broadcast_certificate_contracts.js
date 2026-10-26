'use strict';
const assert=require('node:assert/strict');
const ND=require('../nd_tensor.js'),L=require('../learning_lab.js');
const G=require('../graph_autodiff.js'),Torch=require('../torch_codegen.js');
const {broadcastReduce}=require('./fixtures/nd_broadcast_reduction_graph.js');
const graph=broadcastReduce(),original=JSON.stringify(graph);
const forward=L.evaluate(graph),back=G.backward(graph,forward,{targetId:'loss'});
const values=Object.fromEntries(forward.steps.map(step=>[step.id,step.values]));
assert.deepEqual(ND.shape(values.source),[1,2,3,2]);
assert.deepEqual(ND.shape(values.bias),[1,1,1,2]);
assert.deepEqual(ND.shape(values.offset),[1,2,3,2]);
assert.deepEqual(ND.shape(values.summed),[1,2,1,2]);
assert.deepEqual(ND.shape(values.loss),[1,1]);
assert.deepEqual(ND.shape(back.gradients.source),[1,2,3,2]);
assert.deepEqual(ND.shape(back.gradients.bias),[1,1,1,2]);
const near=(x,y)=>assert.ok(Math.abs(x-y)<1e-11,x+' expected '+y);
for(let d=0;d<2;d++){
 let total=0;
 for(let h=0;h<2;h++){
  const g=back.gradients.summed[0][h][0][d];
  for(let t=0;t<3;t++)near(back.gradients.source[0][h][t][d],g);
  total+=g*3;
 }
 near(back.gradients.bias[0][0][0][d],total);
}
const baseline=ND.at(values.loss,[0,0]);
const perturbed=structuredClone(graph);
const b=perturbed.nodes.find(n=>n.id==='bias').params.values;
b[0][0][0][1]+=1e-6;
const plus=L.evaluate(perturbed).steps.find(x=>x.id==='loss').values[0][0];
b[0][0][0][1]-=2e-6;
const minus=L.evaluate(perturbed).steps.find(x=>x.id==='loss').values[0][0];
assert.ok(Math.abs((plus-minus)/(2e-6)-back.gradients.bias[0][0][0][1])<3e-6);
assert.deepEqual(JSON.stringify(graph),original,'calculation did not mutate graph');
assert.ok(Number.isFinite(baseline));
const compiled=Torch.compile(graph,{lossId:'loss'});
assert.equal(compiled.lossId,'loss');
assert.ok(compiled.source.includes('torch'));
assert.ok(Torch.supportedKinds.includes('nd_reduce_sum'));
assert.ok(Torch.supportedKinds.includes('nd_broadcast_add'));
assert.throws(()=>ND.broadcastBinary(values.source,ND.seedTensor([1,3,2,2],1),(x,y)=>x+y),
 /incompatible/);
console.log('ND_BROADCAST_CERTIFICATE_SEMANTICS_PASS: shape/port, three-token bias VJP, source gradients, finite difference and exporter eligibility');
