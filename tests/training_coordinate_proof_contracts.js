'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),G=require('../graph_autodiff.js');
const Core=require('../learning_experience_core.js');
const Proof=require('../training_coordinate_proof.js'),ND=require('../nd_tensor.js');
const {create}=require('./fixtures/coordinate_training_graph.js');
const graph=create(),original=JSON.stringify(graph);
const options={targetId:'loss',lr:0.01,weightDecay:0.001};
const selection={parameterId:'w',index:[0,1],viaNodeId:'project',viaIndex:[0,1]};
const witness=Proof.inspect(graph,options,selection);
assert.equal(witness.schema,Proof.SCHEMA);
assert.equal(witness.evidence,'EXECUTED_BOUNDED_CPU_TRAINING_COORDINATE');
assert.equal(JSON.stringify(graph),original);
assert.equal(witness.parameterId,'w');
assert.equal(witness.path.status,'VERIFIED_SELECTED_PATH_CONTRIBUTION');
assert.equal(witness.path.selected.nodeId,'project');
assert.ok(witness.path.localTerms.length>=2);
assert.equal(witness.path.loss.targetId,'loss');
assert.ok(Math.abs(witness.path.parameterContribution)>1e-8);
assert.equal(witness.path.sourceProof,'seethetruth-coordinate-vjp-bridge-v1');
const originalRecord=Core.executeLab('training','graph_step',{graph,options});
const result=Core.executeLab('training','coordinate_proof',{graph,options,selection});
assert.deepEqual(result.value,witness);
assert.equal(result.stateTrace,null);
assert.equal(Core.observe(result).traceAvailable,false);
assert.equal(Core.describeActions('training').find(x=>x.name==='coordinate_proof').observation,'none');
assert.equal(Core.replay(result).reproduced,true);
const old=graph.nodes.find(n=>n.id==='w').params.values;
const updated=originalRecord.value.session.graph.nodes.find(n=>n.id==='w').params.values;
assert.equal(witness.parameter.before,old[0][1]);
assert.equal(witness.parameter.after,updated[0][1]);
assert.equal(witness.parameter.delta,updated[0][1]-old[0][1]);
const grad=G.backward(graph,L.evaluate(graph),{targetId:'loss'}).gradients.w;
assert.equal(witness.gradient,ND.at(grad,selection.index));
function fd(paramId,idx){
 const evaluate=sign=>{
  const copy=structuredClone(graph),p=copy.nodes.find(n=>n.id===paramId).params.values;
  p[idx[0]][idx[1]]+=sign*1e-6;
  return L.evaluate(copy).steps.find(s=>s.id==='loss').values[0][0];
 };
 return (evaluate(1)-evaluate(-1))/(2e-6);
}
assert.ok(Math.abs(fd('w',[0,1])-witness.wholeLossGradient)<2e-6);
assert.ok(Math.abs(witness.path.parameterContribution-witness.wholeLossGradient)>1e-8,
 'single selected output route cannot be relabelled entire loss gradient');
const allVia=[];
for(let i=0;i<2;i++)for(let j=0;j<2;j++){
 const item=Proof.inspect(graph,options,{parameterId:'w',index:[0,1],viaNodeId:'project',viaIndex:[i,j]});
 assert.ok(['VERIFIED_SELECTED_PATH_CONTRIBUTION','VERIFIED_ZERO_SELECTED_PATH'].includes(item.path.status));
 assert.ok(Number.isFinite(item.path.parameterContribution));
 allVia.push(item.path.parameterContribution);
}
assert.ok(Math.abs(allVia.reduce((a,b)=>a+b,0)-witness.wholeLossGradient)<2e-10,
 'partition via every projection output equals actual full-loss parameter gradient');
for(const optimizer of ['adamw','adam','sgd_momentum','rmsprop','adagrad','lion']){
 const p=Proof.inspect(graph,{...options,optimizer},{parameterId:'w',index:[0,1]});
 const native=Core.executeLab('training','graph_step',{graph,options:{...options,optimizer}});
 assert.equal(p.optimizer.kind,optimizer);
 assert.equal(p.parameter.after,native.value.session.graph.nodes.find(n=>n.id==='w').params.values[0][1]);
}
for(const bad of [
 {parameterId:'target',index:[0,1]},
 {parameterId:'w',index:[8,0]},
 {parameterId:'missing',index:[0,1]},
 {parameterId:'w',index:[0,1],viaNodeId:'missing',viaIndex:[0,0]},
 {parameterId:'w',index:[0,1],viaNodeId:'project',viaIndex:[3,0]}
 ])assert.throws(()=>Proof.inspect(graph,options,bad));
assert.throws(()=>Core.executeLab('training','coordinate_proof',{graph,options:{targetId:'project'},selection}));
const corrupted=structuredClone(result);corrupted.value.parameter.after+=0.4;
assert.throws(()=>Core.replay(corrupted),/replay mismatch: value/);
console.log('TRAINING_COORDINATE_PROOF_PASS exact parameter update, local VJP, full-loss path partition, finite differences, six optimizers and tamper rejection');
