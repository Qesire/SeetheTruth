'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const M=require('../compute_microscope.js'),L=require('../learning_lab.js'),C=require('../concept_flow.js');
const graph={
 title:'Rowwise native arithmetic microscope',
 nodes:[
  {id:'x',kind:'input',x:10,y:50,params:{rows:3,width:4,seed:3}},
  {id:'rms',kind:'rmsnorm',x:280,y:50,params:{epsilon:.00001}},
  {id:'activation',kind:'silu',x:530,y:50,params:{}},
  {id:'probabilities',kind:'softmax',x:770,y:50,params:{}}
 ],
 edges:[
  {from:'x',to:'rms',port:'x'},
  {from:'rms',to:'activation',port:'x'},
  {from:'activation',to:'probabilities',port:'x'}
 ]
};
const r=L.evaluate(graph),frame=M.learning(graph,r,L);
assert.equal(M.validate(frame),frame);
const byId=Object.fromEntries(frame.steps.map(x=>[x.id,x]));
for(const [id,operator]of [['rms','rmsnorm'],['activation','silu'],['probabilities','softmax']]){
 const step=byId[id],x=step.operands.x.values,cols=step.out.shape[1];
 assert.equal(step.kernel.operator,operator);
 for(let row=0;row<step.out.shape[0];row++){
  const vec=x.slice(row*cols,(row+1)*cols);
  const meanSq=vec.reduce((a,b)=>a+b*b,0)/cols;
  const max=Math.max(...vec),exp=vec.map(v=>Math.exp(v-max)),total=exp.reduce((a,b)=>a+b,0);
  for(let col=0;col<cols;col++){
   const expected=operator==='rmsnorm'?vec[col]/Math.sqrt(meanSq+1e-5):
    operator==='silu'?vec[col]/(1+Math.exp(-vec[col])):exp[col]/total;
   const detail=M.dot(frame,id,row,col);
   assert.equal(detail.kernel,operator);
   assert.equal(detail.reconstructable,true);
   assert.ok(Math.abs(detail.recomputed-expected)<1e-12);
   assert.ok(detail.error<1e-10);
   assert.ok(detail.kernelTerms.length>=1);
  }
 }
 const corrupt=structuredClone(frame);
 corrupt.steps.find(s=>s.id===id).out.values[0]+=.1;
 assert.throws(()=>M.validate(corrupt),/numerical reconstruction contradicts output/,id+' output mutation');
 const badRule=structuredClone(frame);badRule.steps.find(s=>s.id===id).kernel.operator='fake';
 assert.throws(()=>M.validate(badRule),/unsupported rowwise numerical proof/,id+' cannot fake proof kind');
}
const badEpsilon=structuredClone(frame);badEpsilon.steps.find(s=>s.id==='rms').kernel.epsilon=.1;
assert.throws(()=>M.validate(badEpsilon),/rmsnorm numerical reconstruction contradicts output/);
const badPort=structuredClone(frame);badPort.steps.find(s=>s.id==='probabilities').kernel.inputPort='scores';
assert.throws(()=>M.validate(badPort),/rowwise input port provenance mismatch/);
const wrongShape=structuredClone(frame);wrongShape.steps.find(s=>s.id==='rms').operands.x.shape=[2,6];
assert.throws(()=>M.validate(wrongShape),/rowwise input\/output shapes disagree/);
const attention=L.preset('attention'),attn=M.learning(attention,L.evaluate(attention),L);
const step=attn.steps.find(s=>s.kind==='attention'),prob=step.substeps[1];
assert.equal(prob.kernel.operator,'softmax');
assert.equal(prob.kernel.inputPort,'scores');
for(let row=0;row<prob.out.shape[0];row++){
 const w=prob.out.values.slice(row*prob.out.shape[1],(row+1)*prob.out.shape[1]);
 assert.ok(Math.abs(w.reduce((a,b)=>a+b,0)-1)<1e-12);
 for(let col=0;col<prob.out.shape[1];col++)assert.ok(M.dot(attn,prob.id,row,col).error<1e-10);
}
const fakeProbability=structuredClone(attn);
fakeProbability.steps.find(s=>s.kind==='attention').substeps[1].out.values[0]+=.1;
assert.throws(()=>M.validate(fakeProbability),/softmax numerical reconstruction contradicts output/);
const conceptual=M.learning(graph,C.evaluate(graph),L);
assert.equal(conceptual.evidenceTier,M.TIERS.STATIC);
assert.ok(conceptual.steps.every(s=>s.out.values===null&&s.kernel===undefined));
const ui=fs.readFileSync(path.join(__dirname,'../compute_microscope_ui.js'),'utf8');
assert.ok(ui.includes("if(detail.kernel)")&&ui.includes("kernelTerms"));
console.log('NATIVE_ROWWISE_PROOF_PASS RMSNorm/Silu/StableSoftmax, attention inner normalization, mutated output/params/ports refused, symbolic isolated');
