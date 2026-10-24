'use strict';
const assert=require('node:assert/strict');
const M=require('../compute_microscope.js'),L=require('../learning_lab.js'),Concept=require('../concept_flow.js');
const nodes=[],edges=[];
function n(id,kind,params={}){nodes.push({id,kind,params,x:10+nodes.length*18,y:10+nodes.length*12});}
function e(from,to,port){edges.push({from,to,port});}
n('x','input',{rows:3,width:4,seed:1});n('y','input',{rows:3,width:4,seed:3});
for(const kind of ['add','mul','subtract','divide']){n(kind,kind);e('x',kind,'a');e('y',kind,'b');}
for(const kind of ['scale','bias','square','relu','relu_squared','sigmoid','tanh','gelu_tanh','softplus']){
 n(kind,kind,kind==='scale'?{factor:.37}:kind==='bias'?{bias:.42}:{});e('x',kind,'x');
}
for(const [id,kind,params]of [['transpose','transpose2d',{}],['rows','slice_rows',{start:1,length:2}],
 ['cols','slice_cols',{start:1,width:2}],['repeat','repeat_rows',{times:2}]]){
 n(id,kind,params);e('x',id,'x');
}
n('cat','concat_cols');e('cols','cat','a');e('cols','cat','b');
const graph={schema:L.SCHEMA,title:'mixed native proof families',nodes,edges};
const frame=M.learning(graph,L.evaluate(graph),L);assert.equal(M.validate(frame),frame);
const bin=new Set(['add','mul','subtract','divide']),mapped=new Set(['transpose','rows','cols','repeat','cat']);
const targets=[...bin,'scale','bias','square','relu','relu_squared','sigmoid','tanh','gelu_tanh','softplus',...mapped];
for(const id of targets){
 const s=frame.steps.find(v=>v.id===id);assert(s,id);
 for(let r=0;r<s.out.shape[0];r++)for(let c=0;c<s.out.shape[1];c++){
  const proof=M.dot(frame,id,r,c);assert.equal(proof.reconstructable,true,id);
  assert.ok(proof.error<=1e-10,id+' numerical closure');
  if(mapped.has(id))assert.ok(proof.mapped&&proof.sourceCoordinateLabel);
  else if(bin.has(id))assert.equal(proof.operation,s.kind);
  else assert.equal(proof.unary,true);
 }
 const bad=structuredClone(frame);bad.steps.find(v=>v.id===id).out.values[0]+=.1;
 assert.throws(()=>M.validate(bad),/mapped output contradicts|elementwise arithmetic contradicts|scalar numerical reconstruction contradicts/,id);
}
function mutate(id,modify,pattern){const bad=structuredClone(frame);modify(bad.steps.find(s=>s.id===id));assert.throws(()=>M.validate(bad),pattern,id);}
mutate('subtract',s=>s.elementwise.operator='add',/unsupported numerical elementwise/);
mutate('divide',s=>{s.operands.b=structuredClone(s.operands.b);s.operands.b.values[0]=0;},/elementwise division by zero/);
mutate('scale',s=>s.unary.factor=1.1,/scalar numerical reconstruction contradicts/);
mutate('bias',s=>s.unary.operator='square',/unsupported scalar numerical proof/);
mutate('square',s=>s.unary.bias=2,/invalid scalar proof parameter/);
mutate('transpose',s=>s.indexMap[0].index=1,/native source index contract mismatch/);
mutate('rows',s=>s.indexRule.params.start=0,/native source index contract mismatch/);
mutate('cols',s=>s.indexRule.params.start=0,/native source index contract mismatch/);
mutate('repeat',s=>s.indexRule.params.times=1,/native repeated row shape mismatch/);
mutate('cat',s=>s.indexMap[0].port='b',/native source index contract mismatch/);
mutate('cat',s=>delete s.indexMap,/native index rule without source map/);
const spoof=structuredClone(frame);
const scale=spoof.steps.find(s=>s.id==='scale');
scale.operands.x=structuredClone(scale.operands.x);
scale.operands.x.values[0]+=0.25;
scale.out.values[0]=scale.operands.x.values[0]*scale.unary.factor;
assert.throws(()=>M.validate(spoof),/execution dependency operand provenance mismatch/,
 'arithmetic that closes against a forged, non-upstream operand must still fail');
const concept=M.learning(graph,Concept.evaluate(graph),L);
assert.equal(concept.evidenceTier,M.TIERS.STATIC);
assert.ok(concept.steps.every(s=>s.out.values===null&&s.unary===undefined&&s.indexMap===undefined&&s.elementwise===undefined));
console.log('NATIVE_PORTABLE_COORDINATE_PROOFS_PASS: mixed graph closure, evidence falsification, static isolation');
