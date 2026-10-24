'use strict';
const assert=require('node:assert/strict'),C=require('../compute_microscope.js'),L=require('../learning_lab.js'),Concept=require('../concept_flow.js');
const g={schema:L.SCHEMA,title:'native coordinate proof',nodes:[
 {id:'a',kind:'input',params:{rows:2,width:3,seed:1},x:20,y:30},
 {id:'b',kind:'input',params:{rows:2,width:3,seed:2},x:20,y:250},
 {id:'sum',kind:'add',params:{},x:330,y:40},
 {id:'product',kind:'mul',params:{},x:330,y:250}],
 edges:[{from:'a',to:'sum',port:'a'},{from:'b',to:'sum',port:'b'},
 {from:'a',to:'product',port:'a'},{from:'b',to:'product',port:'b'}]};
const result=L.evaluate(g),frame=C.learning(g,result,L);
assert.equal(C.validate(frame),frame);
for(const [node,op] of [['sum','add'],['product','mul']]){
 const stage=frame.steps.find(s=>s.id===node);
 assert.equal(stage.elementwise.operator,op);
 for(let i=0;i<2;i++)for(let j=0;j<3;j++){
  const detail=C.dot(frame,node,i,j);
  assert.equal(detail.elementwise,true);
  assert.equal(detail.reconstructable,true);
  assert.equal(detail.error,0);
  assert.equal(detail.operands.length,2);
  assert.equal(detail.value,op==='add'?detail.operands[0].value+detail.operands[1].value:detail.operands[0].value*detail.operands[1].value);
 }
 const forged=structuredClone(frame);forged.steps.find(s=>s.id===node).out.values[0]+=0.5;
 assert.throws(()=>C.validate(forged),/elementwise arithmetic contradicts/);
}
const brokenOp=structuredClone(frame);brokenOp.steps.find(s=>s.id==='sum').elementwise.operator='fake';
assert.throws(()=>C.validate(brokenOp),/unsupported numerical elementwise/);
const named=L.preset('extension_named_axis_composition'),N=C.learning(named,L.evaluate(named),L);
const typed=Object.fromEntries(N.steps.map(s=>[s.id,s]));
assert.deepEqual(typed.x.out.axes,['T','D']);
assert.deepEqual(typed.bias.out.axes,['One','D']);
assert.deepEqual(typed.add.out.axes,['T','D']);
assert.deepEqual(typed.swap.out.axes,['D','T']);
assert.deepEqual(typed.swap.substeps[0].out.axes,['D','T']);
assert.deepEqual(typed.swap.substeps[0].operands.x.axes,['T','D']);
assert.match(C.dot(N,'swap:td',0,1).coordinateLabel,/D=0,T=1/);
assert.match(C.dot(N,'swap:td',0,1).sourceCoordinateLabel,/T=1,D=0/);
const conceptual=C.learning(named,Concept.evaluate(named),L);
assert.equal(conceptual.evidenceTier,C.TIERS.STATIC);
assert.deepEqual(conceptual.steps.find(s=>s.id==='swap').out.axes,['D','T']);
assert.equal(conceptual.steps.find(s=>s.id==='swap').out.values,null);
const video=L.preset('extension_video_time_composition'),V=C.learning(video,L.evaluate(video),L);
const joined=V.steps.find(s=>s.id==='join');
assert.deepEqual(joined.out.axes,['B','T','H','W','C']);
assert.deepEqual(joined.substeps[0].out.axes,['B','T','H','W','C']);
const map=C.dot(V,joined.substeps[0].id,0,1,[0,2,1]);
assert.match(map.coordinateLabel,/B=0,T=2,H=1,W=0,C=1/);
assert.match(map.sourceCoordinateLabel,/B=0,T=0,H=1,W=0,C=1/);
const bad=structuredClone(V);bad.steps.find(s=>s.id==='join').substeps[0].out.axes=['B','T','C','H','W'];
assert.throws(()=>C.validate(bad),/declared final substep contradicts parent/);
const dup=structuredClone(N);dup.steps.find(s=>s.id==='swap').out.axes=['T','T'];
assert.throws(()=>C.validate(dup),/invalid named axis metadata/);
const falsified=structuredClone(N);falsified.steps.find(s=>s.id==='swap').substeps[0].operands.x.axes=['Wrong','D'];
assert.throws(()=>C.validate(falsified),/named-axis operand/);
const missing=structuredClone(frame);missing.steps.find(s=>s.id==='sum').elementwise.operator='mul';
assert.throws(()=>C.validate(missing),/unsupported numerical elementwise reconstruction|elementwise arithmetic contradicts/);
console.log('MICROSCOPE_AXES_POINTWISE_PASS: T/D and B/T/H/W/C traced coordinates, exact native add/mul, corrupt outputs and axis spoofing rejected');
