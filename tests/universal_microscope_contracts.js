'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const C=require('../compute_microscope.js'),L=require('../learning_lab.js');
const Concept=require('../concept_flow.js'),R=require('../block_extension_registry.js');
assert.equal(C.SCHEMA,'seethetruth-computation-microscope-v1');
for(const name of ['ffn','swiglu','attention','residual']){
 const graph=L.preset(name),run=L.evaluate(graph),frame=C.learning(graph,run,L);
 assert.equal(C.validate(frame),frame);assert.equal(frame.evidenceTier,C.TIERS.MICRO);
 assert.deepEqual(frame.steps.map(x=>x.id),run.steps.map(x=>x.id));
 for(const step of frame.steps){
  const original=run.steps.find(s=>s.id===step.id);
  assert.deepEqual(step.out.shape,original.shape);
  assert.deepEqual(step.out.values,original.values.flat(Infinity));
  for(const dep of step.dependencies)assert.ok(frame.steps.findIndex(x=>x.id===dep.from)<frame.steps.indexOf(step));
  const detail=C.dot(frame,step.id,0,0);assert.ok(Number.isFinite(detail.value));
  if(['linear','matmul'].includes(step.kind)){
   assert.equal(detail.reconstructable,true);
   for(let i=0;i<step.out.shape[0];i++)for(let j=0;j<step.out.shape[1];j++){
    const d=C.dot(frame,step.id,i,j);assert.ok(d.error<1e-9);
   }
  }
 }
 if(name==='attention'){
  const a=frame.steps.find(s=>s.kind==='attention');
  assert.deepEqual(a.substeps.map(s=>s.kind),['scaled_matmul','softmax','matmul']);
  for(const s of [a.substeps[0],a.substeps[2]])for(let i=0;i<s.out.shape[0];i++)
   for(let j=0;j<s.out.shape[1];j++)assert.ok(C.dot(frame,s.id,i,j).error<1e-9);
  const tamper=structuredClone(frame);tamper.steps.find(s=>s.kind==='attention').substeps[0].out.values[0]+=.25;
  assert.throws(()=>C.validate(tamper),/GEMM reconstruction contradicts/);
 }
 const staticFrame=C.learning(graph,Concept.evaluate(graph),L);
 assert.equal(staticFrame.evidenceTier,C.TIERS.STATIC);
 assert.ok(staticFrame.steps.every(s=>s.out.values===null));
 assert.throws(()=>C.dot(staticFrame,staticFrame.steps[0].id,0,0),/numeric execution unavailable/);
 assert.throws(()=>C.learning(graph,{...run,evidence:'real_model_execution'},L),/unrecognized execution evidence/);
}
function verifyMap(frame,kind){
 for(const step of frame.steps.filter(s=>s.kind===kind)){
  assert.equal(step.substeps?.length,1);
  const sub=step.substeps[0];assert.ok(sub.indexMap);
  for(let i=0;i<sub.indexMap.length;i++){
   const src=sub.indexMap[i],v=sub.operands[src.port].values[src.index];
   assert.equal(sub.out.values[i],v);
  }
  if(sub.out.values.length>0){
   const forged=structuredClone(frame);
   forged.steps.find(s=>s.id===step.id).substeps[0].out.values[0]+=0.5;
   assert.throws(()=>C.validate(forged),/mapped output contradicts input coordinate/);
  }
 }
}
const named=L.preset('extension_named_axis_composition'),namedFrame=C.learning(named,L.evaluate(named),L);
assert.equal(C.validate(namedFrame),namedFrame);
verifyMap(namedFrame,'axis_named_transpose');
const video=L.preset('extension_video_time_composition'),videoFrame=C.learning(video,L.evaluate(video),L);
assert.equal(C.validate(videoFrame),videoFrame);verifyMap(videoFrame,'semantic_video_slice');verifyMap(videoFrame,'semantic_video_concat');
const joined=videoFrame.steps.find(s=>s.id==='join');
assert.deepEqual(joined.out.shape,[1,3,2,2,2]);
assert.equal(C.dot(videoFrame,joined.substeps[0].id,0,1,[0,2,1]).mapped,true);
assert.equal(C.dot(videoFrame,joined.substeps[0].id,0,1,[0,2,1]).error,0);
const differentParent=structuredClone(videoFrame);
differentParent.steps.find(s=>s.id==='join').out.values[0]+=.25;
assert.throws(()=>C.validate(differentParent),/declared final substep contradicts parent output/);
const changed=structuredClone(videoFrame);changed.steps.find(s=>s.id==='join').substeps[0].indexMap[0].index=10000;
assert.throws(()=>C.validate(changed),/source index out of bounds/);
const fake=structuredClone(namedFrame);fake.evidenceTier='REAL_MODEL_VERIFIED';assert.throws(()=>C.validate(fake),/unknown evidence tier/);
const badRef=structuredClone(namedFrame);badRef.source={url:'javascript:alert(1)',tier:'PINNED_CODE_STRUCTURE_ONLY'};
assert.throws(()=>C.validate(badRef),/unverified pinned-source metadata/);
const badDep=structuredClone(namedFrame);badDep.steps[1].dependencies.push({from:'not-computed',port:'x'});
assert.throws(()=>C.validate(badDep),/dependencies must point backward/);
const h=fs.readFileSync(path.join(__dirname,'../learn.html'),'utf8');
for(const asset of ['compute_microscope.js','compute_microscope_ui.js'])assert.ok(h.includes('src="'+asset+'"'));
assert.ok(h.includes('id="computeMicroscope"')&&h.includes('Microscope.learning(current,run,L)'));
const allow=fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/);
for(const a of ['compute_microscope.css','compute_microscope.js','compute_microscope_ui.js'])assert.ok(allow.includes(a));
for(const a of ['compute_microscope.js','compute_microscope_ui.js'])new vm.Script(fs.readFileSync(path.join(__dirname,'..',a),'utf8'));
console.log('UNIVERSAL_MICROSCOPE_PASS: original FFN/attention proof, 2D axis and 5D time coordinates, tamper gates, static/no-real tier, shared desktop/Android assets');
