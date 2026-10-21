'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const C=require('../compute_microscope.js');
const L=require('../learning_lab.js'),Concept=require('../concept_flow.js'),W=require('../wan_ffn_trace.js');
assert.equal(C.SCHEMA,'seethetruth-computation-microscope-v1');
for(const name of ['ffn','swiglu','attention','residual']){
 const g=L.preset(name),r=L.evaluate(g),frame=C.learning(g,r,L);
 assert.equal(C.validate(frame),frame);
 assert.equal(frame.steps.length,r.steps.length,name);
 assert.equal(frame.evidenceTier,C.TIERS.MICRO);
 for(const s of frame.steps){
   assert.deepEqual(s.out.shape,r.steps.find(x=>x.id===s.id).shape);
   assert.deepEqual(s.out.values,r.steps.find(x=>x.id===s.id).values.flat());
   assert.equal(typeof s.why,'string');
   for(const dep of s.dependencies)assert.ok(frame.steps.findIndex(x=>x.id===dep.from)<frame.steps.indexOf(s));
   const info=C.dot(frame,s.id,0,0);assert.ok(Number.isFinite(info.value));
   if(s.kind==='linear'||s.kind==='matmul'){
     assert.ok(info.reconstructable, s.kind);
     assert.ok(info.error<1e-9,'full dot must numerically close');
     assert.equal(info.terms.length,s.dot.input.shape[1]);
     for(let row=0;row<s.out.shape[0];row++)for(let col=0;col<s.out.shape[1];col++)
       assert.ok(C.dot(frame,s.id,row,col).error<1e-9);
   }else assert.equal(info.reconstructable,false);
 }
 if(name==='attention'){
   const attn=frame.steps.find(x=>x.kind==='attention');
   assert.equal(attn.substeps.length,3);
   assert.deepEqual(attn.substeps.map(x=>x.kind),['scaled_matmul','softmax','matmul']);
   for(const sub of [attn.substeps[0],attn.substeps[2]])
    for(let t=0;t<sub.out.shape[0];t++)for(let c=0;c<sub.out.shape[1];c++)
     assert.ok(C.dot(frame,sub.id,t,c).error<1e-9,'Attention inner GEMM reconstruction');
   const probs=attn.substeps[1].out;
   for(let t=0;t<probs.shape[0];t++)
     assert.ok(Math.abs(probs.values.slice(t*probs.shape[1],(t+1)*probs.shape[1]).reduce((x,y)=>x+y,0)-1)<1e-10);
   const tamper=JSON.parse(JSON.stringify(frame));
   tamper.steps.find(x=>x.kind==='attention').substeps[0].out.values[0]+=0.1;
   assert.throws(()=>C.validate(tamper),/GEMM reconstruction contradicts/);
 }
 const symbolic=C.learning(g,Concept.evaluate(g),L);
 assert.equal(symbolic.evidenceTier,C.TIERS.STATIC);
 assert.ok(symbolic.steps.every(s=>s.out.values===null));
 assert.throws(()=>C.dot(symbolic,symbolic.steps[0].id,0,0),/numeric execution unavailable/);
 assert.throws(()=>C.learning(g,{...r,evidence:'real_model_execution'},L),/unrecognized execution evidence/);
}
// Rank-4/5 tensor slicing must be a first-class capability, not a Wan-specific view.
const ndGraph={nodes:[{id:'nd',kind:'input',params:{}}],edges:[],title:'4D tensor trace'};
const ndValues=[[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]];
const ndTrace={steps:[{id:'nd',kind:'input',shape:[1,2,2,3],values:ndValues}],evidence:'deterministic_browser_teaching_simulation_not_model_execution'};
const ndFrame=C.learning(ndGraph,ndTrace,L);
assert.deepEqual(ndFrame.steps[0].out.shape,[1,2,2,3]);
assert.equal(C.dot(ndFrame,'nd',1,2,[0,1]).value,12);
assert.equal(C.dot(ndFrame,'nd',0,0,[0,0]).value,1);
assert.throws(()=>C.dot(ndFrame,'nd',0,0,[0,2]),/invalid tensor axis slice/);
assert.throws(()=>C.learning(ndGraph,{...ndTrace,steps:[{...ndTrace.steps[0],values:[[[[1,2],[3,4]],[[5,6],[7,8]]]]}]},L),/tensor shape contradicts/);
const five=C.numericTensor([1,2,1,2,3],Array.from({length:12},(_,i)=>i+.5));
const fiveFrame=JSON.parse(JSON.stringify(ndFrame));fiveFrame.steps[0].out=five;C.validate(fiveFrame);
assert.equal(C.dot(fiveFrame,'nd',1,2,[0,1,0]).value,11.5);
const g=L.preset('ffn'),r=L.evaluate(g);
const original=C.learning(g,r,L);
const first=original.steps.find(s=>s.kind==='linear');
const forged=JSON.parse(JSON.stringify(original));
forged.steps.find(s=>s.id===first.id).out.values[0]+=0.2;
assert.throws(()=>C.validate(forged),/GEMM reconstruction contradicts displayed output/);
const badDep=JSON.parse(JSON.stringify(original));badDep.steps[1].dependencies.push({from:'future_fake',port:'x'});
assert.throws(()=>C.validate(badDep),/dependencies must point backward/);
const badTier=JSON.parse(JSON.stringify(original));badTier.evidenceTier='REAL_MODEL_VERIFIED';
assert.throws(()=>C.validate(badTier),/unknown evidence tier/);
const sourceForged=JSON.parse(JSON.stringify(original));sourceForged.source={url:'javascript:alert(1)',tier:'PINNED_CODE_STRUCTURE_ONLY'};
assert.throws(()=>C.validate(sourceForged),/unverified pinned-source metadata/);
const w=W.trace({seed:19,tokens:3,dim:4,hidden:8});
const wan=C.wan(w);
assert.equal(wan.steps.length,9);assert.equal(wan.evidenceTier,C.TIERS.WAN);
assert.equal(wan.source.revision,W.SOURCE.commit);
for(const idx of [4,6]){
 const s=wan.steps[idx];for(let t=0;t<s.out.shape[0];t++)for(let c=0;c<s.out.shape[1];c++)
 assert.ok(C.dot(wan,s.id,t,c).error<1e-9);
}
assert.deepEqual(wan.steps.map(s=>s.id),W.STEPS.map(s=>s.id));
assert.throws(()=>C.wan({...w,source:{...w.source,commit:'0000000000000000000000000000000000000000'}}),/unsupported Wan source revision/);
const h=fs.readFileSync(path.join(__dirname,'../learn.html'),'utf8'),page=fs.readFileSync(path.join(__dirname,'../wan_ffn_lab.html'),'utf8');
for(const file of ['compute_microscope.js','compute_microscope_ui.js'])assert.ok(h.includes('src="'+file+'"')&&page.includes('src="'+file+'"'));
for(const el of ['computeMicroscope','wanSharedMicroscope'])assert.ok((h+page).includes('id="'+el+'"'));
for(const file of ['compute_microscope.css','compute_microscope.js','compute_microscope_ui.js'])
 assert.ok(fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/).includes(file),file+' Android manifest');
for(const js of ['compute_microscope_ui.js','wan_ffn_lab.js'])new vm.Script(fs.readFileSync(path.join(__dirname,'..',js),'utf8'));
assert.ok(h.includes('Microscope.learning(used,run,L)'));
assert.ok(page.includes('wanSharedMicroscope'));
console.log('UNIVERSAL_MICROSCOPE_PASS 4 unrelated graph templates, symbolic-no-numeric, Linear/MatMul per-coordinate proof, Wan source-pinned adapter, forged-evidence refusal, shared PC/Android UI');