'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const L=require('../learning_lab.js'),C=require('../compute_microscope.js'),R=require('../block_extension_registry.js');
const Concept=require('../concept_flow.js'),Readiness=require('../graph_readiness.js'),P=require('../paper_patches.js');
assert.deepEqual(R.packs,['semantic_video','paper_operators','named_axis']);
assert.equal(R.kinds.length,11);
for(const kind of R.kinds){assert.ok(L.KINDS.includes(kind),kind+' must be in the real free-graph block palette');assert.equal(L.defs[kind].pack,kind.startsWith('paper_')?'paper_operators':kind.startsWith('axis_')?'named_axis':'semantic_video');}
function outcome(name){const graph=L.preset(name);const run=L.evaluate(graph);const frame=C.learning(graph,run,L);assert.equal(C.validate(frame),frame);
 const symbolic=Concept.evaluate(graph),check=Readiness.check(graph,{numeric:true});
 assert.ok(check.executable,name+' should pass generic graph readiness');
 assert.equal(symbolic.steps.length,run.steps.length);
 for(let i=0;i<run.steps.length;i++){assert.deepEqual(run.steps[i].shape,symbolic.steps[i].shape);assert.deepEqual(frame.steps[i].out.shape,run.steps[i].shape);}
 return {graph,run,frame};
}
const v=outcome('extension_video_composition');
assert.deepEqual(v.run.steps.at(-1).shape,[1,3,2,2,2]);
assert.deepEqual(v.frame.steps.at(-1).out.values,v.frame.steps[0].out.values,'semantic source split/reconcat must preserve every pixel');
assert.equal(v.frame.steps.filter(s=>s.substeps).length,4);
assert.equal(v.frame.steps.at(-1).substeps[0].kind,'named_axis_transform');
assert.ok(C.dot(v.frame,'join',0,0,[0,2,0]).value!==null);
const paper=outcome('extension_paper_composition');
const lo=paper.run.steps.find(x=>x.id==='lora'),do_=paper.run.steps.find(x=>x.id==='dora'),blend=paper.run.steps.find(x=>x.id==='blend');
assert.equal(lo.extensionSubsteps.length,5);assert.equal(do_.extensionSubsteps.length,6);
for(let i=0;i<2;i++)for(let j=0;j<2;j++){assert.ok(Math.abs(blend.values[i][j]-lo.values[i][j]-do_.values[i][j])<1e-9);
 const hit=C.dot(paper.frame,'lora:paper_step_0',i,j);assert.ok(hit.reconstructable&&hit.error<=1e-9);}
const online=outcome('extension_online_attention');const op=online.frame.steps.find(x=>x.id==='online');
assert.equal(op.substeps.length,12);assert.ok(op.substeps.some(x=>x.kind==='scaled_matmul'));
for(const item of op.substeps.filter(x=>x.dot)){for(let j=0;j<item.out.shape[1];j++)assert.ok(C.dot(online.frame,item.id,0,j).error<1e-9);}
const original=L.evaluate(L.preset('ffn'));assert.equal(original.steps.length,4,'older native blocks still work');
const bad=JSON.parse(JSON.stringify(paper.graph));bad.nodes.find(x=>x.id==='dora').params.alpha=Infinity;
assert.throws(()=>L.evaluate(bad),/paper alpha/);
const mismatch=JSON.parse(JSON.stringify(v.graph));mismatch.nodes.find(x=>x.id==='last').params.start=3;
assert.throws(()=>L.evaluate(mismatch),/time slice exceeds T/);
assert.throws(()=>R.tensorShape([[[[NaN]]]]),/nonfinite/);
assert.throws(()=>R.tensorShape([[1],[2,3]]),/ragged/);
const malicious={name:'malicious_pack',defs:{test_a:{title:'Test',en:'Test',category:'compute',ports:[],params:{},formula:'1',why:'test',lesson:'test',caution:'test',source:'http://not-an-immutable-reference'}},
 validate(){},infer(){return [1,1]},execute(){return {values:[[1]]}}};
assert.throws(()=>R.createRegistry([malicious]),/real academic or source reference/);
const fake={...malicious,defs:{test_a:{...malicious.defs.test_a,source:'https://example.org/actual-source'}}};
const extension=R.createRegistry([fake]);
assert.deepEqual(extension.infer({kind:'test_a',params:{}},{}).shape,[1,1]);
assert.deepEqual(extension.execute({kind:'test_a',params:{}},{}).values,[[1]]);
assert.throws(()=>R.createRegistry([fake,fake]),/duplicate pack id/);
assert.throws(()=>R.createRegistry([fake,{...fake,name:'another_pack'}]),/invalid\/duplicate registered block kind/);
const html=fs.readFileSync(path.join(__dirname,'../learn.html'),'utf8');
for(const item of ['semantic_block_pack.js','paper_block_pack.js','axis_block_pack.js','block_extension_registry.js'])assert.ok(html.includes('src="'+item+'"'),item);
for(const name of ['extension_video_composition','extension_paper_composition','extension_online_attention'])assert.ok(html.includes('data-preset="'+name+'"'));
const listed=fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/);
for(const item of ['semantic_block_pack.js','paper_block_pack.js','axis_block_pack.js','block_extension_registry.js'])assert.ok(listed.includes(item),item+' offline packaged');
console.log('BLOCK_EXTENSION_REGISTRY_PASS eleven real blocks, four composable graph presets, original FFN parity, actual ND/video/LoRA/DoRA/online numerical evidence and mutated-contract rejection');
