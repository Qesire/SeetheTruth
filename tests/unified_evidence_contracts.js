'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const read=p=>fs.readFileSync(path.join(__dirname,'..',p),'utf8');
const Core=require('../learning_experience_core.js'),Evidence=require('../learning_evidence_core.js');
const Source=require('../source_evidence_levels.js'),Trace=require('../model_trace.js');
const Tele=require('../telemetry_evidence.js');
assert.equal(Evidence.SCHEMA,'seethetruth-evidence-inspection-v1');
const graph={meta:{analysis:'static-python-ast',source:'fixture-only'},nodes:[
 {id:'root',parent:null,label:'Decoder',type:'block'},
 {id:'linear',parent:'root',label:'linear',callee:'linear',certainty:'source_operator'},
 {id:'gelu',parent:'root',label:'gelu',callee:'gelu',certainty:'source_operator'}],
 edges:[{from:'root',to:'linear',kind:'hierarchy'},{from:'root',to:'gelu',kind:'hierarchy'}]};
const original=JSON.stringify(graph),projected=Core.inspectEvidence('source_atlas','project',{graph,scope:'root',level:'matrix'});
assert.equal(projected.kind,'evidence_inspection');assert.equal(projected.executed,false);
assert.equal(projected.independentlyReplayed,false);assert.equal(projected.evidenceTier,'STATIC_SOURCE_ONLY');
assert.deepEqual(projected.data.projection.nodes.map(x=>x.id),['linear']);
assert.deepEqual(projected.data.projection.matchIds,['linear']);
assert.equal(projected.data.projection.matches.has('linear'),true);
assert.equal(JSON.stringify(graph),original);
assert.throws(()=>Core.inspectEvidence('source_atlas','project',{graph:{...graph,meta:{analysis:'synthetic'}},scope:'root'}),/not a static Python AST/);
assert.throws(()=>Core.inspectEvidence('source_atlas','project',{graph,scope:'unknown',level:'matrix'}),/Unknown source scope/);
assert.throws(()=>Core.inspectEvidence('source_atlas','project',{graph,scope:'root',level:'unknown'}),/Invalid source level/);
assert.throws(()=>Core.inspectEvidence('source_atlas','project',{graph,scope:'root',capacity:100000}),/invalid projection capacity/);
const val=x=>({shape:[2,2],count:4,mean:x.reduce((a,b)=>a+b,0)/4,
 rms:Math.sqrt(x.reduce((a,b)=>a+b*b,0)/4),min:Math.min(...x),
 max:Math.max(...x),l2:Math.hypot(...x),
 zeroFraction:x.filter(z=>z===0).length/4,sample:x.slice()});
const trace={schema:'seethetruth-real-model-trace-v1',provenance:{
 kind:'real_model_execution',modelId:'TEST_FIXTURE_NOT_CAPTURED/TinyStories',revision:'a'.repeat(40),
 framework:'pytorch',device:'cpu',dtype:'float32',modulePath:'transformer.h.0.mlp.c_fc',
 moduleType:'torch.nn.Linear',objective:'causal_language_modeling_shifted_cross_entropy',
 promptSha256:'b'.repeat(64),tokenIds:[1,2,3]},loss:.5,
 parameter:{identity:'transformer.h.0.mlp.c_fc.weight',
  weight:val([1,2,3,4]),gradient:val([-1,0,1,0])},
 activation:{input:[1,2,3],output:[2,-1,0]}};
const model=Core.inspectEvidence('real_model','validate_import',{trace});
assert.equal(model.evidenceTier,'IMPORTED_MODEL_PROVENANCE_CLAIM_UNREPLAYED');
assert.equal(model.executed,false);assert.equal(model.independentlyReplayed,false);
assert.deepEqual(model.data.summary,Trace.validate(trace));
assert.equal(model.data.summary.status,'provenance_claim_structurally_valid_not_independently_verified');
assert.throws(()=>Core.inspectEvidence('real_model','validate_import',{trace:{...trace,loss:NaN}}),/invalid model trace/);
assert.throws(()=>Core.inspectEvidence('real_model','validate_import',{trace:{...trace,provenance:{...trace.provenance,revision:'deadbeef'}}}),/provenance/);
const synthetic=Tele.example(),snapshot=JSON.stringify(synthetic);
Tele.validate(synthetic);
const demo=Core.inspectEvidence('telemetry','validate_import',{data:synthetic,synthetic:false});
assert.equal(demo.evidenceTier,'SYNTHETIC_TELEMETRY_DEMO');
assert.equal(demo.data.synthetic,true,'re-imported synthetic telemetry is still synthetic');
assert.equal(demo.data.counts.snapshots,1);assert.equal(JSON.stringify(synthetic),snapshot);
const imported=JSON.parse(snapshot);imported.meta.measurement='UNVERIFIED_PROFILER_IMPORT';
const record=Core.inspectEvidence('telemetry','validate_import',{data:imported});
assert.equal(record.evidenceTier,'IMPORTED_TELEMETRY_UNATTESTED');
assert.equal(record.executed,false);assert.equal(record.independentlyReplayed,false);
const wrong=JSON.parse(snapshot);wrong.snapshots[0].summary.finite_count--;
assert.throws(()=>Core.inspectEvidence('telemetry','validate_import',{data:wrong}),/采样计数|直方图/);
const bad=JSON.parse(snapshot);bad.operators[0].self_cpu_ms=-1;
assert.throws(()=>Core.inspectEvidence('telemetry','validate_import',{data:bad}),/算子耗时/);
assert.throws(()=>Core.inspectEvidence('source_atlas','validate_import',{data:synthetic}),/unregistered evidence operation/);
assert.throws(()=>Core.inspectEvidence('paper','project',{graph}),/not a registered evidence-only specialist/);
for(const [page,deps] of Object.entries({
 'index.html':['source_evidence_levels.js','learning_evidence_core.js','learning_experience_core.js'],
 'real_model_lab.html':['model_trace.js','learning_evidence_core.js','learning_experience_core.js'],
 'telemetry.html':['telemetry_evidence.js','learning_evidence_core.js','learning_experience_core.js']
})){
 const html=read(page);let pos=-1;for(const name of deps){const loc=html.indexOf('src="'+name+'"');assert.ok(loc>pos,'script order '+page+' '+name);pos=loc;}
}
assert.ok(read('index.html').includes('UnifiedCore.inspectEvidence("source_atlas"'));
assert.ok(!read('index.html').includes('SeetheTruthSourceView.project('),'all source projector UI paths must use one inspected evidence contract');
assert.ok(read('real_model_lab.html').includes('UnifiedCore.inspectEvidence("real_model"'));
assert.ok(read('telemetry.html').includes('UnifiedCore.inspectEvidence("telemetry"'));
assert.ok(!read('telemetry.html').includes('function validate(d){'),'no divergent telemetry validator may remain in UI');
for(const name of ['learning_evidence_core.js','telemetry_evidence.js'])
 assert.ok(read('atlas_server.py').includes('"/'+name+'":"'+name+'"'));
console.log('EVIDENCE_UNIFICATION_PASS static AST source projection; unverified model provenance; synthetic/imported telemetry; explicit no-execution boundaries');
