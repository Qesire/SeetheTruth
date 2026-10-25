'use strict';
// Cross-substrate extension contract, derived from the shipping registry and
// deployed pages rather than hardcoded operator counts or hand-maintained flags.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const ROOT=path.resolve(__dirname,'..');
const read=file=>fs.readFileSync(path.join(ROOT,file),'utf8');
const Core=require('../learning_experience_core.js');
const L=require('../learning_lab.js'),X=require('../block_extension_registry.js');
const Operators=require('../operator_contracts.js'),Capabilities=require('../block_capabilities.js');
const C=require('../teaching_curriculum.js');
const inventory=Operators.inventory(),capacity=Capabilities.audit();
assert.equal(inventory.count,L.KINDS.length);
assert.equal(capacity.counts.registered,inventory.count);
assert.equal(C.lessons.length,Core.inventory().counts.courses);
assert.equal(new Set(L.KINDS).size,L.KINDS.length);
assert.equal(new Set(Core.labs.map(l=>l.id)).size,Core.labs.length);
assert.equal(new Set(Core.labs.map(l=>l.href)).size,Core.labs.length);
const actionMaps=new Map();
for(const lab of Core.labs){
 const operations=Core.describeActions(lab.id),identity=Core.inspectLab(lab.id);
 assert.deepEqual(identity.operations,operations,'contract projection is one source: '+lab.id);
 assert.equal(identity.evidence,'NOT_EXECUTED');
 assert.ok(fs.existsSync(path.join(ROOT,lab.href)),lab.id+' page missing');
 assert.ok(fs.existsSync(path.join(ROOT,lab.engine)),lab.id+' source absent');
 assert.equal(new Set(operations.map(op=>op.name)).size,operations.length);
 for(const op of operations){
  assert.match(op.name,/^[a-z_]+$/);
  assert.ok(op.sourceModule&&op.sourceMethod);
  assert.ok(['none','state_transition','tensor_computation'].includes(op.observation));
  assert.equal(op.evidence,'NOT_EXECUTED','introspection cannot become execution evidence');
 }
 actionMaps.set(lab.id,Object.fromEntries(operations.map(o=>[o.name,o.observation])));
}
assert.deepEqual(Core.describeActions('source_atlas'),[]);
assert.deepEqual(Core.describeActions('real_model'),[]);
assert.deepEqual(Core.describeActions('telemetry'),[]);
assert.throws(()=>Core.describeActions('unregistered_domain'),/unknown specialized adapter/);
for(const [id,method,type] of [
 ['wan_ffn','trace','tensor_computation'],['deepinfer_prefill','inspect_pair','state_transition'],
 ['deepinfer_worker','simulate','state_transition'],['deepinfer_kv','replay','state_transition'],
 ['deepinfer_scheduler','simulate','state_transition'],['training','scalar','state_transition'],
 ['cache','replay_steps','state_transition'],['engineering_pipeline','train','state_transition'],
 ['engineering_pipeline','timeline','state_transition']
])assert.equal(actionMaps.get(id)?.[method],type,id+'/'+method+' lost native observation');
assert.equal(actionMaps.get('deepinfer_worker').compare,'none');
assert.equal(actionMaps.get('paper').online_attention,'none');
assert.equal(actionMaps.get('capabilities').audit,'none');
const sourceProofs=[
 Core.executeLab('wan_ffn','trace',{options:{seed:17,tokens:3,dim:4,hidden:8}}),
 Core.executeLab('deepinfer_kv','replay',{config:{numBlocks:7,blockSize:4},operations:[
  {op:'arrive',id:'A',tokenLen:8},{op:'allocate',id:'A'}]}),
 Core.executeLab('engineering_pipeline','timeline',{config:{accumulationSteps:2,microBatchSize:2},
  actions:['micro','micro','optimizer']}),
 Core.executeLab('training','scalar',{config:{steps:1}})
];
for(const record of sourceProofs){
 const action=Core.describeActions(record.id).find(x=>x.name===record.operation);
 assert.ok(action);
 const observed=Core.observe(record);
 assert.equal(observed.sources[0]?.kind,action.observation);
 assert.equal(Core.replay(record).reproduced,true);
}
const staticAudit=Core.executeLab('capabilities','audit',{});
assert.equal(Core.observe(staticAudit).traceAvailable,false);
assert.equal(actionMaps.get('capabilities').audit,'none');
// Every UI consumer shares the same core/codec, with script order explicit.
// This is source loading correctness, not an assertion that all UIs are migrated.
const pages=['learn.html',...Core.labs.map(l=>l.href).filter(p=>
 read(p).includes('src="unified_observation_ui.js"'))];
const consumed=[...new Set(pages)];
const routes=read('atlas_server.py'),assets=new Set(read('android/teaching-assets.txt').split(/\r?\n/));
for(const page of consumed){
 const html=read(page),scripts=[...html.matchAll(/<script[^>]+src="([^"]+)"/g)].map(m=>m[1]);
 const where=x=>scripts.indexOf(x);
 for(const lib of ['learning_experience_core.js','observation_snapshots.js','unified_observation_ui.js'])
  assert.ok(where(lib)>=0,'shared observer missing '+page+' -> '+lib);
 assert.ok(where('learning_experience_core.js')<where('unified_observation_ui.js'));
 assert.ok(where('observation_snapshots.js')<where('unified_observation_ui.js'));
 if(where('unified_state_lab_ui.js')>=0)
  assert.ok(where('unified_observation_ui.js')<where('unified_state_lab_ui.js'));
 if(page==='engineering_pipeline.html')
  assert.ok(where('unified_observation_ui.js')<where('training_timeline_lab_ui.js'));
 for(const script of scripts){
  assert.ok(fs.existsSync(path.join(ROOT,script)),page+' missing script '+script);
  assert.ok(routes.includes('"/'+script+'":"'+script+'"'),page+' missing HTTP route '+script);
  if(assets.has(page))assert.ok(assets.has(script),page+' missing Android dependency '+script);
 }
}
// Exercise a previously unseen standalone extension pack through the actual
// generic extension contract. Installation into LearningLab is separately
// recorded as static bootstrapping work (not dynamic plugin loading).
const probe={
 name:'contract_probe',
 defs:{contract_probe_source:{title:'Probe',en:'Probe',category:'data',ports:[],
  params:{length:2},formula:'out[i] = i + 1',
  why:'Exercise generalized shape and value routing',lesson:'Inspect exact indices',
  caution:'Synthetic diagnostic values',source:'https://example.com/protocol'}},
 validate:(_kind,p)=>{if(!Number.isSafeInteger(p.length)||p.length<1||p.length>8)
  throw Error('probe length out of bounds')},
 infer:(_kind,p)=>[p.length],
 execute:(_kind,p)=>({values:Array.from({length:p.length},(_,i)=>i+1),flops:0})
};
const installed=X.createRegistry([probe]),node={kind:'contract_probe_source',params:{length:3}};
assert.deepEqual(installed.infer(node,{}).shape,[3]);
assert.deepEqual(installed.execute(node,{}).values,[1,2,3]);
assert.throws(()=>installed.execute({kind:node.kind,params:{length:9}},{}),/probe length/);
assert.throws(()=>X.createRegistry([probe,probe]),/duplicate pack/);
assert.ok(!L.KINDS.includes('contract_probe_source'),'registry-only fixture cannot secretly mutate active graph runtime');
const count=Core.labs.reduce((n,lab)=>n+Core.describeActions(lab.id).length,0);
console.log('ARCHITECTURE_EXTENSION_SPINE_PASS '+JSON.stringify({
 operators:L.KINDS.length,extensionKinds:X.kinds.length,courses:C.lessons.length,
 specialists:Core.labs.length,actions:count,sharedObservationPages:consumed.length,
 numpyEligible:capacity.counts.numpyEligible,torchEligible:capacity.counts.torchEligible
}));
