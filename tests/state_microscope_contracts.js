'use strict';
const a=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const S=require('../state_microscope.js'),K=require('../deepinfer_kv_lifecycle.js'),E=require('../deepinfer_worker_events.js');
const Pipeline=require('../engineering_training_pipeline.js'),Scalar=require('../training_flow.js');
const Scheduler=require('../deepinfer_scheduler_workflow.js'),Policy=require('../deepinfer_prefill_policy.js');
const Cache=require('../kv_cache_lab.js');
const ops=[{op:'arrive',id:'A',tokenLen:8},{op:'allocate',id:'A'},{op:'arrive',id:'B',tokenLen:4},{op:'allocate',id:'B'},{op:'fork_pin',id:'C',from:'A',cachedTokens:5,tokenLen:8},{op:'fork_activate',id:'C'},{op:'finish',id:'A'},{op:'finish',id:'C'},{op:'finish',id:'B'}];
const original=K.replay({numBlocks:7,blockSize:4},ops),f=S.kv(original,K);
a.equal(f.evidenceTier,S.TIERS.KV);a.equal(S.validate(f),f);a.equal(f.steps.length,ops.length+1);
a.equal(f.steps[0].changes.length,0);
a.ok(f.steps.some(s=>s.changes.some(d=>d.key==='refs')));
for(let i=1;i<f.steps.length;i++){a.deepEqual(f.steps[i].before,f.steps[i-1].after);a.deepEqual(f.steps[i].dependencies,[f.steps[i-1].id]);}
const bad=JSON.parse(JSON.stringify(f));bad.steps[3].changes.pop();a.throws(()=>S.validate(bad),/inconsistent state changes/);
const invalid=JSON.parse(JSON.stringify(original));invalid.timeline[3].refs[1]=17;
a.throws(()=>S.kv(invalid,K),/reference count|state microscope/);
const run=E.simulate(E.DEFAULT_STEPS,{zeroOverhead:true}),g=S.worker(run);
a.equal(g.evidenceTier,S.TIERS.WORKER);a.equal(S.validate(g),g);a.equal(g.steps.length,run.events.length);
for(let i=0;i<g.steps.length;i++){
 const e=run.events[i],s=g.steps[i];a.equal(s.timing.start,e.start);a.equal(s.timing.end,e.end);a.deepEqual(s.dependencies,e.dependsOn);
}
const forged=JSON.parse(JSON.stringify(run));forged.events[1].dependsOn=['missing'];
a.throws(()=>S.worker(forged),/worker dependency order/);
const overlap=JSON.parse(JSON.stringify(run)),id=overlap.events.findIndex((e,i)=>i>0&&e.resource===overlap.events[0].resource);
if(id>0){overlap.events[id].start=0;overlap.events[id].end=overlap.events[id].duration;a.throws(()=>S.worker(overlap),/overlaps|worker dependency/);}
for(const [name,scenario]of Object.entries(Scheduler.EXAMPLES)){
 const run=Scheduler.simulate(scenario),frame=S.scheduler(run,K);
 a.equal(S.validate(frame),frame);
 a.equal(frame.evidenceTier,S.TIERS.SCHEDULER);
 a.equal(frame.steps.length,run.cycles.length+1);
 for(let i=1;i<frame.steps.length;i++){
  a.equal(frame.steps[i].after.forwardEvent,run.executionProof[i-1].forwardEvent);
  a.deepEqual(frame.steps[i].after.scheduled,run.decisions[i-1].scheduled);
  a.deepEqual(frame.steps[i].dependencies,[frame.steps[i-1].id]);
 }
 const forged=JSON.parse(JSON.stringify(run));forged.executionProof[0].batch=['spoofed'];
 a.throws(()=>S.scheduler(forged,K),/worker dispatch differs/);
 const wrongKV=JSON.parse(JSON.stringify(run));wrongKV.snapshots[1].refs[1]+=1;
 a.throws(()=>S.scheduler(wrongKV,K),/reference count/);
}
const inputs={
 voteInput:{engine:'NORMAL',decodeStep:4,allInterval:2,anyInterval:4,
  ranks:[{waitingSchedulable:true,canActuallySchedule:true,cacheUsage:.2},{waitingSchedulable:false,canActuallySchedule:false,cacheUsage:.2}]},
 chunkInput:{tokens:['TEXT','IMG','PAD','PAD','PAD','TEXT'],maxBatchedTokens:4,cpAlign:1}
};
const policyInputs={...inputs,voteResult:Policy.vote(inputs.voteInput),chunkResult:Policy.chunk(inputs.chunkInput)};
const policyFrame=S.prefill(policyInputs,Policy);
a.equal(policyFrame.evidenceTier,S.TIERS.PREFILL);
a.equal(policyFrame.steps.length,2);
a.deepEqual(policyFrame.steps.map(x=>x.dependencies),[[],[]],'vote and chunk are independent source decisions');
a.equal(policyFrame.steps[0].after.decision.point,'ANY');
a.ok(policyFrame.steps[1].after.decision.q>=0);
const wrongPolicy=JSON.parse(JSON.stringify(policyInputs));wrongPolicy.voteResult.step='prompt';
a.throws(()=>S.prefill(wrongPolicy,Policy),/vote result contradicts/);
const wrongChunk=JSON.parse(JSON.stringify(policyInputs));wrongChunk.chunkResult.q+=1;
a.throws(()=>S.prefill(wrongChunk,Policy),/chunk result contradicts/);
const session=Cache.Session({prefix:3,decode:4,heads:2,kvHeads:1,rope:true});
const cacheSnaps=[session.snapshot()];for(let i=0;i<5;i++)cacheSnaps.push(session.advance());
const cacheFrame=S.cache(cacheSnaps,Cache);
a.equal(cacheFrame.evidenceTier,S.TIERS.CACHE);
a.equal(cacheFrame.steps.length,6);
a.equal(cacheFrame.steps[1].after.phase,'prefill');
a.equal(cacheFrame.steps[2].after.phase,'decode');
for(let i=1;i<cacheFrame.steps.length;i++){
 a.equal(cacheFrame.steps[i].after.cacheTokens,cacheSnaps[i].cursor);
 a.deepEqual(cacheFrame.steps[i].dependencies,[cacheFrame.steps[i-1].id]);
}
const staleCache=JSON.parse(JSON.stringify(cacheSnaps));staleCache[3].cache.key[0][0]+=.0001;
a.throws(()=>S.cache(staleCache,Cache),/KV micro cache state contradicts/);
const wrongConfig=JSON.parse(JSON.stringify(cacheSnaps));wrongConfig[0].config.rope=false;
a.throws(()=>S.cache(wrongConfig,Cache),/KV micro cache state contradicts/);
const scalarRun=Scalar.build({},'scalar'),scalarFrame=S.scalar(scalarRun);
a.equal(scalarFrame.evidenceTier,S.TIERS.SCALAR);
a.equal(scalarFrame.steps.length,scalarRun.events.length);
a.equal(scalarFrame.steps.at(-1).after.optimizerStep,scalarRun.final.step);
a.ok(scalarFrame.steps.some(x=>x.changes.some(v=>v.key==='grad')));
a.throws(()=>S.scalar(Scalar.build({},'concept')),/scalar training requires executed/);
const p=Pipeline.create(),snap0=p.checkpoint(),m=p.microStep(),snap1=p.checkpoint();
p.microStep();p.microStep();p.microStep();p.optimizerStep();const snap2=p.checkpoint();
const training=S.training([snap0,snap1,snap2]);
a.equal(training.evidenceTier,S.TIERS.TRAIN);
a.equal(training.steps.length,3);
a.equal(training.steps[1].after.pendingMicroSteps,1);
a.equal(training.steps[2].after.optimizerStep,1);
a.ok(training.steps[2].changes.some(x=>x.key==='weightsChecksum'));
const corrupted=JSON.parse(JSON.stringify(snap1));corrupted.config.seed+=1;
a.throws(()=>S.training([snap0,corrupted]),/training data\/config identity changed/);
const huge=JSON.parse(JSON.stringify(g));huge.evidenceTier='REAL_GPU_TRACE';a.throws(()=>S.validate(huge),/unknown evidence tier/);
const wrong=JSON.parse(JSON.stringify(g));wrong.steps[1].dependencies=['nonsense'];a.throws(()=>S.validate(wrong),/invalid causal dependencies/);
for(const fn of ['state_microscope_ui.js'])new vm.Script(fs.readFileSync(path.join(__dirname,'..',fn),'utf8'));
for(const page of ['deepinfer_worker_lab.html','deepinfer_kv_lifecycle_lab.html','engineering_pipeline.html','training.html','deepinfer_scheduler_lab.html','deepinfer_prefill_lab.html','cache.html']){
 const s=fs.readFileSync(path.join(__dirname,'..',page),'utf8');
 for(const dep of ['state_microscope.js','state_microscope_ui.js','state_microscope.css'])a.ok(s.includes(dep),page+' missing '+dep);
 a.ok(s.includes('id="stateMicroscope"'));
}
const allow=fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/);
for(const dep of ['state_microscope.js','state_microscope_ui.js','state_microscope.css'])a.ok(allow.includes(dep),'missing android asset '+dep);
console.log('UNIVERSAL_STATE_PASS validated Scheduler, Prefill source policy and GQA numerical cache replay; KV '+f.steps.length+' verified replay steps + Worker '+g.steps.length+' DAG events + Bigram '+training.steps.length+' live checkpoints + scalar '+scalarFrame.steps.length+' derivative/optimizer events');