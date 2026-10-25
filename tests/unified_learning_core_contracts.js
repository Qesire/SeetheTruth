'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const Core=require('../learning_experience_core.js');
const L=require('../learning_lab.js'),C=require('../teaching_curriculum.js');
const X=require('../block_extension_registry.js'),P=require('../deepinfer_prefill_policy.js');
const S=require('../graph_session.js'),B=require('../branch_execution.js');
const studio=read('learn.html'),ids=[
 ...[...studio.matchAll(/data-preset="([^"]+)"/g)].map(m=>m[1]),
 ...X.recipes.map(x=>x.id),
 "stream_state_branches" // explicit standalone shortcut on Learning Studio
];
assert.equal(Core.SCHEMA,'seethetruth-learning-experience-v1');
assert.equal(new Set(ids).size,ids.length);
const indexed=Core.activateGraphLessons(ids);
assert.deepEqual(indexed.map(x=>x.id),ids);
const all=Core.inventory(ids);
assert.equal(all.counts.operators,L.KINDS.length);
assert.equal(all.counts.graphLessons,ids.length);
assert.equal(all.counts.courses,C.lessons.length);
assert.equal(all.counts.labs,Core.labs.length);
assert.equal(all.counts.graphLessons,65+X.recipes.length); // includes source-pinned Wan video RoPE lesson + blank/hidden + pack recipes
assert.ok(all.counts.operators>=146);
for(const kind of L.KINDS)assert.ok(all.operators.operators[kind],
 'every native or separately installed brick needs the same operator contract: '+kind);
for(const id of ids){
 const opened=Core.openPreset(id);
 assert.equal(opened.kind,'graph');
 assert.equal(opened.evidence,'STATIC_ONLY');
 assert.equal(opened.numericOk,false);
 assert.equal(opened.path.steps.length,opened.graph.nodes.length);
 assert.equal(opened.id,id);
 assert.deepEqual(opened.path.steps.map(x=>x.kind),opened.session.nodes.map(x=>x.kind));
}
assert.throws(()=>Core.openPreset('not_an_installed_preset'),/unregistered guided-graph/);
assert.throws(()=>Core.activateGraphLessons(['typo_invalid_mission']),/unknown guided-graph recipe/);
assert.equal(Core.openPreset('stream_state_branches').graph.nodes.length,10,'hidden state demo must not silently return FFN');
const g=L.preset('residual'),old=JSON.stringify(g);
const inspected=Core.inspectGraph(g,{numeric:true,id:'free'});
assert.equal(inspected.evidence,'SYNTHETIC_NUMERIC');
assert.equal(inspected.numericOk,true);
assert.equal(inspected.path.steps.length,inspected.session.ir.nodes.length);
assert.equal(JSON.stringify(g),old,'the unified inspector must never mutate the editable canvas');
const blank=Core.openPreset('blank');
assert.equal(blank.numericOk,false);
const cut=JSON.parse(JSON.stringify(g));
cut.edges.pop();
const incomplete=Core.inspectGraph(cut);
assert.equal(incomplete.numericOk,false);
assert.equal(incomplete.executable,false);
assert.equal(incomplete.evidence,'STATIC_ONLY');
const parent=B.slice(L.preset('residual'),'act');
const scoped=Core.inspectGraph(L.preset('residual'),{targetId:'act',numeric:true});
assert.deepEqual(scoped.graph.nodes.map(n=>n.id),parent.nodes.map(n=>n.id));
assert.deepEqual(scoped.path.steps.map(s=>s.id),scoped.session.nodes.map(n=>n.id));
assert.ok(scoped.path.steps.every(step=>
 JSON.stringify(step.params)===JSON.stringify(scoped.graph.nodes.find(n=>n.id===step.id).params)));
for(const l of C.lessons){
 const course=Core.inspectCourse(l.id);
 assert.equal(course.kind,'curriculum');
 assert.equal(course.title,l.title);
 assert.equal(course.href,l.href);
 assert.equal(course.evidence,'PEDAGOGICAL_DESCRIPTION');
 assert.ok(course.target.id);
}
for(const [id,domain] of [["engineering_journey","guided_training_workflow"],["engineering_pipeline","training_state_machine"]]){
 const lab=Core.inspectLab(id);
 assert.equal(lab.runtime,domain);
 const guide=Core.inspectCourse(id);
 assert.equal(guide.target.id,id);
 assert.equal(guide.href,lab.href);
 assert.equal(guide.evidence,"PEDAGOGICAL_DESCRIPTION");
 const html=read(lab.href);
 assert.ok(html.includes('src="specialized_learning_adapter.js" data-experience="'+id+'"'));
}
const labIds=new Set(Core.labs.map(l=>l.id)),labHrefs=new Set(Core.labs.map(l=>l.href));
assert.equal(labIds.size,Core.labs.length);
assert.equal(labHrefs.size,Core.labs.length);
const allHtml=fs.readdirSync(root).filter(f=>f.endsWith('.html')&&
 !['learn.html','teaching.html','android_teaching.html'].includes(f));
assert.deepEqual([...labHrefs].sort(),allHtml.sort(),
 'every specialized root HTML route must have one explicit shared-core contract');
for(const lab of Core.labs){
 assert.ok(fs.existsSync(path.join(root,lab.href)),'missing lab page '+lab.id);
 assert.ok(fs.existsSync(path.join(root,lab.engine)),'missing real engine source '+lab.engine);
 const page=read(lab.href);
 assert.ok(page.includes('src="learning_experience_core.js"'),lab.id+' missing common registry');
 assert.ok(page.includes('src="specialized_learning_adapter.js" data-experience="'+lab.id+'"'),
  lab.id+' missing explicit typed adapter');
 assert.equal(Core.inspectLab(lab.id).evidence,'NOT_EXECUTED');
 assert.equal(Core.inspectLab(lab.id).runtime,lab.domain);
}
const packaged=read('android/teaching-assets.txt').split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
for(const href of packaged.filter(f=>f.endsWith('.html')&&!['learn.html','teaching.html','android_teaching.html'].includes(f))){
 assert.ok(labHrefs.has(href),'Android specialized page not registered: '+href);
}
for(const asset of ['learning_experience_core.js','specialized_learning_adapter.js']){
 assert.ok(packaged.includes(asset),'offline Android must use the identical core');
 assert.ok(read('atlas_server.py').includes('"/'+asset+'":"'+asset+'"'));
}
const teach=read('teaching.html');
assert.ok(teach.includes('src="learning_experience_core.js"'));
assert.ok(teach.indexOf('src="learning_experience_core.js"')<
 teach.indexOf('src="teaching_hub.js"'));
assert.ok(studio.indexOf('src="graph_learning_path.js"')<
 studio.indexOf('src="learning_experience_core.js"'));
assert.ok(studio.includes('UnifiedCore.activateGraphLessons('));
assert.ok(studio.includes('UnifiedCore.inspectGraph('));
assert.ok(read('teaching_hub.js').includes('Core.inspectCourse(selected)'));
assert.ok(read('teaching_hub.js').includes('Core.executeLab("deepinfer_prefill","vote"'));
const args={decodeStep:4,allInterval:2,anyInterval:4,ranks:[
 {waitingSchedulable:true,canActuallySchedule:true,cacheUsage:.2},
 {waitingSchedulable:false,canActuallySchedule:false,cacheUsage:.2}]};
const vote=Core.executeLab('deepinfer_prefill','vote',args);
assert.deepEqual(vote.value,P.vote(args));
assert.equal(vote.evidence,'SOURCE_DERIVED_DETERMINISTIC');
assert.deepEqual(vote.value.perRank,['prompt','empty-prompt']);
const voteAndChunk={voteInput:args,chunkInput:{tokens:['TEXT','TEXT','TEXT'],cached:0,maxBatchedTokens:2}};
const combined=Core.executeLab('deepinfer_prefill','inspect_pair',voteAndChunk);
assert.deepEqual(combined.value,{vote:P.vote(args),chunk:P.chunk(voteAndChunk.chunkInput)});
assert.equal(combined.stateTrace.evidenceTier,require('../state_microscope.js').TIERS.PREFILL);
assert.equal(combined.stateTrace.steps.length,2,'independent source decisions must not invent a causal edge');
assert.deepEqual(combined.stateTrace.steps[1].dependencies,[]);
const prefillChunk={tokens:['TEXT','TEXT','TEXT'],cached:0,maxBatchedTokens:2};
assert.deepEqual(Core.executeLab('deepinfer_prefill','chunk',prefillChunk).value,P.chunk(prefillChunk));
const Worker=require('../deepinfer_worker_events.js');
const steps=Worker.DEFAULT_STEPS.slice(0,2);
const workerArgs={steps,options:{zeroOverhead:true}};
assert.deepEqual(Core.executeLab('deepinfer_worker','simulate',workerArgs).value,
 Worker.simulate(steps,{zeroOverhead:true}));
const state=require('../state_microscope.js');
const workerFrame=Core.executeLab('deepinfer_worker','simulate',workerArgs).stateTrace;
assert.equal(workerFrame.schema,state.SCHEMA);
assert.equal(workerFrame.evidenceTier,state.TIERS.WORKER);
assert.equal(state.validate(workerFrame),workerFrame);
assert.equal(Core.executeLab('deepinfer_worker','simulate',workerArgs).value.note.includes('not measured'),true);
const KV=require('../deepinfer_kv_lifecycle.js');
const ops=[{op:'arrive',id:'A',tokenLen:5},{op:'allocate',id:'A'}];
const kvArgs={config:{numBlocks:6,blockSize:4},operations:ops};
assert.deepEqual(Core.executeLab('deepinfer_kv','replay',kvArgs).value,
 KV.replay(kvArgs.config,kvArgs.operations));
const kvFrame=Core.executeLab('deepinfer_kv','replay',kvArgs).stateTrace;
assert.equal(kvFrame.evidenceTier,state.TIERS.KV);
assert.equal(kvFrame.steps.length,ops.length+1);
const SW=require('../deepinfer_scheduler_workflow.js');
const sched={spec:SW.EXAMPLES.ample,options:{}};
assert.deepEqual(Core.executeLab('deepinfer_scheduler','simulate',sched).value,
 SW.simulate(sched.spec,sched.options));
const schedFrame=Core.executeLab('deepinfer_scheduler','simulate',sched).stateTrace;
assert.equal(schedFrame.evidenceTier,state.TIERS.SCHEDULER);
assert.equal(schedFrame.steps.length,SW.simulate(sched.spec,sched.options).decisions.length+1);
const Wan=require('../wan_ffn_trace.js'),Mic=require('../compute_microscope.js');
const wo={seed:17,tokens:3,dim:4,hidden:8};
const w=Core.executeLab('wan_ffn','trace',{options:wo});
assert.deepEqual(w.value,Wan.trace(wo));
assert.deepEqual(w.computationTrace,Mic.wan(w.value));
assert.equal(w.computationTrace.evidenceTier,Mic.TIERS.WAN);
assert.throws(()=>Core.executeLab('wan_ffn','trace',{options:{tokens:0}}),/tokens/);
const Trade=require('../architecture_tradeoffs.js');
for(const [kind,input] of Object.entries({
 gqa:{layers:2,heads:8,kvHeads:2,headDim:16,context:10,batch:3,bytes:2},
 moe:{tokens:4,hidden:8,intermediate:16,experts:8,topK:2,sharedExperts:1},
 loop:{width:8,intermediate:16,steps:6,untiedLayers:6}}))
 assert.deepEqual(Core.executeLab('tradeoffs',kind,input).value,Trade[kind](input));
assert.throws(()=>Core.executeLab('tradeoffs','gqa',{layers:1,heads:7,kvHeads:2,headDim:8,context:5}),/divisible/);
const scalar=Core.executeLab('training','scalar',{config:{}});
assert.deepEqual(scalar.value,require('../training_flow.js').build({},'scalar'));
assert.equal(scalar.stateTrace.evidenceTier,state.TIERS.SCALAR);
const Cache=require('../kv_cache_lab.js'),cached=Core.executeLab('cache','replay_steps',{config:{prefix:2,decode:1},steps:2});
const c=Cache.Session({prefix:2,decode:1});
assert.deepEqual(cached.value.snapshots,[c.snapshot(),c.advance(),c.advance()]);
assert.equal(cached.stateTrace.evidenceTier,state.TIERS.CACHE);
assert.throws(()=>Core.executeLab('cache','replay_steps',{config:{prefix:2,decode:1},steps:3}),/cache steps/);
const E=require('../engineering_journey_engine.js'),guide=Core.executeLab('engineering_journey','train',{config:{seed:7},steps:2});
const guideDirect=E.create({seed:7}),trained=guideDirect.train(2);
assert.deepEqual(guide.value.training,trained);
assert.deepEqual(guide.value.checkpoint,guideDirect.checkpoint());
assert.throws(()=>Core.executeLab('engineering_journey','train',{steps:121}),/Bigram steps/);
const Pipeline=require('../engineering_training_pipeline.js');
const p=Core.executeLab('engineering_pipeline','train',{config:{seed:7},steps:2}),p0=Pipeline.create({seed:7});
const snapshots=[p0.checkpoint()];for(let i=0;i<2;i++){p0.train(1);snapshots.push(p0.checkpoint());}
assert.deepEqual(p.value.snapshots,snapshots);
assert.equal(p.stateTrace.evidenceTier,state.TIERS.TRAIN);
assert.equal(p.stateTrace.steps.length,3);
assert.throws(()=>Core.executeLab('engineering_pipeline','train',{steps:13}),/optimizer steps/);
assert.throws(()=>Core.executeLab('deepinfer_worker','vote',args),/no registered executable adapter/);
assert.throws(()=>Core.executeLab('deepinfer_prefill','unknown',args),/no registered executable adapter/);
assert.throws(()=>Core.inspectLab('does_not_exist'),/unknown specialized adapter/);
assert.throws(()=>Core.inspectCourse('does_not_exist'),/unknown curriculum lesson/);
console.log('UNIFIED_LEARNING_CORE_PASS '+L.KINDS.length+' operators / '+ids.length+
 ' guided graphs / '+C.lessons.length+' cross-platform courses / '+Core.labs.length+
 ' typed specialist labs; graph numeric vs static and live Prefill vote evidence separated');
