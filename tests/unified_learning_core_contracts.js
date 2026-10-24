'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const Core=require('../learning_experience_core.js');
const L=require('../learning_lab.js'),C=require('../teaching_curriculum.js');
const X=require('../block_extension_registry.js'),P=require('../deepinfer_prefill_policy.js');
const S=require('../graph_session.js'),B=require('../branch_execution.js');
const studio=read('learn.html'),ids=[
 ...[...studio.matchAll(/data-preset="([^"]+)"/g)].map(m=>m[1]),
 ...X.recipes.map(x=>x.id)
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
assert.equal(all.counts.graphLessons,60+X.recipes.length); // 59 guided + blank canvas + extension recipes
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
const labIds=new Set(Core.labs.map(l=>l.id)),labHrefs=new Set(Core.labs.map(l=>l.href));
assert.equal(labIds.size,Core.labs.length);
assert.equal(labHrefs.size,Core.labs.length);
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
const prefillChunk={tokens:['TEXT','TEXT','TEXT'],cached:0,maxBatchedTokens:2};
assert.deepEqual(Core.executeLab('deepinfer_prefill','chunk',prefillChunk).value,P.chunk(prefillChunk));
const Worker=require('../deepinfer_worker_events.js');
const steps=Worker.DEFAULT_STEPS.slice(0,2);
const workerArgs={steps,options:{zeroOverhead:true}};
assert.deepEqual(Core.executeLab('deepinfer_worker','simulate',workerArgs).value,
 Worker.simulate(steps,{zeroOverhead:true}));
assert.equal(Core.executeLab('deepinfer_worker','simulate',workerArgs).value.note.includes('not measured'),true);
const KV=require('../deepinfer_kv_lifecycle.js');
const ops=[{op:'arrive',id:'A',tokenLen:5},{op:'allocate',id:'A'}];
const kvArgs={config:{numBlocks:6,blockSize:4},operations:ops};
assert.deepEqual(Core.executeLab('deepinfer_kv','replay',kvArgs).value,
 KV.replay(kvArgs.config,kvArgs.operations));
const SW=require('../deepinfer_scheduler_workflow.js');
const sched={spec:SW.EXAMPLES.ample,options:{}};
assert.deepEqual(Core.executeLab('deepinfer_scheduler','simulate',sched).value,
 SW.simulate(sched.spec,sched.options));
assert.throws(()=>Core.executeLab('deepinfer_worker','vote',args),/no registered executable adapter/);
assert.throws(()=>Core.executeLab('deepinfer_prefill','unknown',args),/no registered executable adapter/);
assert.throws(()=>Core.inspectLab('does_not_exist'),/unknown specialized adapter/);
assert.throws(()=>Core.inspectCourse('does_not_exist'),/unknown curriculum lesson/);
console.log('UNIFIED_LEARNING_CORE_PASS '+L.KINDS.length+' operators / '+ids.length+
 ' guided graphs / '+C.lessons.length+' cross-platform courses / '+Core.labs.length+
 ' typed specialist labs; graph numeric vs static and live Prefill vote evidence separated');
