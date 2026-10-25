'use strict';
const assert=require('node:assert/strict');
const Core=require('../learning_experience_core.js');
const input={voteInput:{decodeStep:4,allInterval:2,anyInterval:4,ranks:[
 {waitingSchedulable:true,canActuallySchedule:true,cacheUsage:.2},
 {waitingSchedulable:false,canActuallySchedule:false,cacheUsage:.2}]},
 chunkInput:{tokens:['TEXT','TEXT','TEXT'],cached:0,maxBatchedTokens:2}};
const a=Core.executeLab('deepinfer_prefill','inspect_pair',input);
const frame=Core.observeExecution(a);
assert.equal(frame.schema,'seethetruth-observation-v1');
assert.equal(frame.traceAvailable,true);
assert.equal(frame.sources[0].kind,'state_transition');
assert.equal(frame.sources[0].steps.length,a.stateTrace.steps.length);
assert.deepEqual(frame.sources[0].steps[1].dependencies,[]);
assert.equal(Core.replayLab(a).reproduced,true);
const mutated=structuredClone(a);mutated.value.vote.perRank[0]='tampered';
assert.throws(()=>Core.replayLab(mutated),/replay mismatch: value/);
const invalid=structuredClone(a);delete invalid.inputSnapshot;
assert.throws(()=>Core.replayLab(invalid),/original input/);
const w=Core.executeLab('wan_ffn','trace',{options:{seed:17,tokens:3,dim:4,hidden:8}});
const wf=Core.observeExecution(w);
assert.equal(wf.sources[0].kind,'tensor_computation');
assert.equal(wf.sources[0].steps.length,w.computationTrace.steps.length);
assert.equal(Core.replayLab(w).reproduced,true);
const v=Core.executeLab('deepinfer_prefill','vote',input.voteInput);
assert.equal(Core.observeExecution(v).traceAvailable,false);
assert.deepEqual(Core.observeExecution(v).sources,[]);
assert.equal(Core.replayLab(v).reproduced,true);
assert.throws(()=>Core.observeExecution({}),/accepted specialist execution/);

const stateStep=Core.inspectObservationStep(a,{kind:'state_transition',stepId:a.stateTrace.steps[0].id});
assert.equal(stateStep.replayVerified,true);
assert.deepEqual(stateStep.detail.after,a.stateTrace.steps[0].after);
assert.throws(()=>Core.inspectObservationStep(a,{kind:'state_transition',stepId:a.stateTrace.steps[0].id,coordinate:[0]}),/do not have tensor coordinates/);
const computed=Core.inspectObservationStep(w,{kind:'tensor_computation',stepId:'up',coordinate:[1,2]});
assert.equal(computed.replayVerified,true);
assert.equal(computed.coordinate.join(','),'1,2');
assert.equal(computed.detail.reconstructable,true);
assert.ok(computed.detail.error<1e-9);
assert.throws(()=>Core.inspectObservationStep(w,{kind:'tensor_computation',stepId:'up',coordinate:[1]}),/coordinate must exactly match/);
assert.throws(()=>Core.inspectObservationStep(w,{kind:'tensor_computation',stepId:'unknown',coordinate:[0,0]}),/unknown observation step/);
const ws=Core.observeExecution(w).sources.find(s=>s.kind==='tensor_computation');
assert.deepEqual(ws.steps[1].dependencies,[{from:'condition',port:'previous'}]);
const w2=Core.executeLab('wan_ffn','trace',{options:{seed:18,tokens:3,dim:4,hidden:8}});
const diff=Core.compareExecutions(w,w2);
assert.equal(diff.replayVerified,true);
assert.equal(diff.inputEqual,false);
assert.equal(diff.valueEqual,false);
assert.ok(diff.changedSteps.some(c=>c.kind==='tensor_computation'&&c.stepId==='up'));
const same=Core.compareExecutions(w,w);
assert.equal(same.valueEqual,true);
assert.deepEqual(same.changedSteps,[]);
assert.throws(()=>Core.compareExecutions(w,a),/same engine and operation/);
const badEvidence=structuredClone(w);badEvidence.evidence='ACTUAL_GPU_PROFILER';
assert.throws(()=>Core.replayLab(badEvidence),/replay mismatch: evidence/);
const badTrace=structuredClone(w);badTrace.computationTrace.steps[1].dependencies=[{from:'not_real',port:'previous'}];
assert.throws(()=>Core.observeExecution(badTrace),/dependencies must point backward/);

const L=require('../learning_lab.js');
const graph=L.preset('residual');
const gi=Core.inspectGraph(graph,{numeric:true,id:'free_residual'});
const go=Core.observe(gi);
assert.equal(go.traceAvailable,true);
assert.equal(go.sourceEngine,'learning_lab.js');
assert.equal(go.sources[0].kind,'tensor_computation');
assert.equal(Core.replay(gi).reproduced,true);
const first=gi.session.trace.steps[0],shape=first.shape;
const index=shape.map(()=>0);
const firstCoord=Core.inspectObservationStep(gi,{kind:'tensor_computation',stepId:first.id,coordinate:index});
assert.equal(firstCoord.replayVerified,true);
assert.equal(firstCoord.detail.value,first.values.flat(Infinity)[0]);
const gs=Core.inspectGraph(graph,{numeric:false,id:'free_residual'});
assert.equal(Core.observe(gs).traceAvailable,false);
assert.equal(Core.replay(gs).numericExecuted,false);
assert.throws(()=>Core.inspectObservationStep(gs,{kind:'tensor_computation',stepId:first.id,coordinate:index}),/requested observation trace unavailable/);
const tamperedGraph=structuredClone(gi);tamperedGraph.session.trace.steps[0].values[0][0]+=.25;
assert.throws(()=>Core.replayGraph(tamperedGraph),/graph replay mismatch: trace/);

const edited=structuredClone(graph);edited.nodes.find(n=>n.id==='x').params.seed=4;
const changedGraph=Core.inspectGraph(edited,{numeric:true,id:'free_residual'});
const graphDiff=Core.compareRecords(gi,changedGraph);
assert.equal(graphDiff.replayVerified,true);
assert.equal(graphDiff.numericCompared,true);
assert.equal(graphDiff.inputEqual,false);
assert.equal(graphDiff.valueEqual,false);
assert.ok(graphDiff.changedSteps.some(s=>s.stepId==='sum'));
assert.deepEqual(Core.compareRecords(gi,gi).changedSteps,[]);
const staticDiff=Core.compareRecords(gs,gs);
assert.equal(staticDiff.numericCompared,false);
assert.equal(staticDiff.valueEqual,null);
assert.throws(()=>Core.compareRecords(gi,w),/matching observational domains/);
assert.deepEqual(Core.compareRecords(w,w2).changedSteps,diff.changedSteps);
assert.deepEqual(go.sources[0].steps.find(s=>s.id==='sum').dependencies,
 [{from:'x',port:'a'},{from:'act',port:'b'}]);

// Browser-level dependency order: core captures the real computation microscope.
const html=require('node:fs').readFileSync(require('node:path').join(__dirname,'../learn.html'),'utf8');
assert.ok(html.indexOf('src="compute_microscope.js"')>=0 &&
 html.indexOf('src="compute_microscope.js"')<html.indexOf('src="learning_experience_core.js"'),
 'Learning Studio must load computation microscope before unified core');
console.log('UNIFIED_OBSERVATION_PASS: typed state/computation refs, exact input replay, tamper detection, no invented trace');
