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
console.log('UNIFIED_OBSERVATION_PASS: typed state/computation refs, exact input replay, tamper detection, no invented trace');
