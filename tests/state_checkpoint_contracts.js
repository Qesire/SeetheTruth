'use strict';
const assert=require('node:assert/strict');
const Core=require('../learning_experience_core.js');
const Snap=require('../observation_snapshots.js');
const ops=[
 {op:'arrive',id:'A',tokenLen:8},
 {op:'allocate',id:'A'},
 {op:'arrive',id:'B',tokenLen:4},
 {op:'allocate',id:'B'},
 {op:'finish',id:'A'}
];
const kv=Core.executeLab('deepinfer_kv','replay',{config:{numBlocks:7,blockSize:4},operations:ops});
const base=Snap.create(Core,kv,'KV local replay');
const cursor=Snap.at(Core,base,'kv:3');
assert.equal(cursor.cursor.kind,'state_transition');
assert.equal(cursor.cursor.stepId,'kv:3');
assert.equal(cursor.cursor.index,3);
assert.equal(cursor.cursor.evidence,kv.stateTrace.evidenceTier);
const restored=Snap.restore(Core,cursor);
assert.equal(restored.checkpoint.replayVerified,true);
assert.equal(restored.checkpoint.index,3);
assert.equal(restored.checkpoint.stepId,'kv:3');
assert.deepEqual(restored.checkpoint.detail.after,kv.stateTrace.steps[3].after);
assert.deepEqual(restored.checkpoint.detail.dependencies,kv.stateTrace.steps[3].dependencies);
const continuation=Snap.resumeKV(Core,cursor);
assert.equal(continuation.finalReplayed,true);
assert.equal(continuation.index,3);
assert.equal(continuation.prefix.timeline.length,4);
assert.deepEqual(continuation.prefix.state,kv.value.timeline[3]);
assert.deepEqual(continuation.remainingOperations,ops.slice(3));
assert.deepEqual(Snap.restore(Core,base).checkpoint,null);
const first=Snap.at(Core,base,'kv:0');
assert.equal(Snap.resumeKV(Core,first).prefix.timeline.length,1);
const last=Snap.at(Core,base,'kv:5');
assert.equal(Snap.resumeKV(Core,last).remainingOperations.length,0);
const brokenIndex=structuredClone(cursor);
brokenIndex.cursor.index=4;
assert.throws(()=>Snap.restore(Core,brokenIndex),/state cursor no longer matches/);
const brokenDigest=structuredClone(cursor);
brokenDigest.cursor.stateDigest='fnv1a32:deadbeef:10';
assert.throws(()=>Snap.restore(Core,brokenDigest),/state cursor replay differs/);
const invented=structuredClone(cursor);
invented.cursor.stepId='kv:300';
assert.throws(()=>Snap.restore(Core,invented),/state cursor no longer matches/);
const invalid=structuredClone(cursor);
invalid.cursor.index=-1;
assert.throws(()=>Snap.validate(invalid),/invalid state checkpoint cursor/);
const extra=structuredClone(cursor);
extra.cursor.untrusted='device_memory';
assert.throws(()=>Snap.validate(extra),/invalid state checkpoint cursor/);
const unrelated=Core.executeLab('wan_ffn','trace',{options:{seed:17,tokens:3,dim:4,hidden:8}});
assert.throws(()=>Snap.at(Core,Snap.create(Core,unrelated),'up'),/state cursor requires/);
const kvWithoutCursor=Snap.create(Core,kv);
assert.throws(()=>Snap.resumeKV(Core,kvWithoutCursor),/requires a verified KV cursor/);
const worker=Core.executeLab('deepinfer_worker','simulate',{
 steps:[{mode:'prompt',sequenceIds:['A']},{mode:'decode',sequenceIds:['A']}],
 options:{zeroOverhead:true}});
const workerFrame=worker.stateTrace;
const workerCursor=Snap.at(Core,Snap.create(Core,worker),workerFrame.steps[1].id);
assert.equal(Snap.restore(Core,workerCursor).checkpoint.stepId,workerFrame.steps[1].id);
assert.throws(()=>Snap.resumeKV(Core,workerCursor),/requires a verified KV cursor/);
const prefill=Core.executeLab('deepinfer_prefill','inspect_pair',{
 voteInput:{decodeStep:4,allInterval:2,anyInterval:4,ranks:[
  {waitingSchedulable:true,canActuallySchedule:true,cacheUsage:.2},
  {waitingSchedulable:false,canActuallySchedule:false,cacheUsage:.2}]},
 chunkInput:{tokens:['TEXT','TEXT'],cached:0,maxBatchedTokens:2}});
const p=Snap.at(Core,Snap.create(Core,prefill),'prefill:chunk');
assert.equal(Snap.restore(Core,p).checkpoint.detail.after.decision.q,2);
assert.deepEqual(Snap.restore(Core,p).checkpoint.detail.dependencies,[]);
const training=Core.executeLab('training','scalar',{config:{theta:.4,optimizer:'adamw',steps:2}});
assert.equal(training.stateTrace.evidenceTier,'EXECUTABLE_SCALAR_ANALYTIC_TRAINING');
const trainingStep=training.stateTrace.steps.find(x=>x.changes.length>0);
assert.ok(trainingStep,'training source should contain a nontrivial state transition');
const trainingCursor=Snap.at(Core,Snap.create(Core,training),trainingStep.id);
const trainingRestored=Snap.restore(Core,trainingCursor);
assert.equal(trainingRestored.checkpoint.replayVerified,true);
assert.deepEqual(trainingRestored.checkpoint.detail.after,trainingStep.after);
assert.equal(trainingRestored.checkpoint.detail.changes.length,trainingStep.changes.length);
assert.throws(()=>Snap.resumeKV(Core,trainingCursor),/requires a verified KV cursor/);
console.log('STATE_CHECKPOINT_PASS: source replay, verified step view, KV prefix, worker and independent Prefill, tampering');
