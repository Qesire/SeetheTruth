'use strict';
const assert=require('node:assert/strict');
const Core=require('../learning_experience_core.js');
const Worker=require('../deepinfer_worker_events.js');
const Scheduler=require('../deepinfer_scheduler_workflow.js');

function nativePair(domain,operation,inputA,inputB){
 const A=Core.executeLab(domain,operation,inputA);
 const B=Core.executeLab(domain,operation,inputB);
 assert.ok(A.stateTrace&&B.stateTrace,domain+' must provide native state lineage');
 assert.equal(Core.replayLab(A).reproduced,true);
 assert.equal(Core.replayLab(B).reproduced,true);
 const diff=Core.compareRecords(A,B);
 assert.equal(diff.replayVerified,true);
 assert.equal(diff.inputEqual,false);
 return {A,B,diff};
}
const operations=[{op:'arrive',id:'A',tokenLen:5},{op:'allocate',id:'A'}];
const kv=nativePair('deepinfer_kv','replay',
 {config:{numBlocks:6,blockSize:4},operations},
 {config:{numBlocks:7,blockSize:4},operations});
assert.ok(kv.diff.changedSteps.some(x=>x.kind==='state_transition'));
assert.equal(kv.A.value.timeline.at(-1).numBlocks,6);
assert.equal(kv.B.value.timeline.at(-1).numBlocks,7);
assert.deepEqual(kv.A.inputSnapshot.operations,kv.B.inputSnapshot.operations);

const scheduler=nativePair('deepinfer_scheduler','simulate',
 {spec:Scheduler.EXAMPLES.ample,options:{stopPreempted:false}},
 {spec:Scheduler.EXAMPLES.pressure,options:{stopPreempted:false}});
assert.ok(scheduler.diff.changedSteps.some(x=>x.kind==='state_transition'));
assert.ok(scheduler.B.value.executionProof.every(x=>typeof x.forwardEvent==='string'));
const inspect=Core.inspectObservationStep(scheduler.B,{
 kind:'state_transition',stepId:scheduler.B.stateTrace.steps[1].id});
assert.equal(inspect.replayVerified,true);

const steps=Worker.DEFAULT_STEPS.map(x=>({mode:x.mode,sequenceIds:[...x.sequenceIds]}));
const worker=nativePair('deepinfer_worker','simulate',
 {steps,options:{zeroOverhead:true}},
 {steps,options:{zeroOverhead:true,durations:{forward:Worker.DEFAULT_DURATIONS.forward+2}}});
assert.ok(worker.diff.changedSteps.some(x=>x.kind==='state_transition'));
assert.ok(worker.B.value.makespan>=worker.A.value.makespan);
assert.equal(worker.A.evidence,'ABSTRACT_TIMELINE');

// Explicitly setting the documented default is a legitimate distinct input
// with an unchanged output. Such negative findings must remain representable.
const control=nativePair('deepinfer_worker','simulate',
 {steps,options:{zeroOverhead:true}},
 {steps,options:{zeroOverhead:true,durations:{forward:Worker.DEFAULT_DURATIONS.forward}}});
assert.equal(control.diff.inputEqual,false);
assert.equal(control.diff.valueEqual,true);
assert.equal(control.diff.stateTraceEqual,true);
assert.deepEqual(control.diff.changedSteps,[]);
console.log('NATIVE_AB_GUIDE_PASS 3 engines and one replay-verified no-effect comparison');
