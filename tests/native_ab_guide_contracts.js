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

const Handoff=require('../learning_handoff.js');
for(const [name,pair,target] of [
 ['deepinfer_kv',kv,'decode'],['deepinfer_scheduler',scheduler,'decode'],
 ['deepinfer_worker',worker,'attention'],['deepinfer_worker',control,'attention']
]){
 const selected=pair.B.stateTrace.steps[1].id;
 const handoff=Handoff.create(Core,pair.B,pair.A,selected,
  '哪些真实源事件由一个原生输入变化引起？');
 const storage=new Map();
 Handoff.store({setItem:(key,val)=>storage.set(key,val)},handoff);
 const imported=Handoff.load({getItem:key=>storage.get(key)});
 const restored=Handoff.restore(Core,imported);
 assert.equal(restored.domain,name);
 assert.equal(restored.course,target);
 assert.equal(restored.selectedStepId,selected);
 assert.equal(restored.current.checkpoint.stepId,selected);
 assert.equal(restored.comparison.replayVerified,true);
 assert.equal(restored.comparison.inputEqual,false);
 assert.equal(restored.baseline.record.evidence,pair.A.evidence);
 const changed=JSON.parse(JSON.stringify(imported));
 changed.active.cursor.stateDigest='fnv1a32:00000000:1';
 assert.throws(()=>Handoff.restore(Core,changed),/replay differs|cursor/);
 const mismatched=JSON.parse(JSON.stringify(imported));
 mismatched.domain=name==='deepinfer_kv'?'deepinfer_worker':'deepinfer_kv';
 assert.throws(()=>Handoff.restore(Core,mismatched),/mismatch/);
 assert.throws(()=>Handoff.load({getItem:()=>'{invalid'}),/invalid cross-page JSON/);
}
const single=Handoff.create(Core,worker.B,null,null,'单次实验进入计算图');
assert.equal(Handoff.restore(Core,single).baseline,null);
assert.equal(Handoff.restore(Core,single).current.record.id,'deepinfer_worker');
console.log('NATIVE_GRAPH_HANDOFF_PASS replayed A/B, negative/no-effect, cursor tamper, source mismatch');
