'use strict';
/* Unified underlying engine owns every state, KV revision and Worker event.
 * G0-G4 are replay-verified readonly projections, never execution replicas. */
const assert=require('node:assert/strict');
const Core=require('../learning_experience_core.js');
const Source=require('../deepinfer_scheduler_workflow.js');
const State=require('../state_microscope.js');
const clone=x=>JSON.parse(JSON.stringify(x));
const schema='seethetruth-observation-v1';
const cases=[['pressure',Source.EXAMPLES.pressure],['structured',Source.EXAMPLES.structured],['ample',Source.EXAMPLES.ample]];
for(const [name,spec] of cases){
 const accepted=Core.executeLab('deepinfer_scheduler','simulate',{spec});
 const native=Source.simulate(spec);
 assert.deepEqual(accepted.value,native,name+' must use the native engine');
 const unmodified=JSON.stringify(accepted);
 const observed=Core.observeExecution(accepted);
 assert.equal(observed.sources[0].kind,'state_transition');
 assert.equal(observed.sources[0].steps.length,native.decisions.length+1);
 assert.equal(Core.replayLab(accepted).reproduced,true);
 const g0=Core.projectExecutionGrain(accepted,{grain:'G0'});
 assert.equal(g0.schema,schema);
 assert.equal(g0.grain,'G0');
 assert.deepEqual(g0.value,native.summary);
 assert.deepEqual(g0.nativeStepIds,accepted.stateTrace.steps.map(e=>e.id));
 assert.equal(g0.parentStepId,null);
 assert.equal(g0.replayVerified,true);
 assert.equal(g0.numericalReinterpretation,false);
 const g1=Core.projectExecutionGrain(accepted,{grain:'G1'});
 assert.deepEqual(g1.value.map(c=>c.path),['value.decisions','value.snapshots','value.worker.events']);
 assert.deepEqual(g1.value.map(c=>c.count),[
  native.decisions.length,native.snapshots.length,native.worker.events.length]);
 assert.equal(g1.equivalence,'NOT_CLAIMED');
 for(let cycle=0;cycle<native.decisions.length;cycle++){
  const decision=native.decisions[cycle],proof=native.executionProof[cycle],
   rawStep=accepted.stateTrace.steps[cycle+1],snapshot=native.snapshots[cycle+1];
  const g2=Core.projectExecutionGrain(accepted,{grain:'G2',cycle});
  const raw=State.schedulerGrain(accepted.value,accepted.stateTrace,{grain:'G2',cycle});
  assert.deepEqual(g2.value,raw.value);
  assert.equal(g2.sourcePath,raw.sourcePath);
  assert.equal(g2.parentStepId,raw.parentStepId);
  assert.deepEqual(State.schedulerGrain(accepted.value,accepted.stateTrace,{grain:'G3',cycle}).value,decision.actions);

  assert.deepEqual(g2.value,rawStep);
  assert.equal(g2.parentStepId,rawStep.id);
  assert.equal(g2.kvRevision,snapshot.revision);
  assert.equal(g2.sourceSnapshotPath,'value.snapshots['+(cycle+1)+']');
  assert.equal(g2.forwardEventId,proof.forwardEvent);
  const microscope=Core.inspectObservationStep(accepted,{kind:'state_transition',stepId:g2.parentStepId});
  assert.deepEqual(microscope.detail.after,g2.value.after);
  assert.equal(microscope.evidence,g2.stateEvidence);
  const g3=Core.projectExecutionGrain(accepted,{grain:'G3',cycle});
  assert.deepEqual(g3.value,decision.actions);
  assert.equal(g3.sourcePath,'value.decisions['+cycle+'].actions');
  assert.equal(g3.forwardEventId,proof.forwardEvent);
  assert.equal(g3.kvRevision,snapshot.revision);
  const g4=Core.projectExecutionGrain(accepted,{grain:'G4',cycle});
  const worker=native.worker.events.find(x=>x.id===proof.forwardEvent);
  assert.deepEqual(g4.value,worker);
  assert.equal(g4.nativeEventId,proof.forwardEvent);
  assert.equal(g4.value.step,cycle);
  assert.match(g4.sourcePath,/^value\.worker\.events\[\d+\]$/);
  for(const e of native.worker.events.filter(x=>x.step===cycle)){
    assert.deepEqual(Core.projectExecutionGrain(accepted,{grain:'G4',cycle,eventId:e.id}).value,e);
  }
  for(let i=0;i<decision.actions.length;i++){
   const action=Core.projectExecutionGrain(accepted,{grain:'G4',cycle,actionIndex:i});
   assert.deepEqual(action.value,decision.actions[i]);
   assert.equal(action.kvRevision,decision.actions[i].revision);
   assert.equal(action.originalRequestId,decision.actions[i].id);
   assert.equal(action.scope,'native_kv_action');
   assert.equal(action.sourcePath,'value.decisions['+cycle+'].actions['+i+']');
  }
  assert.throws(()=>Core.projectExecutionGrain(accepted,{grain:'G4',cycle,actionIndex:decision.actions.length}),/outside accepted decision/);
 }
 assert.equal(JSON.stringify(accepted),unmodified,name+' projections may not mutate accepted execution');
 assert.throws(()=>Core.projectExecutionGrain(accepted,{grain:'G2',cycle:-1}),/outside accepted execution/);
 assert.throws(()=>Core.projectExecutionGrain(accepted,{grain:'G2',cycle:native.decisions.length}),/outside accepted execution/);
 assert.throws(()=>Core.projectExecutionGrain(accepted,{grain:'G4',cycle:0,eventId:'not-a-worker-event'}),/not in the selected native cycle/);
 const another=native.worker.events.find(e=>e.step>0);
 if(another)assert.throws(()=>Core.projectExecutionGrain(accepted,{grain:'G4',cycle:0,eventId:another.id}),/not in the selected native cycle/);
 for(const invalid of [{grain:'G9'},{grain:'G4',actionIndex:-1},{grain:'G4',actionIndex:0,eventId:'x'},{grain:'G2',unknown:1}]){
  assert.throws(()=>Core.projectExecutionGrain(accepted,invalid));
 }
 const malicious=clone(accepted);
 malicious.value.decisions[0].kvRevision=-1;
 assert.throws(()=>Core.projectExecutionGrain(malicious,{grain:'G2'}),/replay mismatch/);
 const maliciousEvent=clone(accepted);
 maliciousEvent.value.worker.events[0].stage='faked';
 assert.throws(()=>Core.projectExecutionGrain(maliciousEvent,{grain:'G4'}),/replay mismatch/);
 const maliciousTrace=clone(accepted);
 maliciousTrace.stateTrace.steps[1].after.forwardEvent='faked';
 assert.throws(()=>Core.projectExecutionGrain(maliciousTrace,{grain:'G2'}),/replay mismatch/);
}
assert.throws(()=>Core.projectExecutionGrain(Core.executeLab('deepinfer_prefill','vote',{
 decodeStep:2,allInterval:2,anyInterval:4,ranks:[{waitingSchedulable:true,canActuallySchedule:true,cacheUsage:0}]
}),{grain:'G0'}),/adapter unavailable/);
const fs=require('node:fs'),path=require('node:path');
const ui=fs.readFileSync(path.join(__dirname,'../unified_observation_ui.js'),'utf8');
assert.ok(ui.includes("core.projectExecutionGrain(current,opts)"));
assert.ok(ui.includes("id='nativeGranularityView'"));
assert.ok(ui.includes("grainResult.dataset.sourcePath=result.sourcePath"));
assert.ok(ui.includes("current.value.executionProof[cycle].forwardEvent"));
console.log('NATIVE_GRANULARITY_SPINE_PASS source parity × 3, G0-G4 real IDs/revisions, replay, hostile mutation rejection');
