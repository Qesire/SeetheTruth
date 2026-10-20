'use strict';
const assert=require('node:assert/strict');
const W=require('../deepinfer_scheduler_workflow.js'),K=require('../deepinfer_kv_lifecycle.js');
const pressure=W.simulate(W.EXAMPLES.pressure);
assert.equal(pressure.decisions.length,4);
assert.deepEqual(pressure.decisions[0].scheduled,['B','C'].sort((a,b)=>pressure.snapshots[1].requests[b].tokenLen-pressure.snapshots[1].requests[a].tokenLen||a.localeCompare(b)));
assert.deepEqual(pressure.decisions[0].preempted.map(x=>[x.id,x.outcome]),[['A','recompute']]);
assert.equal(pressure.snapshots[1].requests.A.status,'WAITING');
assert.equal(pressure.snapshots[1].requests.A.cachedTokens,0);
assert.equal(pressure.snapshots[1].requests.A.recomputes,1);
assert.equal(pressure.decisions[0].freeBeforeEnd.length,0,'all remaining KV blocks consumed by surviving decode batch');
assert.deepEqual(pressure.decisions[1].preempted.map(x=>[x.id,x.outcome]),[['B','recompute']]);
assert.equal(pressure.decisions[2].actions.filter(x=>x.op==='allocate'&&x.id==='A').length,1,
 'WAITING recompute request admitted only through explicit prefill action');
assert.deepEqual(pressure.decisions[2].scheduled,['A','C'].sort((a,b)=>
 pressure.snapshots[3].requests[b].tokenLen-pressure.snapshots[3].requests[a].tokenLen||a.localeCompare(b)));
assert.equal(pressure.summary.preemptions,2);
assert.equal(pressure.summary.recomputes,2);
assert.equal(pressure.summary.finalFreeBlocks,6);
assert.equal(pressure.worker.steps.length,4);
assert.equal(pressure.worker.released.length,4);
pressure.decisions.forEach((d,i)=>{
 assert.deepEqual(d.scheduled,pressure.worker.steps[i].sequenceIds,'Worker batch must equal scheduled KV ownership batch');
 assert.ok(!d.scheduled.some(id=>d.preempted.some(p=>p.id===id)), 'preempted cannot forward in current step');
});
assert.equal(pressure.worker.pendingStep,null,'final worker state flushed');
const structured=W.simulate(W.EXAMPLES.structured);
assert.deepEqual(structured.decisions[0].preempted.map(x=>[x.id,x.outcome]),[['S','stop']],
 'source structured-output preemption terminates, does not recompute');
assert.equal(structured.snapshots[1].requests.S.status,'FINISHED_PREEMPTED');
assert.equal(structured.summary.terminalPreemptions,1);
assert.equal(structured.summary.recomputes,0);
assert.equal(structured.snapshots.at(-1).free.length,4);
const ample=W.simulate(W.EXAMPLES.ample);
assert.equal(ample.summary.preemptions,0);
assert.equal(ample.summary.finalFreeBlocks,9);
assert.deepEqual(ample.decisions.map(d=>d.scheduled),[['A','B'],['A','B'],['B']],
 'free capacity should never evict a healthy decode batch');
const stopping=W.simulate(W.EXAMPLES.pressure,{stopPreempted:true,flush:false});
assert.equal(stopping.decisions[0].preempted[0].outcome,'stop');
assert.equal(stopping.snapshots[1].requests.A.status,'FINISHED_PREEMPTED');
assert.equal(stopping.worker.pendingStep,3);
for(const spec of [W.EXAMPLES.pressure,W.EXAMPLES.structured,W.EXAMPLES.ample]){
 const out=W.simulate(spec);
 out.snapshots.forEach(K.verify);
 for(const step of out.worker.steps)assert.ok(step.sequenceIds.every(id=>typeof id==='string'));
 for(const event of out.worker.events)for(const dep of event.dependsOn)
  assert.ok(out.worker.events.some(e=>e.id===dep&&e.end<=event.start),'missing Worker prerequisite');
}
assert.throws(()=>W.simulate({...W.EXAMPLES.ample,cycles:[{outputLens:{A:1}}]}),/missing outputLen/);
assert.throws(()=>W.simulate({...W.EXAMPLES.ample,cycles:[{
 outputLens:{A:1,B:2},workloads:{A:{tokenLen:19}}}]}),/only appends one/);
assert.throws(()=>W.simulate({...W.EXAMPLES.ample,cycles:[{outputLens:{A:1,B:2},admit:[{id:'B'}]}]}),/WAITING/);
assert.throws(()=>W.simulate({...W.EXAMPLES.ample,cycles:[]}),/cycles/);
assert.throws(()=>W.simulate(W.EXAMPLES.ample,{stopPreempted:'yes'}),/boolean/);
const noChange=K.replay(W.EXAMPLES.ample.config,W.EXAMPLES.ample.setup).state;
assert.equal(noChange.free.length,7,'no caller state mutated by failed composite run');
console.log('DEEPINFER_SCHEDULER_WORKER_KV_PASS source LCFS eviction, free-block guard, recompute/admission, SO stop, causal Worker batch');
