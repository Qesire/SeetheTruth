'use strict';
const assert=require('node:assert/strict'),M=require('../deepinfer_worker_events.js');
const steps=[
 {mode:"prompt",sequenceIds:["A","B"]},
 {mode:"decode",sequenceIds:["A","B"]},
 {mode:"decode",sequenceIds:["B","C"]},
 {mode:"decode",sequenceIds:["B","C"]}
];
const make=(zeroOverhead,flush=true,durations={})=>M.simulate(steps,{zeroOverhead,flush,durations});
const zos=make(true),serial=make(false);
assert.equal(zos.released.length,4,'all results must be released after flush');
assert.equal(serial.released.length,4,'non-ZOS always releases current step');
assert.deepEqual(zos.released.map(x=>x.via),['immediate','deferred','deferred','flush']);
assert.deepEqual(serial.released.map(x=>x.via),['immediate','immediate','immediate','immediate']);
assert.equal(zos.pendingStep,null);
assert.ok(zos.events.some(e=>e.id==='s2-post_prev'));
assert.ok(zos.events.some(e=>e.id==='s2-correct_zos_metadata'));
assert.ok(!serial.events.some(e=>e.stage==='post_prev'));
assert.ok(!serial.events.some(e=>e.stage==='correct_zos_metadata'));
assert.deepEqual(zos.events.find(e=>e.id==='s2-correct_zos_metadata').meta.remap,[1,-1]);
const pending=make(true,false);
assert.equal(pending.released.length,3);
assert.equal(pending.pendingStep,3);
assert.ok(!pending.events.some(e=>e.stage==='flush_pending'));
const pendingSample=pending.events.find(e=>e.id==='s3-sample');
assert.ok(pendingSample&&pendingSample.end>=0);
const idmap=new Map(zos.events.map(e=>[e.id,e]));
for(const run of [zos,serial,pending]){
 const deps=new Map(run.events.map(e=>[e.id,e]));
 for(const event of run.events){
  assert.ok(event.start>=0&&event.end>=event.start);
  for(const dep of event.dependsOn)assert.ok(deps.get(dep)?.end<=event.start,event.id+' must wait '+dep);
 }
 for(const resource of ['CPU','GPU']){
  const events=run.events.filter(e=>e.resource===resource);
  for(let i=1;i<events.length;i++)
   assert.ok(events[i-1].end<=events[i].start,resource+' double-booked');
 }
 const ids=run.released.map(x=>x.step);
 assert.equal(ids.length,new Set(ids).size,'requests may be delivered only once');
 for(const release of run.released)assert.equal(deps.get(release.event).end,release.at);
}
const launched=idmap.get('s2-launch_forward'),forward=idmap.get('s2-forward'),post=idmap.get('s2-post_prev');
assert.ok(launched.end<=post.start,'previous postprocessing follows current forward launch');
assert.ok(post.start<forward.end,'postprocessing overlaps current GPU forward in illustrative scenario');
assert.equal(post.meta.stateFromStep,1);
assert.ok(zos.makespan<=serial.makespan,'chosen hypothetical durations should show possible overlap');
assert.equal(M.compare(steps,M.DEFAULT_DURATIONS).delta,serial.makespan-zos.makespan);
const noOverlap=M.simulate([
 {mode:'decode',sequenceIds:['A']},{mode:'decode',sequenceIds:['B']}
],{zeroOverhead:true});
assert.ok(!noOverlap.events.some(e=>e.stage==='correct_zos_metadata'));
const placeholder=M.simulate([
 {mode:'decode',sequenceIds:['A']},{mode:'placeholder',sequenceIds:[]},{mode:'decode',sequenceIds:['A']}
],{zeroOverhead:true});
assert.equal(placeholder.released.length,2,'placeholder has no user-facing output');
assert.equal(placeholder.events.filter(e=>e.stage==='forward').length,3,'upstream invokes placeholder forward');
const promptAfterPending=M.simulate([
 {mode:'decode',sequenceIds:['A']},{mode:'prompt',sequenceIds:['A']}
],{zeroOverhead:true});
assert.equal(promptAfterPending.released.length,2);
assert.equal(promptAfterPending.released[0].via,'deferred');
assert.equal(promptAfterPending.released[1].via,'immediate');
assert.throws(()=>M.simulate([{mode:'decode',sequenceIds:['A','A']}],{zeroOverhead:true}),/repeated/);
assert.throws(()=>M.simulate([{mode:'placeholder',sequenceIds:['A']}],{zeroOverhead:true}),/placeholder/);
assert.throws(()=>M.simulate([{mode:'decode',sequenceIds:[]}],{zeroOverhead:true}),/sequenceIds/);
assert.throws(()=>M.simulate(steps,{zeroOverhead:true,durations:{forward:-1}}),/duration/);
assert.throws(()=>M.simulate(steps,{zeroOverhead:true,durations:{extra:2}}),/duration/);
assert.equal(zos.note.includes('not measured'),true);
console.log('DEEPINFER_WORKER_EVENTS_PASS causal DAG, cross-step correction, deferred/flush output, event resources, no invented timing');
