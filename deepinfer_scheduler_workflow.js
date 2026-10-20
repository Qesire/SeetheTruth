/* DeepInfer source-grounded bounded *decode-only* scheduler + KV/Worker integration.
 * C++ source: scheduler.hpp::_schedule_decode, _preempt and block_manager.hpp.
 * Explicitly excludes _schedule_prefill, DP votes, MTP slot pool, device kernels.
 */
(function(root,factory){"use strict";const api=factory(
 typeof module==="object"&&module.exports?require("./deepinfer_kv_lifecycle.js"):root.SeetheTruthKVLifecycle,
 typeof module==="object"&&module.exports?require("./deepinfer_worker_events.js"):root.SeetheTruthDeepInferEvents
);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthSchedulerWorkflow=api;})(typeof window!=="undefined"?window:{},function(K,Worker){"use strict";
 const fail=s=>{throw Error(s)},idOk=id=>typeof id==="string"&&/^[A-Za-z0-9_-]{1,24}$/.test(id);
 const EXAMPLES=Object.freeze({
  pressure:{config:{numBlocks:7,blockSize:4},setup:[
   {op:"arrive",id:"A",tokenLen:8},{op:"allocate",id:"A"},
   {op:"arrive",id:"B",tokenLen:8},{op:"allocate",id:"B"},
   {op:"arrive",id:"C",tokenLen:8},{op:"allocate",id:"C"}],
   cycles:[{outputLens:{A:8,B:4,C:1},workloads:{B:{tokenLen:9},C:{tokenLen:9}}},
    {outputLens:{B:5,C:2},after:[{op:"finish",id:"B"}]},
    {admit:[{id:"A"}],outputLens:{A:9,C:3},workloads:{A:{tokenLen:9}},
     after:[{op:"finish",id:"A"}]},
    {outputLens:{C:4},after:[{op:"finish",id:"C"}]}]},
  structured:{config:{numBlocks:5,blockSize:4},setup:[
   {op:"arrive",id:"S",tokenLen:8},{op:"allocate",id:"S"},
   {op:"arrive",id:"B",tokenLen:8},{op:"allocate",id:"B"},
   {op:"set_structured",id:"S",status:1}],
   cycles:[{outputLens:{S:8,B:1},workloads:{B:{tokenLen:9}}},
    {outputLens:{B:2},after:[{op:"finish",id:"B"}]}]},
  ample:{config:{numBlocks:10,blockSize:4},setup:[
   {op:"arrive",id:"A",tokenLen:4},{op:"allocate",id:"A"},
   {op:"arrive",id:"B",tokenLen:4},{op:"allocate",id:"B"}],
   cycles:[{outputLens:{A:1,B:2},workloads:{A:{tokenLen:5},B:{tokenLen:5}}},
    {outputLens:{A:2,B:3},after:[{op:"finish",id:"A"}]},
    {outputLens:{B:4},after:[{op:"finish",id:"B"}]}]}
 });
 function validate(spec){
  if(!spec||typeof spec!=="object"||!spec.config||!Array.isArray(spec.setup)||
    spec.setup.length>50||!Array.isArray(spec.cycles)||!spec.cycles.length||spec.cycles.length>12)
   fail("workflow requires config, setup<=50 and cycles 1..12");
  let initial=K.replay(spec.config,spec.setup).state;
  if(Object.values(initial.requests).some(x=>x.branch))fail("forked waiting branch requires upstream prefill; unsupported by decode-only harness");
  const cycles=spec.cycles.map((c,i)=>{
   if(!c||typeof c!=="object"||!c.outputLens||typeof c.outputLens!=="object"||
      Array.isArray(c.outputLens)||!Object.keys(c.outputLens).length)fail("cycle "+i+" outputLens required");
   for(const [id,len] of Object.entries(c.outputLens))
    if(!idOk(id)||!Number.isSafeInteger(len)||len<0||len>10000)fail("invalid outputLen");
   for(const key of ["admit","before","after"])if(c[key]!==undefined&&!Array.isArray(c[key]))fail(key+" must be a list");
   if(c.admit?.length>16||c.before?.length>16||c.after?.length>16)fail("cycle has too many operations");
   if(c.workloads!==undefined&&(!c.workloads||typeof c.workloads!=="object"||Array.isArray(c.workloads)))fail("invalid workloads");
   return c;
  });
  return {initial,cycles};
 }
 function simulate(spec,{stopPreempted=false,flush=true,durations}={}){
  if(typeof stopPreempted!=="boolean"||typeof flush!=="boolean")fail("boolean controls required");
  const {initial,cycles}=validate(spec);
  let s=initial,steps=[],snapshots=[initial],decisions=[];
  const run=(op,actions)=>{s=K.advance(s,op);actions.push({op:op.op,id:op.id??null,operation:op,
    free:s.free.slice(),revision:s.revision});};
  cycles.forEach((cyc,idx)=>{
   let actions=[],reason=[],preempted=[],scheduled=[],waiting=[],blocked=[];
   for(const op of cyc.before||[])run(op,actions);
   for(const entry of cyc.admit||[]){
    if(!entry||!idOk(entry.id))fail("invalid admission");
    const r=s.requests[entry.id];if(!r||r.status!=="WAITING")fail("admit requires WAITING request");
    const status=K.canAllocate(s,K.blocksNeeded(r,s.blockSize));
    if(status==="OK")run({op:"allocate",id:entry.id},actions);
    else blocked.push({id:entry.id,status,reason:"prefill admission requires separate _schedule_prefill"});
   }
   for(const [id,values] of Object.entries(cyc.workloads||{})){
    if(!values||typeof values!=="object"||Array.isArray(values))fail("invalid workload");
    run({op:"update_workload",id,...values},actions);
   }
   const candidates=Object.keys(s.requests).filter(id=>s.requests[id].status==="RUNNING");
   for(const id of candidates)if(!Object.hasOwn(cyc.outputLens,id))
    fail("missing outputLen for active request "+id+" at cycle "+idx);
   // C++ sort descending outputLen; schedule from BACK -> shortest generated
   // outputs first. Victims are at FRONT -> longest generated output first.
   const sorted=candidates.sort((a,b)=>cyc.outputLens[b]-cyc.outputLens[a]||a.localeCompare(b));
   let left=0,right=sorted.length;
   while(left<right){
    const candidate=sorted[right-1];
    while(s.free.length===0&&right-left>1){
     const victim=sorted[left++];
     run({op:"preempt",id:victim,stop:stopPreempted},actions);
     preempted.push({id:victim,outcome:s.requests[victim].status==="WAITING"?"recompute":"stop",
      trigger:"Scheduler._schedule_decode free-block guard"});
    }
    if(s.free.length>0){
     // Real Scheduler guard requires a free block even if _append_slot itself
     // would not grow this request. Simulated _append_slot may free a tail.
     run({op:"append_slot",id:candidate},actions);
     scheduled.push(candidate);
    }else{
     run({op:"preempt",id:candidate,stop:stopPreempted},actions);
     preempted.push({id:candidate,outcome:s.requests[candidate].status==="WAITING"?"recompute":"stop",
      trigger:"no free block for remaining request"});
    }
    right--;
   }
   // C++ sorts scheduled by input token length DESC for actual Worker batch.
   scheduled.sort((a,b)=>s.requests[b].tokenLen-s.requests[a].tokenLen||a.localeCompare(b));
   for(const id of scheduled){
    if(s.requests[id].status!=="RUNNING"||!s.requests[id].blocks.length)
     fail("scheduled request lacks active physical KV blocks: "+id);
   }
   steps.push({mode:scheduled.length?"decode":"placeholder",sequenceIds:scheduled.slice()});
   const freeBeforeEnd=s.free.slice();
   for(const op of cyc.after||[])run(op,actions);
   K.verify(s);
   const waitingIDs=Object.keys(s.requests).filter(id=>s.requests[id].status==="WAITING");
   waiting.push(...waitingIDs);
   decisions.push({cycle:idx,scheduled:scheduled.slice(),preempted,blocked,
    waiting:waiting.slice(),freeBeforeEnd,freeAfter:s.free.slice(),
    kvRevision:s.revision,actions,reason:reason.concat("decode-only: preempted WAITING requests require explicit subsequent prefill admission")});
   snapshots.push(s);
  });
  const worker=Worker.simulate(steps,{zeroOverhead:true,flush,durations});
  if(worker.steps.length!==decisions.length)fail("Worker batch mismatch");
  decisions.forEach((d,i)=>{
   if(worker.steps[i].sequenceIds.join(",")!==d.scheduled.join(","))fail("Worker executed unscheduled requests");
  });
  // Resource and owner conservation was checked by K.advance at every change;
  // additionally verify all stored snapshots as independently replayable.
  snapshots.forEach(K.verify);
  return {config:spec.config,setup:spec.setup,cycles:spec.cycles,stopPreempted,
   decisions,snapshots,worker,
   summary:{preemptions:decisions.reduce((sum,d)=>sum+d.preempted.length,0),
    recomputes:decisions.reduce((sum,d)=>sum+d.preempted.filter(p=>p.outcome==="recompute").length,0),
    terminalPreemptions:decisions.reduce((sum,d)=>sum+d.preempted.filter(p=>p.outcome==="stop").length,0),
    executedDecodeSteps:steps.filter(x=>x.mode==="decode").length,
    finalFreeBlocks:s.free.length},
   disclaimer:"Source-derived bounded decode scheduling and Worker event state, NOT actual C++ scheduler, model execution, Prefill votes, GPU allocations or measured throughput."};
 }
 return Object.freeze({EXAMPLES,validate,simulate});
});
