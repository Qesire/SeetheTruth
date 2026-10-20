/* DeepInfer Worker.execute_model_step source-grounded cross-step event simulator.
 * CPU/GPU logical resources; times are assumed abstract durations, NEVER measured.
 * Does not execute an actual ML model or DeepInfer C++ runtime.
 */
(function(root,factory){"use strict";const api=factory(typeof module==="object"&&module.exports?require("./deepinfer_zos_metadata.js"):root.SeetheTruthZOSMetadata);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthDeepInferEvents=api;})(typeof window!=="undefined"?window:{},function(ZOS){"use strict";
 const DEFAULT_DURATIONS=Object.freeze({schedule:2,launch:1,correct:1,forward:8,post:4,launchSample:1,sample:2});
 const DEFAULT_STEPS=Object.freeze([
  {mode:"prompt",sequenceIds:["A","B"]},
  {mode:"decode",sequenceIds:["A","B"]},
  {mode:"decode",sequenceIds:["B","C"]},
  {mode:"decode",sequenceIds:["B","C"]}
 ]);
 const MAX_STEPS=12;
 const fail=s=>{throw Error(s);};
 function validate(steps,opts={}){
  if(!Array.isArray(steps)||!steps.length||steps.length>MAX_STEPS)fail("steps must contain 1.."+MAX_STEPS+" entries");
  if(typeof opts.zeroOverhead!=="boolean")fail("zeroOverhead must be boolean");
  if(opts.flush!==undefined&&typeof opts.flush!=="boolean")fail("flush must be boolean");
  const durations={...DEFAULT_DURATIONS,...(opts.durations||{})};
  for(const [name,value] of Object.entries(durations)){
   if(!Object.hasOwn(DEFAULT_DURATIONS,name))fail("unrecognised duration "+name);
   if(!Number.isSafeInteger(value)||value<0||value>100)fail(name+" duration must be 0..100 abstract ticks");
  }
  return Object.freeze(steps.map((step,i)=>{
   if(!step||!["prompt","decode","placeholder"].includes(step.mode))fail("invalid step mode at "+i);
   if(!Array.isArray(step.sequenceIds)||step.sequenceIds.length>8||
      (step.mode!=="placeholder"&&!step.sequenceIds.length))fail("sequenceIds required for real steps");
   if(step.mode==="placeholder"&&step.sequenceIds.length)fail("placeholder cannot contain live sequences");
   const set=new Set();
   for(const id of step.sequenceIds){
    if(typeof id!=="string"||!/^[A-Za-z0-9_-]{1,24}$/.test(id)||set.has(id))fail("invalid or repeated sequence id");
    set.add(id);
   }
   return Object.freeze({mode:step.mode,sequenceIds:Object.freeze(step.sequenceIds.slice())});
  }));
 }
 function simulate(steps,opts={zeroOverhead:true}){
  const accepted=validate(steps,opts),config={...DEFAULT_DURATIONS,...opts.durations};
  const fixtures=opts.metadataFixture;
  if(fixtures!==undefined){
   if(!ZOS)fail("DeepInfer source metadata module missing");
   if(!fixtures||typeof fixtures!=="object"||!Number.isSafeInteger(fixtures.nextn)||
     fixtures.nextn<0||fixtures.nextn>8||typeof fixtures.variableMtp!=="boolean"||
     !Array.isArray(fixtures.records)||fixtures.records.length!==accepted.length)
    fail("metadataFixture must have nextn, variableMtp and one record per step");
   for(let i=0;i<accepted.length;i++){
    const r=fixtures.records[i],ids=accepted[i].sequenceIds;
    if(!r||!Array.isArray(r.sequences)||r.sequences.length!==ids.length||
      r.sequences.some((seq,j)=>seq.seqId!==ids[j]))fail("metadataFixture sequenceIds mismatch at step "+i);
    if(!Array.isArray(r.accepted)||r.accepted.length!==ids.length||
      r.accepted.some(v=>!Number.isSafeInteger(v)||v<0||v>fixtures.nextn))
     fail("metadataFixture accepted counts invalid at step "+i);
    if(fixtures.variableMtp&&(!Array.isArray(r.confidenceRates)||r.confidenceRates.length!==ids.length||
       r.confidenceRates.some(row=>!Array.isArray(row)||row.length!==fixtures.nextn||
        row.some(v=>typeof v!=="number"||!Number.isFinite(v)||v<0||v>1))))
      fail("metadataFixture confidence rates invalid at step "+i);
   }
  }
  const zeroOverhead=opts.zeroOverhead,flush=opts.flush!==false;
  const events=[],completed=new Map(),resourceReady={CPU:0,GPU:0},resourceLast={CPU:null,GPU:null},requestReady=[];
  let prior=null,previousGpuSample=null;
  const metadataTrace=[];
  const add=(step,stage,resource,duration,depends=[],meta={})=>{
   depends=depends.slice();
   if(resourceLast[resource]&&!depends.includes(resourceLast[resource]))
    depends.push(resourceLast[resource]); // serialize resource conflicts as explicit event edges
   for(const dep of depends)if(!completed.has(dep))fail("dependency was not scheduled: "+dep);
   const after=depends.reduce((t,dep)=>Math.max(t,completed.get(dep)),0);
   const start=Math.max(resourceReady[resource],after);
   const id="s"+step+"-"+stage;
   if(completed.has(id))fail("repeated event id "+id);
   const end=start+duration;
   events.push(Object.freeze({id,step,stage,resource,start,end,duration,
     dependsOn:Object.freeze(depends.slice()),meta:Object.freeze({...meta})}));
   completed.set(id,end);resourceReady[resource]=end;resourceLast[resource]=id;return id;
  };
  const deliver=(step,postEvent,kind)=>{
   const state=accepted[step];
   if(state.mode!=="placeholder")requestReady.push(Object.freeze({
    step,sequenceIds:state.sequenceIds.slice(),event:postEvent,at:completed.get(postEvent),via:kind}));
  };
  for(let i=0;i<accepted.length;i++){
   const step=accepted[i],scheduled=add(i,"schedule","CPU",config.schedule,[],{mode:step.mode,
    batch:step.sequenceIds.slice(),source:"LLMEngine.step → Scheduler.schedule"});
   const launch=add(i,"launch_forward","CPU",config.launch,[scheduled],{
    source:"Worker.execute_model_step launches ModelRunner.execute_main_model"});
   let correction=null;
   if(prior&&zeroOverhead&&step.mode!=="placeholder"&&
     prior.sequenceIds.some(id=>step.sequenceIds.includes(id))){
    let numerical=null;
    if(fixtures){
     const r=fixtures.records[i],prev=fixtures.records[prior.step];
     const prepared=ZOS.prepare(r.sequences,prev.sequences,{
      nextn:fixtures.nextn,paddedLength:r.paddedLength??r.sequences.length});
     if(prepared){
      const result=ZOS.correct(prepared,{
       accepted:prev.accepted,variableMtp:fixtures.variableMtp,
       confidenceRates:fixtures.variableMtp?prev.confidenceRates:null,
       totalQ:fixtures.variableMtp?r.totalQ:null});
      const sampler=r.cachedTokens?ZOS.samplingView(prepared,result,r.cachedTokens):null;
      numerical={prepared,result,sampler};
      metadataTrace.push({step:i,previousStep:prior.step,
       remap:prepared.remap.slice(),seqlensQ:result.seqlensQ.slice(),
       seqlensK:result.seqlensK.slice(),seqlensKBase:prepared.seqlensKBase.slice(),
       previousAccepted:prev.accepted.slice(),d2hSeqlensQ:result.d2hSeqlensQ,
       sampler:result&&sampler,notes:"Source C++ remap + exact accepted-token length reference; sampler refresh not executed"});
     }
    }
    if(!fixtures||numerical){
     correction=add(i,"correct_zos_metadata","GPU",config.correct,[launch,prior.sample],{
      priorStep:prior.step,remap:numerical?numerical.prepared.remap.slice():
         step.sequenceIds.map(id=>prior.sequenceIds.indexOf(id)),
      ...(numerical?{seqlensQ:numerical.result.seqlensQ.slice(),
        seqlensK:numerical.result.seqlensK.slice(),prevAccepted:numerical.result.acceptedPrevious.slice(),
        d2h:numerical.result.d2hSeqlensQ!==null}:{}),
      source:"Worker._correct_seqlens_qk: C++ remap + varlen Q + accepted-token K; D2H timing abstract"
     });
    }
   }
   const forward=add(i,"forward","GPU",config.forward,[correction||launch,...(previousGpuSample?[previousGpuSample]:[])],{
    source:"ModelRunner.execute_main_model",mode:step.mode,
    note:step.mode==="placeholder"?"placeholder forward is still invoked in upstream source":"abstract device execution"
   });
   if(prior){
    const postPrev=add(i,"post_prev","CPU",config.post,[launch,prior.sample],{
      stateFromStep:prior.step,source:"Worker._post_process_step(self.prev_step_state) after sample_end_evt.synchronize"
    });
    deliver(prior.step,postPrev,"deferred");
    prior=null;
   }
   // The sample launch follows the completed CPU post_previous and, when present,
   // the GPU correction's D2H event. Kernel runs after the forward on GPU.
   const sampleLaunch=add(i,"launch_sample","CPU",config.launchSample,
     correction?[forward]:[launch],{source:"ModelRunner.execute_sample enqueue; correction sync simplified"});
   const sample=add(i,"sample","GPU",config.sample,[forward,sampleLaunch],{
     source:"ModelRunner.execute_sample + sample_end_evt.record",mode:step.mode
   });
   previousGpuSample=sample;
   if(step.mode==="prompt"||!zeroOverhead){
    const post=add(i,"post_current","CPU",config.post,[sample,sampleLaunch],{
      source:"prompt/non-ZOS: _post_process_step(curr_step_state) in same invocation"
    });
    deliver(i,post,"immediate");
   }else{
    prior={step:i,sample,sequenceIds:step.sequenceIds};
   }
  }
  if(prior&&flush){
   const id=add(accepted.length,"flush_pending","CPU",config.post,[prior.sample],{
    source:"Worker.flush_inflight_reqs → _post_process_step(prev_step_state)",
    stateFromStep:prior.step
   });
   deliver(prior.step,id,"flush");prior=null;
  }
  const makespan=Math.max(...events.map(e=>e.end),0);
  const readySteps=new Set(requestReady.map(e=>e.step));
  if(requestReady.some(e=>e.at>makespan))fail("request after timeline");
  if(!flush&&prior&&readySteps.has(prior.step))fail("unflushed output unexpectedly delivered");
  const frozenEvents=Object.freeze(events);
  const serialGpu=events.filter(x=>x.resource==="GPU");
  for(let k=1;k<serialGpu.length;k++)if(serialGpu[k].start<serialGpu[k-1].end)fail("GPU resource overlap");
  const serializedCpu=events.filter(x=>x.resource==="CPU");
  for(let k=1;k<serializedCpu.length;k++)if(serializedCpu[k].start<serializedCpu[k-1].end)fail("CPU resource overlap");
  return Object.freeze({zeroOverhead,flush,
    durations:Object.freeze(config),steps:accepted,events:frozenEvents,
    metadataTrace:Object.freeze(metadataTrace),
    released:Object.freeze(requestReady),pendingStep:prior?prior.step:null,
    makespan,resourceBusy:Object.freeze(Object.fromEntries(["CPU","GPU"].map(resource=>
       [resource,events.filter(e=>e.resource===resource).reduce((sum,e)=>sum+e.duration,0)]))),
    note:"All durations are user-specified abstract units; not measured DeepInfer latency or GPU utilization."
  });
 }
 function compare(steps,durations){
  const overlapped=simulate(steps,{zeroOverhead:true,durations});
  const immediate=simulate(steps,{zeroOverhead:false,durations});
  return Object.freeze({overlapped,immediate,delta:immediate.makespan-overlapped.makespan,
   note:"Timeline difference is a logical scheduling scenario; not a performance benchmark."});
 }
 return Object.freeze({DEFAULT_DURATIONS,DEFAULT_STEPS,validate,simulate,compare});
});
