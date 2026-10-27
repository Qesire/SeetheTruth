/* Shared, bounded coordinate selection contract.
 * A selection always carries native execution identity and evidence tier.
 * No synthetic numerical values are introduced by this module.
 */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthCoordinateFocus=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const SCHEMA="seethetruth-coordinate-focus-v1",MAX=16384;
 const TIERS=Object.freeze({
  STATE:"EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS",
  GRAPH:"EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS",
  PINNED:"PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64",
  TRAINING:"EXECUTED_BOUNDED_CPU_TRAINING_COORDINATE",
  STATIC:"STATIC_SHAPES_NO_EXECUTION"
 });
 const domains=Object.freeze(["semantic_state","guided_graph","computation_microscope","unified_observation","training_parameter"]);
 const numeric=new Set([TIERS.STATE,TIERS.GRAPH,TIERS.PINNED,TIERS.TRAINING]);
 const fail=msg=>{throw Error("coordinate focus: "+msg);};
 const product=shape=>shape.reduce((p,n)=>p*n,1);
 function bound(shape,index,axes){
  if(!Array.isArray(shape)||shape.length<1||shape.length>5||
     !shape.every(n=>Number.isSafeInteger(n)&&n>0)||product(shape)>MAX)fail("bounded rank 1..5 shape required");
  if(!Array.isArray(index)||index.length!==shape.length||
     !index.every((n,i)=>Number.isSafeInteger(n)&&n>=0&&n<shape[i]))fail("coordinate outside tensor shape");
  if(axes!==undefined&&(!Array.isArray(axes)||axes.length!==shape.length||
    new Set(axes).size!==axes.length||!axes.every(a=>typeof a==="string"&&/^[A-Za-z][A-Za-z0-9_]{0,15}$/.test(a))))
   fail("axis metadata must match shape");
  return Object.freeze({shape:Object.freeze(shape.slice()),coordinate:Object.freeze(index.slice()),
   axes:axes===undefined?null:Object.freeze(axes.slice())});
 }
 function flatten(shape,index){bound(shape,index);return index.reduce((acc,v,i)=>acc*shape[i]+v,0);}
 function unflatten(shape,index){
  if(!Array.isArray(shape)||shape.length<1||shape.length>5||
    !shape.every(n=>Number.isSafeInteger(n)&&n>0)||product(shape)>MAX||
    !Number.isSafeInteger(index)||index<0||index>=product(shape))fail("flat index outside tensor shape");
  const result=new Array(shape.length);
  for(let i=shape.length-1;i>=0;i--){result[i]=index%shape[i];index=Math.floor(index/shape[i]);}
  return Object.freeze(result);
 }
 function create(info){
  if(!info||!domains.includes(info.domain))fail("invalid observation domain");
  if(typeof info.executionId!=="string"||!info.executionId||info.executionId.length>160)
   fail("execution identity required");
  if(typeof info.nodeId!=="string"||!info.nodeId||info.nodeId.length>160)
   fail("node identity required");
  if(typeof info.evidence!=="string"||!Object.values(TIERS).includes(info.evidence))
   fail("explicit evidence tier required");
  const c=bound(info.shape,info.coordinate,info.axes??undefined);
  if(info.value!==undefined&&(!numeric.has(info.evidence)||typeof info.value!=="number"||!Number.isFinite(info.value)))
   fail("numeric coordinate needs actual finite native execution evidence");
  if(info.branch!==undefined&&(!["a","b"].includes(info.branch)))fail("invalid A/B branch");
  const transition=info.transition;
  if(transition!==undefined){
   if(info.domain!=="training_parameter"||info.evidence!==TIERS.TRAINING||
      !transition||["before","after","delta","wholeLossGradient"].every(k=>
       typeof transition[k]==="number"&&Number.isFinite(transition[k]))===false||
      info.value!==transition.after||
      Math.abs(transition.after-transition.before-transition.delta)>
       1e-10*Math.max(1,Math.abs(transition.after),Math.abs(transition.before)))
    fail("training coordinate transition must come from a finite native parameter update");
  }
  const path=info.path===undefined?[]:info.path;
  if(!Array.isArray(path)||path.length>256||path.some(p=>typeof p!=="string"||!p||p.length>160))
   fail("invalid dependency path");
  if(typeof info.note!=="string"||!info.note)fail("coordinate evidence explanation required");
  return Object.freeze({schema:SCHEMA,domain:info.domain,executionId:info.executionId,nodeId:info.nodeId,
   ...c,flatIndex:flatten(c.shape,c.coordinate),evidence:info.evidence,
   ...(info.value!==undefined?{value:info.value}:{}),
   ...(info.branch?{branch:info.branch}:{}),
   ...(transition?{transition:Object.freeze({...transition})}:{}),
   path:Object.freeze(path.slice()),note:info.note});
 }
 function compare(a,b){
  for(const s of [a,b])if(!s||s.schema!==SCHEMA)fail("two coordinate selections required");
  // Only the immutable state-example A/B pair currently has one shared
  // accepted execution. Graph baseline differences belong to Core.compareRecords.
  if(a.domain!=="semantic_state"||b.domain!=="semantic_state")
   fail("graph/other execution comparison requires native compareRecords");
  if(a.domain!==b.domain||a.executionId!==b.executionId||
     JSON.stringify(a.shape)!==JSON.stringify(b.shape)||
     JSON.stringify(a.coordinate)!==JSON.stringify(b.coordinate))
   fail("coordinate comparison requires same native execution, shape and index");
  if(a.value===undefined||b.value===undefined)fail("comparison requires two observed numeric values");
  if(a.evidence!==b.evidence)fail("evidence tier mismatch");
  return Object.freeze({schema:"seethetruth-coordinate-comparison-v1",coordinate:a.coordinate,
   executionId:a.executionId,delta:a.value-b.value,evidence:a.evidence});
 }
 function store(){
  let selection=null;
  const listeners=new Set();
  const notify=()=>{for(const callback of listeners)callback(selection);};
  return Object.freeze({
   select:input=>{selection=create(input);notify();return selection;},
   clear:()=>{selection=null;notify();},
   subscribe:callback=>{
    if(typeof callback!=="function")fail("subscriber must be a function");
    listeners.add(callback);return()=>listeners.delete(callback);
   },
   get current(){return selection;}
  });
 }
 function fromState(proof){
  if(!proof||!Array.isArray(proof.shape)||!Array.isArray(proof.coordinate)||
    !proof.name||!["a","b"].includes(proof.branch))fail("accepted state witness required");
  return create({domain:"semantic_state",executionId:"semantic_state/scenario/"+proof.name,
   nodeId:proof.nodeId,shape:proof.shape,axes:proof.axes,coordinate:proof.coordinate,
   evidence:TIERS.STATE,value:proof.value,branch:proof.branch,path:proof.traceNodes,
   note:proof.note});
 }
 function fromMicroscope(frame,step,coordinate,value,parentId){
  if(!frame||!step||!Array.isArray(step.out?.shape)||!Number.isFinite(value))
   fail("native microscope tensor step required");
  return create({domain:"computation_microscope",executionId:frame.title+" / "+(frame.origin||"teaching"),
   nodeId:parentId||step.id,shape:step.out.shape,axes:step.out.axes,
   coordinate,evidence:frame.evidenceTier,value,
   path:[...(parentId&&parentId!==step.id?[parentId]:[]),step.id],
   note:"selected native microscope output; original execution provenance "+frame.evidenceTier});
 }
 function fromObservation(record,proof){
  if(record?.kind!=="graph"||record.numericOk!==true||
     proof?.kind!=="tensor_computation"||proof.replayVerified!==true)
   fail("replayed native numeric graph observation required");
  const step=record.path?.steps?.find(s=>s.id===proof.stepId);
  if(!step?.shape||!Array.isArray(proof.coordinate)||
    typeof proof.detail?.value!=="number"||!Number.isFinite(proof.detail.value))
   fail("graph observation is missing native coordinate or numeric value");
  const evidence=proof.evidence;
  if(!numeric.has(evidence))fail("unrecognized numerical evidence tier");
  // Used for local selection identity only. Native Core.compareRecords is the
  // authority for comparing two separately accepted graph executions.
  const executionId="guided_graph/"+(record.id||"learning_studio")+"/accepted_numeric";
  return create({domain:"unified_observation",executionId,nodeId:proof.stepId,
   shape:step.shape,coordinate:proof.coordinate,evidence,value:proof.detail.value,
   path:[proof.stepId],note:"replayed accepted graph step; per-execution comparison uses native compareRecords"});
 }
 function fromTraining(record,replay){
  if(record?.kind!=="specialized_execution"||record.id!=="training"||
     record.operation!=="coordinate_proof"||replay?.reproduced!==true||
     record.value?.schema!=="seethetruth-training-coordinate-proof-v1"||
     record.value.evidence!==TIERS.TRAINING)
   fail("replay-verified native training coordinate proof required");
  const proof=record.value;
  if(typeof proof.parameterId!=="string"||!proof.parameterId||
     !Array.isArray(proof.index)||!Array.isArray(proof.shape)||
     !proof.parameter||!Number.isFinite(proof.wholeLossGradient))
   fail("original parameter, shape and whole-loss gradient required");
  return create({domain:"training_parameter",executionId:"training/coordinate_proof/"+
    proof.parameterId+"/"+proof.index.join(",")+"/step_"+proof.optimizer.stepAfter,
   nodeId:proof.parameterId,shape:proof.shape,coordinate:proof.index,
   evidence:TIERS.TRAINING,value:proof.parameter.after,path:[proof.parameterId],
   transition:{before:proof.parameter.before,after:proof.parameter.after,
     delta:proof.parameter.delta,wholeLossGradient:proof.wholeLossGradient,
     optimizerStepBefore:proof.optimizer.stepBefore,optimizerStepAfter:proof.optimizer.stepAfter},
   note:"来自已重放的原生小模型训练更新；完整 Loss 梯度与选定局部 VJP 路径贡献需分别解释。"});
 }
 return Object.freeze({SCHEMA,TIERS,bound,flatten,unflatten,create,compare,store,fromState,fromMicroscope,fromObservation,fromTraining});
});
