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
  STATIC:"STATIC_SHAPES_NO_EXECUTION"
 });
 const domains=Object.freeze(["semantic_state","guided_graph","computation_microscope","unified_observation"]);
 const numeric=new Set([TIERS.STATE,TIERS.GRAPH,TIERS.PINNED]);
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
  const path=info.path===undefined?[]:info.path;
  if(!Array.isArray(path)||path.length>256||path.some(p=>typeof p!=="string"||!p||p.length>160))
   fail("invalid dependency path");
  if(typeof info.note!=="string"||!info.note)fail("coordinate evidence explanation required");
  return Object.freeze({schema:SCHEMA,domain:info.domain,executionId:info.executionId,nodeId:info.nodeId,
   ...c,flatIndex:flatten(c.shape,c.coordinate),evidence:info.evidence,
   ...(info.value!==undefined?{value:info.value}:{}),
   ...(info.branch?{branch:info.branch}:{}),path:Object.freeze(path.slice()),note:info.note});
 }
 function compare(a,b){
  for(const s of [a,b])if(!s||s.schema!==SCHEMA)fail("two coordinate selections required");
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
 return Object.freeze({SCHEMA,TIERS,bound,flatten,unflatten,create,compare,store,fromState,fromMicroscope});
});
