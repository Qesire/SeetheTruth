/* Portable, deliberately small observation recipe. Original engines retain
 * arithmetic authority. The checksum detects ordinary corruption; it is NOT a
 * signature, hardware attestation, or authentication of a user-supplied file.
 */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;if(root)root.SeetheTruthObservationSnapshots=api;})(typeof window!=="undefined"?window:null,function(){"use strict";
const SCHEMA="seethetruth-observation-snapshot-v1",MAX_BYTES=196608;
const fail=m=>{throw Error("observation snapshot: "+m)};
const plain=x=>x!==null&&typeof x==="object"&&!Array.isArray(x);
const clone=x=>JSON.parse(JSON.stringify(x));
function safeJSON(value){
 const json=JSON.stringify(value,(_k,v)=>{
  if(typeof v==="number"&&!Number.isFinite(v))fail("non-finite numeric value");
  if(typeof v==="bigint"||typeof v==="function"||typeof v==="symbol")fail("unsupported value type");
  return v;
 });
 if(typeof json!=="string")fail("snapshot exceeds bounded JSON size");
 let bytes=0;
 for(const character of json){const v=character.codePointAt(0);bytes+=v<=127?1:v<=2047?2:v<=65535?3:4;
  if(bytes>MAX_BYTES)fail("snapshot exceeds bounded JSON size");
 }
 return json;
}
// Deterministic integrity diagnostic, not a cryptographic seal.
function checksum(value){
 const raw=JSON.stringify(value);let h=2166136261;
 for(let i=0;i<raw.length;i++){h=Math.imul(h^raw.charCodeAt(i),16777619);}
 return "fnv1a32:"+("00000000"+(h>>>0).toString(16)).slice(-8)+":"+raw.length;
}
function witness(record){
 if(record.kind==="graph")return {kind:record.kind,id:record.id,evidence:record.evidence,
  numericOk:record.numericOk,graph:record.graph,trace:record.session?.trace??null};
 return {kind:record.kind,id:record.id,operation:record.operation,
  engine:record.engine,evidence:record.evidence,input:record.inputSnapshot,
  value:record.value,stateTrace:record.stateTrace,computationTrace:record.computationTrace};
}
function validate(input){
 if(!plain(input))fail("expected plain JSON object");
 const raw=safeJSON(input),s=JSON.parse(raw);
 const allowed=["schema","kind","id","operation","engine","evidence","numeric","input","checksum","label","parent","cursor"];
 if(Object.keys(s).some(key=>!allowed.includes(key)))fail("unexpected envelope field");
 if(s.schema!==SCHEMA||!["graph","specialized_execution"].includes(s.kind))fail("invalid schema/domain");
 if(typeof s.id!=="string"||s.id.length<1||s.id.length>100||!/^[a-zA-Z0-9_.:-]+$/.test(s.id))
  fail("invalid original execution identity");
 if(typeof s.operation!=="string"||s.operation.length<1||s.operation.length>80||
   !/^[a-zA-Z0-9_]+$/.test(s.operation))fail("invalid operation");
 if(typeof s.engine!=="string"||s.engine.length<1||s.engine.length>100||
   !/^[a-zA-Z0-9_.-]+$/.test(s.engine))fail("invalid source engine");
 if(typeof s.evidence!=="string"||s.evidence.length<1||s.evidence.length>120||
   !/^[A-Za-z0-9_]+$/.test(s.evidence))fail("invalid evidence");
 if(typeof s.numeric!=="boolean")fail("missing numeric mode");
 if(typeof s.label!=="string"||s.label.length>80)fail("invalid snapshot label");
 if(s.parent!==undefined&&s.parent!==null&&
    (typeof s.parent!=="string"||!/^fnv1a32:[a-f0-9]{8}:[0-9]{1,9}$/.test(s.parent)))
  fail("invalid parent snapshot reference");
 if(s.cursor!==undefined&&s.cursor!==null){
  const c=s.cursor;
  if(s.kind!=="specialized_execution"||!plain(c)||
    Object.keys(c).some(k=>!["kind","stepId","index","evidence","stateDigest"].includes(k))||
    c.kind!=="state_transition"||typeof c.stepId!=="string"||
    !/^[a-zA-Z0-9_:-]{1,80}$/.test(c.stepId)||
    !Number.isSafeInteger(c.index)||c.index<0||c.index>249||
    typeof c.evidence!=="string"||!/^[A-Z0-9_]{1,120}$/.test(c.evidence)||
    typeof c.stateDigest!=="string"||
    !/^fnv1a32:[a-f0-9]{8}:[0-9]{1,9}$/.test(c.stateDigest))
   fail("invalid state checkpoint cursor");
 }
 if(typeof s.checksum!=="string"||!/^fnv1a32:[a-f0-9]{8}:[0-9]{1,9}$/.test(s.checksum))
  fail("invalid diagnostic checksum");
 if(!plain(s.input)||s.kind==="specialized_execution"&&s.numeric)fail("invalid source input");
 if(s.kind==="graph"&&(s.operation!=="inspectGraph"||s.engine!=="learning_lab.js"))
  fail("graph source identity mismatch");
 if(s.kind==="specialized_execution"&&s.operation==="inspectGraph")
  fail("specialist operation collides with graph");
 return Object.freeze(s);
}
function create(core,record,label=""){
 if(!core||typeof core.replay!=="function"||typeof core.observe!=="function")
  fail("unified execution core unavailable");
 core.replay(record);
 const observation=core.observe(record);
 const isGraph=record?.kind==="graph";
 if(!isGraph&&record?.kind!=="specialized_execution")fail("unsupported original record");
 const entry={
  schema:SCHEMA,kind:record.kind,id:record.id,
  operation:isGraph?"inspectGraph":record.operation,
  engine:isGraph?"learning_lab.js":record.engine,
  evidence:observation.evidence,
  numeric:isGraph?record.numericOk===true:false,
  input:clone(isGraph?record.graph:record.inputSnapshot),
  checksum:checksum(witness(record)),label:String(label),parent:null
 };
 return validate(entry);
}
// Checkpoint selection is a view cursor over a verified *native* state frame,
// not serialized mutable runtime memory or an unearned device checkpoint.
function at(core,source,stepId){
 const parent=validate(source),replayed=restore(core,parent),record=replayed.record;
 if(parent.kind!=="specialized_execution"||!record.stateTrace)
  fail("state cursor requires a state-transition source");
 if(typeof stepId!=="string")fail("state stepId required");
 const index=record.stateTrace.steps.findIndex(s=>s.id===stepId);
 if(index<0)fail("state step does not exist in original trace");
 const proof=core.inspectObservationStep(record,{kind:"state_transition",stepId});
 if(!proof.replayVerified||proof.evidence!==record.stateTrace.evidenceTier)
  fail("state cursor could not be revalidated");
 const cursor={kind:"state_transition",stepId,index,
  evidence:proof.evidence,
  stateDigest:checksum({stepId,index,evidence:proof.evidence,detail:proof.detail})};
 return validate({...parent,cursor});
}
function resumeKV(core,source){
 const replayed=restore(core,source),s=validate(source),cursor=replayed.cursor;
 if(s.id!=="deepinfer_kv"||s.operation!=="replay"||!cursor||
    !plain(s.input)||!Array.isArray(s.input.operations))
  fail("KV continuation requires a verified KV cursor");
 const prefix=core.executeLab("deepinfer_kv","replay",{
  config:s.input.config,operations:s.input.operations.slice(0,cursor.index)});
 core.replay(prefix);
 if(prefix.stateTrace.steps.length!==cursor.index+1||
    JSON.stringify(prefix.stateTrace.steps.at(-1)?.after)!==
    JSON.stringify(replayed.checkpoint.detail.after)||
    JSON.stringify(prefix.value.timeline.at(-1))!==
    JSON.stringify(replayed.record.value.timeline[cursor.index]))
  fail("KV prefix state diverged from imported native replay");
 // The original operations after the cursor are replayed by the same engine.
 // The previous full-run source checksum is verified in restore().
 return Object.freeze({schema:SCHEMA,kind:"local_kv_continuation",
  index:cursor.index,stepId:cursor.stepId,evidence:replayed.evidence,
  checkpoint:replayed.checkpoint,prefix:prefix.value,
  remainingOperations:clone(s.input.operations.slice(cursor.index)),
  finalReplayed:true});
}
function fork(core,original,nextInput,label=""){
 const base=validate(original);
 const confirmed=restore(core,base);
 if(!plain(nextInput))fail("fork requires a complete source input object");
 const input=clone(nextInput);
 if(JSON.stringify(input)===JSON.stringify(base.input))
  fail("fork needs a changed source input");
 const candidate=base.kind==="graph"
  ?core.inspectGraph(input,{numeric:base.numeric,id:base.id})
  :core.executeLab(base.id,base.operation,input);
 const diff=core.compareRecords(confirmed.record,candidate);
 if(!diff.replayVerified)fail("branch did not pass source replay");
 const next=create(core,candidate,label);
 if(next.engine!==base.engine||next.evidence!==base.evidence||next.operation!==base.operation)
  fail("branch changed source domain/evidence");
 return validate({...next,parent:base.checksum});
}
function route(core,snapshot){
 const s=validate(snapshot);
 if(s.kind==="graph")return "learn.html";
 const lab=core?.labs?.find(item=>item.id===s.id&&item.engine===s.engine);
 if(!lab)fail("unregistered source engine");
 return lab.href;
}
function restore(core,snapshot){
 const s=validate(snapshot);
 if(!core||typeof core.inspectGraph!=="function"||typeof core.executeLab!=="function"||
  typeof core.replay!=="function")fail("unified execution core unavailable");
 // Require recognized source even before executing any user-controlled scenario.
 route(core,s);
 const record=s.kind==="graph"
  ?core.inspectGraph(s.input,{numeric:s.numeric,id:s.id})
  :core.executeLab(s.id,s.operation,s.input);
 core.replay(record);
 const observed=core.observe(record);
 if(record.evidence!==s.evidence||observed.evidence!==s.evidence)
  fail("source evidence changed");
 if(checksum(witness(record))!==s.checksum)
  fail("replayed source result differs from exported snapshot");
 let checkpoint=null;
 if(s.cursor){
  if(!record.stateTrace||record.stateTrace.evidenceTier!==s.cursor.evidence||
     record.stateTrace.steps[s.cursor.index]?.id!==s.cursor.stepId)
   fail("state cursor no longer matches the source trace");
  const proof=core.inspectObservationStep(record,{kind:"state_transition",stepId:s.cursor.stepId});
  const digest=checksum({stepId:s.cursor.stepId,index:s.cursor.index,
   evidence:proof.evidence,detail:proof.detail});
  if(!proof.replayVerified||digest!==s.cursor.stateDigest)
   fail("state cursor replay differs from exported snapshot");
  checkpoint=Object.freeze({...proof,index:s.cursor.index});
 }
 return Object.freeze({schema:SCHEMA,restored:true,evidence:s.evidence,
  record,observation:observed,checkpoint,sourceHref:route(core,s)});
}
function describe(core,snapshot){
 const s=validate(snapshot);
 return Object.freeze({schema:SCHEMA,kind:s.kind,id:s.id,operation:s.operation,
  label:s.label,evidence:s.evidence,cursor:s.cursor||null,sourceHref:route(core,s),
  replayVerified:false,imported:true});
}
return Object.freeze({SCHEMA,MAX_BYTES,create,fork,at,resumeKV,validate,restore,describe,route});
});
