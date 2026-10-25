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
 if(typeof json!=="string"||json.length>MAX_BYTES)fail("snapshot exceeds bounded JSON size");
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
 const allowed=["schema","kind","id","operation","engine","evidence","numeric","input","checksum","label"];
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
  checksum:checksum(witness(record)),label:String(label)
 };
 return validate(entry);
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
 return Object.freeze({schema:SCHEMA,restored:true,evidence:s.evidence,
  record,observation:observed,sourceHref:route(core,s)});
}
function describe(core,snapshot){
 const s=validate(snapshot);
 return Object.freeze({schema:SCHEMA,kind:s.kind,id:s.id,operation:s.operation,
  label:s.label,evidence:s.evidence,sourceHref:route(core,s),
  replayVerified:false,imported:true});
}
return Object.freeze({SCHEMA,MAX_BYTES,create,validate,restore,describe,route});
});
