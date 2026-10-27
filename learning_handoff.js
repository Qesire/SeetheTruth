/* Pedagogical cross-page handoff of native A/B evidence to an existing graph lesson.
 * Every imported run and state cursor is restored through ObservationSnapshots.
 * No native state is treated as a tensor graph or a mutable engine checkpoint.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const api=factory(isNode?require("./observation_snapshots.js"):root.SeetheTruthObservationSnapshots);
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthLearningHandoff=api;
})(typeof window!=="undefined"?window:null,function(S){"use strict";
 const SCHEMA="seethetruth-native-graph-handoff-v1";
 const KEY="seethetruth:native-graph-handoff-v1",LIMIT=180000;
 const target=Object.freeze({
  deepinfer_kv:Object.freeze({course:"decode",node:"gqa",question:"解码时读取的 K/V 前缀与物理块生命周期是什么关系？"}),
  deepinfer_scheduler:Object.freeze({course:"decode",node:"gqa",question:"调度准入如何决定哪些 token 可以进入 Attention 计算？"}),
  deepinfer_worker:Object.freeze({course:"attention",node:"attn",question:"依赖 DAG 的先后关系和 Attention 中 Q/K/V 的数据依赖有什么异同？"})
 });
 function fail(m){throw Error("learning handoff: "+m);}
 function bounded(x){
  const json=JSON.stringify(x);
  if(typeof json!=="string"||json.length>LIMIT)fail("cross-page teaching evidence exceeds size limit");
  return json;
 }
 function check(input){
  if(!input||typeof input!=="object"||Array.isArray(input))fail("invalid package");
  if(Object.keys(input).some(k=>!["schema","domain","active","baseline","question"].includes(k)))
   fail("unexpected package member");
  if(input.schema!==SCHEMA||!Object.hasOwn(target,input.domain))fail("unregistered native lab");
  if(typeof input.question!=="string"||input.question.length>300)fail("invalid learner question");
  if(!input.active||!input.active.cursor)fail("source event cursor is required");
  S.validate(input.active);
  if(input.baseline!==null&&input.baseline!==undefined)S.validate(input.baseline);
  bounded(input);
  return input;
 }
 function create(core,active,baseline=null,selectedId=null,question=""){
  if(!active||active.kind!=="specialized_execution"||!Object.hasOwn(target,active.id)||
     !active.stateTrace?.steps?.length)fail("replayable source state required");
  const snapshot=S.create(core,active,"native selected B");
  const stepId=selectedId&&active.stateTrace.steps.some(x=>x.id===selectedId)?
   selectedId:active.stateTrace.steps[0].id;
  const cursor=S.at(core,snapshot,stepId);
  const A=baseline?S.create(core,baseline,"native baseline A"):null;
  const payload={schema:SCHEMA,domain:active.id,active:cursor,baseline:A,
    question:String(question||"").trim().slice(0,300)};
  restore(core,payload);
  return Object.freeze(payload);
 }
 function restore(core,payload){
  const p=check(payload);
  if(p.active.kind!=="specialized_execution"||p.active.id!==p.domain||
     p.active.cursor.kind!=="state_transition")fail("source lab and event mismatch");
  const current=S.restore(core,p.active);
  const baseline=p.baseline?S.restore(core,p.baseline):null;
  if(baseline&&(baseline.record.id!==current.record.id||
      baseline.record.operation!==current.record.operation||
      baseline.record.engine!==current.record.engine||
      baseline.record.evidence!==current.record.evidence))
   fail("A and B use different native execution semantics");
  let comparison=null;
  if(baseline){
   comparison=core.compareRecords(baseline.record,current.record);
   if(!comparison.replayVerified||comparison.inputEqual)fail("independent A/B input replay required");
  }
  const lesson=target[p.domain];
  return Object.freeze({schema:SCHEMA,domain:p.domain,course:lesson.course,
   focusNode:lesson.node,question:p.question||lesson.question,sourceHref:current.sourceHref,
   current,baseline,comparison,selectedStepId:current.checkpoint.stepId,
   selectedStepIndex:current.checkpoint.index,evidence:current.evidence});
 }
 function store(storage,payload){
  const data=bounded(check(payload));
  if(!storage||typeof storage.setItem!=="function")fail("per-tab teaching storage unavailable");
  storage.setItem(KEY,data);
 }
 function load(storage){
  if(!storage||typeof storage.getItem!=="function")fail("per-tab teaching storage unavailable");
  const raw=storage.getItem(KEY);
  if(!raw)fail("no active learner-initiated handoff");
  if(raw.length>LIMIT)fail("cross-page payload exceeds size limit");
  let parsed;
  try{parsed=JSON.parse(raw);}catch(_){fail("invalid cross-page JSON");}
  return check(parsed);
 }
 return Object.freeze({SCHEMA,KEY,LIMIT,create,restore,store,load});
});
