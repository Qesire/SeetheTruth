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

 // The mechanism lens is a read-only join of two independently executed
 // contracts. Native event dependencies are precedence relations, graph ports
 // are data dependencies; neither represents a cross-domain arithmetic edge.
 function explain(core,payload,currentGraph,numericRecord=null){
  const native=restore(core,payload);
  const graph=core.inspectGraph(currentGraph,{numeric:false,id:native.course});
  const node=graph.path.steps.find(step=>step.id===native.focusNode);
  const expectedKind=native.domain==="deepinfer_worker"?"attention":"gqa";
  if(!node||node.kind!==expectedKind)
   fail("current graph lesson no longer contains the intended Attention operation");
  const ports=node.ports.filter(p=>["q","k","v"].includes(p.port)).map(p=>Object.freeze({
   port:p.port,sourceId:p.sourceId,sourceShape:p.sourceShape,
   destinationId:node.id,destinationKind:node.kind
  }));
  if(ports.length!==3||ports.some(p=>!p.sourceId||!Array.isArray(p.sourceShape)))
   fail("the current lesson needs three actual Q/K/V source ports");
  let numeric=null;
  if(numericRecord){
   if(numericRecord.kind!=="graph"||numericRecord.numericOk!==true||
      JSON.stringify(numericRecord.graph)!==JSON.stringify(graph.graph))
    fail("numeric trace belongs to a different or non-executed graph");
   core.replay(numericRecord);
   const computed=numericRecord.path?.steps?.find(step=>step.id===native.focusNode);
   if(!computed||computed.evidence!=="SYNTHETIC_NUMERIC"||
      !numericRecord.session?.trace?.steps?.some(step=>step.id===node.id))
    fail("selected graph node lacks an accepted numeric calculation");
   numeric=Object.freeze({shape:computed.shape,sample:computed.sample,
    evidence:computed.evidence,observedNode:computed.id});
  }
  const step=native.current.record.stateTrace.steps[native.selectedStepIndex];
  if(!step||step.id!==native.selectedStepId)fail("native selected event changed");
  const changed=native.comparison?.changedSteps.some(item=>
   item.kind==="state_transition"&&item.stepId===step.id)??null;
  const details=Object.freeze({
   stepId:step.id,title:step.title,
   dependencies:Object.freeze(step.dependencies.slice()),
   changes:Object.freeze(step.changes.map(item=>item.key)),
   timing:step.timing?Object.freeze({...step.timing}):null,
   selectedStepChangedInAB:changed,
   totalChangedNativeSteps:native.comparison?.changedSteps.length??null
  });
  const topic=native.domain==="deepinfer_worker"
   ?"Worker 的边是事件先后约束；Q/K/V 的边是张量输入依赖。试比较两者的可并行条件。"
   :native.domain==="deepinfer_kv"
   ?"物理块队列与引用计数管理 K/V 存储所有权；图中的 k/v 是逻辑张量数据输入。试区分存储地址与内容。"
   :"Scheduler 的准入决定请求能否进入 Forward；图的 Q/K/V 端口说明已准入的一次 Attention 如何计算。试区分调度与算术。";
  return Object.freeze({
   schema:"seethetruth-native-graph-mechanism-v1",domain:native.domain,
   course:native.course,question:native.question,sourceEvidence:native.evidence,
   native:details,graph:Object.freeze({id:graph.id,kind:node.kind,
    nodeId:node.id,shape:node.shape,
    ports:Object.freeze(ports),evidence:numeric?"SYNTHETIC_NUMERIC":"STATIC_ONLY",
    numeric}),
   contrast:topic,
   status:numeric?"REPLAYED_NATIVE_AND_GRAPH_NUMERIC":"REPLAYED_NATIVE_AND_GRAPH_STATIC"
  });
 }
 return Object.freeze({SCHEMA,KEY,LIMIT,create,restore,store,load,explain});
});
