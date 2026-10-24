/* SeetheTruth unified learning entrypoint v1.
 * One versioned identity/evidence contract for free-form graphs, guided graph
 * recipes, shared curriculum and specialized *domain* lessons.
 * A scheduler or KV state machine is NOT disguised as a tensor DAG.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const deps=isNode?{
  learning:require("./learning_lab.js"),
  operators:require("./operator_contracts.js"),
  session:require("./graph_session.js"),
  path:require("./graph_learning_path.js"),
  branch:require("./branch_execution.js"),
  curriculum:require("./teaching_curriculum.js"),
  prefill:require("./deepinfer_prefill_policy.js"),
  worker:require("./deepinfer_worker_events.js"),
  kv:require("./deepinfer_kv_lifecycle.js"),
  scheduler:require("./deepinfer_scheduler_workflow.js"),
  state:require("./state_microscope.js")
 }:{
  learning:root.SeetheTruthLearning,operators:root.SeetheTruthOperatorContracts,
  session:root.SeetheTruthGraphSession,path:root.SeetheTruthGraphLearningPath,
  branch:root.SeetheTruthBranchExecution,curriculum:root.SeetheTruthCurriculum,
  prefill:root.SeetheTruthPrefillPolicy,
  worker:root.SeetheTruthDeepInferEvents,kv:root.SeetheTruthKVLifecycle,
  scheduler:root.SeetheTruthSchedulerWorkflow,
  state:root.SeetheTruthStateMicroscope
 };
 const api=factory(deps);
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthUnifiedCore=api;
})(typeof window!=="undefined"?window:null,function(deps){"use strict";
 const SCHEMA="seethetruth-learning-experience-v1";
 // Pages in the shipped teaching surface which have an independent numerical
 // or state-machine engine. Each entry names its REAL source, not a fictitious
 // graph operator; adding a lab must explicitly declare its contract.
 const labs=Object.freeze([
  ["source_atlas","index.html","source_graph","atlas_parser.py","SOURCE_OR_DECLARED"],
  ["engineering_journey","engineering_journey.html","guided_training_workflow","engineering_journey_engine.js","SYNTHETIC_NUMERIC"],
  ["engineering_pipeline","engineering_pipeline.html","training_state_machine","engineering_training_pipeline.js","SYNTHETIC_NUMERIC"],
  ["tradeoffs","tradeoffs.html","analytical_cost","architecture_tradeoffs.js","ANALYTICAL_ESTIMATE"],
  ["paper","paper_lab.html","graph_rewrite","paper_graph_rewrites.js","DECLARED_OR_SYNTHETIC"],
  ["training","training.html","training_state","training_flow.js","SYNTHETIC_NUMERIC"],
  ["micro_training","micro_training.html","training_state","micro_training_flows.js","SYNTHETIC_NUMERIC"],
  ["cache","cache.html","cache_state","kv_cache_lab.js","SYNTHETIC_NUMERIC"],
  ["semantic_state","semantic_state.html","tensor_state","semantic_state_lab.js","SYNTHETIC_NUMERIC"],
  ["wan_ffn","wan_ffn_lab.html","source_operator","wan_ffn_lab.js","PINNED_SOURCE_SYNTHETIC"],
  ["deepinfer_worker","deepinfer_worker_lab.html","event_dag","deepinfer_worker_events.js","ABSTRACT_TIMELINE"],
  ["deepinfer_kv","deepinfer_kv_lifecycle_lab.html","kv_state_machine","deepinfer_kv_lifecycle.js","SYNTHETIC_NUMERIC"],
  ["deepinfer_scheduler","deepinfer_scheduler_lab.html","scheduler_state_machine","deepinfer_scheduler_workflow.js","SYNTHETIC_NUMERIC"],
  ["deepinfer_prefill","deepinfer_prefill_lab.html","scheduler_decision","deepinfer_prefill_policy.js","SOURCE_DERIVED_DETERMINISTIC"],
  ["real_model","real_model_lab.html","recorded_model_trace","model_trace_bridge.js","EXTERNAL_EVIDENCE_REQUIRES_PROVENANCE"],
  ["capabilities","capabilities.html","operator_capability","block_capabilities.js","STATIC_ONLY"],
  ["telemetry","telemetry.html","recorded_performance","model_trace.js","EXTERNAL_EVIDENCE_REQUIRES_PROVENANCE"]
 ].map(([id,href,domain,engine,evidence])=>Object.freeze({
  id,href,domain,engine,evidence,kind:"specialized",
  execution:"independent_domain_adapter"})));
 const byId=new Map(labs.map(x=>[x.id,x])),byHref=new Map(labs.map(x=>[x.href,x]));
 if(byId.size!==labs.length||byHref.size!==labs.length)throw Error("duplicate specialized learning identity");
 const fail=message=>{throw Error("unified learning core: "+message);};
 const copy=v=>JSON.parse(JSON.stringify(v));
 const bound=()=>!!(deps.learning&&deps.operators&&deps.session&&deps.path&&deps.branch);
 const sourceName=s=>typeof s==="string"&&/^[a-z0-9_.-]+$/.test(s);
 function inspectGraph(raw,options={}){
  if(!bound())fail("graph runtime is not installed in this teaching surface");
  const targetId=options.targetId??null;
  const numerical=options.numeric===true;
  if(options.numeric!==undefined&&typeof options.numeric!=="boolean")fail("numeric must be boolean");
  // Work on a sliced identity graph so all teaching steps, ports and parameters
  // refer to the same closure, even when unrelated nodes are still incomplete.
  const selected=targetId?deps.branch.slice(raw,targetId):deps.learning.validateGraph(raw);
  const session=deps.session.inspect(selected,{numeric:numerical});
  const path=deps.path.build(selected,{session});
  if(path.numericOk&&!session.trace)fail("a numeric learning path requires a real bounded execution trace");
  if(path.steps.some(step=>!deps.operators.resolve(step.kind)))fail("unknown graph contract");
  return Object.freeze({schema:SCHEMA,kind:"graph",id:options.id||"free_graph",
   runtime:"learning_dag",evidence:path.evidence,
   executable:path.executable,numericOk:path.numericOk,
   graph:copy(selected),session,path,
   note:"SYNTHETIC_NUMERIC is a bounded JS teaching trace; never a checkpoint or GPU measurement."});
 }
 let activeGraphIds=null;
 function presetIds(ids){
  if(!bound())fail("graph registry unavailable");
  if(!Array.isArray(ids)||!ids.length||ids.length>256)fail("preset IDs must be a nonempty bounded list");
  const seen=new Set();
  return Object.freeze(ids.map(id=>{
   if(!sourceName(id)||seen.has(id))fail("invalid or duplicate graph lesson ID "+id);
   seen.add(id);
   const graph=deps.learning.preset(id);
   const checked=deps.learning.validateGraph(graph);
   return Object.freeze({id,kind:"guided_graph",nodeCount:checked.nodes.length,
    operatorKinds:Object.freeze([...new Set(checked.nodes.map(n=>n.kind))]),
    evidence:"STATIC_ONLY",href:"learn.html?mission="+encodeURIComponent(id)});
  }));
 }
 function activateGraphLessons(ids){
  const listed=presetIds(ids);
  activeGraphIds=new Set(listed.map(item=>item.id));
  return listed;
 }
 function openPreset(id,options={}){
  if(!sourceName(id))fail("invalid preset name");
  if(activeGraphIds&&!activeGraphIds.has(id))fail("unregistered guided-graph preset "+id);
  // The actual graph recipe is the *only* source of computational truth.
  return inspectGraph(deps.learning.preset(id),{...options,id});
 }
 function inspectLab(id){
  const lab=byId.get(id);
  if(!lab)fail("unknown specialized adapter "+id);
  return Object.freeze({schema:SCHEMA,id,kind:"specialized",
   runtime:lab.domain,engine:lab.engine,href:lab.href,
   declaredEvidence:lab.evidence,evidence:"NOT_EXECUTED",
   executable:null,numericOk:false,
   note:"Use the lab's declared domain engine. This contract alone is not execution evidence."});
 }
 // Domain adapters preserve the engines' real input semantics instead of
 // treating a KV state or CPU/GPU event as an ordinary numeric tensor.
 const domainRunners=Object.freeze({
  deepinfer_prefill:Object.freeze({
   vote:Object.freeze({module:"prefill",method:"vote",run:(m,x)=>m.vote(x)}),
   chunk:Object.freeze({module:"prefill",method:"chunk",run:(m,x)=>m.chunk(x)}),
   inspect_pair:Object.freeze({module:"prefill",method:"vote",run:(m,x)=>({vote:m.vote(x.voteInput),chunk:m.chunk(x.chunkInput)})})
  }),
  deepinfer_worker:Object.freeze({
   simulate:Object.freeze({module:"worker",method:"simulate",
    run:(m,x)=>m.simulate(x.steps,x.options||{zeroOverhead:true})}),
   compare:Object.freeze({module:"worker",method:"compare",
    run:(m,x)=>m.compare(x.steps,x.durations)})
  }),
  deepinfer_kv:Object.freeze({
   replay:Object.freeze({module:"kv",method:"replay",
    run:(m,x)=>m.replay(x.config,x.operations)})
  }),
  deepinfer_scheduler:Object.freeze({
   simulate:Object.freeze({module:"scheduler",method:"simulate",
    run:(m,x)=>m.simulate(x.spec,x.options||{})})
  })
 });
 function executeLab(id,operation,input){
  const lab=byId.get(id),runner=domainRunners[id]?.[operation];
  if(!lab||!runner)fail("no registered executable adapter for "+id+"/"+operation);
  if(!input||typeof input!=="object"||Array.isArray(input))
   fail("domain operation requires an input object");
  const engine=deps[runner.module];
  if(!engine||typeof engine[runner.method]!=="function")
   fail("domain source engine unavailable: "+lab.engine);
  const value=runner.run(engine,copy(input));
  const eventAdapters={
   "deepinfer_prefill/inspect_pair":()=>deps.state.prefill({voteInput:input.voteInput,chunkInput:input.chunkInput,voteResult:value.vote,chunkResult:value.chunk},deps.prefill),
   "deepinfer_worker/simulate":()=>deps.state.worker(value),
   "deepinfer_kv/replay":()=>deps.state.kv(value,deps.kv),
   "deepinfer_scheduler/simulate":()=>deps.state.scheduler(value,deps.kv)
  };
  const eventKey=id+"/"+operation;
  if(eventAdapters[eventKey]&&!deps.state)
   fail("state microscope dependency unavailable for "+eventKey);
  const stateTrace=eventAdapters[eventKey]?eventAdapters[eventKey]():null;
  return Object.freeze({schema:SCHEMA,kind:"specialized_execution",
   id,operation,runtime:lab.domain,engine:lab.engine,
   evidence:lab.evidence,value,stateTrace,
   note:"Observed deterministic domain engine output; does not imply GPU performance, model execution or cross-domain equivalence."});
 }
 function inspectCourse(id){
  const curriculum=deps.curriculum;
  if(!curriculum)fail("shared teaching curriculum unavailable");
  const lesson=curriculum.lookup(id);
  if(!lesson)fail("unknown curriculum lesson "+id);
  const domain=byHref.get(lesson.href)||null;
  if(lesson.href!=="learn.html"&&!domain)fail("course has no registered executable-domain destination "+lesson.href);
  const defaultPresets={building:"ffn",multimodal:"image_vit_patch"};
  return Object.freeze({schema:SCHEMA,id,kind:"curriculum",
   title:lesson.title,href:lesson.href,group:lesson.group,level:lesson.level,
   theory:lesson.theory,why:lesson.why,check:lesson.check,answer:lesson.answer,
   target:domain?Object.freeze({kind:"specialized",id:domain.id,href:domain.href}):
    Object.freeze({kind:"guided_graph",id:defaultPresets[id]||"ffn",href:"learn.html"}),
   evidence:"PEDAGOGICAL_DESCRIPTION",
   note:"Curriculum text links to an executable lesson; it never claims to have executed it."});
 }
 function inventory(ids=[]){
  const courses=deps.curriculum?.lessons?.map(l=>inspectCourse(l.id))||[];
  const graphLessons=ids.length?presetIds(ids):Object.freeze([]);
  const registered=bound()?deps.operators.inventory():null;
  if(registered&&registered.count!==deps.learning.KINDS.length)fail("operator catalog coverage mismatch");
  return Object.freeze({schema:SCHEMA,operators:registered,
   graphLessons:Object.freeze(graphLessons),courses:Object.freeze(courses),
   labs,counts:Object.freeze({operators:registered?.count??null,graphLessons:graphLessons.length,
    courses:courses.length,labs:labs.length})});
 }
 return Object.freeze({SCHEMA,labs,inspectGraph,openPreset,presetIds,activateGraphLessons,inspectLab,
  executeLab,inspectCourse,inventory});
});
