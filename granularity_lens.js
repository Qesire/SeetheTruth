/* Orthogonal granularity lens for system-to-atomic drilldown.
 * Presentation references canonical course/lab/composite/operator identities.
 * No new engine, inference graph, unobserved event, or numeric equivalence.
 */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthGranularityLens=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const SCHEMA="seethetruth-granularity-lens-v1";
 const GRAINS=Object.freeze([
  Object.freeze({id:"system",code:"G0",name:"系统",question:"端到端目标、成本、可靠性与质量约束是什么？"}),
  Object.freeze({id:"subsystem",code:"G1",name:"子系统",question:"谁负责路由、数据、执行和持久状态？"}),
  Object.freeze({id:"component",code:"G2",name:"组件",question:"一个边界明确的模块如何接收输入、更新状态和产生输出？"}),
  Object.freeze({id:"operation",code:"G3",name:"算子 / 动作",question:"单步计算或工具动作遵循何种具体语义？"}),
  Object.freeze({id:"atomic",code:"G4",name:"坐标 / 事件",question:"哪个真实数值坐标或因果事件可以被定位和复核？"})
 ]);
 // Each entry: [grain, title, objective, targetKind, canonicalId, evidence,
 //               relationToParent, containmentClaim].
 // An educational route is not a proof that parent functionality is decomposed.
 const DATA=Object.freeze({
  video:Object.freeze([
   ["system","生成视频的端到端质量与算力","真实视频质量、运动、语义、GPU 成本是终局目标","course","research_video_sparse","DESIGN_ONLY","ROOT","NOT_ESTABLISHED"],
   ["subsystem","去噪与稀疏推理流水线","完整专家切换、Block 和 GPU 执行链尚需实测","course","research_video_sparse","PLANNED_RUNTIME","PEDAGOGICAL_ZOOM","NOT_ESTABLISHED"],
   ["component","Wan FFN（局部源码）","定位真实 FFN 数学依赖，不能代表整段视频","lab","wan_ffn","PINNED_SOURCE_SYNTHETIC","PEDAGOGICAL_ZOOM","PARTIAL_SUBSYSTEM"],
   ["operation","门控及通道贡献","检查 GELU、W1/W2、e5 门控后的贡献选择","lab","wan_ffn","PINNED_SOURCE_SYNTHETIC","SAME_LAB_REFOCUS","SOURCE_ALIGNED_LOCAL"],
   ["atomic","单 Token / Hidden 贡献","在有界合成张量上逐坐标验算，非视频帧证据","lab","wan_ffn","SYNTHETIC_COORDINATE","SAME_LAB_REFOCUS","SOURCE_ALIGNED_LOCAL"]
  ]),
  posttrain:Object.freeze([
   ["system","后训练能力与预算","定义泛化、奖励稳定性、任务质量与训练成本","course","research_llm_posttrain","DESIGN_ONLY","ROOT","NOT_ESTABLISHED"],
   ["subsystem","训练流水线与恢复","Shuffle → Microbatch → Update → Checkpoint；教学小模型","lab","engineering_pipeline","SYNTHETIC_NUMERIC","PEDAGOGICAL_ZOOM","ANALOGY_ONLY"],
   ["component","优化器与训练状态","在 CPU 小模型中检查梯度、权重和动量状态","lab","training","SYNTHETIC_NUMERIC","PEDAGOGICAL_ZOOM","ANALOGY_ONLY"],
   ["operation","AdamW 真实算子语义","由注册算子检查输入端口、更新公式和导出资格","operator","opt_adamw","REGISTERED_OPERATOR","PEDAGOGICAL_ZOOM","UNPROVEN_EQUIVALENCE"],
   ["atomic","参数坐标与训练事件","使用真实小图 VJP/训练状态显微镜按坐标观察","lab","training","BOUNDED_NUMERIC_TRACE","PEDAGOGICAL_ZOOM","ANALOGY_ONLY"]
  ]),
  agent:Object.freeze([
   ["system","Agent 任务成功与安全","定义交付质量、工具权限、错误预算和人工审批","course","research_agent_arch","DESIGN_ONLY","ROOT","NOT_ESTABLISHED"],
   ["subsystem","Planner / Executor / Verifier","教学演示 ReAct / Plan-Execute / Supervisor 的事件编排","research","agent","SCRIPTED_STATE_MACHINE","PEDAGOGICAL_ZOOM","SCRIPTED_MODEL"],
   ["component","工具操作与状态门禁","跟踪审批、写操作、失败重试和验证的状态条件","research","agent","SCRIPTED_STATE_MACHINE","SAME_SIMULATION_REFOCUS","SCRIPTED_MODEL"],
   ["operation","单次 Tool Call","观察脚本事件中的 action；不执行外部工具","research","agent","SCRIPTED_STATE_MACHINE","SAME_SIMULATION_REFOCUS","SCRIPTED_MODEL"],
   ["atomic","事件 ID / 依赖 / 结果","检查确定性事件序列与阻断条件；不虚构张量坐标","research","agent","SCRIPTED_STATE_MACHINE","SAME_SIMULATION_REFOCUS","SCRIPTED_MODEL"]
  ])
 });
 const COURSE_FOCUS=Object.freeze({
  building:"operation",papers:"subsystem",multimodal:"component",loop:"subsystem",
  cache:"component",prefill:"component",scheduler:"subsystem",worker:"component",
  engineering_journey:"system",engineering_pipeline:"subsystem",
  research_video_sparse:"system",research_llm_posttrain:"system",research_agent_arch:"system"
 });
 function curriculumIndex(curriculum){
  if(!curriculum||!Array.isArray(curriculum.lessons)||typeof curriculum.lookup!=="function")
   throw Error("granularity course index requires canonical curriculum");
  if(Object.keys(COURSE_FOCUS).length!==curriculum.lessons.length)
   throw Error("curriculum granularity coverage mismatch");
  return Object.freeze(curriculum.lessons.map(course=>{
   const grain=COURSE_FOCUS[course.id];
   if(!GRAINS.some(g=>g.id===grain)||!curriculum.lookup(course.id))
    throw Error("course missing granularity classification "+course.id);
   return Object.freeze({id:course.id,grain,code:GRAINS.find(g=>g.id===grain).code,
    title:course.title,sourceId:"course:"+course.id,href:course.href,
    note:"主讲解粒度；课程中可以跨层观察，不限制工作台探索"});
  }));
 }
 const nameKinds=Object.freeze({course:1,lab:1,operator:1,composite:1,research:1});
 const rootTracks=Object.freeze(["video","posttrain","agent"]);
 function resolve(track,grain,curriculum,core,extra={}){
  if(!rootTracks.includes(track))throw Error("unknown granularity track");
  const i=GRAINS.findIndex(g=>g.id===grain);
  if(i<0)throw Error("unknown granularity");
  if(!curriculum||typeof curriculum.lookup!=="function")throw Error("missing canonical curriculum");
  if(!core||typeof core.inspectLab!=="function")throw Error("missing unified learning core");
  const [_,title,objective,targetKind,id,evidence,relation,containment]=DATA[track][i];
  if(!nameKinds[targetKind])throw Error("unregistered granularity target kind");
  let href,sourceId;
  if(targetKind==="course"){
   const c=curriculum.lookup(id);if(!c)throw Error("missing course "+id);
   href="teaching.html?focus="+encodeURIComponent(c.id)+"#cards";sourceId="course:"+c.id;
  }else if(targetKind==="lab"){
   const l=core.inspectLab(id);if(l.kind!=="specialized"||!l.href||l.executable!==null)
    throw Error("invalid lab contract "+id);
   href=l.href;sourceId="lab:"+l.id;
  }else if(targetKind==="operator"){
   const d=extra.operators?.defs?.[id];
   if(!d||!extra.operators.KINDS.includes(id))throw Error("unknown registered operator "+id);
   href="learn.html?block="+encodeURIComponent(id);sourceId="operator:"+id;
  }else if(targetKind==="composite"){
   if(!extra.composites?.get(id))throw Error("unknown composite "+id);
   href="learn.html?composite="+encodeURIComponent(id);sourceId="composite:"+id;
  }else{
   // Existing teaching micro-experiment, not a new specialized lab.
   if(id!=="agent"||!curriculum.lookup("research_agent_arch"))throw Error("unknown research experiment");
   href="teaching.html#track-agent";sourceId="research:agent";
  }
  return Object.freeze({schema:SCHEMA,track,grain,code:GRAINS[i].code,title,objective,
   targetKind,sourceId,href,evidence,relation,containment,
   note:containment==="NOT_ESTABLISHED"||containment==="ANALOGY_ONLY"?
    "教学路径不等于父系统的执行分解或功能等价":"目标绑定到同一真实来源；不由缩放产生额外执行证据"});
 }
 function inspectTrack(track,curriculum,core,extra={}){
  return Object.freeze(GRAINS.map(g=>resolve(track,g.id,curriculum,core,extra)));
 }
 function navigation(track,grain,curriculum,core,extra={}){
  const all=inspectTrack(track,curriculum,core,extra),idx=GRAINS.findIndex(g=>g.id===grain);
  if(idx<0)throw Error("unknown granularity");
  return Object.freeze({current:all[idx],parent:idx?all[idx-1]:null,
   child:idx+1<all.length?all[idx+1]:null,path:all.slice(0,idx+1),
   // Explicitly no automatic parent↔child function equivalence certification.
   identityPreserved:all[idx].sourceId===all[0].sourceId,
   numericallyEquivalent:"NOT_CLAIMED"});
 }
 return Object.freeze({SCHEMA,GRAINS,rootTracks,COURSE_FOCUS,curriculumIndex,resolve,inspectTrack,navigation});
});
