/* Architecture-first presentation model. No alternate graph executor or lab registry.
 * References existing curriculum/lab routes and source-defined operator categories.
 * Classifications are pedagogical lenses, not execution/verification claims. */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthArchitecturePerspective=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const LEVELS=Object.freeze([
  Object.freeze({id:"system",name:"系统目标",en:"SYSTEM / SLO",question:"工作负载、质量门槛与资源预算是什么？"}),
  Object.freeze({id:"architecture",name:"架构权衡",en:"DESIGN / TOPOLOGY",question:"如何划分控制面、数据面和执行边界？"}),
  Object.freeze({id:"runtime",name:"运行时与状态",en:"RUNTIME / STATE",question:"请求、资源和持久状态如何随时间变化？"}),
  Object.freeze({id:"operator",name:"计算内核",en:"KERNEL / MATH",question:"哪些张量或算子真正承担成本与功能？"}),
  Object.freeze({id:"evidence",name:"观测与验证",en:"TRACE / EVIDENCE",question:"怎样构造可证伪对照，并判别模拟与实测？"})
 ]);
 const roles=Object.freeze([
  Object.freeze({id:"all",name:"全部张量积木",explain:"保留完整自由组合算子库；以下角色仅改变浏览方式。"}),
  Object.freeze({id:"operator",name:"算子 / Kernel",explain:"逐坐标计算、形状、精度和反向传播；不是 CUDA kernel 性能证明。"}),
  Object.freeze({id:"representation",name:"表示 / 数据路径",explain:"Token、模态、位置和张量形状的变换与依赖。"}),
  Object.freeze({id:"runtime",name:"路由 / 缓存语义",explain:"张量图中的选择、索引和缓存子操作；完整 Scheduler、Worker 与内存状态机需要进入专项实验室。"}),
  Object.freeze({id:"training",name:"训练 / 优化",explain:"Loss、梯度及更新的图算子；分布式训练、恢复和 rollout 由独立状态实验刻画。"})
 ]);
 const roleByCategory=Object.freeze({
  tensor:"representation",data:"representation",vision:"representation",audio:"representation",
  video:"representation",generative:"representation",position:"representation",shape:"representation",
  memory:"runtime",inference:"runtime",routing:"runtime",sampling:"runtime",
  loss:"training",training:"training",regularize:"training",
  compute:"operator",norm:"operator",activate:"operator",combine:"operator",
  attention:"operator",reduce:"operator",gated:"operator"
 });
 function roleFor(kind,def){
  if(typeof kind!=="string"||!def||typeof def.category!=="string")
   throw Error("architecture lens requires registered operator identity and category");
  return roleByCategory[def.category]||"operator";
 }
 const TRACKS=Object.freeze([
  Object.freeze({id:"video",name:"视频模型稀疏",course:"research_video_sparse",
   goal:"降低实际 FFN 成本，同时维持视频时序与文本语义质量",
   metrics:"完整视频质量 / 运动与时序 / GPU 延迟、显存、吞吐",
   steps:Object.freeze([
    ["system","明确质量门槛和配对视频实验矩阵","课程：研究问题与终端质量目标","research_video_sparse","DESIGN_ONLY"],
    ["architecture","比较静态通道、逐 Token 和条件调度的成本可达性","解析成本：设计与模型开销","loop","ANALYTICAL_ONLY"],
    ["runtime","检查双专家、跨 Block 和稀疏执行的运行前提","Wan FFN 源码微观剖析（不是完整 Worker）","research_video_sparse","SOURCE_PINNED_SYNTHETIC"],
    ["operator","查看 e5 门控、GELU、W1/W2 的真实数学依赖","Wan FFN 逐坐标显微镜","research_video_sparse","SOURCE_PINNED_SYNTHETIC"],
    ["evidence","对照 Dense/Sparse 完整生成、运动和硬件性能","证据边界与代码能力审计","papers","PROPOSED_REAL_EVALUATION"]
   ])}),
  Object.freeze({id:"posttrain",name:"LLM 后训练",course:"research_llm_posttrain",
   goal:"设计可靠、可恢复并能证明能力提升的训练流水线",
   metrics:"任务成功率 / 偏好胜率 / KL 与长度偏置 / 训练成本",
   steps:Object.freeze([
    ["system","定义数据分布、能力目标和评价基准","研究问题：SFT、DPO 与 GRPO","research_llm_posttrain","DESIGN_ONLY"],
    ["architecture","组织采样、奖励、Reference/Old Policy 与参数更新","训练流水线：批次、调度、Checkpoint","engineering_pipeline","SYNTHETIC_NUMERIC"],
    ["runtime","检查梯度累积、恢复、状态和数据谱系","工程训练流程与续训","engineering_journey","SYNTHETIC_NUMERIC"],
    ["operator","比较 Token CE、偏好对数概率及 GRPO 梯度边界","后训练微型数学与反传","research_llm_posttrain","TOY_ANALYTIC"],
    ["evidence","验证中心差分、独立梯度及真实任务泛化","数学验证与实验方案","papers","PROPOSED_REAL_EVALUATION"]
   ])}),
  Object.freeze({id:"agent",name:"Agent 架构",course:"research_agent_arch",
   goal:"让工具编排具备可验证、可恢复、可审计的执行语义",
   metrics:"任务完成率 / 工具成功率 / 恢复率 / 成本与时延",
   steps:Object.freeze([
    ["system","定义工具权限、交付标准与故障预算","研究问题：Agent 任务与 SLO","research_agent_arch","DESIGN_ONLY"],
    ["architecture","选择 ReAct、Plan-Execute 或 Supervisor 的控制边界","脚本化 Agent 编排对照","research_agent_arch","SCRIPTED_STATE_MACHINE"],
    ["runtime","对照队列、资源和状态迁移；注意不同系统语义","状态实验室：版本与持久状态","research_agent_arch","SCRIPTED_STATE_MACHINE"],
    ["operator","进入具体操作：工具调用不是 Tensor MatMul","基础图：张量计算与状态区别","building","STATIC_GRAPH"],
    ["evidence","注入 Tool Read 失败、拒绝审批与验证故障","可重放事件链与后续真实工具验收","research_agent_arch","SCRIPTED_STATE_MACHINE"]
   ])})
 ]);
 function paths(curriculum){
  if(!curriculum||typeof curriculum.lookup!=="function")
   throw Error("architecture paths require the shared curriculum");
  return TRACKS.map(track=>Object.freeze({...track,steps:Object.freeze(track.steps.map(([level,question,artifact,course,evidence])=>{
   const lesson=curriculum.lookup(course);
   if(!lesson)throw Error("unknown shared course "+course);
   if(!LEVELS.some(l=>l.id===level))throw Error("unknown architecture layer");
   return Object.freeze({level,question,artifact,course,href:lesson.href,evidence});
  }))}));
 }
 return Object.freeze({VERSION:"2026.11.01",LEVELS,roles,roleFor,paths});
});
