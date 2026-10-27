/* Coordinate-focus drilldown into *native* free-graph lineage/VJP engines.
 * The accepted focus is an address. Exact values and gradients are obtained
 * only by replaying the graph's own executed trace.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const d=isNode?{
  Focus:require("./coordinate_focus.js"),L:require("./learning_lab.js"),
  E:require("./element_lineage.js"),V:require("./coordinate_vjp_bridge.js"),
  M:require("./compute_microscope.js"),Ready:require("./graph_readiness.js"),
  Core:require("./learning_experience_core.js")
 }:{
  Focus:root.SeetheTruthCoordinateFocus,L:root.SeetheTruthLearning,
  E:root.SeetheTruthElementLineage,V:root.SeetheTruthCoordinateVJP,
  M:root.SeetheTruthComputeMicroscope,Ready:root.SeetheTruthGraphReadiness,
  Core:root.SeetheTruthUnifiedCore
 };
 const api=factory(d);
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthCoordinateDrilldown=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
 const SCHEMA="seethetruth-coordinate-drilldown-v1";
 const fail=why=>{throw Error("coordinate drilldown: "+why);};
 const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 function inspect(input,context={}){
  if(!d.Focus)fail("coordinate focus engine unavailable");
  if(!input||input.schema!==d.Focus.SCHEMA)fail("a verified focus record is required");
  const focus=d.Focus.create(input);
  // A separate state scenario is paired with a graph at the operator level.
  // Cross-sample node IDs do not imply identical scalar values or VJPs.
  if(focus.domain==="semantic_state")
   return Object.freeze({schema:SCHEMA,scope:"mechanism_correspondence",focus,
    lineage:null,vjp:null,complete:null,
    note:"状态实验与教学计算图是独立数值执行；请在课程图上选择一个真实输出坐标后检验其局部导数。"});
  if(focus.domain==="training_parameter"){
   if(!focus.transition)fail("training coordinate requires verified native optimizer transition");
   return Object.freeze({schema:SCHEMA,scope:"native_training_transaction",focus,
    lineage:null,vjp:null,complete:null,
    note:"原生参数更新 "+focus.transition.before+" → "+focus.transition.after+
      "；全 Loss 梯度 "+focus.transition.wholeLossGradient+
      "。局部路径贡献以训练坐标证明面板中的真实 VJP 项为准。"});
  }
  // Specialist state and training results have already been verified by
  // their own native engines and do not require a browser graph runtime.
  if(!d.E||!d.L||!d.Ready||!d.V||!d.M)
   fail("native graph coordinate engines unavailable");
  const raw=context.graph;
  if(!raw||!Array.isArray(raw.nodes)||!Array.isArray(raw.edges))fail("current graph required");
  const graph=d.L.validateGraph(raw),node=graph.nodes.find(n=>n.id===focus.nodeId);
  if(!node)fail("focused node not in current graph");
  const readiness=d.Ready.check(graph);
  const status=readiness.nodes.find(n=>n.id===focus.nodeId);
  if(status?.status!=="ready"||!same(status.shape,focus.shape))
   fail("coordinate does not match current graph shape");
  let run=null,frame=null;
  if(focus.value!==undefined){
   if(focus.domain!=="computation_microscope"&&focus.domain!=="unified_observation")
    fail("numerical graph focus must come from a native observed result");
   run=context.run;frame=context.frame;
   if(!run||!frame||run.mode==="concept"||
      frame.evidenceTier!==d.M.TIERS.MICRO)fail("current bounded numeric graph execution required");
   const step=frame.steps.find(s=>s.id===focus.nodeId);
   if(!step||!same(step.out.shape,focus.shape))fail("selected numeric node does not match execution");
   if(focus.domain==="computation_microscope"){
    const source=d.Focus.fromMicroscope(frame,step,focus.coordinate,focus.value);
    if(source.executionId!==focus.executionId||!same(source.coordinate,focus.coordinate)||
       !same(source.shape,focus.shape))fail("microscope execution identity changed");
   }else{
    if(!context.accepted||!d.Core||typeof d.Core.inspectObservationStep!=="function")
     fail("accepted unified observation record required");
    const observed=d.Core.inspectObservationStep(context.accepted,{
     kind:"tensor_computation",stepId:focus.nodeId,coordinate:focus.coordinate});
    const original=d.Focus.fromObservation(context.accepted,observed);
    if(original.executionId!==focus.executionId||original.value!==focus.value||
       !same(original.coordinate,focus.coordinate))fail("unified observation proof is stale");
   }
  }else if(focus.domain!=="guided_graph"||focus.evidence!==d.Focus.TIERS.STATIC)
   fail("structural graph drilldown requires a static graph focus");
  const lineage=d.E.inspect(graph,{targetId:focus.nodeId,outputIndex:focus.coordinate,
   ...(run?{trace:run}:{}),maxVertices:512,maxEdges:2048});
  let vjp=null;
  if(run){
   vjp=d.V.inspect(graph,run,frame,{stepId:focus.nodeId,coordinate:focus.coordinate});
   const target=lineage.vertices.find(v=>v.key===focus.nodeId+"@"+focus.coordinate.join(","));
   if(lineage.mode!=="bounded-numeric"||!target||target.value!==focus.value||
      vjp.selected.value!==focus.value||!same(vjp.selected.index,focus.coordinate))
    fail("native forward, graph lineage, and VJP disagree on selected value");
  }
  return Object.freeze({schema:SCHEMA,scope:run?"native_numeric_graph":"native_structural_graph",
   focus,lineage,vjp,complete:lineage.complete,
   note:run
    ?"单位输出种子下的真实微型图依赖与局部 VJP；损失梯度仅在原生 loss 验证明确成功时成立。"
    :"仅检查当前真实图的逐坐标形状和依赖；暂无数值与局部导数。"});
 }
 return Object.freeze({SCHEMA,inspect});
});
