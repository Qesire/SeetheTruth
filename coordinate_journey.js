/* Native coordinate explanation timeline.
 * Consume element_lineage + coordinate_vjp_bridge outputs without re-evaluating
 * the model. Unit-seed sensitivities are accumulated in DAG order and checked
 * against the native engine's recorded sensitivities.
 */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthCoordinateJourney=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const SCHEMA="seethetruth-coordinate-journey-v1",MAX_VERTICES=8192,MAX_EDGES=32768;
 const fail=m=>{throw Error("coordinate journey: "+m);};
 const close=(a,b)=>Math.abs(a-b)<=2e-9+1e-8*Math.max(Math.abs(a),Math.abs(b));
 const valid=x=>x===null||typeof x==="number"&&Number.isFinite(x);
 const immutable=o=>Object.freeze(o);
 function fromDrilldown(report){
  if(!report||report.schema!=="seethetruth-coordinate-drilldown-v1"||!report.focus)
   fail("accepted coordinate drilldown required");
  if(report.scope==="mechanism_correspondence")
   return immutable({schema:SCHEMA,scope:report.scope,focus:report.focus,
    steps:immutable([{kind:"correspondence",title:"机制与执行实例",detail:report.note,nodeId:null}]),
    complete:null});
  if(report.scope==="native_training_transaction")return training(report.focus);
  if(!["native_structural_graph","native_numeric_graph"].includes(report.scope))
   fail("unsupported drilldown scope");
  const g=report.lineage,focus=report.focus;
  if(g?.schema!=="seethetruth-element-lineage-v1"||!Array.isArray(g.vertices)||
     !Array.isArray(g.edges)||g.vertices.length<1||g.vertices.length>MAX_VERTICES||
     g.edges.length>MAX_EDGES||g.targetId!==focus.nodeId||
     JSON.stringify(g.outputIndex)!==JSON.stringify(focus.coordinate))
   fail("native element lineage does not match selected coordinate");
  if((report.scope==="native_numeric_graph")!==(g.mode==="bounded-numeric")||
     (report.scope==="native_numeric_graph")!==!!report.vjp)
   fail("numeric and structural evidence tiers disagree");
  const key=focus.nodeId+"@"+focus.coordinate.join(",");
  const nodes=new Map(g.vertices.map(v=>[v.key,v]));
  if(nodes.size!==g.vertices.length||!nodes.has(key))fail("duplicate or absent coordinate node");
  const outgoing=new Map(),incoming=new Map(g.vertices.map(v=>[v.key,0]));
  for(const edge of g.edges){
   if(!nodes.has(edge.from)||!nodes.has(edge.to)||!valid(edge.localDerivative))
    fail("malformed native dependency edge");
   if(!outgoing.has(edge.from))outgoing.set(edge.from,[]);
   outgoing.get(edge.from).push(edge);
   incoming.set(edge.to,incoming.get(edge.to)+1);
  }
  const roots=[...incoming].filter(([,count])=>count===0).map(([id])=>id);
  if(roots.length!==1||roots[0]!==key)fail("lineage must have one selected output seed");
  const acc=new Map([[key,1]]),initial=immutable({key,kind:"seed",title:"输出坐标 · 单位种子",
   nodeId:focus.nodeId,index:immutable([...focus.coordinate]),value:focus.value??null,
   sensitivity:1,detail:report.scope==="native_numeric_graph"
    ?"已选择原生数值输出，反向传播从该坐标单位种子 1 开始"
    :"形状依赖以单位种子 1 表示结构上的反向传播起点"});
  const steps=[initial],queue=[key],ordered=[];
  const numeric=report.scope==="native_numeric_graph";
  while(queue.length){
   const id=queue.shift(),v=nodes.get(id),before=acc.get(id);
   if(!v||before===undefined)fail("lineage contains unreachable nodes");
   if(!valid(before)||!valid(v.sensitivity))fail("invalid sensitivity");
   if(numeric&&before!==null&&v.sensitivity!==null&&!close(before,v.sensitivity)||
      before===null&&v.sensitivity!==null||before!==null&&v.sensitivity===null)
    fail("accumulated sensitivity differs from native element lineage at "+id);
   ordered.push(id);
   if(id!==key)steps.push(immutable({kind:"merge",title:(incoming.get(id)===0?"输出":"汇合")+" · "+v.id,
    key:id,nodeId:v.id,index:immutable([...v.index]),value:v.value,
    sensitivity:before,rule:v.rule??null,
    detail:v.status==="boundary"?"算子依赖规则存在精确边界；未知局部导数保留为未求值":
     before===null?"上游含尚未求值的局部导数":
     "已按所有进入此元素的依赖边汇合灵敏度"}));
   for(const edge of outgoing.get(id)||[]){
    const old=acc.get(edge.to),term=before===null||edge.localDerivative===null?
     null:before*edge.localDerivative;
    const total=old===undefined?term:old===null||term===null?null:old+term;
    if(!valid(total)||!valid(term))fail("nonfinite path contribution");
    acc.set(edge.to,total);
    const source=nodes.get(edge.to);
    steps.push(immutable({kind:"edge",title:v.id+" → "+source.id,
     nodeId:source.id,sourceNodeId:v.id,sourceKey:id,targetKey:edge.to,
     index:immutable([...source.index]),port:edge.port,term:edge.term,
     rule:edge.rule,upstreamSensitivity:before,localDerivative:edge.localDerivative,
     contribution:term,accumulatedBefore:old===undefined?null:old,
     accumulatedAfter:total,firstArrival:old===undefined,
     detail:term===null?"局部导数尚无数值证据；未求值不同于零":
       "上游灵敏度 × 该边局部导数 = 本路径贡献；同一元素的多个路径贡献相加"}));
    incoming.set(edge.to,incoming.get(edge.to)-1);
    if(incoming.get(edge.to)===0)queue.push(edge.to);
   }
  }
  if(ordered.length!==nodes.size||steps.length!==nodes.size+g.edges.length)
   fail("native lineage has a cycle or disconnected coordinate dependency");
  if(report.vjp){
   if(report.vjp.selected.nodeId!==focus.nodeId||
      JSON.stringify(report.vjp.selected.index)!==JSON.stringify(focus.coordinate)||
      report.vjp.selected.value!==focus.value||
      report.vjp.backward.nodeCount!==nodes.size||report.vjp.backward.edgeCount!==g.edges.length)
    fail("native VJP and element graph disagree");
  }
  return immutable({schema:SCHEMA,scope:report.scope,focus,
   complete:g.complete,steps:immutable(steps),
   sourceCount:g.vertices.filter(v=>v.status==="source").length,
   boundaryCount:g.frontiers?.length||0,
   note:report.note});
 }
 function training(focus){
  const tr=focus?.transition;
  if(focus?.domain!=="training_parameter"||!tr||!valid(tr.before)||!valid(tr.after)||
     !valid(tr.delta)||!valid(tr.wholeLossGradient)||tr.before===null||
     tr.after===null||tr.delta===null||tr.wholeLossGradient===null)
   fail("verified training transition required");
  const base={nodeId:focus.nodeId,index:immutable([...focus.coordinate])};
  const steps=[
   immutable({kind:"parameter_before",...base,title:"参数坐标 · 更新前",value:tr.before,
    detail:"原生训练事务中的初始参数值"}),
   immutable({kind:"loss_gradient",...base,title:"全 Loss 梯度",
    value:tr.wholeLossGradient,detail:"从完整训练目标对该参数求导；区别于任意单条中间路径的贡献"}),
  ];
  if(tr.optimizerKind)
   steps.push(immutable({kind:"optimizer_state",...base,title:"优化器状态 · "+tr.optimizerKind,
    value:null,detail:tr.optimizerNote||"原生优化器计算已完成；参数位移以事务输出为准",
    moment1:tr.moment1||null,moment2:tr.moment2||null}));
  steps.push(immutable({kind:"parameter_after",...base,title:"参数坐标 · 更新后",
   value:tr.after,delta:tr.delta,detail:"参数更新前后差值由原生训练事务提供"}));
  if(tr.lossBefore!==undefined&&tr.lossAfter!==undefined)
   steps.push(immutable({kind:"loss_transition",...base,title:"训练目标 · 提交前后",
    value:tr.lossAfter,lossBefore:tr.lossBefore,
    detail:"目标值来自完整训练事务，不能归因于单独一个参数坐标的更新"}));
  if(tr.selectedPathStatus)
   steps.push(immutable({kind:"selected_path",...base,title:"指定路径 · 局部贡献",
    value:tr.selectedPathContribution??null,status:tr.selectedPathStatus,
    detail:"选定中间坐标到该参数的路径贡献；全 Loss 梯度还可能包含其他路径"}));
  return immutable({schema:SCHEMA,scope:"native_training_transaction",focus,
   complete:true,steps:immutable(steps)});
 }
 return immutable({SCHEMA,fromDrilldown});
});
