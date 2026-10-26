/* Graph-derived learning route. No per-model assumptions or synthetic evidence promotion. */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?{
  L:require("./learning_lab.js"),S:require("./graph_session.js"),
  O:require("./operator_contracts.js"),K:require("./block_knowledge.js")
 }:{L:root.SeetheTruthLearning,S:root.SeetheTruthGraphSession,
  O:root.SeetheTruthOperatorContracts,K:root.SeetheTruthBlockKnowledge});
 if(node)module.exports=api;if(root)root.SeetheTruthGraphLearningPath=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
 function preview(value){const out=[];
  function visit(x){if(out.length>=6)return;if(Array.isArray(x)){for(const y of x)visit(y);}
   else if(typeof x==="number"&&Number.isFinite(x))out.push(x);}
  visit(value);return Object.freeze(out);
 }
 function build(raw,options={}){
  const s=options.session||d.S.inspect(raw,{numeric:options.numeric===true});
  if(s.schema!=="seethetruth-graph-session-v1")throw Error("invalid session");
  let g,index;
  try{g=d.L.validateGraph(raw);index=d.L.indexGraph(g);}
  catch(e){return Object.freeze({schema:"seethetruth-graph-learning-path-v1",
   executable:false,numericOk:false,firstError:s.readiness?.structuralError||e.message,
   evidence:"NO_EXECUTION",steps:Object.freeze([])});}
  const sessionNodes=s.nodes||s.readiness.nodes; // GraphSession numeric error returns readiness but no node projection.
  const nodeById=new Map(g.nodes.map(node=>[node.id,node]));
  const nodes=new Map(sessionNodes.map(n=>[n.id,n]));
  const trace=new Map((s.trace?.steps||[]).map(n=>[n.id,n]));
  const ir=new Map((s.ir?.nodes||[]).map(n=>[n.id,n]));
  const numericOk=!!(s.ir&&s.trace&&!s.error);
  const steps=sessionNodes.map((n,i)=>{
   const op=d.O.resolve(n.kind),card=d.K.get(n.kind);
   if(!card)throw Error("missing knowledge "+n.kind);
   const ports=op.ports.map(port=>{
    const sourceId=index.incomingByPort.get(n.id+":"+port)||null,up=nodes.get(sourceId);
    return Object.freeze({port,sourceId,sourceShape:up?.shape?[...up.shape]:null,
     sourceStatus:up?.status||"missing"});
   });
   const seen=ir.get(n.id),t=trace.get(n.id);
   const observed=!!(numericOk&&n.numericObserved&&seen?.valueAvailable&&t?.values!==undefined);
   if(observed&&JSON.stringify(t.shape)!==JSON.stringify(n.shape))
    throw Error("static/numeric mismatch "+n.id);
   return Object.freeze({ordinal:i+1,id:n.id,kind:n.kind,title:op.title,
    status:n.status,reason:n.reason||null,ports:Object.freeze(ports),
    shape:n.shape?[...n.shape]:null,formula:op.formula,
    why:card.intuition,flow:card.flow,practice:card.practice,trap:card.trap,exercise:card.exercise,
    params:Object.freeze({...nodeById.get(n.id).params}),
    evidence:observed?"SYNTHETIC_NUMERIC":"STATIC_ONLY",
    sample:observed?preview(t.values):null,
    numpyCandidate:op.backends.numpyEligible,torchCandidate:op.backends.torchEligible});
  });
  return Object.freeze({schema:"seethetruth-graph-learning-path-v1",
   executable:s.executable,numericOk,evidence:numericOk?"SYNTHETIC_NUMERIC":"STATIC_ONLY",
   firstError:s.error?.reason||s.readiness.firstError?.reason||null,
   steps:Object.freeze(steps)});
 }

 // Curriculum hints select actual source-graph nodes. Missing anchors fall back
 // to the current graph's real topology, never to a substitute computation.
 const guidedRoutes=Object.freeze({
  ffn:{focus:["up","act","down"],next:"attention",questions:[
   "为什么上投影要扩大中间通道？",
   "这条 FFN 的形状推断能否由实际张量计算支持？",
   "SiLU 改变了上投影中的哪些数值关系？",
   "下投影如何恢复输出通道，并将非线性带给后续层？"]},
  attention:{focus:["q","attn","proj"],next:"trainable_core",questions:[
   "Q、K、V 的信息分别来自哪里？",
   "Attention 的 Q/K/V 路径能否执行并产生可核对的输出？",
   "为什么查询与键的匹配会影响 Value 的汇合？",
   "Attention 输出投影如何接回 Transformer 的特征空间？"]},
  trainable_core:{focus:["shared_weight","project","loss"],next:"nd_wan_i2v_full_block",questions:[
   "同一份可训练参数如何被多个计算路径引用？",
   "当前参数与 Loss 之间是否具有真实可执行的前向路径？",
   "共享参数经过投影时，梯度由哪些下游路径贡献？",
   "Loss 怎样定义优化器实际更新的目标？"]},
  nd_wan_i2v_full_block:{focus:["after_self","ffn_gelu_tanh","after_ffn"],next:"image_vit_patch",questions:[
   "视频 Self-Attention 后，e2 门控残差保留了什么信息？",
   "源码顺序连接的 Self、Cross、FFN 是否形成可执行的同一计算图？",
   "条件调制后的 FFN 在 GELU(tanh) 处引入了什么非线性？",
   "e5 门控的 FFN 结果如何写回原有视频状态？"]},
  image_vit_patch:{focus:["patches","image_embedding","image_attend"],next:"audio_spectrogram",questions:[
   "如何把图像网格转化成可处理的 Patch Token？",
   "Patch Token 能否沿着实际积木图计算？",
   "可训练特征投影怎样改变每个 Patch 的表示？",
   "不同 Patch 为什么可以通过 Attention 交换信息？"]},
  audio_spectrogram:{focus:["spectrum","log_power","audio_attend"],next:"video_causal_conv3d",questions:[
   "Hann STFT 把时间序列变成了什么表征？",
   "时频特征能否沿着真实数值图传播？",
   "对数功率谱怎样改变频谱值之间的相对尺度？",
   "Attention 在时频 Token 表示上聚合什么信息？"]},
  video_causal_conv3d:{focus:["causal_conv","activate","temporal_attend"],questions:[
   "时间因果卷积怎样只读取当前位置及其历史帧？",
   "相同输入能否通过实际五维卷积教学算子？",
   "SiLU 对时空卷积输出产生哪些逐坐标变化？",
   "时间注意力如何继续组合卷积得到的特征？"]},
  nd_gqa_attention:{focus:["key_groups","probabilities","context"],questions:[
   "KV Head 如何被共享到更多 Query Head？",
   "四轴 QK 与 PV 路径能否在真实形状上执行？",
   "Softmax 如何在最后一个轴形成 Attention 权重？",
   "Value 的加权结果如何接到后续 Loss？"]}
 });
 const computeKinds=new Set(["input","nd_input","parameter","nd_parameter","token_ids"]);
 function firstSentence(value,fallback){
  const s=String(value||"").replace(/\s+/g," ").trim();
  if(!s)return fallback;
  const part=s.split(/[。！？]/)[0].slice(0,110);
  return part?part+(part.endsWith("?")||part.endsWith("？")?"":"。"):fallback;
 }
 function plan(raw,options={}){
  const lessonId=options.id||"free_graph";
  const route=build(raw);
  const byId=new Map(route.steps.map(x=>[x.id,x]));
  const actual=route.steps.filter(x=>!computeKinds.has(x.kind));
  const chosen=route.steps.length?actual.length?actual:route.steps:[];
  if(!chosen.length)return Object.freeze({schema:"seethetruth-learning-guide-v1",
   id:lessonId,steps:Object.freeze([]),next:null});
  const candidate=guidedRoutes[lessonId];
  const distinct=[],add=id=>{if(byId.has(id)&&!distinct.includes(id))distinct.push(id);};
  if(candidate&&candidate.focus.every(id=>byId.has(id)))
   candidate.focus.forEach(add);
  else {
   add(chosen[0].id);
   add(chosen[Math.floor((chosen.length-1)/2)].id);
   add(chosen[chosen.length-1].id);
  }
  if(!distinct.length)add(chosen[0].id);
  const up=byId.get(distinct[0]),focus=byId.get(distinct[Math.min(1,distinct.length-1)]),
    final=byId.get(distinct[distinct.length-1]);
  const source=step=>Object.freeze({nodeId:step.id,kind:step.kind,
   shape:step.shape?Object.freeze([...step.shape]):null,
   ports:Object.freeze(step.ports.map(x=>Object.freeze({
     name:x.port,sourceId:x.sourceId,sourceShape:x.sourceShape?
      Object.freeze([...x.sourceShape]):null}))),
   formula:step.formula});
  const questions=candidate?.questions;
  const steps=[
   {kind:"select",title:"从计算结构提出问题",node:up.id,
    question:questions?.[0]||"为什么需要 "+up.title+"，它的输入来自哪些节点？",
    instruction:"选中 "+up.id+"，查看真实连接、形状与计算公式。",proof:source(up)},
   {kind:"numeric",title:"用数值检验结构预测",node:null,
    question:questions?.[1]||"这张图的形状关系是否对应真实可执行的张量运算？",
    instruction:"切换小矩阵数值模式，点击「运行」，对照本图实际张量形状与数值。",proof:null},
   {kind:"numeric-select",title:"解释关键变换",node:focus.id,
    question:questions?.[2]||firstSentence(focus.why,
     focus.title+" 如何变换来自上游的数值？"),
    instruction:"定位 "+focus.id+"，用本次数值轨迹和公式解释变换；"+firstSentence(focus.exercise,"预测一个输出坐标的变化。"),
    proof:source(focus)},
   {kind:"numeric-select",title:"追踪下游计算",node:final.id,
    question:questions?.[3]||"为什么 "+final.title+" 出现在这条计算链的下游？",
    instruction:"定位 "+final.id+"，对比输入源、输出形状和本次实际计算。",proof:source(final)}
  ];
  // A one-node lesson has one meaningful selection plus a numeric observation.
  const pruned=distinct.length===1?steps.slice(0,2):
    distinct.length===2?[steps[0],steps[1],steps[3]]:steps;
  return Object.freeze({schema:"seethetruth-learning-guide-v1",
   id:lessonId,source:"graph_learning_path",executable:route.executable,
   next:candidate?.next||null,steps:Object.freeze(pruned.map((step,i)=>
    Object.freeze({...step,ordinal:i+1})))});
 }

 return Object.freeze({build,plan});
});
