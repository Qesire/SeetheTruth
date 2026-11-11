/* VLM teaching pack: bounded visual-language token plumbing on the shared
 * LearningLab DAG. The numerics are synthetic Float64, not pretrained inference.
 * The decoder Attention, mask, and language head remain ordinary native blocks.
 */
(function(root,factory){"use strict";const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthVLMBlockPack=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const SRC_CAT="https://pytorch.org/docs/stable/generated/torch.cat.html";
 const SRC_POOL="https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html";
 const SRC_VIEW="https://pytorch.org/docs/stable/tensor_view.html";
 const SRC_GRID="https://docs.pytorch.org/docs/stable/generated/torch.unravel_index.html";
 const def=(title,en,ports,params,formula,why,lesson,caution,source)=>({
  title,en,category:"multimodal",ports,params,formula,why,lesson,caution,source,
  outputAxes:["T","D"],
  knowledge:{flow:"图像 Patch Token 和文本 Embedding 具有不同来源；只有特征维度对齐后才可以组成有序 Token 序列。",
   practice:lesson,trap:"相同的特征宽度不证明语义对齐；物理拼接也不等于预训练的多模态能力。",
   exercise:"运行两个视觉前缀图，核对每个源 Token 的去向、因果 Mask 与最终文本行。",
   refs:"visual_instruction_tuning,torch_cat"}
 });
 const defs=Object.freeze({
  vlm_token_concat:def("视觉前缀与文本 Token 拼接","Vision-first Token Concatenation",
   ["visual","text"],{},"Y=concat(V_visual,E_text,axis=T)",
   "按视觉 Token → 文本 Token 的顺序逐行拼接；保留每一行的来源坐标。",
   "改变视觉数量，比较 Prefill 长度、后续 Attention Mask 和输出文本位置。",
   "两个输入必须有相同 D，T 总长最多 8；这是嵌入拼接，不会自行计算语义对齐。",SRC_CAT),
  vlm_visual_pair_pool:def("视觉 Token 两两平均压缩","Adjacent Visual Token Mean Pooling",
   ["visual"],{},"Y[j,d]=(V[2j,d]+V[2j+1,d])/2",
   "以固定的相邻两行均值减少视觉 Token 长度，用可复现数值研究分辨率与 Prefill 成本的取舍。",
   "改变同一对 Token 中的两个值，观察均值保留什么、抹去什么。",
   "要求偶数视觉 Token；固定顺序平均不是训练过的 Token 剪枝、动态分辨率或信息无损压缩。",SRC_POOL),
  vlm_text_tail:def("从多模态上下文提取文本行","Text Readout After Visual Prefix",
   ["context","visual"],{prefixTokens:4},"Y[t,d]=X[prefixTokens+t,d]",
   "按视觉前缀长度提取对应文本位置的上下文表示；让语言 Head 只消费文本行。",
   "移动前缀边界，检查输出行的原始绝对位置是否同步变化。",
   "必须显式提供视觉前缀长度，严格小于序列长度；不自动识别多模态特殊 Token。",SRC_VIEW),
  vlm_patch_grid_lookup:{
   ...def("视觉 Patch 序号还原二维坐标","Patch Index to Spatial Grid",["indices","visual"],{gridHeight:2,gridWidth:2},
    "id=y·gridWidth+x; y=floor(id/gridWidth), x=id mod gridWidth",
    "把每个文本 Query 选中的 Patch 行号转换成二维网格位置，保持 Patchify 的行优先顺序。",
    "改变文本 Token 与 Top-K 预算，观察所选 Patch 的二维位置如何变化。",
    "坐标基于已知 Patchify 网格，而不是检测框或语义定位证明；必须保持原始行优先 Patch 排列。",SRC_GRID),
   outputAxes:["Selection","Coordinate"],
   knowledge:{flow:"Top-K 的行是文本 Query，列是命中的视觉 Patch 序号；输出按 Query→选中位置次序展开 [y,x]。",
    practice:"将 2×2 Patch 网格对应的四个 Patch 编号映射回 y 和 x，再通过换 Query 观察选择位置的变化。",
    trap:"这些坐标只表示被模型注意力选中的 Patch；语义是否正确必须有独立的 Grounding 标注。",
    exercise:"人为置换输入 Patch 次序，保持网格标注不变，核验错误位置或重新建立合法索引映射。",
    refs:"vit_patch,torch_unravel_index"}
  }
 });
 const must=(ok,why)=>{if(!ok)throw Error("VLM block: "+why);};
 function validShape(s,name){must(Array.isArray(s)&&s.length===2&&
  s.every((x,i)=>Number.isSafeInteger(x)&&x>=1&&x<=(i===0?8:16)),name+" requires [T<=8,D<=16]");}
 function validate(kind,p){
  must(Object.hasOwn(defs,kind),"unknown VLM block "+kind);
  if(kind==="vlm_text_tail")
   must(Number.isSafeInteger(p.prefixTokens)&&p.prefixTokens>=1&&p.prefixTokens<=7,
    "prefixTokens must be integer 1..7");
  if(kind==="vlm_patch_grid_lookup"){
   must(Number.isSafeInteger(p.gridHeight)&&p.gridHeight>=1&&p.gridHeight<=8,
    "gridHeight must be an integer 1..8");
   must(Number.isSafeInteger(p.gridWidth)&&p.gridWidth>=1&&p.gridWidth<=8,
    "gridWidth must be an integer 1..8");
   must(p.gridHeight*p.gridWidth<=8,"grid exceeds eight visual Patch tokens");
  }
 }
 function infer(kind,p,inputs){
  validate(kind,p);
  if(kind==="vlm_token_concat"){
   const v=inputs.visual,t=inputs.text;validShape(v,"visual");validShape(t,"text");
   must(v[1]===t[1],"visual/text feature widths must match");
   must(v[0]+t[0]<=8,"combined visual/text token length exceeds 8");
   return [v[0]+t[0],v[1]];
  }
  if(kind==="vlm_visual_pair_pool"){
   const v=inputs.visual;validShape(v,"visual");must(v[0]%2===0,"visual pooling needs even token count");
   return [v[0]/2,v[1]];
  }
  if(kind==="vlm_patch_grid_lookup"){
   const indices=inputs.indices,v=inputs.visual;
   validShape(indices,"indices");validShape(v,"visual");
   must(v[0]===p.gridHeight*p.gridWidth,
    "Patch grid dimensions must match actual visual token count");
   must(indices[0]*indices[1]<=8,"selected Patch coordinates exceed eight teaching rows");
   return [indices[0]*indices[1],2];
  }
  const x=inputs.context,v=inputs.visual;validShape(x,"context");validShape(v,"visual");
  must(p.prefixTokens<x[0],"prefixTokens must leave at least one text token");
  must(v[0]===p.prefixTokens,"prefixTokens must match connected visual token count");
  must(v[1]===x[1],"visual/context feature widths must match");
  return [x[0]-p.prefixTokens,x[1]];
 }
 const dims=v=>[v.length,v[0].length],flatten=v=>v.flat();
 function execute(kind,p,inputs){
  const expected=infer(kind,p,Object.fromEntries(Object.entries(inputs).map(([k,v])=>[k,dims(v)])));
  let values,internals;
  if(kind==="vlm_token_concat"){
   values=[...inputs.visual.map(row=>row.slice()),...inputs.text.map(row=>row.slice())];
   internals={visualTokenRange:[0,inputs.visual.length],textTokenRange:[inputs.visual.length,values.length],
    sourceOrder:["visual","text"]};
  }else if(kind==="vlm_visual_pair_pool"){
   values=Array.from({length:expected[0]},(_,i)=>inputs.visual[2*i].map((v,d)=>
    (v+inputs.visual[2*i+1][d])/2));
   internals={groups:Array.from({length:expected[0]},(_,i)=>[2*i,2*i+1]),weights:[0.5,0.5]};
  }else if(kind==="vlm_patch_grid_lookup"){
   const ids=inputs.indices.flat();
   values=ids.map(id=>{
    must(Number.isSafeInteger(id)&&id>=0&&id<p.gridHeight*p.gridWidth,
     "Patch index outside selected image Patch grid");
    return [Math.floor(id/p.gridWidth),id%p.gridWidth];
   });
   internals={grid:[p.gridHeight,p.gridWidth],selectedPatchIds:ids.slice(),
    layout:"row-major image Patch grid, zero-based [y,x]"};
  }else{
   values=inputs.context.slice(p.prefixTokens).map(row=>row.slice());
   internals={prefixTokens:p.prefixTokens,sourceRows:Array.from({length:values.length},(_,i)=>p.prefixTokens+i)};
  }
  must(values.length===expected[0]&&values.every(r=>r.length===expected[1]&&r.every(Number.isFinite)),
   "computed values violate shape/finite contract");
  return {values,internals};
 }
 function trace(kind,p,inputs,run,node){
  const out={shape:dims(run.values),values:flatten(run.values)};
  const operands=Object.fromEntries(Object.entries(inputs).map(([port,v])=>[port,{shape:dims(v),values:flatten(v)}]));
  let indexMap=null;
  if(kind==="vlm_token_concat"){
   const vd=inputs.visual[0].length,vt=inputs.visual.length;
   indexMap=out.values.map((_,i)=>i<vt*vd?{port:"visual",index:i}:{port:"text",index:i-vt*vd});
  }else if(kind==="vlm_text_tail"){
   const D=out.shape[1];
   indexMap=out.values.map((_,i)=>({port:"context",index:p.prefixTokens*D+i}));
  }
  // Provenance of the selected grid position: Query row, rank, original Patch.
  // A spatial index is an observed choice, not evidence of semantic grounding.
  const selectedRows=kind==="vlm_patch_grid_lookup"
   ?run.values.map(([y,x],flatIndex)=>{
     const k=inputs.indices[0].length,query=Math.floor(flatIndex/k),rank=flatIndex%k;
     return {query,rank,patchId:inputs.indices[query][rank],y,x};
   }):null;
  return [{id:node.id+":vlm",kind:"multimodal_token_transform",title:defs[kind].title,
   formula:defs[kind].formula,why:defs[kind].why,
   out,operands,outputIsParent:true,...(selectedRows?{selectedRows}:{}),...(indexMap?{indexMap}:{})}];
 }
 const N=(id,kind,x,y,params={})=>({id,kind,x,y,params});
 const E=(from,to,port)=>({from,to,port});
 function early(compressed){
  const nodes=[
   N("pixels","input",20,30,{rows:4,width:4,seed:1}),
   N("patches","vision_patchify",280,30),
   N("vision_projector","linear",530,30,{out:4}),
   N("prompt_ids","token_ids",30,360,{rows:2,vocab:8,seed:4}),
   N("text_embedding","embedding",290,360,{width:4,vocab:8}),
   N("prefix","vlm_token_concat",830,180),
   N("query","linear",1100,20,{out:4}),
   N("key","linear",1100,185,{out:4}),
   N("value","linear",1100,355,{out:4}),
   N("key_transpose","transpose2d",1350,200),
   N("scores","matmul",1570,80),
   N("scaled","scale",1785,80,{factor:0.5}),
   N("mask","causal_mask",1785,335),
   N("probabilities","masked_softmax",2010,110),
   N("context","weighted_sum",2250,155),
   N("text_context","vlm_text_tail",2490,155,{prefixTokens:compressed?2:4}),
   N("language_head","linear",2730,155,{out:8})
  ];
  const edges=[
   E("pixels","patches","x"),E("patches","vision_projector","x"),
   E("prompt_ids","text_embedding","ids"),E("text_embedding","prefix","text"),
   E("prefix","query","x"),E("prefix","key","x"),E("prefix","value","x"),
   E("key","key_transpose","x"),E("query","scores","a"),E("key_transpose","scores","b"),
   E("scores","scaled","x"),E("scores","mask","x"),
   E("scaled","probabilities","logits"),E("mask","probabilities","mask"),
   E("probabilities","context","weights"),E("value","context","values"),
   E("context","text_context","context"),E("text_context","language_head","x")
  ];
  if(compressed){
   nodes.push(N("visual_pool","vlm_visual_pair_pool",780,0));
   edges.push(E("vision_projector","visual_pool","visual"),
    E("visual_pool","prefix","visual"),E("visual_pool","text_context","visual"));
  }else edges.push(E("vision_projector","prefix","visual"),
   E("vision_projector","text_context","visual"));
  return {title:compressed?"VLM · 视觉 Token 平均压缩后的单序列 Prefill":"VLM · 视觉前缀 + 文本 Token 的因果 Prefill",
   nodes,edges};
 }
 const cross={title:"VLM · 文本 Query 读取视觉 Key/Value 的 Cross-Attention",
  nodes:[N("pixels","input",30,30,{rows:4,width:4,seed:2}),
   N("patches","vision_patchify",270,30),N("visual_projector","linear",510,30,{out:4}),
   N("prompt_ids","token_ids",35,360,{rows:2,vocab:8,seed:5}),
   N("text_embedding","embedding",285,360,{width:4,vocab:8}),
   N("cross_attention","attention",800,160),N("text_residual","add",1050,160),
   N("language_head","linear",1300,160,{out:8})],
  edges:[E("pixels","patches","x"),E("patches","visual_projector","x"),
   E("prompt_ids","text_embedding","ids"),E("text_embedding","cross_attention","q"),
   E("visual_projector","cross_attention","k"),E("visual_projector","cross_attention","v"),
   E("cross_attention","text_residual","a"),E("text_embedding","text_residual","b"),
   E("text_residual","language_head","x")]};
 const patchSelection={
  title:"VLM · 文本条件视觉 Patch 选择与 Dense 读出误差",
  nodes:[
   N("pixels","input",20,40,{rows:4,width:4,seed:3}),
   N("patches","vision_patchify",280,40),
   N("visual_projector","linear",540,40,{out:4}),
   N("prompt_ids","token_ids",20,370,{rows:2,vocab:8,seed:7}),
   N("text_embedding","embedding",285,370,{width:4,vocab:8}),
   N("visual_transpose","transpose2d",790,20),
   N("query_patch_scores","matmul",1010,170),
   N("scale_scores","scale",1240,170,{factor:0.5}),
   N("top_patches","inference_topk",1470,170,{topk:2}),
   N("patch_coordinates","vlm_patch_grid_lookup",1730,455,{gridHeight:2,gridWidth:2}),
   N("dense_visual","attention",1330,0),
   N("sparse_visual","inference_sparse_attention",1760,130,{causal:false,queryOffset:0}),
   N("readout_delta","subtract",2010,0),
   N("readout_error","mse",2260,95),
   N("text_residual","add",2000,340),
   N("language_head","linear",2260,340,{out:8})
  ],
  edges:[
   E("pixels","patches","x"),E("patches","visual_projector","x"),
   E("prompt_ids","text_embedding","ids"),
   E("visual_projector","visual_transpose","x"),
   E("text_embedding","query_patch_scores","a"),E("visual_transpose","query_patch_scores","b"),
   E("query_patch_scores","scale_scores","x"),
   E("scale_scores","top_patches","scores"),
   E("top_patches","patch_coordinates","indices"),E("visual_projector","patch_coordinates","visual"),
   E("text_embedding","dense_visual","q"),
   E("visual_projector","dense_visual","k"),E("visual_projector","dense_visual","v"),
   E("text_embedding","sparse_visual","q"),
   E("visual_projector","sparse_visual","k"),E("visual_projector","sparse_visual","v"),
   E("top_patches","sparse_visual","indices"),
   E("dense_visual","readout_delta","a"),E("sparse_visual","readout_delta","b"),
   E("sparse_visual","readout_error","prediction"),E("dense_visual","readout_error","target"),
   E("sparse_visual","text_residual","a"),E("text_embedding","text_residual","b"),
   E("text_residual","language_head","x")
  ]
 };
 const recipes=Object.freeze([
  {id:"extension_vlm_early_fusion",title:"VLM · 视觉前缀与因果语言解码",
   summary:"Patchify → Projector → [Image, Text] → QK/Mask/PV → 文本行 Logits；逐 Token 检查来源和因果性",
   graph:early(false)},
  {id:"extension_vlm_compressed_prefix",title:"VLM · 视觉 Token 压缩对照",
   summary:"对相邻视觉 Token 取均值，减少 Prefill 行数；真实对照信息损失、Mask 规模与文本读出",
   graph:early(true)},
  {id:"extension_vlm_cross_attention",title:"VLM · 文本到图像 Cross-Attention",
   summary:"文本 Query 读取独立视觉 Key/Value，再与原文本残差相加；与视觉前缀路径对照",
   graph:cross},
  {id:"extension_vlm_query_patch_selection",title:"VLM · 文本条件视觉 Patch 选择",
   summary:"真实 QK 分数 → Top-K Patch → 二维坐标 → 稀疏 / Dense 注意力差分及 Loss；保留数学误差证据",
   graph:patchSelection}
 ]);
 return Object.freeze({name:"vlm",defs,validate,infer,execute,trace,recipes});
});
