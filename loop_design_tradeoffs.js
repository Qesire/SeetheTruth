/* Evidence-bounded "WHY" layer for architecture decisions.
 * Derived numbers are exact for the validated miniature DAG, never model
 * throughput, CPU wall-clock timing, GPU memory or benchmark accuracy.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthDesignTradeoffs=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const Loop=typeof module==="object"&&module.exports?require("./loop_transformer.js"):root.SeetheTruthLoopTransformer;
const SOURCES=Object.freeze({
 ouro:"https://arxiv.org/abs/2510.25741",
 ouro_code:"https://github.com/Ouro-LLM/Ouro",
 huginn:"https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py",
 recipes:"https://arxiv.org/abs/2610.00673",
 loopformer:"https://arxiv.org/abs/2602.11451",
 loopformer_code:"https://github.com/armenjeddi/loopformer/blob/master/train.py",
 mor:"https://arxiv.org/abs/2507.10524",
 hyperloop:"https://arxiv.org/abs/2604.21254",
 residual:"https://arxiv.org/abs/2606.18524",
 deeploop:"https://arxiv.org/abs/2607.13491"
});
const DECISIONS=Object.freeze({
 residual_scaling:Object.freeze({
 title:"为什么共享循环 Transformer 的残差要按循环次数缩放？",
 goal:"在共享同一个 Transformer Block 多次执行时，控制残差更新沿循环深度累加造成的状态和梯度放大；保持参数存储规模不变",
 mechanism:"本实验对每次 Attention/FFN 的残差分别计算 H←H+ε·Fθ(H)。比较 ε∈{1,1/√N,1/N}，N=2；原始主路径和全部 Parameter ID 恒定。梯度经 Scale 的系数与共享参数的所有访问次数求和",
 alternative:"ε=1（不缩放）、传统 1/√N 缩放、与循环长度相关的 1/N；DeepLoop 则采用 Post-LN DeepNorm 的 α=√(2N), β=1/√(8N)，不是直接给 Pre-Norm 输出乘 ε",
 cost:"1/N 可能压低单轮有效特征更新而减慢表示变化；1/√N 可能仍不足以抑制共享访问的相关累积；更多循环仍带来更多矩阵计算和训练激活",
 failure:"任务需要大的单轮非线性跳跃、梯度相关性不显著，或模型已有稳健深度缩放/归一化时，统一 1/N 未必优于对照；本教学实验不能推断大模型的收敛改进",
 test:"同一初始 E/Wq/Wk/Wv/Wo/Wup/Wgate/Wdown 与同一 Token/监督，对比 ε=1、1/√2、1/2：逐轮 CE、状态变化 RMS、Wq 梯度范数；再做有限差分、JS/PyTorch Autograd、AdamW 续训；论文的验证集收益需真实训练",
 sources:[["On the Residual Scaling of Looped Transformers (2026)",SOURCES.residual],["DeepLoop: alternative Post-LN α/β scaling (2026)",SOURCES.deeploop]]
 }),
 sharing:Object.freeze({
 title:"为什么跨深度共享权重，而不直接堆更多不同层？",
 goal:"在独立权重参数量／模型加载内存受限时，增加同一 Token 的潜空间推理步骤",
 mechanism:"Hᵣ₊₁=Fθ(Hᵣ)，所有轮次消费同一组 Parameter。BPTT 将各次调用的梯度汇合至唯一 θ，AdamW 仅保留一套状态",
 alternative:"不共享权重的同深度 Transformer：每层独立参数，允许更高知识存储容量与逐层特化；参数匹配的浅层 Transformer：更少串行计算",
 cost:"同深度的 GEMM 数量并没有因为权重共享减少；循环步骤增加训练激活、反向工作量、推理串行深度。不能把省参数宣传成推理加速",
 failure:"若任务主要受事实记忆／模型容量而非逐步算法推理限制，或延迟/算力而非显存是主要瓶颈，共享循环可能不合适；论文结果不构成通用保证",
 test:"在同数据和等训练 FLOPs 下比较：同参数浅层、同深度不共享层、共享循环；分别报告推理 FLOPs、tokens/s、知识题与算法题",
 sources:[["Ouro: reasoning vs knowledge capacity",SOURCES.ouro],["Closing the Loop: compute/data controlled comparison",SOURCES.recipes]]
 }),
 multidepth:Object.freeze({
 title:"为什么每一轮都给监督，而不只优化最深一轮？",
 goal:"让浅循环也能产生可用的 logits；在预算被迫提前结束时维持可解释的学习信号",
 mechanism:"每步 CEr 以 αr≥0 加权组成 L=Σαr·CEr；梯度在同一共享参数上汇合。改变 α 会改变优化目标，不能将其视为固定的深度等价变换",
 alternative:"仅末轮交叉熵监督：与最终最大深度推理目标更一致，浅轮输出不受直接约束；随机深度训练：可暴露可变预算，但需要额外采样分布",
 cost:"中间 LM Head / CE 增加训练端算子与监督计算；浅层损失可能与最深层表示目标竞争",
 failure:"中间表示未准备好预测输出时，强行每轮监督可能限制表征细化；极端 α 可能导致浅层主导或最终深度学不到",
 test:"相同初始状态对比末轮 CE-only、等权 CE、深度递增 CE；绘制每轮 loss 及共享参数的梯度方向相似度",
 sources:[["Ouro: latent reasoning and learned depth",SOURCES.ouro],["Closing the Loop: exit loss / regularization",SOURCES.recipes]]
 }),
 consistency:Object.freeze({
 title:"为什么用深层 Detach 教师指导浅层，而不是仅加两个 CE？",
 goal:"将较深预算取得的隐藏表征作为短预算路径的连续目标，减少短路径表示漂移",
 mechanism:"L=Ldepth+λ·MSE(Hshort, stop_gradient(Hdeep))；教师数值真实参与 MSE，但不沿一致性目标更新教师分支。共享参数仍可从浅路径接收梯度",
 alternative:"双深度 CE-only 不直接约束隐藏表征；无 Detach 对称 MSE 同时推动教师与学生，可能出现目标互相迁移；真正 LoopFormer 使用不同长度且经时间/步长条件的轨迹",
 cost:"多一个 MSE 目标、额外梯度；只做相同轨迹两轮状态匹配不等同于不同步长的短程一致性训练，可能反而逼近过早收敛",
 failure:"较深状态本身不比浅状态更好、各步表征处于不同语义阶段，或共享参数使深层目标不稳定时，MSE 可能伤害主任务",
 test:"固定 CPU 随机种子比较 λ=0/0.1。数值梯度检验必须固定深层教师张量，只扰动浅层学生及主任务路径；若同时重算教师得到的是另一目标的全导数。再比浅/深 CE、表征距离及验证质量",
 sources:[["LoopFormer official training code: long.detach() → short",SOURCES.loopformer_code],["LoopFormer ICLR 2026",SOURCES.loopformer]]
 }),
 exit:Object.freeze({
 title:"为什么需要自适应退出，而不能把循环数永远定为 3？",
 goal:"简单 Token 花更少计算，困难 Token 保留更多循环，以满足不同请求延迟预算",
 mechanism:"学习退出概率或 Token 路由必须同时定义质量目标、成本惩罚与剩余概率质量；真实推理还要保证不同轮缓存不能串用",
 alternative:"全体 Token 固定深度（本项目已实现）简单且训练/缓存语义清晰；逐请求整句动态深度；逐 Token 动态路由更灵活但调度复杂",
 cost:"Router 引入离散/近似梯度、负载不均衡与 KV Cache 按轮管理成本；自适应算力是否带来延迟收益需另测",
 failure:"全部 Token 都选择最大深度，或者早退导致质量显著劣化/Cache 错乱；GPU Token 稀疏未高效成批时 FLOPs 减少却不一定减少延迟",
 test:"给每个 Token 记录 chosen_depth，比较按层 FLOPs、实际 wall time、答案质量、KV 命中和早退比例；固定深度是必需对照",
 sources:[["Mixture-of-Recursions: dynamic token depth",SOURCES.mor],["Ouro: learned exit depth",SOURCES.ouro]]
 }),
 caching:Object.freeze({
 title:"为什么循环模型的 KV Cache 不能按普通单层 Transformer 随意复用？",
 goal:"在递归深度与时间步上同时保留语义一致的 K/V，避免把不同轮的隐藏状态混成一份",
 mechanism:"K/KV=WK·H_{token,loop}，因此通常要将 cache key 绑定到 loop/layer 等价身份；能安全共享时需证明相应状态或投影等价",
 alternative:"完整 Prefill 逐轮重算（本项目可执行的数学路径）；独立 (depth,layer) cache；跨轮压缩或特定管线设计",
 cost:"按轮存储增加 KV 容量、可能抵消权重省内存的部分好处；重算则增加算力与延迟",
 failure:"跨轮误用 cache 产生看似正确的 shape 但实际用错状态；预填充数学等价不保证增量 decode 一致",
 test:"同一确定性模型比较完整 Prefill 与逐 Token decode，逐轮 KV 值及 logits 一致才算通过；本项目尚未实现这种服务链",
 sources:[["Huginn official recurrent model",SOURCES.huginn],["Mixture-of-Recursions",SOURCES.mor]]
 })
});
const SKETCH=Object.freeze({
 sharing:["Wq","Wk","Wv","Wo","Wup","Wgate","Wdown"],
 nonrecurrent:["embedding"]
});
const elems=shape=>shape.reduce((a,b)=>a*b,1);
function profile(raw){
 const g=L.validateGraph(raw);
 const isOld=Boolean(g.nodes.some(n=>n.id==="r1_state")&&!g.nodes.some(n=>n.id==="r0_state"));
 let info;
 if(isOld){
  const active=[1,2,3].filter(i=>g.nodes.some(n=>n.id==="r"+i+"_state"));
  const parents=new Map(g.edges.map(e=>[e.to+":"+e.port,e.from]));
  if(active.length===0||!active.every((i,j)=>i===j+1&&parents.get("r"+i+"_norm:x")===(i===1?"pos":"r"+(i-1)+"_state")))
   throw Error("not a correctly connected legacy recurrent stack");
  info={loops:active.length,linked:true};
 }else info=Loop.inspect(g);
 if(!info.linked||!info.loops)throw Error("loop profiling requires connected recurrent graph");
 const by=new Map(g.nodes.map(n=>[n.id,n]));
 const parameters=g.nodes.filter(n=>n.kind==="parameter");
 const unique=parameters.reduce((s,n)=>s+elems([n.params.values.length,n.params.values[0].length]),0);
 const ids=isOld?["wq","wk","wv","wo","wup","wgate","wdown"]:SKETCH.sharing;
 const core=ids.reduce((s,id)=>{
  const n=by.get(id);return s+(n&&n.kind==="parameter"?elems([n.params.values.length,n.params.values[0].length]):0);
 },0);
 const fixed=unique-core;
 const trace=L.evaluate(g),observed=new Map(trace.steps.map(s=>[s.id,s]));
 const inputs=new Map(g.edges.map(e=>[e.to+":"+e.port,e.from]));
 let matmulMAC=0,count=0;
 for(const n of g.nodes){
  if(n.kind!=="matmul")continue;
  const a=observed.get(inputs.get(n.id+":a")),bb=observed.get(inputs.get(n.id+":b"));
  if(!a||!bb||a.shape.length!==2||bb.shape.length!==2||a.shape[1]!==bb.shape[0])
   throw Error("cannot calculate actual MatMul shapes for "+n.id);
  matmulMAC+=a.shape[0]*a.shape[1]*bb.shape[1];count++;
 }
 const outputElements=trace.steps.reduce((s,n)=>s+elems(n.shape),0);
 return Object.freeze({
  loops:info.loops,parameterTensors:parameters.length,physicalParameterElements:unique,
  coreParameterElements:core,nonRecurrentParameterElements:fixed,
  depthMatchedUntiedParameterElements:fixed+info.loops*core,
  relativeWeightElementsVsUntied:unique/(fixed+info.loops*core),
  matmulNodes:count,matmulMAC,matmulFlopTerms:matmulMAC*2,
  sumMaterializedNodeOutputElements:outputElements,
  graphNodes:g.nodes.length,graphEdges:g.edges.length,
  scope:"current validated micro-tensor graph, MatMul-only arithmetic; no kernel, memory allocation, throughput or factual benchmark"
 });
}
function observations(raw){
 const graph=L.validateGraph(raw),about=Loop.inspect(graph);
 if(!about.linked||!about.loops)throw Error("not a connected loop for observations");
 const trace=L.evaluate(graph),by=new Map(trace.steps.map(x=>[x.id,x]));
 const ce=about.perIteration.map(r=>({depth:r.index+1,ce:by.get(r.lossId)?.values?.[0]?.[0]}));
 if(ce.some(x=>!Number.isFinite(x.ce)))throw Error("missing real per-loop CE tensor output");
 const output=(id)=>by.get(id)?.values;
 const rmsDifference=(a,b)=>{
  if(!a||!b||a.length!==b.length||a[0].length!==b[0].length)throw Error("incompatible loop state tensors");
  let sum=0,count=0;
  for(let i=0;i<a.length;i++)for(let j=0;j<a[i].length;j++){let d=a[i][j]-b[i][j];sum+=d*d;count++;}
  return Math.sqrt(sum/count);
 };
 const changes=about.perIteration.map((r,i)=>({depth:i+1,
   stateRMSChange:rmsDifference(output(r.finalStateId),output(i===0?"lookup":about.perIteration[i-1].finalStateId))}));
 const m=by.get("short_consistency"),base=by.get("base_loss");
 const consistency=m?{mse:m.values[0][0],baseLoss:base?.values?.[0]?.[0],
    totalLoss:by.get("loss")?.values?.[0]?.[0],
    lambda:graph.nodes.find(n=>n.id==="cons_weight")?.params.factor}:null;
 return Object.freeze({perDepth:ce,changes,consistency,
   evidence:"actual finite Float64 graph outputs only; not model accuracy or held-out validation"});
}
let residualCache=null;
function residualComparisons(){
 if(!residualCache){
  const Grad=typeof module==="object"&&module.exports?
    require("./graph_autodiff.js"):root.SeetheTruthGraphAutodiff;
  if(!Grad)throw Error("graph gradient engine unavailable for real scaling evidence");
  const data=Loop.RESIDUAL_RULES.map(rule=>{
   const graph=Loop.buildResidualScaling({rule}),valid=Loop.inspectResidualScaling(graph);
   const trace=L.evaluate(graph),by=new Map(trace.steps.map(x=>[x.id,x]));
   const gradient=Grad.backward(graph,trace),elements=gradient.gradients.Wq.flat();
   const initial=by.get("lookup").values,second=by.get("r1_state").values;
   const rms=Math.sqrt(second.reduce((acc,row,i)=>acc+row.reduce((s,v,j)=>s+(v-initial[i][j])**2,0),0)/
     (initial.length*initial[0].length));
   const meta=profile(graph);
   return Object.freeze({rule,epsilon:valid.coefficient,loss:by.get("loss").values[0][0],
    stateRmsFromInput:rms,wqGradientL2:Math.hypot(...elements),
    paramElements:meta.physicalParameterElements,matmulMAC:meta.matmulMAC,
    nodes:graph.nodes.length,claim:"fixed deterministic micro-tensor calculation, no convergence or GPU performance claim"});
  });
  if(new Set(data.map(x=>x.paramElements)).size!==1||new Set(data.map(x=>x.matmulMAC)).size!==1)
    throw Error("residual scaling rules changed physical parameter size or underlying MatMul work");
  residualCache=Object.freeze(data);
 }
 return residualCache;
}
let cache=null;
function comparisons(){
 if(!cache)cache=Object.freeze([1,2,3].map(loops=>profile(Loop.build({loops}))));
 return cache;
}
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function keyFor(graph,id){
 if(!graph||!graph.title?.includes("Loop")||!graph.nodes?.some(n=>n.id===id))return null;
 if(graph.title.includes("任务 43")){
   try{Loop.inspectResidualScaling(graph);}catch(e){return null;}
   return "residual_scaling";
 }
 if(/任务 (?:39|40)/.test(graph.title)&&["loss","logits"].includes(id))return "sharing"; // legacy fixed-depth recipes have final-only CE
 if(graph.nodes.some(n=>n.id==="short_consistency")){
  if(["loss","base_loss","deep_teacher","short_consistency","cons_weight"].includes(id))return "consistency";
 }
 if(["loss","base_loss"].includes(id)||/^weight\d$/.test(id)||/^r\d_(?:ce|logits)$/.test(id))return "multidepth";
 if(/^r\d_(?:mask|scaled|probs|context)$/.test(id))return "caching";
 if(/^r\d_(?:state|attn_norm|ffn_norm|gate|swiglu)$/.test(id))return "sharing";

 if(/^(Wq|Wk|Wv|Wo|Wup|Wgate|Wdown|embedding|tied_head)$/.test(id))return "sharing";
 return "exit";
}
function render(raw,id){
 let key=keyFor(raw,id);if(!key)return "";
 let d=DECISIONS[key],out='<section class="knowledge-part" data-design-why="'+esc(key)+'">'+
 '<h3>WHY THIS DESIGN · 为什么这么设计？</h3><strong>'+esc(d.title)+'</strong>';
 for(const [title,field] of [["设计目标","goal"],["计算机制","mechanism"],["工业界常用替代方案","alternative"],
  ["必须付出的代价","cost"],["什么时候不该这么设计","failure"],["最小对照实验","test"]])
  out+='<p><b>'+title+'：</b>'+esc(d[field])+'</p>';
 out+='<p><b>证据依据：</b>'+d.sources.map(([name,url])=>'<a target="_blank" rel="noopener noreferrer" href="'+esc(url)+'">'+esc(name)+' ↗</a>').join(" · ")+'</p>';
 if(key==="sharing"||key==="multidepth"){
  try{const g=profile(raw),rows=comparisons();
  out+='<details open><summary>真实教学图的参数量与计算量 · ×1/×2/×3</summary>'+
   '<p>表格是第 41 课标准循环图的真实 ×1/×2/×3 对照；下方“所选图”数字独立从当前图推导。不同课程权重尺寸不同，不能将表格逐行当作其他课程的参数统计。非 GPU profiling，不含非 GEMM 算子和峰值内存。</p>'+
   '<table style="width:100%;font-size:11px"><thead><tr><th>循环</th><th>独立参数元素</th><th>不共享同深度对照</th><th>MatMul MAC</th></tr></thead><tbody>'+
    rows.map(r=>'<tr'+(g.loops===r.loops?' style="font-weight:bold"':'')+'><td>'+r.loops+
      '</td><td>'+r.physicalParameterElements+'</td><td>'+r.depthMatchedUntiedParameterElements+
      '</td><td>'+r.matmulMAC+'</td></tr>').join('')+'</tbody></table>'+
   '<p>所选图：'+g.graphNodes+' 节点、'+g.parameterTensors+' 套独立参数、'+g.matmulNodes+
   ' 个 MatMul；参数元素='+g.physicalParameterElements+'，核心共享权重元素='+
   g.coreParameterElements+'；参数共享节省的是权重元素，不是循环 GEMM 的执行次数。</p></details>';
  }catch(err){out+='<p>资源对照暂停：需要先修复循环连线及张量合同（'+esc(err.message)+'）。不会伪造无效统计。</p>';}
 }
 if(key==="consistency"||key==="sharing"||key==="multidepth"){
  try{const obs=observations(raw);
  out+='<details open><summary>当前参数下真实执行的逐轮损失／表征变化</summary>'+
   '<p>由当前教学图执行的 Float64 值，非训练精度或验证集质量；CE 变低不自动说明泛化改善。</p>'+
   '<table style="width:100%;font-size:11px"><thead><tr><th>轮次</th><th>Next-token CE</th><th>状态变化 RMS</th></tr></thead><tbody>'+
   obs.perDepth.map((row,i)=>'<tr><td>'+row.depth+'</td><td>'+Number(row.ce).toFixed(6)+
     '</td><td>'+obs.changes[i].stateRMSChange.toFixed(6)+'</td></tr>').join('')+'</tbody></table>'+
   (obs.consistency?'<p>λ='+obs.consistency.lambda+
     '；原多深度 CE='+obs.consistency.baseLoss.toFixed(6)+
     '；浅/深状态 MSE='+obs.consistency.mse.toFixed(6)+
     '；实际总损失='+obs.consistency.totalLoss.toFixed(6)+'</p>':'')+
   '</details>';
  }catch(err){out+='<p>逐轮数值观察暂停：需要可执行的当前计算图（'+esc(err.message)+'）。</p>';}
 }
 if(key==="residual_scaling"){
  try{
   const current=Loop.inspectResidualScaling(raw),data=residualComparisons(),
    rows=data.map(x=>'<tr'+(Math.abs(x.epsilon-current.coefficient)<1e-12?' style="font-weight:750"':'')+
    '><td>'+esc(x.rule)+'</td><td>'+x.epsilon.toFixed(6)+'</td><td>'+x.loss.toFixed(7)+
    '</td><td>'+x.stateRmsFromInput.toFixed(7)+'</td><td>'+x.wqGradientL2.toFixed(7)+'</td></tr>').join("");
   out+='<details open><summary>同权重、同输入、同循环数的实际数值对照</summary>'+
    '<p>只改变四个 residual Scale 系数；三种规则的参数元素数='+data[0].paramElements+
    '，MatMul MAC='+data[0].matmulMAC+'，不随 ε 改变。以下是微型张量的初始前向/梯度数据，不是训练质量或稳定性结论。</p>'+
    '<table style="width:100%;font-size:11px"><thead><tr><th>规则</th><th>ε</th><th>Loss</th><th>状态变化 RMS</th><th>∥∇Wq∥₂</th></tr></thead><tbody>'+
    rows+'</tbody></table>'+
    '<p>当前所选图的残差系数='+current.coefficient.toFixed(6)+
    '；可直接修改四个 Scale 节点或使用顶部残差方案切换，全部真实参数及 AdamW 状态保留。</p></details>';
  }catch(err){out+='<p>真实对照不可用：'+esc(err.message)+'；不会虚构未执行数据。</p>';}
 }
 if(key==="consistency")out+='<p><strong>精确实现边界：</strong>本课程固定 2 轮，共享参数，不含 LoopFormer 原版的 t/Δt 条件或独立采样的 shortcut trajectory。只能验证 Detach + MSE 消融的数学与梯度。</p>';
 out+='</section>';return out;
}
return Object.freeze({profile,comparisons,residualComparisons,observations,keyFor,render,decisions:DECISIONS,sources:SOURCES});
});