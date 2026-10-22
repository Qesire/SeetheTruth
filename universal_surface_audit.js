/* Maintainable, falsifiable inventory of teaching screens. Not every independent
 * semantic runtime belongs inside the tensor microscope: classify precisely. */
(function(root,make){'use strict';const api=make();if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthSurfaceAudit=api;})(typeof window!=='undefined'?window:{},function(){'use strict';
const screens=Object.freeze([
['learn.html','自由积木与训练/推理','tensor','结构、数学、张量、Linear/MatMul、Attention 内部算子'],
['wan_ffn_lab.html','Wan FFN 官方源码','tensor','固定源码步骤、张量、逐乘加和通道删除反事实'],
['deepinfer_kv_lifecycle_lab.html','KV 物理块生命周期','state','FIFO、引用计数、逐操作状态差分'],
['deepinfer_worker_lab.html','Worker 执行时序','state','事件因果 DAG、CPU/GPU 假设排队状态'],
['training.html','标量训练循环','state','标量 Forward/Backward/Gradient/AdamW 状态差分已接入通用事件显微镜；概念模式不生成数值'],
['micro_training.html','MoE / PPO / VJP 微实验','tensor','共享 Tape/MoE/PPO 执行轨迹：前向矩阵点积、参数梯度归并、Top-1 专家派发和 PPO Clip 反向梯度可逐坐标检查'],
['cache.html','KV Cache 数学','state','实际 GQA/RoPE 数值缓存与 Prefill/Decode 全会话可重放核查；共享前后状态显微镜'],
['deepinfer_scheduler_lab.html','Scheduler + KV + Worker','state','已通过真实决策与 KV 修订验证接入共用透视；Worker Forward 绑定物理块所有权'],
['deepinfer_prefill_lab.html','Prefill Vote 与多模态 Chunk','state','独立 ALL/ANY 投票与 IMG/PAD Chunk 源码参考决策进入通用审计，输出重新计算防篡改'],
['paper_lab.html','论文图变换','specialized','专用数值算子；部分 LoRA 重写已进入主编辑器，但完整 DoRA/OnlineAttn 对应独立实验'],
['semantic_state.html','语义轴与持久状态','specialized','独立 1–5 维 TensorIR/状态操作，尚未映射全部核心 event/coordinate 接口'],
['tradeoffs.html','架构成本权衡','specialized','解析算术/交互页面，不是执行 Tensor Trace；不能强行提升证据级别'],
['capabilities.html','积木能力矩阵','specialized','全局注册/后端白名单审计，职责是状态与覆盖统计，不是计算实验'],
['engineering_journey.html','12 关训练到推理','specialized','使用真实微型 Bigram 数学，但教学关卡未复用通用 Trace UI'],
['engineering_pipeline.html','累积、Shuffle、续训','state','已接入通用 State Frame：Microbatch、Sampler、Optimizer Step、检查点差分'],
['teaching.html','教学中心','navigation','课程路由和问题索引，无数值执行职责'],
['android_teaching.html','Android 教学导航','navigation','离线教育导航入口，不执行模型计算，也不提供训练产物或文件导出'],
['real_model_lab.html','真实 CPU 小模型证据','desktop_only','结构化只读观测，不能被误认为完整算子输入或已验证复跑'],
['index.html','源码结构图与 Runtime','desktop_only','AST 源码图＋独立 Runtime Trace，不应与合成数值混合'],
['telemetry.html','Profiler 和 Tensor 观测','desktop_only','GPU Profiler 是独立证据源，未加入通用数值算子接口']
].map(([page,title,status,note])=>Object.freeze({page,title,status,note})));
// Functional bindings are checked separately from <script> presence.
// The UI coverage count alone is not a mathematical or behavioral guarantee.
const runtimeContracts=Object.freeze([
 ['learn.html','learn.html',['Microscope.learning(used,run,L)','microscopeView.show(']],
 ['wan_ffn_lab.html','wan_ffn_lab.js',['commonMicroscope.wan(d)','sharedMicroscope.show(']],
 ['micro_training.html','micro_training.html',['Trace.tape(t)','Trace.moe(args,result)','Trace.ppo(args,result)','microscope.show(frame)']],
 ['deepinfer_kv_lifecycle_lab.html','deepinfer_kv_lifecycle_lab.js',['State.kv(result,M)','stateView.show(']],
 ['deepinfer_worker_lab.html','deepinfer_worker_lab.js',['State.worker(zos)','stateView.show(']],
 ['engineering_pipeline.html','engineering_pipeline_ui.js',['State.training(s.history)','stateView.show(']],
 ['training.html','training.html',['State.scalar(run)','view.show(']],
 ['deepinfer_scheduler_lab.html','deepinfer_scheduler_lab.js',['State.scheduler(run,KV)','stateView.show(']],
 ['deepinfer_prefill_lab.html','deepinfer_prefill_lab.js',['State.prefill(evidence,P)','stateView.show(']],
 ['cache.html','cache.html',['State.cache(runHistory,Lab)','stateView.show(']]
].map(([page,module,needles])=>Object.freeze({page,module,needles:Object.freeze(needles)})));
function verifyBindings(files){
 if(!files||typeof files!=='object')throw Error('audit runtime bindings require actual source files');
 const found=new Set();
 for(const item of runtimeContracts){
  if(found.has(item.page))throw Error('duplicate runtime binding '+item.page);found.add(item.page);
  const code=files[item.module];
  if(typeof code!=='string')throw Error('runtime module missing '+item.module);
  for(const needle of item.needles)if(!code.includes(needle))throw Error('runtime adapter invocation missing '+item.page+': '+needle);
 }
 const shared=screens.filter(x=>x.status==='tensor'||x.status==='state').map(x=>x.page).sort();
 if(JSON.stringify([...found].sort())!==JSON.stringify(shared))throw Error('every shared view must have a runtime adapter binding');
 return Object.freeze({verified:found.size,modules:runtimeContracts.map(x=>x.module)});
}
const coreFiles=Object.freeze({tensor:['compute_microscope.js','compute_microscope_ui.js'],state:['state_microscope.js','state_microscope_ui.js']});
function classify(html){
 const scripts=new Set([...String(html).matchAll(/<script\b[^>]*\bsrc=["']([^"']+)["']/g)].map(x=>x[1].split(/[?#]/)[0]));
 if(coreFiles.tensor.every(x=>scripts.has(x)))return'tensor';
 if(coreFiles.state.every(x=>scripts.has(x)))return'state';
 return'none';
}
function analyze(sources){
 if(!sources||typeof sources!=='object')throw Error('audit requires actual page HTML texts');
 const rows=screens.map(item=>{
  const html=sources[item.page];if(typeof html!=='string'||!html.includes('<html'))throw Error('missing actual page '+item.page);
  const actual=classify(html);
  if(['tensor','state'].includes(item.status)&&actual!==item.status)throw Error('universal capability regression '+item.page);
  if(!['tensor','state'].includes(item.status)&&actual!=='none')throw Error('audit status stale: '+item.page+' now '+actual);
  return {...item,actual};
 });
 const counts={total:rows.length,tensor:rows.filter(x=>x.actual==='tensor').length,state:rows.filter(x=>x.actual==='state').length,
  specialized:rows.filter(x=>x.status==='specialized').length,navigation:rows.filter(x=>x.status==='navigation').length,
  desktopOnly:rows.filter(x=>x.status==='desktop_only').length};
 return Object.freeze({schema:'seethetruth-universal-surface-audit-v1',counts,rows:Object.freeze(rows)});
}
return Object.freeze({screens,coreFiles,runtimeContracts,classify,analyze,verifyBindings});
});