/* Paper operators are ordinary composable blocks. The pack delegates numerics to
 * the same source-grounded teaching engines used by Paper Lab, not to a mockup. */
(function(root,factory){'use strict';
 const api=factory(typeof module==='object'&&module.exports?require('./paper_patches.js'):root.SeetheTruthPaperPatches,
  typeof module==='object'&&module.exports?require('./paper_compute_trace.js'):root.SeetheTruthPaperMicroscope);
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.SeetheTruthPaperBlockPack=api;
})(typeof window!=='undefined'?window:{},function(P,Trace){'use strict';
 if(!P||!Trace)throw Error('paper block pack requires real paper numeric engines');
 const def=(title,en,ports,params,formula,why,lesson,source)=>({
  title,en,category:'compute',ports,params,formula,why,lesson,source,
  caution:'微型 Float64 教学数值与源码/论文数学一致，执行图可自由连线。',knowledge:{
   flow:'由当前图中接入的实际 X、权重、Q/K/V 运行矩阵运算；详细子阶段复用公共计算显微镜。',
   practice:lesson,trap:'形状一致但含义错误仍可能产生可计算张量，需检查端口与参数对应关系。',
   exercise:'保持输入及其它模块不变，改变一个结构参数，检查内部计算及后继输出。',
   refs:'source_mathematics'
  }
 });
 const defs=Object.freeze({
  paper_lora_forward:def('LoRA · 低秩输出更新','LoRA Forward',['x','w','a','b'],{alpha:2},
   'Y=XW₀+(α/r)(XA)B','从冻结基础路径及低秩更新路径生成最终输出；所有输入来自用户连线。',
   '调整 α 或替换 A/B，观察同一矩阵输出的增量通道。','https://arxiv.org/abs/2106.09685'),
  paper_dora_forward:def('DoRA · 权重方向与幅度','DoRA Forward',['x','w','a','b','m'],{alpha:1},
   'V=W₀+(α/r)AB; Y=X[m⊙V/||V||col]','先更新 W 的方向并逐输出列归一化，再由独立幅度 m 控制。',
   '固定 A/B 只修改 m，逐列复核 W_eff 的 L₂ 范数。','https://proceedings.mlr.press/v235/liu24bn.html'),
  paper_online_attention:def('Online Softmax · K/V Tile','Tiled Exact Attention',['q','k','v'],{chunk:1},
   'O=OnlineSoftmax(QKᵀ/√d)V','通过真正的逐 Tile 稳定递推实现不物化完整 Attention 概率矩阵的精确数学结果。',
   '改变 chunk；对比计算中间值与最终 Dense Attention 精确等价。','https://arxiv.org/abs/2307.08691')
 });
 const validShape=(shape,label)=>{if(!Array.isArray(shape)||shape.length!==2||shape.some(v=>!Number.isSafeInteger(v)||v<1||v>16))throw Error(label+' requires rank-2 size ≤16');return shape;};
 function validate(kind,p){
  if(!Object.hasOwn(defs,kind))throw Error('unknown paper block');
  if(!Number.isFinite(p.alpha??1)||Math.abs(p.alpha??1)>64)throw Error('invalid paper alpha');
  if(kind==='paper_online_attention'&&(!Number.isSafeInteger(p.chunk)||p.chunk<1||p.chunk>16))throw Error('invalid attention tile width');
 }
 function infer(kind,p,shapes){
  const get=k=>validShape(shapes[k],k);
  if(kind==='paper_online_attention'){
   const q=get('q'),k=get('k'),v=get('v');
   if(q[1]!==k[1]||k[0]!==v[0])throw Error('online attention Q/K/V dimensions differ');
   return [q[0],v[1]];
  }
  const x=get('x'),w=get('w'),a=get('a'),b=get('b');
  if(x[1]!==w[0]||a[0]!==w[0]||a[1]!==b[0]||b[1]!==w[1])throw Error('paper low-rank contraction mismatch');
  if(kind==='paper_dora_forward'){
   const m=get('m');if(m[0]!==1||m[1]!==w[1])throw Error('DoRA magnitude must be [1,out]');
  }
  return [x[0],w[1]];
 }
 function args(kind,p,inputs){
  if(kind==='paper_online_attention')return {q:inputs.q,k:inputs.k,v:inputs.v,chunk:p.chunk};
  const o={x:inputs.x,w:inputs.w,a:inputs.a,b:inputs.b,alpha:p.alpha};
  if(kind==='paper_dora_forward')o.m=inputs.m;
  return o;
 }
 function run(kind,p,inputs){
  const a=args(kind,p,inputs);
  if(kind==='paper_lora_forward')return {a,result:P.lora(a.x,a.w,a.a,a.b,a.alpha)};
  if(kind==='paper_dora_forward')return {a,result:P.dora(a.x,a.w,a.a,a.b,a.m,a.alpha)};
  return {a,result:P.onlineAttention(a.q,a.k,a.v,a.chunk)};
 }
 function execute(kind,p,inputs){
  infer(kind,p,Object.fromEntries(Object.entries(inputs).map(([k,v])=>[k,[v.length,v[0].length]])));
  const result=run(kind,p,inputs);
  return {values:result.result.output,internals:{paperMethod:kind,execution:result}};
 }
 function trace(kind,p,inputs,execution,node){
  const real=execution?.internals?.execution;
  if(!real)throw Error('missing real paper method execution');
  const check=run(kind,p,inputs);
  if(JSON.stringify(real)!==JSON.stringify(check))throw Error('paper trace is inconsistent with the live numerical execution');
  const frame=kind==='paper_lora_forward'?Trace.lora(real.a,real.result):
   kind==='paper_dora_forward'?Trace.dora(real.a,real.result):Trace.online(real.a,real.result);
  // The generic microscope supports one nested level. Select mathematically
  // meaningful verified stages; for online attention expose actual tile updates.
  const primary=frame.steps.filter(s=>!['input','parameter'].includes(s.kind));
  const steps=kind==='paper_online_attention'?
   primary.flatMap(s=>s.substeps?s.substeps:[s]):primary;
  return steps.slice(0,12).map((s,i)=>({
   id:node.id+':paper_step_'+i,kind:s.kind,title:s.title,formula:s.formula,why:s.why,
   out:s.out,operands:s.operands,dot:s.dot||null
  }));
 }
 return Object.freeze({name:'paper_operators',defs,validate,infer,execute,trace});
});