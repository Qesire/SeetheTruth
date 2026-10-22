/* Paper method executions -> shared source-agnostic operator microscope.
 * Every numeric intermediate is captured by the real paper_patches engine;
 * a fresh independent invocation verifies the reported execution transcript. */
(function(root,factory){'use strict';const api=factory(root);if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthPaperMicroscope=api;})(typeof window!=='undefined'?window:{},function(root){'use strict';
const C=typeof module==='object'&&module.exports?require('./compute_microscope.js'):root.SeetheTruthComputeMicroscope;
const P=typeof module==='object'&&module.exports?require('./paper_patches.js'):root.SeetheTruthPaperPatches;
function need(x,msg){if(!x)throw Error('paper trace: '+msg);}
const tensor=x=>C.numericTensor([x.length,x[0].length],x.flat());
const row=x=>tensor([x]);
const point=x=>row([x]);
const dep=(from,port)=>({from,port});
function stage(id,kind,title,arr,dependencies=[],operands={},formula='',why='',dot=null){
 return {id,kind,title,out:tensor(arr),dependencies,operands,formula,why,dot};
}
function dot(A,B){
 need(A.shape.length===2&&B.shape.length===2&&A.shape[1]===B.shape[0],'matmul incompatible operands');
 const [D,H]=B.shape,weights=C.numericTensor([H,D],Array.from({length:H*D},(_,i)=>B.values[(i%D)*H+Math.floor(i/D)]));
 return {type:'dot',orientation:'out_in',input:A,weights,bias:C.numericTensor([1,H],Array(H).fill(0))};
}
function materialize(s,why,source){
 return C.validate({schema:C.SCHEMA,title:source,evidenceTier:C.TIERS.MICRO,source:null,steps:s,
  origin:'paper_patches_executed_numeric_engine',limits:why});
}
function verify(input,result,actual){
 need(JSON.stringify(result)===JSON.stringify(actual),'numerical result differs from the actual paper execution');
}
function inputs(X,W,A,B,m=null){
 const out=[stage('x','input','X · Input token features',X,[],{},'X[T,C]','所有后续 Token 表示均从此开始。'),
  stage('w','parameter','W₀ · Frozen base weight',W,[],{},'W₀[C,H]','先观察冻结基座，再看低秩更新如何改变实际运算。'),
  stage('a','parameter','A · Low-rank input projection',A,[],{},'A[C,r]','低秩瓶颈的第一矩阵。'),
  stage('b','parameter','B · Low-rank output projection',B,[],{},'B[r,H]','低秩瓶颈的第二矩阵。')];
 if(m)out.push(stage('m','parameter','m · Output-column magnitudes',m,[],{},'m[1,H]','每个输出列单独设置长度。'));
 return out;
}
function lora(args,result){
 const {x,w,a,b,alpha=1}=args;verify(args,result,P.lora(x,w,a,b,alpha));
 const nx=tensor(x),nw=tensor(w),na=tensor(a),nb=tensor(b),steps=inputs(x,w,a,b);
 const base=tensor(result.base),down=tensor(result.internals.down),up=tensor(result.internals.up);
 steps.push({...stage('base','matmul','Base GEMM · XW₀',result.base,[dep('x','X'),dep('w','W0')],{X:nx,W:nw},
   'Y_base = X · W₀','冻结基座仍参与最终前向结果；这里可按坐标展开每次乘加。'),dot:dot(nx,nw)});
 steps.push({...stage('down','matmul','Rank-down · XA',result.internals.down,[dep('x','X'),dep('a','A')],{X:nx,A:na},
   'H_rank = X · A','让输入投影到 r 维，而不是原 C→H 全量可训练矩阵。'),dot:dot(nx,na)});
 steps.push({...stage('up','matmul','Rank-up · (XA)B',result.internals.up,[dep('down','rankInput'),dep('b','B')],{rank:down,B:nb},
   'ΔY_raw = H_rank · B','从低秩空间回投到输出通道，可以查看重建所用的每个 r 坐标。'),dot:dot(down,nb)});
 steps.push(stage('scaled','scale','Scaling · α/r · (XA)B',result.update,[dep('up','lowRankOutput')],{raw:up,factor:point(result.internals.scaling)},
   'ΔY = (α/r) ΔY_raw','α 与 r 改变输出扰动的幅度，不改变参与计算的 token/channel 路由。'));
 steps.push(stage('output','add','Output · Base + LoRA update',result.output,[dep('base','base'),dep('scaled','lowRankUpdate')],
   {base,update:tensor(result.update)},'Y = Y_base + ΔY','直接比较基座与低秩新增项的各输出坐标。'));
 return materialize(steps,'bounded Float64 LoRA execution: actual XW0, XA, (XA)B, scale and residual','LoRA · 权重低秩增量如何进入一次前向');
}
function dora(args,result){
 const {x,w,a,b,m,alpha=1}=args;verify(args,result,P.dora(x,w,a,b,m,alpha));
 const nx=tensor(x),na=tensor(a),nb=tensor(b),steps=inputs(x,w,a,b,m);
 const raw=tensor(result.internals.rawDelta);
 steps.push({...stage('ab','matmul','Low-rank weight delta · AB',result.internals.rawDelta,[dep('a','A'),dep('b','B')],
   {A:na,B:nb},'ΔW_raw=A·B','DoRA 在权重空间构造方向增量；这不是 LoRA 在 Token 输出上直接相加。'),dot:dot(na,nb)});
 steps.push(stage('scaled','scale','Scaled delta · (α/r)AB',result.internals.delta,[dep('ab','rawDelta')],
   {raw,factor:point(result.internals.scale)},'ΔW=(α/r)AB','观察每个权重坐标的实际扰动。'));
 steps.push(stage('direction','add','Direction · W₀ + ΔW',result.directions,[dep('w','baseWeight'),dep('scaled','delta')],
   {W0:tensor(w),delta:tensor(result.internals.delta)},'V=W₀+ΔW','不同于直接相加后的输出，接下来每个列还要重新计算范数。'));
 steps.push(stage('norm','reduction','Output-column norms · ||V[:,j]||₂',result.internals.norms,[dep('direction','V')],
   {directions:tensor(result.directions)},'n[j]=sqrt(ΣᵢV[i,j]²)','每个输出通道沿输入维求 L₂ 范数，保留列的方向信息。'));
 steps.push(stage('effective','normalize_scale','Reparameterized weight · m × V / ||V||',result.effectiveWeight,
   [dep('direction','V'),dep('norm','norms'),dep('m','magnitude')],
   {V:tensor(result.directions),norms:tensor(result.internals.norms),m:tensor(m)},
   'W_eff[i,j]=m[j] V[i,j]/n[j]','真正进入最终 GEMM 的是重参数化后的 W_eff，每一列的 L₂ 范数可以由 m 控制。'));
 steps.push({...stage('output','matmul','Forward · X W_eff',result.output,[dep('x','X'),dep('effective','W_eff')],
   {X:nx,effective:tensor(result.effectiveWeight)},'Y = X · W_eff',
   '用同一个逐乘加 Inspector 直接核算最终输出，区分方向更新与幅度更新。'),dot:dot(nx,tensor(result.effectiveWeight))});
 return materialize(steps,'bounded Float64 DoRA: AB, V, per-column norm, effective weight and XW','DoRA · 方向和幅度如何重参数化权重');
}
function denseAttentionReference(q,k,v){
 const d=q[0].length;
 return q.map(query=>{
  const score=k.map(key=>query.reduce((a,x,i)=>a+x*key[i],0)/Math.sqrt(d));
  const peak=Math.max(...score),p=score.map(x=>Math.exp(x-peak));
  const denom=p.reduce((a,b)=>a+b,0);
  return v[0].map((_,c)=>p.reduce((a,w,i)=>a+w*v[i][c],0)/denom);
 });
}
function online(args,result){
 const {q,k,v,chunk=2}=args;verify(args,result,P.onlineAttention(q,k,v,chunk));
 const dense=denseAttentionReference(q,k,v);
 const error=Math.max(...dense.flatMap((r,i)=>r.map((x,j)=>Math.abs(x-result.output[i][j]))));
 need(error<=1e-10,'tiled online-softmax output contradicts independent dense attention');
 const steps=[
  stage('q','input','Queries Q',q,[],{},'Q[Tq,d]','每一行 Query 依次处理属于它的 K/V tiles。'),
  stage('k','input','Keys K',k,[],{},'K[Tk,d]','Tile 遍历 Key 轴，计算局部注意力分数。'),
  stage('v','input','Values V',v,[],{},'V[Tk,dv]','Value 的每个通道用同一组指数权重更新累积器。')
 ];
 need(result.tileTrace.length+4<=160,'too many trace nodes; reduce token count or enlarge tile');
 let prior=null;
 const fullK=tensor(k),fullV=tensor(v),D=q[0].length;
 for(const [i,t]of result.tileTrace.entries()){
  const keyRows=k.slice(t.start,t.end),valRows=v.slice(t.start,t.end);
  const scaledQ=row(q[t.query].map(x=>x/Math.sqrt(D))),kt=tensor(keyRows),vt=tensor(valRows);
  const deps=[dep('q','query'),dep('k','keyTile'),dep('v','valueTile')];
  if(prior)deps.push(dep(prior,'previousTile'));
  const id='tile'+i,acc=row(t.accumulator),previous=row(t.previousAccumulator),exps=row(t.weights);
  const part=stage(id,'online_softmax','Query '+t.query+' · K ['+t.start+','+t.end+')', [t.partialOutput],
    deps,{query:row(q[t.query]),keyTile:kt,valueTile:vt,priorAccumulator:previous},
    'm←max(m,s); l←l·exp(m_old−m)+Σexp(s−m); acc←acc·exp(m_old−m)+Σexp(s−m)V',
    '点击下层真实子阶段，逐项观察一次 Tile 如何改变 running max、normalizer 和 Value 累积器。');
  part.substeps=[
   {...stage(id+':score','scaled_matmul','Scores · QK_tileᵀ / √d', [t.scores],[],
      {qScaled:scaledQ,keyTile:kt},'s_j = Σ_k Q[k]K_j[k]/√d',
      '对当前 Query 与 Tile 每个 Key 的 dot-product 进行完整重建。'),
    dot:dot(scaledQ,tensor(keyRows.map(r=>r).length?keyRows[0].map((_,j)=>keyRows.map(r=>r[j])):[]))},
   stage(id+':max','maximum','Running maximum',[[t.runningMax]],[],
      {scores:row(t.scores),previousMax:point(t.previousMax??-1e9)},
      'm_new=max(m_old,max_j s_j)','使用新的最大值重新缩放旧累计量以实现稳定指数运算。'),
   stage(id+':weights','exp','Tile exp(s−m_new)',[t.weights],[],
      {scores:row(t.scores),runningMax:point(t.runningMax)},
      'p_unorm[j]=exp(s[j]−m_new)','局部指数权重未经过最终归一化，不能当作所有 Keys 的 Softmax。'),
   stage(id+':denom','accumulate','Denominator / row normalizer',[[t.normalizer]],[],
      {weights:exps,correction:point(t.correction),previousNormalizer:point(t.previousDenom)},
      'l_new=l_old·correction+Σp_unorm','每个 Tile 改变同一 Query 的全局归一化常数。'),
   stage(id+':acc','accumulate','Weighted V accumulator',[t.accumulator],[],
      {previousAccumulator:previous,weights:exps,values:vt,correction:point(t.correction)},
      'acc_new=acc_old·correction+Σp_unorm[j]V[j]','Value 矩阵按 Tile 加权累加，不需要物化完整 Attention 矩阵。'),
   stage(id+':output','division','Partial normalized output',[t.partialOutput],[],
      {accumulator:acc,denominator:point(t.normalizer)},'o_partial=acc/l',
      '所有 Tile 完成之后，该值就是这个 Query 的确切注意力输出。')
  ];
  steps.push(part);prior=id;
 }
 steps.push(stage('output','gather','Online attention · final output',result.output,
  [dep(prior,'finalTile')],{Q:tensor(q),K:fullK,V:fullV},
  'O[t,:]=final_accumulator[t,:]/final_normalizer[t]',
  '完整 Dense QK→Softmax→PV 独立计算的最大绝对差：'+error.toExponential(3)+'。调节 Chunk 观察实际逐块归一化如何仍重建相同输出。'));
 return materialize(steps,'bounded Float64 tiled online softmax reference; no causal mask/dropout/hardware kernel','Online Softmax · 逐 Tile 重标定与分块汇聚');
}
return Object.freeze({lora,dora,online});
});