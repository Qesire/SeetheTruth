/* Deep-trace adapters for exact micro training execution, not a second
 * hand-authored toy. All tensors come from the executed Tape/MoE/PPO engines.
 * Output dots are reconstructed by the universal Compute Microscope. */
(function(root,factory){'use strict';const api=factory(root);if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthMicroTrainingTrace=api;})(typeof window!=='undefined'?window:{},function(root){'use strict';
const Core=typeof module==='object'&&module.exports?require('./compute_microscope.js'):root.SeetheTruthComputeMicroscope;
const M=typeof module==='object'&&module.exports?require('./micro_training_flows.js'):root.SeetheTruthMicroTrain;
function need(ok,msg){if(!ok)throw Error('micro training trace: '+msg);}
const values=a=>Core.numericTensor([a.length,a[0].length],a.flat());
const point=(x,y)=>Core.numericTensor([1,1],[x]);
const row=a=>Core.numericTensor([1,a.length],a);
const node=(id,kind,title,out,dependencies=[],operands={},formula='',why='',dot=null)=>({id,kind,title,formula,why,dependencies,out,operands,dot});
const edge=(from,port)=>({from,port});
function finalize(title,steps){
 return Core.validate({schema:Core.SCHEMA,title,evidenceTier:Core.TIERS.MICRO,source:null,steps,origin:'micro_training_flows_engine_numerically_executed',limits:'actual_float64_training_tape_or_bounded_MoE_PPO_reference'});
}
function tape(t){
 need(t&&t.nodes instanceof Map&&t.parameters instanceof Map&&Array.isArray(t.events),'actual Tape required');
 need(t.events.some(e=>e.phase==='backward'),'run Tape.backward() before opening the microscope');
 const original=[...t.nodes.values()],seen=new Set(),steps=[];
 for(const n of original){
  const deps=n.parents.map((p,i)=>edge(p.id,'input'+i));
  need(deps.every(x=>seen.has(x.from)),'Tape forward parents not topological');
  const operands={};
  n.parents.forEach((p,i)=>{operands['input'+i]=values(t.nodes.get(p.id).value);});
  const x=values(n.value),kind=n.name==='matmul'?'matmul':n.name;
  let dot=null;
  if(kind==='matmul'){
   const A=operands.input0,B=operands.input1;need(A&&B&&A.shape[1]===B.shape[0],'matmul needs true operands');
   const D=B.shape[0],H=B.shape[1],W=Core.numericTensor([H,D],
    Array.from({length:H*D},(_,p)=>B.values[(p%D)*H+Math.floor(p/D)]));
   const bias=Core.numericTensor([1,H],Array(H).fill(0));
   dot={type:'dot',input:A,weights:W,bias,orientation:'out_in'};
   operands.transposedRhs=W;
  }
  steps.push(node(n.id,kind,'Forward · '+n.name,x,deps,operands,
   kind==='matmul'?'Y[i,j]=Σₖ A[i,k]×B[k,j]':kind==='mse'?'loss=mean((prediction−target)²)':n.name,
   n.identity?'Parameter ID '+n.identity+'：相同 ID 重复引用同一份 W，并在反向传播时归并梯度。':'该步骤读取上述真实前驱矩阵并交给微型 Tape 执行器。',dot));
  seen.add(n.id);
 }
 const history=t.events.filter(e=>e.phase==='backward'||e.phase==='vjp');
 // Every inner VJP edge has a real numerical contribution from the tape;
 // each step follows the engine's own recorded reverse execution order.
 need(history.some(e=>e.phase==='vjp'),'no executed derivative contributions');
 let previous=null;
 const currentGrad=new Map();
 for(let i=0;i<history.length;i++){
  const e=history[i],id='adjoint_'+i,dependencies=[];
  if(previous)dependencies.push(edge(previous,'previousExecutionEvent'));
  let out,operands,title,formula,why,kind;
  if(e.phase==='backward'){
   need(seen.has(e.id),'backward references nonexistent forward tensor');
   const n=t.nodes.get(e.id);
   need(Array.isArray(e.incomingValues)&&Array.isArray(e.accumulatedValues),'missing real gradients at backward event');
   const contribution=values(e.incomingValues),accumulated=values(e.accumulatedValues);
   if(currentGrad.has(e.id)){
    const before=currentGrad.get(e.id);
    need(before.length===accumulated.values.length,'gradient accumulation dimensionality changed');
    for(let j=0;j<before.length;j++)need(Math.abs(before[j]+contribution.values[j]-accumulated.values[j])<1e-10,'cumulative gradient is inconsistent');
   }else need(accumulated.values.every((z,j)=>Math.abs(z-contribution.values[j])<1e-10),'initial node gradient not equal to actual seed');
   currentGrad.set(e.id,accumulated.values);
   if(JSON.stringify(t.nodes.get(e.id).grad)!==JSON.stringify(e.accumulatedValues)){
    // Multiple backward() calls can visit the same node again; the final
    // stored gradient is only expected to match its last accumulated event.
    const last=history.filter(z=>z.phase==='backward'&&z.id===e.id).at(-1);
    if(last===e)need(false,'last accumulated gradient differs from Tape node');
   }
   dependencies.push(edge(e.id,'forwardActivation'));
   out=accumulated;operands={incomingGradient:contribution,forwardActivation:values(n.value)};
   title='Backward · accumulated dL/d'+e.id;kind='vjp_accumulation';
   formula='accumulated_grad += incoming_grad';
   why='显示本次真实反向事件收到的梯度与累计后矩阵。共享参数的梯度相加可以逐次观察。';
  }else{
   const n=t.nodes.get(e.from),parent=t.nodes.get(e.to);
   need(n&&parent&&Array.isArray(e.incomingValues)&&Array.isArray(e.gradientValues),'VJP missing source or actual gradient matrix');
   const link=n.parents.find(p=>p.id===e.to);
   need(link&&typeof link.vjp==='function','VJP target is not a parent');
   need(JSON.stringify(link.vjp(e.incomingValues))===JSON.stringify(e.gradientValues),'VJP branch contribution contradicts executed derivative');
   const incoming=values(e.incomingValues);
   out=values(e.gradientValues);operands={outputGradient:incoming,forwardInput:values(parent.value),forwardOutput:values(n.value)};
   dependencies.push(edge(e.from,'downstreamNode'),edge(e.to,'parentNode'));
   title='VJP · '+e.from+' → '+e.to;kind='vjp_edge';
   formula='∂L/∂'+e.to+' += VJP('+n.name+',∂L/∂'+e.from+')';
   why='这个数值是执行器对该条计算边独立计算的梯度贡献，能区分两条路径各自的贡献与最终累积值。';
  }
  steps.push(node(id,kind,title,out,dependencies,operands,formula,why));
  previous=id;
 }
 return finalize('真实 Tape：前向矩阵乘法 → MSE → VJP 梯度归并',steps);
}
function moe(args,result){
 need(args&&Array.isArray(args.tokens)&&Array.isArray(args.expertWeights)&&
  Array.isArray(args.routerLogits)&&Array.isArray(args.target),'complete MoE inputs required');
 const ref=M.moeBackward(args.tokens,args.expertWeights,args.routerLogits,args.target,args.options||{});
 need(JSON.stringify(result)===JSON.stringify(ref),'MoE results disagree with real reference engine');
 const steps=[],tokens=values(args.tokens),router=values(args.routerLogits);
 steps.push(node('tokens','input','Input tokens',tokens,[],{},'X[t,d]','每一行是一枚 Token 的特征向量。'));
 steps.push(node('router','input','Router logits',router,[],{},'R[t,e]','该路由器使用 softmax + 离散 top-1 选择。'));
 steps.push(node('probs','softmax','Router probabilities',values(ref.probabilities),[edge('router','logits')],{logits:router},
  'p[t,e]=softmax(R[t,:])','沿专家维计算全概率，再选出每个 Token 的最大概率专家。'));
 const choice=values(ref.routed.map(r=>[r.expert,r.accepted?1:0,r.probability]));
 steps.push(node('dispatch','top1','Expert ID / accepted / probability',choice,[edge('probs','probabilities')],{probabilities:values(ref.probabilities)},
  '[expert, accepted, p_selected]','容量是每专家的独立计数；超额 Token 输出置零而不是隐式迁移到次优专家。'));
 const output=values(ref.output);
 const combined=node('out','moe','Experts produce token outputs',output,[edge('tokens','tokenFeatures'),edge('dispatch','chosenExpert')],{tokens,dispatch:choice},
  'y[t] = accepted[t] ? X[t]W[expert[t]] : 0','查看任意被接收 Token 的真实专家 W 和局部乘加；超容量 Token 直接走零输出，不会暗中计算专家层。');
 combined.substeps=ref.routed.slice(0,12).map((route,i)=>{
  const feature=values([args.tokens[i]]),actual=values([ref.output[i]]);
  const inputW=args.expertWeights[route.expert],D=inputW.length,H=inputW[0].length;
  const W=Core.numericTensor([H,D],Array.from({length:H*D},(_,p)=>inputW[p%D][Math.floor(p/D)]));
  const bias=Core.numericTensor([1,H],Array(H).fill(0));
  return {id:'out:token'+i,kind:route.accepted?'linear':'capacity_overflow',
   title:'Token '+i+' → Expert '+route.expert+(route.accepted?' · GEMM':' · capacity blocked'),
   formula:route.accepted?'y[0,j] = Σₖ token[k] W_expert[k,j]':'overflow(token) → zero output',
   why:route.accepted?'当前 Token 的实际路由专家权重参与点积；点击输出通道可展开全部加权乘积。':'该 Token 超出已选专家容量，实际执行器没有进行本次专家矩阵乘。',
   out:actual,operands:{input:feature,selectedExpertWeight:W},
   dot:route.accepted?{type:'dot',input:feature,weights:W,bias,orientation:'out_in'}:null};
 });
 steps.push(combined);
 const loss=point(ref.loss,0);
 steps.push(node('loss','mse','MSE loss',loss,[edge('out','prediction')],{prediction:output,target:values(args.target)},
  'L=mean((y−target)²)','将所有 Token 和输出通道的平方差平均。'));
 for(let i=0;i<ref.gradExpert.length;i++)steps.push(node('expert_grad_'+i,'vjp','Expert '+i+' weight gradient',values(ref.gradExpert[i]),
  [edge('loss','loss'),edge('dispatch','routedTokens')],{tokens,output},
  'dW_e=Σ_{t:selected(e)} X_tᵀ·dY_t','只有真正进入该专家的 Token 对其权重梯度有贡献。'));
 steps.push(node('input_grad','vjp','Input gradient',values(ref.gradInput),[edge('loss','loss')],{tokens},
  'dX[t]=dY[t]·W[expert[t]]ᵀ','被容量拒绝的 Token 不产生专家计算路径的输入梯度。'));
 return finalize('Top-1 MoE：Token→Router→Dispatch→Expert→梯度',steps);
}
function ppo(args,result){
 need(args&&Array.isArray(args.logits)&&Array.isArray(args.oldLogits),'complete PPO inputs required');
 const ref=M.ppo(args.logits,args.oldLogits,args.action,args.advantage,args.epsilon);
 need(JSON.stringify(ref)===JSON.stringify(result),'PPO results disagree with real reference engine');
 const steps=[
 node('new_logits','input','New policy logits',row(args.logits),[],{},'z_new','当前策略将被求导。'),
 node('old_logits','input','Old policy frozen logits',row(args.oldLogits),[],{},'z_old','固定旧策略产生概率比分母，不传播新策略梯度。'),
 node('prob_new','softmax','New policy probabilities',row(ref.probabilities),[edge('new_logits','logits')],{},
  'p_new=softmax(z_new)','稳定 softmax 归一化整个 Action 分布。'),
 node('prob_old','softmax','Old policy probabilities',row(ref.oldProbabilities),[edge('old_logits','logits')],{},
  'p_old=softmax(z_old)','相同 Action 的快照概率定义离策略 Ratio。'),
 node('ratio','division','Policy probability ratio',point(ref.ratio),[edge('prob_new','numerator'),edge('prob_old','denominator')],{},
  'r=p_new[action]/p_old[action]','比值只比较选定 Action 的概率。'),
 node('clipped','clamp','Clipped surrogate branch',point(ref.clippedRatio),[edge('ratio','r')],{},
  'clip(r,1−ε,1+ε)','该分支对 Ratio 做区间裁剪。'),
 node('loss','minimum','PPO clipped surrogate loss',point(ref.loss),[edge('ratio','unclipped'),edge('clipped','clipped')],{},
  'L=−min(r A,clip(r) A)','优势符号改变目标的有效裁剪方向。'),
 node('gradient','vjp','dL/d policy logits',row(ref.gradLogits),[edge('loss','loss'),edge('prob_new','distribution')],{},
  '∂L/∂z = active ? −A r (1_action−p) : 0','观察裁剪激活与否对所有 logits 梯度的直接影响。')
 ];
 return finalize('PPO：策略概率比、Clip 分支与 Logit VJP',steps);
}
return Object.freeze({tape,moe,ppo});
});