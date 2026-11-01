/* Shared bounded research exercises: video FFN sparsity, LLM post-training,
 * and Agent orchestration. Distinct evidence tiers: exact synthetic Wan FFN
 * channel ablation; toy scalar preference math; scripted state transitions.
 * Nothing here executes Wan video generation, LLM training, or real tools. */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?require("./wan_ffn_trace.js"):root.SeetheTruthWanFFNTrace);
 if(node)module.exports=api;
 if(root)root.SeetheTruthResearchTracks=api;
})(typeof window!=="undefined"?window:null,function(Wan){"use strict";
 if(!Wan||typeof Wan.trace!=="function")throw Error("Wan FFN trace engine unavailable");
 const finite=(x,min,max,label)=>{
  if(typeof x!=="number"||!Number.isFinite(x)||x<min||x>max)
   throw Error(label+" must be a finite number in ["+min+","+max+"]");
  return x;
 };
 const select=(scores,k)=>scores.map((score,h)=>({score,h}))
  .sort((a,b)=>b.score-a.score||a.h-b.h).slice(0,k).map(x=>x.h).sort((a,b)=>a-b);
 const sumsq=xs=>xs.reduce((s,v)=>s+v*v,0);
 function videoSparsity({seed=17,keep=4,selection="gated",scope="dynamic"}={}){
  finite(seed,1,2147483647,"seed");
  if(!Number.isSafeInteger(seed))throw Error("seed must be integer");
  const d=Wan.trace({seed,tokens:3,dim:4,hidden:8}),{tokens:T,dim:C,hidden:H}=d.config;
  if(!Number.isSafeInteger(keep)||keep<1||keep>H)throw Error("keep must be 1..hidden");
  if(!["gated","activation"].includes(selection)||!["static","dynamic"].includes(scope))
   throw Error("invalid sparsity policy");
  const act=d.steps[5].main.values,w2=d.config.w2,e5=d.e[5],gate=d.config.gate;
  const contribution=(t,h,c)=>act[t*H+h]*w2[c*H+h]*gate*e5[t*C+c];
  const score=(t,h)=>selection==="activation"?act[t*H+h]**2:
   Array.from({length:C},(_,c)=>contribution(t,h,c)**2).reduce((a,b)=>a+b,0);
  const global=select(Array.from({length:H},(_,h)=>
   Array.from({length:T},(_,t)=>score(t,h)).reduce((a,b)=>a+b,0)),keep);
  const kept=Array.from({length:T},(_,t)=>scope==="static"?global.slice():
   select(Array.from({length:H},(_,h)=>score(t,h)),keep));
  const dense=d.final.values.slice(),sparse=[],dropped=[],perToken=[];
  let separateSquared=0;
  for(let t=0;t<T;t++){
   const missing=Array.from({length:C},()=>0),set=new Set(kept[t]);
   for(let h=0;h<H;h++)if(!set.has(h)){
    const vector=Array.from({length:C},(_,c)=>contribution(t,h,c));
    separateSquared+=sumsq(vector);
    vector.forEach((v,c)=>{missing[c]+=v;});
   }
   const values=missing.map((v,c)=>dense[t*C+c]-v);
   sparse.push(...values);dropped.push(...missing);
   perToken.push({token:t,kept:kept[t],errorL2:Math.sqrt(sumsq(missing)),
    denseDeltaL2:Math.sqrt(sumsq(d.steps[7].main.values.slice(t*C,(t+1)*C)))});
  }
  const residualSquared=sumsq(dropped),deltaSquared=sumsq(d.steps[7].main.values);
  return Object.freeze({
   evidence:"SOURCE_PINNED_TOPOLOGY_SYNTHETIC_FLOAT64",
   source:d.source,shape:[T,C],hidden:H,keep,selection,scope,kept,
   dense,sparse,dropped,perToken,
   errorL2:Math.sqrt(residualSquared),
   relativeDeltaError:Math.sqrt(residualSquared/Math.max(deltaSquared,1e-24)),
   cancellationRatio:separateSquared>0?residualSquared/separateSquared:0,
   idealMacFraction:scope==="static"?keep/H:(1+keep/H)/2,
   costNote:scope==="static"?
    "假定 W1/W2 都实际进行结构化通道裁剪；理想 MAC 比例不包含选择和访存开销。":
    "逐 Token 选择依赖完整 W1/GELU 激活；W1 必须 Dense。特别是 gated 评分还读取全部 W2 通道贡献，是诊断 oracle，不是低成本可执行选择器。此比例是假定另有免费 Mask 的反事实 MAC，非可实现加速。",
   note:"真实 Wan FFN 计算拓扑 + 小型合成权重。Token 不是视频帧；输出误差不是视频质量或时序一致性。"
  });
 }
 const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
 function dpo({chosen=-0.7,rejected=-1.5,refChosen=-1,refRejected=-1,beta=0.5}={}){
  for(const [k,v] of Object.entries({chosen,rejected,refChosen,refRejected}))
   finite(v,-30,0,k);
  finite(beta,0,10,"beta");
  const margin=(chosen-rejected)-(refChosen-refRejected),z=beta*margin;
  const loss=Math.log1p(Math.exp(-z)),gradChosen=-beta*sigmoid(-z);
  return Object.freeze({objective:"DPO",margin,logit:z,loss,
   gradChosen,gradRejected:-gradChosen,preferenceProbability:sigmoid(z),
   evidence:"TOY_ANALYTIC_LOGPROB_OBJECTIVE",note:"完整应答 logπ 的配对标量目标；没有模型参数、Tokenizer、批次或真实更新。"});
 }
 function grpo({rewardGap=1,ratioShift=0.3,epsilon=0.2}={}){
  finite(rewardGap,0,3,"rewardGap");finite(ratioShift,-0.9,0.9,"ratioShift");
  finite(epsilon,0.01,0.9,"epsilon");
  const rewards=[rewardGap,0,-rewardGap],mean=rewards.reduce((a,b)=>a+b,0)/3;
  const variance=rewards.reduce((a,b)=>a+(b-mean)**2,0)/3;
  const sd=Math.sqrt(variance),advantage=rewards.map(r=>sd>1e-12?(r-mean)/sd:0);
  const ratios=[1+ratioShift,1,1-ratioShift];
  const terms=ratios.map((ratio,i)=>{
   const a=advantage[i],clipped=Math.max(1-epsilon,Math.min(1+epsilon,ratio));
   const raw=ratio*a,guarded=clipped*a;
   return {ratio,advantage:a,surrogate:Math.min(raw,guarded),
    clipActive:guarded<raw && (ratio<1-epsilon||ratio>1+epsilon)};
  });
  return Object.freeze({objective:"GRPO_CLIPPED_CORE",rewards,mean,std:sd,terms,
   loss:-terms.reduce((a,b)=>a+b.surrogate,0)/terms.length,
   evidence:"TOY_ANALYTIC_GROUP_ADVANTAGES",
   note:"单组 3 应答、每应答一个等权 token 的裁剪代理项；未包含参考 KL、长度 Mask、真实 rollout 或参数更新。"});
 }
 // Scripted transitions make control-flow and tool/approval boundaries visible.
 // They do not model LLM reasoning accuracy or execute external tools.
 function agent({architecture="react",fault=false,approved=true}={}){
  if(!["react","plan_execute","supervisor"].includes(architecture))
   throw Error("unknown agent architecture");
  if(typeof fault!=="boolean"||typeof approved!=="boolean")
   throw Error("fault and approved must be boolean");
  let phase="start",hasObservation=false,hasApproval=false,hasWrite=false,verified=false;
  const events=[];
  const emit=(actor,action,next,detail,ok=true)=>{
   const before=phase,id="e"+events.length;
   events.push(Object.freeze({id,dependsOn:events.length?["e"+(events.length-1)]:[],
    actor,action,before,after:next,ok,detail}));
   phase=next;
  };
  const planner=architecture==="supervisor"?"supervisor":"assistant";
  emit(planner,architecture==="react"?"decide":"plan","planned",
   architecture==="react"?"观察目标，决定下一次工具调用":"列出读取、审批、写入和核验步骤");
  emit(architecture==="supervisor"?"worker-A":"assistant","read_tool",
   fault?"read_failed":"observed",fault?"注入一次暂时性工具失败":"读取了模拟的必要证据",!fault);
  if(fault){
   if(architecture==="plan_execute")
    return Object.freeze({architecture,fault,approved,events,status:"failed",verified:false,
     evidence:"SCRIPTED_STATE_MACHINE",note:"Plan-Execute 的此教学策略不重试；实际系统可配置重试。"});
   emit(architecture==="supervisor"?"worker-B":"assistant","retry_read","observed",
    architecture==="supervisor"?"Supervisor 把工具任务移交给 worker-B":"ReAct 依据错误反馈重试读取");
  }
  hasObservation=true;
  emit(planner,"request_approval",approved?"approved":"blocked",
   approved?"获得模拟写操作审批":"没有审批，不执行写入",approved);
  if(!approved)return Object.freeze({architecture,fault,approved,events,status:"blocked",
   verified:false,evidence:"SCRIPTED_STATE_MACHINE",note:"没有真实外部写入；审批是显式控制边界。"});
  hasApproval=true;
  if(!hasObservation||!hasApproval)throw Error("write invariant violated");
  emit(architecture==="supervisor"?"worker-B":"assistant","write_tool","written","执行模拟写入");
  hasWrite=true;
  if(!hasWrite)throw Error("verification invariant violated");
  emit(architecture==="supervisor"?"reviewer":"assistant","verify","verified","独立核对模拟写入结果");
  verified=true;
  if(!verified)throw Error("commit invariant violated");
  emit(planner,"finalize","completed","仅在核验后完成任务");
  return Object.freeze({architecture,fault,approved,events,status:"completed",verified,
   evidence:"SCRIPTED_STATE_MACHINE",
   note:"同一固定任务的脚本化控制策略对照，非真实 LLM 决策、Agent 成功率或工具调用。"});
 }
 return Object.freeze({videoSparsity,dpo,grpo,agent});
});
