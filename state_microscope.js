/* Universal, bounded EVENT/STATE microscope. Provenance is always a scoped
 * simulation contract; events and transitions are obtained from validated
 * engines, never from a fabricated tensor/GPU telemetry stream. */
(function(root,make){'use strict';const api=make();if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthStateMicroscope=api;})(typeof window!=='undefined'?window:{},function(){'use strict';
const SCHEMA='seethetruth-state-microscope-v1';
const TIERS=Object.freeze({KV:'SOURCE_GROUNDED_KV_CPU_STATE_SIMULATION',WORKER:'SOURCE_GROUNDED_WORKER_ASSUMED_DURATIONS',SCHEDULER:'SOURCE_GROUNDED_DECODE_SCHEDULER_CPU_REPLAY',PREFILL:'SOURCE_GROUNDED_PREFILL_DECISION_CPU_REFERENCE',CACHE:'EXECUTED_KV_GQA_MICRO_NUMERIC_CACHE',TRAIN:'EXECUTABLE_BIGRAM_OPTIMIZER_STATE_SIMULATION',SCALAR:'EXECUTABLE_SCALAR_ANALYTIC_TRAINING'});
const fail=m=>{throw Error('state microscope: '+m);};
const need=(cond,m)=>{if(!cond)fail(m);};
const plain=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const finite=v=>typeof v==='number'&&Number.isFinite(v);
function values(x,depth=0){
 need(depth<=5,'state field nesting too deep');
 if(x===null||typeof x==='string'||typeof x==='boolean')return true;
 if(typeof x==='number'){need(finite(x),'nonfinite state value');return true;}
 if(Array.isArray(x)){need(x.length<=160,'oversized state array');for(const v of x)values(v,depth+1);return true;}
 need(plain(x)&&Object.keys(x).length<=48,'invalid state structure');
 for(const [k,v]of Object.entries(x)){need(/^[A-Za-z0-9_\-]{1,72}$/.test(k),'invalid state key');values(v,depth+1);}
 return true;
}
function delta(a,b){
 const keys=[...new Set([...Object.keys(a),...Object.keys(b)])].sort();
 return keys.filter(k=>JSON.stringify(a[k])!==JSON.stringify(b[k])).map(key=>({key,before:Object.hasOwn(a,key)?a[key]:null,after:Object.hasOwn(b,key)?b[key]:null}));
}
function validate(frame){
 need(frame?.schema===SCHEMA,'wrong schema');
 need(Object.values(TIERS).includes(frame.evidenceTier),'unknown evidence tier');
 need(typeof frame.title==='string'&&frame.title.length>0&&frame.title.length<140,'invalid title');
 need(Array.isArray(frame.steps)&&frame.steps.length>0&&frame.steps.length<=250,'invalid steps');
 need(typeof frame.disclaimer==='string'&&frame.disclaimer.length>0,'missing epistemic boundary');
 let prev=-1;const ids=new Set();
 for(const [index,e]of frame.steps.entries()){
  need(e&&typeof e.id==='string'&&/^[A-Za-z0-9_:-]{1,80}$/.test(e.id)&&!ids.has(e.id),'invalid/duplicate event ID');
  need(e.index===index&&e.index>prev,'steps out of order');
  need(typeof e.title==='string'&&e.title.length>0&&typeof e.why==='string'&&e.why.length>0,'missing event explanation');
  need(plain(e.before)&&plain(e.after)&&Array.isArray(e.changes),'state before/after missing');
  values(e.before);values(e.after);
  need(JSON.stringify(e.changes)===JSON.stringify(delta(e.before,e.after)),'inconsistent state changes');
  need(Array.isArray(e.dependencies)&&e.dependencies.every(id=>typeof id==='string'&&ids.has(id)),'invalid causal dependencies');
  if(e.timing){need(finite(e.timing.start)&&finite(e.timing.end)&&e.timing.start>=0&&e.timing.end>=e.timing.start,'invalid simulated timestamp');}
  ids.add(e.id);prev=e.index;
 }
 return frame;
}
function kv(replay,KV){
 need(replay&&Array.isArray(replay.timeline)&&replay.timeline.length>=2&&replay.timeline.length<=81&&KV&&typeof KV.verify==='function','invalid KV replay');
 const steps=[];for(let i=0;i<replay.timeline.length;i++){
  const s=replay.timeline[i];KV.verify(s);
  const b=i?replay.timeline[i-1]:s;
  need(i===0||s.revision===b.revision+1,'KV revisions must advance by one');
  const op=s.events.at(-1);
  const project=x=>({revision:x.revision,freeQueue:x.free.slice(),refs:x.refs.slice(),
   owners:Object.fromEntries(Object.entries(x.requests).map(([id,v])=>[id,{status:v.status,blocks:v.blocks.slice(),cachedTokens:v.cachedTokens,recomputes:v.recomputes}]))});
  const before=project(b),after=project(s);
  steps.push({id:'kv:'+i,index:i,title:i?'操作 '+op?.op+' · '+(op?.id??''):'初始化 Block Pool',
   why:i?'每次操作都由原引擎校验物理块引用计数和 FIFO 空闲列表，展示的是 CPU 参考逻辑，不是设备显存实际分配。':'物理块 0 永远保留；其它块按 FIFO 维护引用计数。',
   dependencies:i?['kv:'+(i-1)]:[],before,after,changes:delta(before,after),timing:null});
 }
 return validate({schema:SCHEMA,title:'KV BlockManager · 状态转换的通用显微镜',evidenceTier:TIERS.KV,steps,
   disclaimer:'Verified small CPU block/refcount transitions, NOT GPU blocks, runtime Scheduler or measured time'});
}
function worker(run){
 need(run&&Array.isArray(run.events)&&run.events.length>=1&&run.events.length<=250,'invalid worker event run');
 const completed=new Map(),seen=new Set(),steps=[],busy={CPU:0,GPU:0};
 for(let i=0;i<run.events.length;i++){
  const e=run.events[i];
  need(e&&(e.resource==='CPU'||e.resource==='GPU')&&typeof e.stage==='string'&&Number.isSafeInteger(e.step),'invalid worker event');
  need(finite(e.start)&&finite(e.end)&&finite(e.duration)&&e.start>=0&&e.end>=e.start&&Math.abs(e.end-e.start-e.duration)<1e-9,'invalid assumed event timing');
  need(!seen.has(e.id),'duplicate worker event');
  need(Array.isArray(e.dependsOn)&&e.dependsOn.every(dep=>completed.has(dep)&&completed.get(dep)<=e.start+1e-9),'worker dependency order violated');
  need(e.start+1e-9>=busy[e.resource],'same resource overlaps');
  const before={completedEvents:i,resourceBusy:{...busy}},after={completedEvents:i+1,resourceBusy:{...busy,[e.resource]:e.end}};
  steps.push({id:e.id,index:i,title:'Step '+e.step+' · '+e.stage,
    why:'演示 CPU/GPU 资源排队和跨步因果依赖。Duration 来自可修改的假设输入，不能解释为实测硬件延迟。',
    dependencies:e.dependsOn.slice(),before,after,changes:delta(before,after),timing:{start:e.start,end:e.end,resource:e.resource}});
  busy[e.resource]=e.end;seen.add(e.id);completed.set(e.id,e.end);
 }
 return validate({schema:SCHEMA,title:'DeepInfer Worker · 跨 Step 假设时序',evidenceTier:TIERS.WORKER,steps,
   disclaimer:'Durations are assumed logical units, NOT profiler data, GPU utilization or production throughput'});
}
function scheduler(run,K){
 need(run&&Array.isArray(run.decisions)&&run.decisions.length>0&&
  Array.isArray(run.snapshots)&&run.snapshots.length===run.decisions.length+1&&
  Array.isArray(run.executionProof)&&run.executionProof.length===run.decisions.length&&
  run.worker&&Array.isArray(run.worker.events)&&K&&typeof K.verify==='function','scheduler requires full verified decision/worker lineage');
 const steps=[],project=x=>({revision:x.revision,freeQueue:x.free.slice(),refs:x.refs.slice(),
  owners:Object.fromEntries(Object.entries(x.requests).map(([id,v])=>[id,{status:v.status,blocks:v.blocks.slice(),cachedTokens:v.cachedTokens,recomputes:v.recomputes}]))});
 const allEvents=new Map(run.worker.events.map(e=>[e.id,e]));
 for(let i=0;i<run.snapshots.length;i++){
  const s=run.snapshots[i];K.verify(s);
  if(i===0){
   const state=project(s);steps.push({id:'scheduler:0',index:0,title:'初始化 Scheduler / BlockManager',
    why:'先复原输入的 KV 资源所有权和等待队列；与实际 GPU BlockManager 无关。',
    dependencies:[],before:state,after:state,changes:delta(state,state),timing:null});continue;
  }
  const before=project(run.snapshots[i-1]),after=project(s),d=run.decisions[i-1],proof=run.executionProof[i-1];
  need(d.cycle===i-1&&d.kvRevision===s.revision,'decision/source KV revision mismatch');
  need(Array.isArray(d.actions)&&Array.isArray(d.scheduled)&&Array.isArray(d.preempted)&&Array.isArray(d.waiting)&&
   Array.isArray(d.blocked)&&plain(d.dispatchBlocks)&&Array.isArray(d.freeAfter),'incomplete source scheduler decision');
  need(JSON.stringify(d.freeAfter)===JSON.stringify(s.free),'scheduler free-list mismatch');
  need(proof.cycle===i-1&&JSON.stringify(proof.batch)===JSON.stringify(d.scheduled)&&
   JSON.stringify(proof.physicalBlockTables)===JSON.stringify(d.dispatchBlocks),'worker dispatch differs from proven scheduler decision');
  const evt=allEvents.get(proof.forwardEvent);
  need(evt&&evt.stage==='forward'&&evt.step===i-1,'Worker Forward event does not match scheduling decision');
  need(run.worker.steps[i-1]?.sequenceIds.join(',')===d.scheduled.join(','),'Worker sequence mismatch');
  const next={...after,scheduled:d.scheduled.slice(),preempted:d.preempted.map(x=>({id:x.id,outcome:x.outcome})),
   waiting:d.waiting.slice(),blocked:d.blocked.map(x=>({id:x.id,status:x.status})),
   forwardEvent:proof.forwardEvent,dispatchBlocks:d.dispatchBlocks};
  const prev={...before,scheduled:[],preempted:[],waiting:[],blocked:[],forwardEvent:null,dispatchBlocks:{}};
  steps.push({id:'scheduler:'+i,index:i,title:'Decode Cycle '+(i-1)+' · Queue → KV → Worker',
   why:'由 Scheduler CPU 判定、BlockManager refs/FIFO 和 Worker Forward 同步导出；blocked 的 Prefill Admission 仍是明确未实现的独立路径。',
   dependencies:['scheduler:'+(i-1)],before:prev,after:next,changes:delta(prev,next),timing:null});
 }
 return validate({schema:SCHEMA,title:'Decode Scheduler · 资源及 Worker 派发因果透视',evidenceTier:TIERS.SCHEDULER,steps,
  disclaimer:'Verified bounded scheduler decisions + CPU KV ownership + logical Worker Forward link; no real C++ execution, GPU allocation, prompt scheduling or throughput measurement'});
}
function prefill(execution,P){
 need(execution&&plain(execution.voteInput)&&plain(execution.chunkInput)&&plain(execution.voteResult)&&
  plain(execution.chunkResult)&&P&&typeof P.vote==='function'&&typeof P.chunk==='function','prefill needs complete policy input/result evidence');
 const vote=P.vote(execution.voteInput),chunk=P.chunk(execution.chunkInput);
 need(JSON.stringify(vote)===JSON.stringify(execution.voteResult),'Prefill vote result contradicts source reference');
 need(JSON.stringify(chunk)===JSON.stringify(execution.chunkResult),'Prefill chunk result contradicts source reference');
 const inputs=[
  {id:'prefill:vote',title:'NORMAL ALL / ANY · DP Rank 投票',why:'检查当前 Decode Step、间隔、各 Rank 是否想执行及是否真能排上；第二轮投票避免空 Prompt。',
   before:{input:execution.voteInput},after:{input:execution.voteInput,decision:vote}},
  {id:'prefill:chunk',title:'IMG / PAD Token · Chunk 预算与边界',why:'逐 Token 判断图片 PAD 的完整性，受剩余 Batch 和 CP Align 约束；这是单独的 Chunk 决策，和投票不强行伪造时序连接。',
   before:{input:execution.chunkInput},after:{input:execution.chunkInput,decision:chunk}}
 ];
 const steps=inputs.map((p,i)=>({id:p.id,index:i,title:p.title,why:p.why,dependencies:[],
  before:p.before,after:p.after,changes:delta(p.before,p.after),timing:null}));
 return validate({schema:SCHEMA,title:'DeepInfer Prefill · 投票和安全 Chunk 独立决策',evidenceTier:TIERS.PREFILL,steps,
  disclaimer:'Exact bounded source-derived policy outputs re-executed and checked; not GPU KV allocation, distributed collectives, image embeddings or real serving latency'});
}
function cache(snapshots,Cache){
 need(Array.isArray(snapshots)&&snapshots.length>0&&snapshots.length<=13&&
  snapshots[0]?.config&&Cache&&typeof Cache.Session==='function','cache needs coherent complete session snapshots');
 const session=Cache.Session(snapshots[0].config);
 const selected=[];
 for(let i=0;i<snapshots.length;i++){
  const actual=i?session.advance():session.snapshot();
  need(JSON.stringify(actual)===JSON.stringify(snapshots[i]),'KV micro cache state contradicts deterministic source replay at '+i);
  const last=actual.last;
  selected.push({phase:actual.phase,cursor:actual.cursor,cacheTokens:actual.cache.length,
   keyShape:actual.cache.keyShape,valueShape:actual.cache.valueShape,
   newPositions:last?.kAppendedPositions||[],queryPositions:last?.queryPositions||[],
   appendCount:last?.appended||0,flops:last?.mathMatmulFlops||0});
 }
 const steps=selected.map((state,i)=>{
  const prior=i?selected[i-1]:state;
  return {id:'cache:'+i,index:i,title:i===0?'初始化数学 KV Cache':'Step '+i+' · '+state.phase+' / Query '+state.queryPositions.join(','),
   why:i===0?'没有做 Token 投影；实际 K/V 缓存为空。':i===1?'Prefill 生成全部前缀 Q/K/V，一次性写入全部新 K/V。':'Decode 只投影新 token 的 K/V，再与全部缓存 Key/Value 计算真实 GQA/RoPE 小张量。',
   dependencies:i?['cache:'+(i-1)]:[],before:prior,after:state,changes:delta(prior,state),timing:null};
 });
 return validate({schema:SCHEMA,title:'数学 GQA · 持久 K/V 缓存与 Token 读写',evidenceTier:TIERS.CACHE,steps,
  disclaimer:'Byte-exact deterministic small projected K/V replay with RoPE/GQA reference; no production model weights, GPU KV page allocator or true device speed'});
}
function training(snapshots){
 need(Array.isArray(snapshots)&&snapshots.length>=1&&snapshots.length<=32,'invalid training transcript');
 let source=null,params=null,previous=null;
 const steps=snapshots.map((snap,i)=>{
  need(snap?.schema==='seethetruth-training-pipeline-v1'&&snap.dataFingerprint&&snap.model&&snap.sampler&&snap.pending,'invalid checkpoint schema');
  need(Array.isArray(snap.model.weights)&&snap.model.weights.length===64&&snap.model.weights.every(finite),'invalid Bigram checkpoint weights');
  need(Array.isArray(snap.model.firstMoment)&&snap.model.firstMoment.length===64&&
   snap.model.firstMoment.every(finite)&&Array.isArray(snap.model.secondMoment)&&
   snap.model.secondMoment.length===64&&snap.model.secondMoment.every(finite),
   'invalid native optimizer moment arrays');
  need(Array.isArray(snap.pending.gradientSum)&&snap.pending.gradientSum.length===64&&
   snap.pending.gradientSum.every(finite)&&Array.isArray(snap.pending.indices)&&
   snap.pending.indices.length<=64&&snap.pending.indices.every(Number.isSafeInteger),
   'invalid native pending gradient provenance');
  need(Number.isSafeInteger(snap.model.step)&&snap.model.step>=0&&snap.model.step<=1e6,'invalid optimizer step');
  need(Number.isSafeInteger(snap.sampler.epoch)&&snap.sampler.epoch>=0&&Number.isSafeInteger(snap.sampler.cursor)&&snap.sampler.cursor>=0,'invalid sampler position');
  need(Number.isSafeInteger(snap.pending.microSteps)&&snap.pending.microSteps>=0&&Number.isSafeInteger(snap.pending.samples)&&snap.pending.samples>=0&&finite(snap.pending.lossSum),'invalid pending gradients');
  const contract=JSON.stringify({dataFingerprint:snap.dataFingerprint,config:snap.config});
  if(source===null)source=contract;else need(contract===source,'training data/config identity changed');
  if(previous)need(snap.model.step>=previous.model.step,'optimizer step went backwards');
  const x={optimizerStep:snap.model.step,epoch:snap.sampler.epoch,cursor:snap.sampler.cursor,
   pendingMicroSteps:snap.pending.microSteps,pendingSamples:snap.pending.samples,
   pendingLossSum:snap.pending.lossSum,pendingIndices:snap.pending.indices.slice(),
   weightsChecksum:snap.model.weights.reduce((acc,v,j)=>acc+(j+1)*v,0),
   firstMomentChecksum:snap.model.firstMoment.reduce((acc,v,j)=>acc+(j+1)*v,0),
   secondMomentChecksum:snap.model.secondMoment.reduce((acc,v,j)=>acc+(j+1)*v,0),
   pendingGradientChecksum:snap.pending.gradientSum.reduce((acc,v,j)=>acc+(j+1)*v,0)};
  const before=params||x,after=x;
  const evt={id:'train:'+i,index:i,title:'检查点阶段 '+(i+1)+' · optimizer.step='+x.optimizerStep,
   why:'比较真实 Bigram 教学训练引擎生成的 W/Step、Sampler 和梯度缓冲状态；不是 PyTorch/GPU 模型权重或训练时间。',
   dependencies:i?['train:'+(i-1)]:[],before,after,changes:delta(before,after),timing:null};
  params=x;previous=snap;return evt;
 });
 return validate({schema:SCHEMA,title:'Bigram · 训练/续训状态转换',evidenceTier:TIERS.TRAIN,steps,
  disclaimer:'Deterministic small Bigram checkpoints with optimizer/sampler pending state; NOT a large-model training trace'});
}
function scalar(run){
 need(run&&run.mode==='scalar'&&run.evidence==='deterministic_scalar_analytic_oracle_not_autograd'&&
  Array.isArray(run.events)&&run.events.length>=1&&run.events.length<=250,'scalar training requires executed analytic reference events');
 const steps=run.events.map((event,i)=>{
  need(event&&event.id==='e'+i&&plain(event.state)&&typeof event.label==='string'&&
   typeof event.formula==='string'&&typeof event.detail==='string','scalar event evidence missing');
  const copy=x=>({theta:x.theta,grad:x.grad,m:x.m,v:x.v,optimizerStep:x.step});
  const after=copy(event.state),before=i?copy(run.events[i-1].state):after;
  return {id:'scalar:'+i,index:i,title:(i+1)+'. '+event.label,
   why:event.detail+' · '+event.formula,
   dependencies:i?['scalar:'+(i-1)]:[],before,after,changes:delta(before,after),timing:null};
 });
 return validate({schema:SCHEMA,title:'标量训练 · 前向/反向/梯度缓冲/Optimizer',evidenceTier:TIERS.SCALAR,steps,
  disclaimer:'Actual scalar analytic derivative and SGD/AdamW state arithmetic; not PyTorch autograd or GPU training'});
}
return Object.freeze({SCHEMA,TIERS,validate,kv,worker,scheduler,prefill,cache,training,scalar,delta});
});