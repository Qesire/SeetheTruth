/* Interactive six-step bridge between the introductory journey and real training.
 * Every successful task executes source-backed numerical model transitions.
 */
(function(){'use strict';
const P=window.SeetheTruthTrainingPipeline,E=window.SeetheTruthEngineeringEngine;
if(!P||!E)throw Error('pipeline or numerical engine missing');
const $=id=>document.getElementById(id), fmt=x=>Number(x).toFixed(7),trace=a=>a.join('\n');
const tasks=[
 {category:'01 · DATALOADER',title:'Shuffle、Epoch 和 Batch Cursor',
  goal:'在固定 samplerSeed 的条件下恢复 Epoch 内的样本排列；同一种子/同一 Epoch 必须获得相同顺序。',
  task:'初始化一个独立训练流水线，观察当前 Epoch 前 8 个 Pair 索引和下一个 Microbatch。只做预览，不可改变权重或游标。',
  question:'仅保存模型权重和优化器状态，能精确恢复 DataLoader 下一个 Batch 吗？',
  options:['不能；还需要 Sampler/Epoch/Cursor 状态','能；只要固定 batch size','能；数据文件不变即可'],correct:0,
  math:'order = FisherYates(seed, epoch)\nbatch = order[cursor : cursor + microBatchSize]\nstate = (samplerSeed, epoch, cursor)',
  why:'工业训练中的 Sample 顺序会改变梯度轨迹。精确断点续训需要记录 DataLoader 的采样状态以及数据版本，不能仅靠 seed。',
  failure:'恢复模型后重新洗牌但把游标置零：样本重复或漏读，训练轨迹无法复现。',links:[['Tokenizer 与标签','engineering_journey.html']]},
 {category:'02 · BACKWARD',title:'一次 Microbatch 只做梯度积累',
  goal:'验证一次 microStep 产生可计算 CE 与梯度，却不会执行 AdamW、增加 optimizer.step 或修改权重。',
  task:'执行第一批前向/反向并累计梯度，同时检查样本索引、Gradient Buffer 状态及当前权重摘要。',
  question:'正确的 Gradient Accumulation 中，每处理一个 Microbatch 就更新 W 吗？',
  options:['是，每个 micro 都做 optimizer.step','否，只在累积窗口完成后更新','训练不需要 Gradient Buffer'],correct:1,
  math:'micro_loss, micro_grad = forward_backward(microbatch)\ngrad_sum += micro_grad * microbatch_size\n# W, m, v and optimizer_step stay unchanged',
  why:'有效 Batch 可拆成更小的激活内存占用单元。每个 Microbatch 贡献梯度；只有有效 Batch 的全部样本处理完才做一次更新。',
  failure:'每个 Microbatch 都调用 optimizer.step，得到的是多个参数更新，而不是等价的大 Batch 反向计算。',links:[['Autograd 积木','training.html']]},
 {category:'03 · ACCUMULATION',title:'完整有效 Batch 与等价性证明',
  goal:'把尚未完成的 Microbatch 累积至指定次数；仅执行一次 AdamW，然后与单个等大小 Batch 的独立运行逐参数对照。',
  task:'剩余 microStep 完成后除以总样本数，执行梯度裁剪与优化器更新。比较 split 与 full Batch 的 Loss/64 个权重。',
  question:'比较大小不同的 Microbatch 时，怎样聚合才能等价于整体平均 Loss？',
  options:['不加权相加 micro 平均梯度','按每组样本数加权求和，再除以总样本数','每个 micro 独立更新动量'],correct:1,
  math:'g = sum(micro_count_i * grad_mean_i) / sum(micro_count_i)\ng = g * min(1,clip_norm / ||g||)\nAdamW.step(g) // exactly once',
  why:'多 Microbatch 与大 Batch 等价，需要对每组梯度按样本数加权，保持参数不在累积中改变，且保持相同样本顺序。浮点求和顺序可带来微小差异。',
  failure:'丢掉尾部 Batch 或遗漏梯度归一化，会改变学习率等效大小和参数更新。',links:[['可训练 Tensor Graph','learn.html']]},
 {category:'04 · LR SCHEDULER',title:'Warmup、Cosine 与梯度裁剪',
  goal:'实际推进 10 次优化器更新，同时观察 Warmup/Cosine 的每步学习率、梯度裁剪及验证集 CE。',
  task:'执行 10 个有效 Batch；核对优化器 step 与 Microstep 的差异、每步 LR 轨迹以及裁剪前后梯度范数。',
  question:'学习率调度器通常按什么推进才能与 optimizer 更新保持一致？',
  options:['按真实 optimizer.step 次数','按每个 Token 数无条件加一','按每次 UI 点击数'],correct:0,
  math:'lr(step) = lr_base * warmup_scale(step)     // early steps\nlr(step) = lr_base * cosine_decay(step)     // later\nclip(g,max_norm) happens before AdamW.step',
  why:'更新次数而非 Forward 次数决定优化器状态和学习率时序。梯度裁剪约束更新前梯度幅度，不等于限制参数更新范数。',
  failure:'在 Microbatch 上推进 Scheduler 会让 LR 衰减过快；先更新再裁剪则无法保护该次更新。',links:[['训练优化器','training.html']]},
 {category:'05 · RESUMABLE CHECKPOINT',title:'在累积到一半时中断并精确续训',
  goal:'在下一次 optimizer.step 之前主动中断，保存 model W/m/v/step 与 sampler epoch/cursor、pending 梯度及进度；重建后与无中断轨迹逐项相同。',
  task:'从当前状态复制两条训练分支：一条不停止连续执行 3 步；另一条处理部分 Microbatch 后 JSON 快照→恢复→继续 3 步。',
  question:'中途保存只含 W、m、v、step 的检查点是否足以精确恢复梯度累积？',
  options:['足够，因为梯度在模型里','不够，必须保留待提交的梯度与游标','只要额外存储验证 Loss 就够'],correct:1,
  math:'snapshot = {model:{W,m,v,step},sampler:{epoch,cursor},\n            pending:{gradientSum,samples,microSteps,indices}}\nrestore(snapshot); finish_pending(); step();',
  why:'现代大模型训练往往受抢占、故障重启或作业时限影响。真正的“续训”必须恢复整个可影响未来计算的状态，不只是推理权重。',
  failure:'没有 pending 梯度或错误的 sampler cursor，将把当前有效 Batch 静默改成不同的数据集合。',links:[['模型与状态合约','semantic_state.html']]},
 {category:'06 · INFERENCE HANDOFF',title:'从恢复后的 Checkpoint 进入 Decode',
  goal:'验证同一冻结训练快照产生相同的 Seeded Decode；训练实验与推理结果按模型 Step、数据指纹和来源绑定。',
  task:'对已恢复模型进行 Greedy 和带温度采样；核对来源标签、推理权重 step、固定随机种子的一致性，以及非法输入拒绝。',
  question:'单纯记住训练 Loss，能确认线上推理真正加载的是同一模型版本吗？',
  options:['不能，需要 Tokenizer/权重/模型版本与检查点绑定','能；Loss 相同就是同一权重','能；只有采样温度需要记录'],correct:0,
  math:'artifact = checkpoint(model, optimizer, sampler, config, data_id)\ninference = generate(weights=artifact.model.W, tokenizer=artifact.vocab)\nassert(replay_same_seed(inference))',
  why:'交接到 Serving 不只是函数调用，还涉及权重版本、Tokenizer、精度、导出校验、加载日志、回滚和流量控制。此处是数学交接，不是网络 API。',
  failure:'混用旧版词表或检查点，即使推理程序正常返回文本，语义也可能已经失真。',links:[['Decode / KV Cache','cache.html'],['Serving 调度','deepinfer_scheduler_lab.html']]}
];
const s={index:0,done:new Set(),responses:new Map(),logs:new Map(),pipeline:null,config:null,provenance:null,history:[]};
const State=window.SeetheTruthStateMicroscope;
const stateView=window.SeetheTruthStateMicroscopeUI.mount(document.getElementById('stateMicroscope'));
const getOpts=()=>({seed:Number($('modelSeed').value),samplerSeed:Number($('samplerSeed').value),lr:Number($('lr').value),
 microBatchSize:Number($('microBatchSize').value),accumulationSteps:Number($('accumulationSteps').value),
 maxGradNorm:Number($('maxGradNorm').value)});
const assert=(valid,msg)=>{if(!valid)throw Error(msg);};
function get(){assert(s.pipeline,'先初始化 Sampler 与模型');assert(JSON.stringify(getOpts())===JSON.stringify(s.config),'训练配置已改变；必须重置重新初始化');return s.pipeline;}
function updateMetrics(){
 const p=s.pipeline,d=p?.diagnostic();
 const rows=d?[['optimizer.step',d.modelStep],['micro.step pending',d.pendingMicroSteps+'/'+s.config.accumulationSteps],['epoch/cursor',d.epoch+'/'+d.cursor],
 ['effective batch',d.effectiveBatch],['next LR',fmt(d.nextLearningRate)],['train CE',fmt(p.evaluate('train').loss)],
 ['validation CE',fmt(p.evaluate('validation').loss)],['dataset fingerprint',d.dataFingerprint]]:
 [['optimizer.step','—'],['micro.step pending','—'],['epoch/cursor','—'],['effective batch','—'],['next LR','—'],['train CE','—'],['validation CE','—'],['dataset fingerprint',P.DATA_FINGERPRINT]];
 $('kpis').replaceChildren(...rows.map(([label,value])=>{const m=document.createElement('div');m.className='metric';
 const dt=document.createElement('dt');dt.textContent=label;const dd=document.createElement('dd');dd.textContent=String(value);m.append(dt,dd);return m;}));
 const rates=Array.from({length:24},(_,i)=>P.scheduledLr(s.config||P.defaults,i+1)),maximum=Math.max(...rates);
 $('schedule').replaceChildren(...rates.map((lr,i)=>{const bar=document.createElement('span');bar.style.height=Math.max(3,Math.round(lr/maximum*92))+'px';
 bar.style.background=i<(d?.modelStep||0)?'#e8c689':'#3bafa5';bar.title='optimizer step '+(i+1)+' · lr='+lr.toFixed(7);return bar;}));
 const weights=p?.checkpoint().model.weights||Array(64).fill(0),scale=Math.max(...weights.map(Math.abs),0.000001);
 $('weights').replaceChildren(...weights.map(x=>{const box=document.createElement('span');const v=Math.min(0.9,Math.abs(x)/scale*0.9);
 box.style.background=x>=0?'rgba(102,230,193,'+(0.12+v)+')':'rgba(236,159,112,'+(0.12+v)+')';
 box.title=x.toFixed(7);return box;}));
}
function render(){
 const l=tasks[s.index];$('category').textContent=l.category;$('lessonTitle').textContent=l.title;
 $('goal').textContent=l.goal;$('task').textContent=l.task;$('question').textContent=l.question;
 $('why').textContent=l.why;$('failure').textContent=l.failure;$('math').textContent=l.math;
 $('config').hidden=s.index!==0;
 $('answer').replaceChildren(new Option('先作出机制判断…',''));
 l.options.forEach((o,i)=>$('answer').add(new Option(o,String(i))));
 $('answer').value=s.responses.get(s.index)??'';
 const record=s.logs.get(s.index);$('log').textContent=record?.text||'请选择机制判断，然后执行。';
 $('log').dataset.status=record?.status||'idle';
 $('next').disabled=!s.done.has(s.index)||s.index>=tasks.length-1;
 const max=s.done.size?Math.min(tasks.length-1,Math.max(...s.done)+1):0;
 $('stages').replaceChildren(...tasks.map((task,i)=>{const b=document.createElement('button');b.className='stage';b.disabled=i>max;
 b.setAttribute('aria-current',String(i===s.index?'step':'false'));
 const n=document.createElement('span');n.className='stepid';n.textContent=String(i+1).padStart(2,'0');
 const title=document.createElement('span');title.textContent=task.title;b.append(n,title);
 if(s.done.has(i)){const mark=document.createElement('span');mark.className='done';mark.textContent='✓';b.append(mark);}
 b.addEventListener('click',()=>{s.responses.set(s.index,$('answer').value);s.index=i;render();});return b;}));
 $('progressText').textContent=s.done.size+' / '+tasks.length+' 通过';
 $('progressFill').style.width=(s.done.size/tasks.length*100)+'%';
 $('links').replaceChildren(...l.links.map(([title,url])=>{const a=document.createElement('a');a.href=url;a.textContent=title+' ↗';return a;}));
 updateMetrics();
}
function perform(i){
 if(i===0){
  const config=P.optionsOf(getOpts());s.pipeline=P.create(config);s.config=getOpts();s.provenance=null;
  const p=s.pipeline,peek=p.previewBatch(),order=P.shuffledIndices(config.samplerSeed,0);
  assert(new Set(order).size===E.pairs(E.TRAIN).length,'Sampler index permutation invalid');
  assert(p.diagnostic().cursor===0&&p.modelStep===0,'Preview must not modify state');
  return trace(['model seed = '+config.seed,'sampler seed = '+config.samplerSeed,
   'epoch 0 first 8 pair indices = '+order.slice(0,8).join(', '),
   'next microbatch indices = '+peek.indices.map(x=>x.index).join(', '),
   'effective batch = '+p.diagnostic().effectiveBatch,'sampler cursor after peek = '+p.diagnostic().cursor,
   'dataset identity = '+p.diagnostic().source,'dataset fingerprint (non-cryptographic) = '+P.DATA_FINGERPRINT]);
 }
 const p=get();
 if(i===1){const before=p.checkpoint(),r=p.microStep(),after=p.checkpoint();
  assert(before.model.step===after.model.step&&JSON.stringify(before.model.weights)===JSON.stringify(after.model.weights),'W changed before optimizer step');
  return trace(['micro forward/backward finished','pair IDs = '+r.indices.join(', '),'pairs = '+r.tokenPairs.map(x=>x.join('→')).join(' | '),
   'micro CE = '+fmt(r.microLoss),'micro grad L2 = '+fmt(r.gradNorm),
   'pending microsteps = '+r.microStep+'/'+r.requiredMicroSteps,'optimizer.step = '+after.model.step,'weights unchanged = true']);}
 if(i===2){
  const reference=P.create({...s.config,microBatchSize:s.config.microBatchSize*s.config.accumulationSteps,accumulationSteps:1});
  const full=reference.train(1);
  const lines=[];
  while(p.pendingMicroSteps<s.config.accumulationSteps){const m=p.microStep();lines.push('micro '+m.microStep+': indices '+m.indices.join(','));}
  const r=p.optimizerStep(),weights=p.checkpoint().model.weights,refW=reference.checkpoint().model.weights;
  const error=Math.max(...weights.map((x,k)=>Math.abs(x-refW[k])));
  assert(error<1e-12,'Large batch vs accumulated microbatches mismatch');
  return trace([...lines,'effective examples = '+r.examples,'optimizer.step = '+r.step,
   'CE micro/full = '+fmt(r.batchLoss)+' / '+fmt(full.batchLoss),'max |Wsplit-Wfull| = '+error.toExponential(3),
   'grad norm before/after clip = '+fmt(r.rawGradNorm)+' / '+fmt(r.clippedGradNorm),
   'clip scale = '+fmt(r.clipScale),'parameter update norm = '+fmt(r.updateNorm)]);
 }
 if(i===3){const before=p.evaluate('train'),logs=[];
  for(let k=0;k<10;k++){const u=p.train(1);logs.push('step '+u.step+': lr='+fmt(u.learningRate)+', grad='+fmt(u.rawGradNorm)+'→'+fmt(u.clippedGradNorm)+', CE='+fmt(u.train.loss));}
  const after=p.evaluate('train');assert(p.modelStep===11,'LR schedule must advance optimizer steps');
  return trace(['optimizer update count = 11 (previous 1 + additional 10)',...logs,
   'full train CE = '+fmt(before.loss)+' → '+fmt(after.loss),'validation CE = '+fmt(p.evaluate('validation').loss),
   'next step LR = '+fmt(p.diagnostic().nextLearningRate)]);
 }
 if(i===4){
  const start=JSON.parse(JSON.stringify(p.checkpoint())),control=P.create(p.config);control.restore(start);
  const interruption=Math.max(1,Math.floor(p.config.accumulationSteps/2));const consumed=[];
  for(let j=0;j<interruption;j++){const event=p.microStep();consumed.push(...event.indices);}
  const mid=JSON.parse(JSON.stringify(p.checkpoint()));
  const restarted=P.create(p.config),state=restarted.restore(mid);
  control.train(3);restarted.train(3);
  const actual=restarted.checkpoint(),expected=control.checkpoint();
  assert(JSON.stringify(actual)===JSON.stringify(expected),'Full state checkpoint failed exact interrupted replay');
  s.pipeline=restarted;s.provenance=JSON.parse(JSON.stringify(actual));
  return trace(['checkpoint at optimizer.step = '+start.model.step,
   'interrupted after '+interruption+' microsteps (NO optimizer update)','consumed IDs = '+consumed.join(','),
   'restored pending = '+state.pendingMicroSteps+'/'+p.config.accumulationSteps,
   'restored epoch/cursor = '+state.epoch+'/'+state.cursor,
   'control uninterrupted +3 optimizer steps = '+control.modelStep,
   'restarted +3 optimizer steps = '+restarted.modelStep,
   'bit-exact serialized model/sampler/pending/momentum equality = true']);
 }
 if(i===5){
  assert(s.provenance,'training checkpoint missing');
  const latest=p.checkpoint();assert(JSON.stringify(latest)===JSON.stringify(s.provenance),'model drift since checkpoint');
  const a=p.generate({temperature:0,seed:5}),b=p.generate({temperature:0,seed:997}),c=p.generate({temperature:0.7,topK:3,seed:23}),
    d=p.generate({temperature:0.7,topK:3,seed:23});
  assert(JSON.stringify(a.tokens)===JSON.stringify(b.tokens),'greedy mismatch');
  assert(JSON.stringify(c)===JSON.stringify(d),'seeded sampling mismatch');
  let rejected=false;try{p.generate({maxNewTokens:25});}catch(err){rejected=true;}
  assert(rejected,'length safety missing');
  return trace(['model optimizer.step = '+p.modelStep,'source = '+latest.dataSource,
   'fingerprint = '+latest.dataFingerprint,'tokenizer length = '+latest.model.vocab.length,
   'greedy: '+a.tokens.join(' ')+' [EOS '+a.endedWithEOS+']',
   'seeded top-K: '+c.tokens.join(' ')+' [EOS '+c.endedWithEOS+']',
   'repeat same seed stable = true','invalid decode rejected = '+rejected,
   'HTTP endpoint / GPU throughput / real model loading = NOT CLAIMED']);
 }
 throw Error('unknown stage');
}
function run(){
 const i=s.index,value=$('answer').value;
 if(s.done.has(i)){render();return;}
 if(value===''){s.logs.set(i,{status:'fail',text:'先选择机制判断，再提交实验。'});render();return;}
 s.responses.set(i,value);
 if(Number(value)!==tasks[i].correct){s.logs.set(i,{status:'fail',text:'机制判断未通过：'+tasks[i].failure});render();return;}
 try{if(i)assert(s.done.has(i-1),'先通过前一关');
  const evidence=perform(i);s.done.add(i);s.logs.set(i,{status:'pass',text:'PASS · 数值验证成功\n'+evidence});
  if(s.pipeline){s.history.push(s.pipeline.checkpoint());stateView.show(State.training(s.history),s.history.length-1);}
 }catch(error){s.logs.set(i,{status:'fail',text:'BLOCKED · '+error.message+'\n请检查状态、配置及前一关。'});}
 render();
}
function counterexample(){
 let out='';try{
  if(s.index===0)P.create({accumulationSteps:0});
  else if(s.index===2){const p=get();if(p.pendingMicroSteps<p.config.accumulationSteps)p.optimizerStep();else throw Error('在 Microbatch 完成前推进 optimizer 将改变有效 Batch 语义');}
  else if(s.index===4){
   const p=get(),c=JSON.parse(JSON.stringify(p.checkpoint()));
   c.sampler.cursor=(c.sampler.cursor+1)%(E.pairs(E.TRAIN).length+1);p.restore(c);
  }else if(s.index===5)get().generate({temperature:-1});
  else out=tasks[s.index].failure+'（机制故障描述，不冒充实际错误日志）';
 }catch(error){out='真实数值状态校验器拒绝：'+error.message;}
 $('log').textContent=out+'\n\n反例不更改已通过的工程状态。';$('log').dataset.status='fail';
}
$('answer').addEventListener('change',v=>s.responses.set(s.index,$('answer').value));
$('run').addEventListener('click',run);$('counterexample').addEventListener('click',counterexample);
$('next').addEventListener('click',()=>{if(s.done.has(s.index)&&s.index<tasks.length-1){s.index++;render();window.scrollTo({top:0,behavior:'smooth'});}});
$('reset').addEventListener('click',()=>{s.index=0;s.done.clear();s.responses.clear();s.logs.clear();s.pipeline=null;s.config=null;s.provenance=null;s.history=[];stateView.clear();render();});
render();
window.SeetheTruthEngineeringPipeline={get stage(){return s.index;},get completed(){return s.done.size;},get optimizerStep(){return s.pipeline?.modelStep||0;},get pendingMicroSteps(){return s.pipeline?.pendingMicroSteps||0;},get provenance(){return s.provenance?.model?.step??null;},lessons:tasks.length};
})();