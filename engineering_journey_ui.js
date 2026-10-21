/* 12-stage guided lab. All values below are computed by engineering_journey_engine;
 * written explanations are evidence labels, not fabricated measurements.
 * Shared PC/Android; no persistence, downloads, runtime backend or dynamic evaluation.
 */
(function(){'use strict';
const E=window.SeetheTruthEngineeringEngine;
if(!E)throw Error('Engineering engine missing');
const $=id=>document.getElementById(id);
const lessons=[
 {category:'00 / PROJECT BOOTSTRAP',title:'环境、版本、种子与复现',
  goal:'在同一代码版本、随机种子和超参数下，初始化权重逐元素相同；明确教学执行环境不等于生产硬件。',
  operations:'检查页面的运行边界，修改种子/学习率/Batch/Decay 后，先判断如何验证初始化复现，再执行两次构建。',
  question:'哪一项是初始化可复现性的直接证据？',options:['两次同种子权重完全一致','Loss 自然下降','采样文本总相同'],correct:0,
  code:'const a = create({seed:7});\nconst b = create({seed:7});\nassert.deepEqual(a.checkpoint().weights,b.checkpoint().weights);',
  principle:'真实工程应记录 Git revision、依赖锁、种子、模型配置、数据版本、设备和精度；仅固定种子不保证跨设备算子确定性。',
  pitfall:'只记录随机种子，忘记提交版本、数据集版本和非确定性内核。',links:[['训练细节','training.html']]},
 {category:'01 / DATA GOVERNANCE',title:'数据来源、划分与防泄漏',
  goal:'确认训练与验证样本严格不重叠。这里使用离线内置的合成中文小语料；真实项目还需检查数据授权、PII、分布和去重。',
  operations:'检查训练/验证原句、Token 对数与重复样本，判断能否让验证样本参与梯度计算。',
  question:'哪个操作会破坏验证集独立性？',options:['分别报告训练和验证 Loss','将验证集并入训练 Batch','固定数据切分'],correct:1,
  code:'train = [6 条原句]; val = [2 条未见原句];\nassert(intersection(train,val).size === 0);',
  principle:'数据合同包含来源、许可、去重、划分键、预处理、泄漏审计和质量筛查；模型评估必须与训练梯度路径隔离。',
  pitfall:'不同文件中的同一段内容同时进入 train / val 会抬高评价的可信度错觉。',links:[]},
 {category:'02 / TOKENIZER & LABELS',title:'Tokenization、BOS/EOS 与标签右移',
  goal:'看清文本编码以及 X[t] 预测 X[t+1] 的监督位置，避免把当前 Token 当作标签。',
  operations:'查看「我 爱 数学」的 [BOS, 我, 爱, 数学, EOS] 编码，并核对第一个训练对。',
  question:'BOS 在 Next-token 监督中预测谁？',options:['BOS 自身','第一个内容 Token','EOS'],correct:1,
  code:'ids = [BOS, 我, 爱, 数学, EOS];\nX = ids[:-1]; Y = ids[1:];',
  principle:'词表、特殊 Token、位置、Mask、Packing 和 Padding 必须同时遵循数据与 Loss 合同；生产 Tokenizer 需冻结版本。',
  pitfall:'标签没有 shift，导致模型学习复制当前 Token；或把 padding 当有效目标。',links:[['可训练 Token LM','learn.html']]},
 {category:'03 / ARCHITECTURE',title:'参数形状与模型设计预算',
  goal:'确认 Bigram 参数矩阵 W[8,8] 的语义：输入 Token 选择一行，输出 8 个 Logits，参数总数 64。',
  operations:'从源码式表达推导 shape、参数数目和 Bigram 没有注意力这一结构限制。',
  question:'Bigram 当前步 Logits 由什么决定？',options:['整个上下文的 Attention','前一 Token 对应的 W 行','KV Block Table'],correct:1,
  code:'W.shape = [vocab_size, vocab_size];  // 8 × 8\nlogits = W[last_token_id];',
  principle:'先定义张量轴、参数共享、序列依赖和推理状态，再谈模型大小、显存与 FLOPs。此微模型无法替代 Transformer。',
  pitfall:'把“模型能生成 Token”理解成“已经复刻了真实大模型”。',links:[['积木能力边界','capabilities.html']]},
 {category:'04 / INITIALIZATION',title:'权重初始化和 AdamW 优化器状态',
  goal:'区分模型参数 W、梯度 dW 和优化器一阶/二阶矩 m/v。未训练时 t=0，m=v=0。',
  operations:'查看权重摘要与第一批梯度长度；检查实际优化器超参数和零状态。',
  question:'AdamW 续训正确恢复需要什么？',options:['只有最终 Loss','W + m + v + step 与超参数','只有词表'],correct:1,
  code:'m=zeros_like(W); v=zeros_like(W); step=0;\n// 参数量 64；优化器附加 m/v 共 128 个数',
  principle:'优化器状态通常显著增加训练内存；混合精度与分片优化器对真实训练影响大，但此实验用 Float64。',
  pitfall:'续训时只恢复参数不恢复动量，会改变之后的参数轨迹。',links:[['训练回路','training.html']]},
 {category:'05 / FORWARD',title:'Forward：Logits → 稳定 Softmax',
  goal:'查看给定 Token 的 8 个 Logits，实际计算归一化概率，并验证概率和为 1。',
  operations:'前向输入“爱”，预测 W[爱] 这一行；观察 softmax 概率而不是直接把 logits 当概率。',
  question:'为什么 Softmax 先减去最大 Logit？',options:['改善数值稳定性','自动减少计算复杂度','实现注意力缓存'],correct:0,
  code:'z = W[token];\np = exp(z - max(z)) / sum(exp(z - max(z)));',
  principle:'训练中的前向值、激活、数据类型和算子精度是误差定位的基本单元。',
  pitfall:'直接 exp(极大Logit) 可能溢出，最后出现 Inf/NaN。',links:[['逐积木运行','learn.html']]},
 {category:'06 / BACKPROP',title:'CrossEntropy、解析梯度与有限差分',
  goal:'比较解析导数 ∂L/∂W 与有限差分数值梯度，至少在当前 Batch 的一个坐标达到 1e-6 误差阈值。',
  operations:'先判断 CE 对正确类别的梯度形式，检查解析梯度、有限差分与 L2 梯度范数。',
  question:'Softmax+CE 的对应 Logit 梯度形式是？',options:['p - one_hot(y)','p + one_hot(y)','对所有类别都为 0'],correct:0,
  code:'L = -mean(log(p[target]));\ndW[row,:] += (p-one_hot(target))/batch_size;',
  principle:'复杂新算子应先用有限差分、参考后端和退化边界测试验证 VJP，再进入大规模训练。',
  pitfall:'把 CE 直接当作训练完成；没有 backward 和 optimizer.step 时参数并未更新。',links:[['梯度教学','micro_training.html']]},
 {category:'07 / OPTIMIZER STEP',title:'真实训练：Batch、AdamW 与 Loss 曲线',
  goal:'执行 120 次实际参数更新，并检查全训练集平均交叉熵相对初始化下降。',
  operations:'观察训练前后 CE、Perplexity、首 Batch Loss、梯度范数、参数更新范数和 step。这个模型只包含 64 个权重。',
  question:'梯度累积或参数更新在哪个环节真正改变 W？',options:['forward','optimizer.step','eval'],correct:1,
  code:'for step in range(120):\n  loss, grad = forward_backward(train_batch)\n  adamw_update(W, m, v, grad, lr, decay)',
  principle:'真实训练还需要数据加载、梯度裁剪、混合精度、调度器、Checkpoint、监控与恢复；这里仅覆盖数学最小闭环。',
  pitfall:'只记录最后一个 Batch Loss，不对固定全集重算，容易误判是否真的收敛。',links:[['完整训练步骤','training.html']]},
 {category:'08 / EVALUATION',title:'验证集、PPL 与泛化诊断',
  goal:'分别评估 train/validation 的 CE、Perplexity、Next-token Accuracy，不在验证集上做任何反向传播。',
  operations:'读取两个互斥数据集的指标，比较训练误差与泛化误差；不能从一次微型实验断言真实语言能力。',
  question:'validation loss 比 train loss 高说明什么？',options:['必然存在代码错误','可能存在泛化差距，需继续验证','应该将验证集用于 optimizer'],correct:1,
  code:'with no_grad():\n  train_ce = eval(train_pairs)\n  validation_ce = eval(validation_pairs)',
  principle:'上线前还需固定评测集、任务指标、统计区间、数据污染检测和与基线比较。',
  pitfall:'以训练 Loss 代替 held-out 评测；或为降低验证 Loss 而持续调参却不保留最终测试集。',links:[['指标与能力边界','capabilities.html']]},
 {category:'09 / CHECKPOINT',title:'Checkpoint：序列化、扰动、恢复',
  goal:'保存 W/m/v/step/Tokenizer/配置的内存快照；临时继续训练，再恢复并逐元素验证完全一致。',
  operations:'本关做一次真正的状态 round-trip；检查 t、权重和同输入概率能否精确复原。',
  question:'只恢复 W 而未恢复 m/v，会怎样？',options:['保证后续更新完全一致','不能保证后续 AdamW 轨迹相同','词表会自动变大'],correct:1,
  code:'snapshot = {vocab, config, weights:W, m, v, step};\ntrain_one_step(); restore(snapshot); assert(equal(W,snapshot.W));',
  principle:'生产快照还应包含 RNG 状态、DataLoader 位置、分布式分片方案、配置和元信息，且需检验原子写入。',
  pitfall:'错配 Tokenizer 或裁断 Tensor，却仍把其载入为同一模型。Android 本实验只保存内存状态。',links:[['PyTorch 训练导出','learn.html']]},
 {category:'10 / DECODING',title:'用训练后的同一权重进行 Decode',
  goal:'从 BOS 开始逐 Token 计算 W[last] → Softmax/Top-K/采样，记录每一次条件输入、输出、概率和 EOS。',
  operations:'调节温度、Top-K 和采样种子；温度 0 时 Greedy，可对照温度高于 0 的多样性。',
  question:'本 Bigram 模型推理时是否实际使用 KV Cache？',options:['是，使用物理页缓存','否，它没有 Attention/KV','会自动调用 DeepInfer'],correct:1,
  code:'id = BOS\nwhile not EOS: logits = W[id]\n  id = sample(logits, temperature, top_k)',
  principle:'真正 Transformer 在线推理要处理 Prompt Tokenization、Prefill、KV 生命周期、Batch、采样、停止条件、请求取消与流式输出。',
  pitfall:'将教学 Bigram 自回归 Decode 的时间线误判为生产 Transformer 实测性能。',links:[['KV Cache','cache.html'],['调度实验','deepinfer_scheduler_lab.html']]},
 {category:'11 / SERVING & QA',title:'上线前合同：边界输入、重试和观测',
  goal:'检查同一 Checkpoint 的确定性 Greedy、未知 Token 拒绝、输出长度保护；明确尚未覆盖的生产部署工作。',
  operations:'验证重复推理与异常输入，完成训练→评估→快照→恢复→Decode 的全链路审计。',
  question:'CPU 数值教学成功后，哪项仍需单独验证？',options:['真实 GPU Serving 吞吐和并发行为','8×8 参数形状','BOS 的索引'],correct:0,
  code:'assert(greedy(seed=1) == greedy(seed=999))\nreject_unknown_token(); reject_length_over_limit();',
  principle:'真实上线还需要服务 API、鉴权、限流、热更新、监控告警、回滚、资源利用率与安全评测。能力边界必须对用户可见。',
  pitfall:'把逻辑正确、离线可执行和线上吞吐达标视为同一证据级别。',links:[['调度/内存','deepinfer_scheduler_lab.html'],['推理事件','deepinfer_worker_lab.html']]}
];
const state={idx:0,done:new Set(),engine:null,snapshot:null,logs:new Map(),choices:new Map()};
const explain=()=>lessons[state.idx];
const f=x=>Number(x).toFixed(5);
const textLines=lines=>lines.join('\n');
function params(){return{seed:Number($('seed').value),lr:Number($('lr').value),batchSize:Number($('batchSize').value),weightDecay:Number($('weightDecay').value)};}
function assert(ok,message){if(!ok)throw Error(message);}
function currentEngine(){assert(state.engine,'请先完成关卡 00');assert(JSON.stringify(state.engine.config)===JSON.stringify(params()),'参数配置已更改；请点击“重置训练与所有关卡”以重新开始');return state.engine;}
function metrics(){
 const e=state.engine,rows=e?[
  ['optimizer.step',e.step],['参数 W','8 × 8 = 64'],
  ['Train CE',f(e.evaluate('train').loss)],['Valid CE',f(e.evaluate('validation').loss)],
  ['Train PPL',f(e.evaluate('train').perplexity)],['Valid PPL',f(e.evaluate('validation').perplexity)]
 ]:[['optimizer.step','0'],['参数 W','未初始化'],['Train CE','—'],['Valid CE','—'],['Train PPL','—'],['Valid PPL','—']];
 $('metrics').replaceChildren(...rows.map(([name,value])=>{
  const el=document.createElement('div');el.className='metric';
  const n=document.createElement('div');n.className='name';n.textContent=name;
  const v=document.createElement('div');v.className='value';v.textContent=String(value);el.append(n,v);return el;
 }));
}
function render(){
 $('category').textContent=explain().category;$('title').textContent=explain().title;$('goal').textContent=explain().goal;
 $('operations').textContent=explain().operations;$('principle').textContent=explain().principle;$('pitfall').textContent=explain().pitfall;
 $('source').textContent=explain().code;$('question').textContent=explain().question;
 const sel=$('answerSelect');sel.replaceChildren(new Option('先作出判断…',''));
 explain().options.forEach((option,i)=>sel.add(new Option(option,String(i))));
 sel.value=state.choices.get(state.idx)??'';
 $('configForm').hidden=state.idx!==0 && state.idx!==4;
 $('inferenceForm').hidden=state.idx!==10;
 $('result').textContent=state.logs.get(state.idx)?.message||'选择答案后执行；本关尚未验证。';
 $('result').dataset.status=state.logs.get(state.idx)?.status||'idle';
 $('next').disabled=!state.done.has(state.idx)||state.idx===lessons.length-1;
 const max=state.done.size?Math.min(lessons.length-1,Math.max(...state.done)+1):0;
 $('stages').replaceChildren(...lessons.map((lesson,i)=>{
  const b=document.createElement('button');b.className='stage';b.disabled=i>max;
  b.setAttribute('aria-current',i===state.idx?'step':'false');
  const n=document.createElement('span');n.className='num';n.textContent=String(i).padStart(2,'0');
  const label=document.createElement('span');label.textContent=lesson.title;
  b.append(n,label);if(state.done.has(i)){const done=document.createElement('span');done.className='done';done.textContent='✓';b.append(done);}
  b.addEventListener('click',()=>{state.choices.set(state.idx,sel.value);state.idx=i;render();});return b;
 }));
 $('progressText').textContent=state.done.size+' / '+lessons.length+' 已验证';
 $('progressFill').style.width=(state.done.size/lessons.length*100)+'%';
 $('related').replaceChildren(...explain().links.map(([name,href])=>{
  const a=document.createElement('a');a.href=href;a.textContent=name+' ↗';return a;
 }));
 metrics();
}
function execution(i){
 if(i===0){
  // Re-initialization invalidates every result obtained with prior weights/config.
  state.done.clear();state.logs.clear();state.choices=new Map([[0,$('answerSelect').value]]);
  state.engine=E.create(params());state.snapshot=null;
  const same=E.create(params()).checkpoint().weights;
  assert(JSON.stringify(same)===JSON.stringify(state.engine.checkpoint().weights),'初始化不可复现');
  return textLines(['ENGINE: Float64 JS CPU · BIGRAM','seed = '+state.engine.config.seed,'learning_rate = '+state.engine.config.lr,'batch = '+state.engine.config.batchSize,'weight_decay = '+state.engine.config.weightDecay,'parameters = 64','same-seed weights equal: true']);
 }
 const e=currentEngine();
 if(i===1){const a=E.auditDataset();assert(a.overlap.length===0,'存在数据泄漏');return textLines(['train sentences = '+a.trainSentences+'; valid sentences = '+a.validationSentences,'train token pairs = '+a.trainPairs+'; valid token pairs = '+a.validationPairs,'intersection = '+JSON.stringify(a.overlap),'train: '+a.examples.train.join(' | '),'valid: '+a.examples.validation.join(' | ')]);}
 if(i===2){const ids=E.encode('我 爱 数学');assert(ids.join(',')==='0,1,3,5,7','tokenizer mismatch');return textLines(['text = 我 爱 数学','tokens = '+ids.map(j=>E.VOCAB[j]).join(' / '),'x = '+ids.slice(0,-1).map(j=>E.VOCAB[j]).join(', '),'y = '+ids.slice(1).map(j=>E.VOCAB[j]).join(', '),'pairs = '+(ids.length-1)]);}
 if(i===3){const s=e.checkpoint();assert(s.weights.length===E.VOCAB.length**2,'shape mismatch');return textLines(['W.shape = ['+E.VOCAB.length+','+E.VOCAB.length+']','parameters = '+s.weights.length,'first row logits = '+s.weights.slice(0,8).map(f).join(' '),'no Attention / no KV memory']);}
 if(i===4){const s=e.checkpoint(),g=e.inspectGradient();assert(s.step===0&&s.firstMoment.every(x=>x===0)&&s.secondMoment.every(x=>x===0),'expected zero optimizer state');return textLines(['W elements = '+s.weights.length,'m/v lengths = '+s.firstMoment.length+'/'+s.secondMoment.length,'step = '+s.step,'first batch gradient L2 = '+f(g.gradNorm),'optimizer = AdamW (β1=.9, β2=.999, eps=1e-8)']);}
 if(i===5){const p=e.forward('爱');assert(Math.abs(p.probs.reduce((a,b)=>a+b,0)-1)<1e-12,'probability normalization');return textLines(['input token = '+p.token,'logits = '+p.logits.map(f).join(' '),'probabilities = '+p.probs.map(f).join(' '),'sum(p) = '+p.probs.reduce((a,b)=>a+b,0).toFixed(12)]);}
 if(i===6){const g=e.inspectGradient(),check=e.gradientCheck();assert(check.absoluteError<1e-6,'finite difference gradient mismatch');return textLines(['batch CE = '+f(g.loss),'dW shape = [8,8]','gradient norm = '+f(g.gradNorm),'index = '+check.index,'analytic = '+check.analytic.toFixed(9),'finite diff = '+check.finiteDifference.toFixed(9),'abs error = '+check.absoluteError.toExponential(3)]);}
 if(i===7){const before=e.evaluate('train'),after=e.train(120);assert(after.train.loss<before.loss-0.05,'训练集 CE 未显著下降');state.snapshot=null;for(let j=8;j<lessons.length;j++){state.done.delete(j);state.logs.delete(j);}return textLines(['optimizer updates = '+after.step,'train CE: '+f(before.loss)+' → '+f(after.train.loss),'train PPL: '+f(before.perplexity)+' → '+f(after.train.perplexity),'validation CE = '+f(after.validation.loss),'last batch CE = '+f(after.batchLoss),'last gradient norm = '+f(after.gradNorm),'last parameter delta norm = '+f(after.updateNorm)]);}
 if(i===8){const train=e.evaluate('train'),valid=e.evaluate('validation');assert(Number.isFinite(train.loss)&&Number.isFinite(valid.loss),'nonfinite evaluation');return textLines(['SPLIT      CE      PPL     TOP1','train      '+f(train.loss)+'   '+f(train.perplexity)+'   '+f(train.nextTokenAccuracy),'validation '+f(valid.loss)+'   '+f(valid.perplexity)+'   '+f(valid.nextTokenAccuracy),'generalization gap (CE) = '+f(valid.loss-train.loss),'Validation only: NO optimizer step; checkpoint step='+e.step]);}
 if(i===9){state.snapshot=e.checkpoint();const before=e.forward('我').probs,step=state.snapshot.step;e.trainStep();e.restore(state.snapshot);assert(e.step===step,'checkpoint step restoration failed');assert(JSON.stringify(before)===JSON.stringify(e.forward('我').probs),'checkpoint logits mismatch');return textLines(['checkpoint.schema = '+state.snapshot.schema,'tokenizer matches = true','W/m/v = '+state.snapshot.weights.length+'/'+state.snapshot.firstMoment.length+'/'+state.snapshot.secondMoment.length,'disturb: optimizer.step() -> '+(step+1),'restore: step -> '+e.step,'byte-exact Float64 array JSON roundtrip = true','storage = in-memory (no APK file or exported model)']);}
 if(i===10){assert(state.snapshot&&e.step===state.snapshot.step,'未加载训练完成的 checkpoint');const args={seed:Number($('decodeSeed').value),temperature:Number($('temperature').value),topK:Number($('topK').value),maxNewTokens:10},g=e.generate(args);assert(g.trace.length>0,'empty decode');return textLines(['checkpoint update step = '+e.step,'temperature = '+args.temperature+'; top_k = '+args.topK,'tokens: '+g.tokens.join(' ')+' '+(g.endedWithEOS?'<eos>':'[max tokens stop]'),'decode steps: '+g.trace.length,...g.trace.map(x=>'#'+x.step+' '+x.context+' → '+x.token+' (p='+f(x.probability)+')')]);}
 if(i===11){assert(state.snapshot,'missing checkpoint');const a=e.generate({temperature:0,seed:1}),b=e.generate({temperature:0,seed:888});assert(JSON.stringify(a.tokens)===JSON.stringify(b.tokens),'greedy nondeterministic');let unknown=false,overflow=false;try{e.forward('未知词');}catch(err){unknown=true;}try{e.generate({maxNewTokens:25});}catch(err){overflow=true;}assert(unknown&&overflow,'invalid inputs not rejected');return textLines(['full journey checkpoints = '+(state.done.size+1)+' / 12','same checkpoint greedy stable = true','unknown vocabulary rejected = '+unknown,'generation length limit enforced = '+overflow,'trained checkpoint step = '+e.step,'production API / GPU throughput / safety evaluation = NOT IMPLEMENTED']);}
 throw Error('unknown stage');
}
function run(){
 const i=state.idx,selected=$('answerSelect').value;
 if(selected===''){state.logs.set(i,{status:'fail',message:'请先选择机制判断的答案。每一步都需要主动预测结果。'});render();return;}
 state.choices.set(i,selected);
 if(Number(selected)!==explain().correct){state.logs.set(i,{status:'fail',message:'机制判断不成立。提示：'+explain().pitfall+'\n请重新判断后执行。'});render();return;}
 try{
  if(i>0)assert(state.done.has(i-1),'须先通过前一关');
  const out=execution(i);state.done.add(i);state.logs.set(i,{status:'pass',message:'PASS · 数值/合同检查成功\n'+out});
 }catch(err){state.logs.set(i,{status:'fail',message:'BLOCKED · '+err.message+'\n修改输入或检查前置关卡后重试。'});}
 render();
}
function failureExample(){
 const i=state.idx;let msg='';
 try{
  if(i===0){E.create({seed:0});}
  else if(i===2){E.encode('未知 Token');}
  else if(i===4){E.create({lr:2});}
  else if(i===9){const snap=currentEngine().checkpoint();snap.weights.pop();currentEngine().restore(snap);}
  else if(i===10){currentEngine().generate({temperature:-1});}
  else if(i===11){currentEngine().generate({maxNewTokens:999});}
  else if(i===1||i===8){const d=E.auditDataset();msg='人为注入验证样本到训练集合后，重叠样本数 = '+(new Set([...d.examples.train,d.examples.validation[0]]).has(d.examples.validation[0])?1:0)+'；本实验真实分割未污染。';}
  else{msg=explain().pitfall+'（概念反例；不作为真实运行的失败日志）';}
 }catch(err){msg='真实校验器拒绝：'+err.message;}
 $('result').textContent=msg+'\n\n此反例不会解锁关卡，也不改写已训练的模型。';$('result').dataset.status='fail';
}
$('answerSelect').addEventListener('change',()=>state.choices.set(state.idx,$('answerSelect').value));
$('run').addEventListener('click',run);$('example').addEventListener('click',failureExample);
$('next').addEventListener('click',()=>{if(state.done.has(state.idx)&&state.idx<lessons.length-1){state.idx++;render();window.scrollTo({top:0,behavior:'smooth'});}});
$('reset').addEventListener('click',()=>{state.idx=0;state.done.clear();state.logs.clear();state.choices.clear();state.engine=null;state.snapshot=null;render();});
render();
window.SeetheTruthEngineeringJourney={get stage(){return state.idx;},get completed(){return state.done.size;},get modelStep(){return state.engine?.step||0;},get snapshotReady(){return !!state.snapshot;},lessons:lessons.length};
})();