/* SeetheTruth full-lifecycle teaching micro-engine.
 * Real Float64 CPU arithmetic for an 8-token BIGRAM conditional language model.
 * Not a Transformer, GPU implementation, or profiler. Shared PC/Android.
 */
(function(root,factory){'use strict';const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthEngineeringEngine=api;})(typeof window!=='undefined'?window:{},function(){'use strict';
const VOCAB=Object.freeze(['<bos>','我','你','爱','学','数学','代码','<eos>']);
const TRAIN=Object.freeze(['我 爱 数学','我 爱 代码','你 爱 数学','你 爱 代码','我 学 数学','你 学 代码']);
const VALID=Object.freeze(['你 学 数学','我 学 代码']);
const V=VOCAB.length, BOS=0, EOS=7, SCHEMA='seethetruth-bigram-training-v1';
function ensure(ok,msg){if(!ok)throw Error(msg);}
function rng(seed){let s=seed>>>0;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
function configOf(opts={}){
 const c={seed:opts.seed??7,lr:opts.lr??0.08,batchSize:opts.batchSize??8,weightDecay:opts.weightDecay??0.01};
 ensure(Number.isSafeInteger(c.seed)&&c.seed>=1&&c.seed<=2147483647,'seed 必须是正整数');
 ensure(Number.isFinite(c.lr)&&c.lr>0&&c.lr<=1,'学习率范围 (0,1]');
 ensure(Number.isSafeInteger(c.batchSize)&&c.batchSize>=1&&c.batchSize<=64,'batchSize 必须为 1–64');
 ensure(Number.isFinite(c.weightDecay)&&c.weightDecay>=0&&c.weightDecay<=0.2,'weightDecay 范围 [0,0.2]');
 return c;
}
function encode(sentence){
 const words=sentence.trim().split(/\s+/),ids=[BOS];
 for(const word of words){const id=VOCAB.indexOf(word);ensure(id>0&&id<EOS,'词表外 token: '+word);ids.push(id);}
 ids.push(EOS);return ids;
}
function pairs(samples){const out=[];for(const text of samples){const tokens=encode(text);for(let i=0;i<tokens.length-1;i++)out.push({x:tokens[i],y:tokens[i+1],sentence:text,position:i});}return out;}
const trainPairs=pairs(TRAIN),validPairs=pairs(VALID);
function auditDataset(){
 const uniqueTrain=new Set(TRAIN),uniqueVal=new Set(VALID);
 return {trainSentences:TRAIN.length,validationSentences:VALID.length,trainPairs:trainPairs.length,validationPairs:validPairs.length,overlap:[...uniqueVal].filter(s=>uniqueTrain.has(s)),vocabSize:V,examples:{train:TRAIN.slice(),validation:VALID.slice()}};
}
function softmax(row){
 const max=Math.max(...row),exp=row.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);
 return exp.map(x=>x/sum);
}
function batchLoss(weights,items){
 let sum=0;for(const p of items){const probs=softmax(weights.slice(p.x*V,(p.x+1)*V));sum-=Math.log(Math.max(probs[p.y],1e-300));}return sum/items.length;
}
function grad(weights,items){
 const g=Array(V*V).fill(0);let ce=0;let sample=null;
 for(const p of items){
  const row=weights.slice(p.x*V,(p.x+1)*V),probs=softmax(row);
  ce-=Math.log(Math.max(probs[p.y],1e-300));
  for(let j=0;j<V;j++)g[p.x*V+j]+=(probs[j]-(j===p.y?1:0))/items.length;
  if(!sample)sample={input:VOCAB[p.x],target:VOCAB[p.y],logits:row,probs};
 }
 return {loss:ce/items.length,gradient:g,gradNorm:Math.hypot(...g),sample,batchCount:items.length};
}
function create(opts={}){
 const config=configOf(opts),random=rng(config.seed);
 let W=Array.from({length:V*V},()=> (random()-0.5)*0.08),m=Array(V*V).fill(0),v=Array(V*V).fill(0),t=0;
 function forward(token){const id=typeof token==='number'?token:VOCAB.indexOf(token);ensure(Number.isInteger(id)&&id>=0&&id<V,'unknown token');const logits=W.slice(id*V,(id+1)*V);return {token:VOCAB[id],logits,probs:softmax(logits)};}
 function evaluate(split='validation'){
  ensure(split==='train'||split==='validation','unknown split');
  const items=split==='train'?trainPairs:validPairs,ce=batchLoss(W,items);
  const acc=items.filter(p=>forward(p.x).probs.indexOf(Math.max(...forward(p.x).probs))===p.y).length/items.length;
  return {split,loss:ce,perplexity:Math.exp(ce),nextTokenAccuracy:acc,pairs:items.length};
 }
 function currentBatch(){return Array.from({length:config.batchSize},(_,i)=>trainPairs[(t*config.batchSize+i)%trainPairs.length]);}
 function inspectGradient(){return grad(W,currentBatch());}
 function gradientCheck(){
  const batch=currentBatch(),analytic=grad(W,batch),idx=batch[0].x*V+batch[0].y,eps=1e-5;
  const original=W[idx];W[idx]=original+eps;const plus=batchLoss(W,batch);
  W[idx]=original-eps;const minus=batchLoss(W,batch);W[idx]=original;
  return {index:idx,analytic:analytic.gradient[idx],finiteDifference:(plus-minus)/(2*eps),absoluteError:Math.abs(analytic.gradient[idx]-(plus-minus)/(2*eps))};
 }
 function trainStep(){
  const g=inspectGradient(),update=applyGradient(g.gradient);
  return {step:update.step,batchLoss:g.loss,gradNorm:g.gradNorm,updateNorm:update.updateNorm,
   firstSample:g.sample,train:evaluate('train'),validation:evaluate('validation')};
 }
 function gradientForPairs(items){
  ensure(Array.isArray(items)&&items.length>=1&&items.length<=512,'microbatch must contain 1–512 examples');
  for(const p of items)ensure(p&&Number.isInteger(p.x)&&p.x>=0&&p.x<V&&Number.isInteger(p.y)&&p.y>=0&&p.y<V,'invalid training token pair');
  return grad(W,items);
 }
 // Exactly one optimizer update; uses externally accumulated gradient, not microbatch mean loss.
 function applyGradient(gradient,learningRate=config.lr){
  ensure(Array.isArray(gradient)&&gradient.length===V*V&&gradient.every(Number.isFinite),'invalid accumulated gradient');
  ensure(Number.isFinite(learningRate)&&learningRate>0&&learningRate<=1,'invalid scheduled learning rate');
  const before=W.slice(),beta1=0.9,beta2=0.999,eps=1e-8;
  t++;
  for(let i=0;i<W.length;i++){
   m[i]=beta1*m[i]+(1-beta1)*gradient[i];
   v[i]=beta2*v[i]+(1-beta2)*gradient[i]**2;
   const mh=m[i]/(1-beta1**t),vh=v[i]/(1-beta2**t);
   W[i]-=learningRate*(mh/(Math.sqrt(vh)+eps)+config.weightDecay*W[i]);
  }
  return {step:t,learningRate,gradientNorm:Math.hypot(...gradient),updateNorm:Math.hypot(...W.map((x,i)=>x-before[i]))};
 }
 function train(steps){ensure(Number.isSafeInteger(steps)&&steps>=1&&steps<=1000,'训练步数必须为 1–1000');let last;for(let i=0;i<steps;i++)last=trainStep();return last;}
 function checkpoint(){return {schema:SCHEMA,model:'bigram-transition-logits',config:{...config},vocab:VOCAB.slice(),step:t,weights:W.slice(),firstMoment:m.slice(),secondMoment:v.slice()};}
 function restore(snapshot){
  ensure(snapshot&&snapshot.schema===SCHEMA&&snapshot.model==='bigram-transition-logits','checkpoint schema/model mismatch');
  ensure(Array.isArray(snapshot.vocab)&&JSON.stringify(snapshot.vocab)===JSON.stringify(VOCAB),'checkpoint tokenizer mismatch');
  ensure(JSON.stringify(configOf(snapshot.config))===JSON.stringify(config),'checkpoint config mismatch');
  ensure(Number.isSafeInteger(snapshot.step)&&snapshot.step>=0&&snapshot.step<=1000000,'checkpoint step invalid');
  for(const key of ['weights','firstMoment','secondMoment'])ensure(Array.isArray(snapshot[key])&&snapshot[key].length===V*V&&snapshot[key].every(Number.isFinite),'invalid checkpoint '+key);
  ensure(snapshot.secondMoment.every(x=>x>=0),'second moment must be nonnegative');
  W=snapshot.weights.slice();m=snapshot.firstMoment.slice();v=snapshot.secondMoment.slice();t=snapshot.step;return {step:t,restored:true};
 }
 function generate({seed=17,maxNewTokens=10,temperature=0,topK=V}={}){
  ensure(Number.isSafeInteger(seed)&&seed>0,'generation seed invalid');
  ensure(Number.isSafeInteger(maxNewTokens)&&maxNewTokens>=1&&maxNewTokens<=24,'maxNewTokens range 1–24');
  ensure(Number.isFinite(temperature)&&temperature>=0&&temperature<=3,'temperature range 0–3');
  ensure(Number.isInteger(topK)&&topK>=1&&topK<=V,'topK out of bounds');
  const random=rng(seed),trace=[];let current=BOS;
  for(let step=0;step<maxNewTokens;step++){
   const f=forward(current),candidates=f.logits.map((logit,id)=>({id,logit})).filter(x=>x.id!==BOS);
   candidates.sort((a,b)=>b.logit-a.logit||a.id-b.id);
   const keep=candidates.slice(0,topK),scores=temperature===0?null:softmax(keep.map(x=>x.logit/temperature));
   let selected=keep[0],prob=1;
   if(scores){let draw=random();for(let i=0;i<keep.length;i++){draw-=scores[i];if(draw<=0||i===keep.length-1){selected=keep[i];prob=scores[i];break;}}}
   trace.push({step:step+1,context:VOCAB[current],token:VOCAB[selected.id],probability:prob,topCandidates:keep.slice(0,3).map((x,i)=>({token:VOCAB[x.id],probability:scores?scores[i]:Number(x.id===selected.id)}))});
   current=selected.id;if(current===EOS)break;
  }
  return {tokens:trace.filter(x=>x.token!=='<eos>').map(x=>x.token),endedWithEOS:current===EOS,trace,step:t};
 }
 return {config:{...config},vocab:VOCAB.slice(),forward,inspectGradient,gradientForPairs,applyGradient,gradientCheck,trainStep,train,evaluate,checkpoint,restore,generate,get step(){return t;}};
}
return Object.freeze({VERSION:'2026.10.21',SCHEMA,VOCAB,TRAIN,VALID,encode,pairs,auditDataset,create});
});
