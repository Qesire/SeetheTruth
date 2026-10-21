/* Deterministic CPU training-pipeline lab: shuffling, accumulation, warmup/cosine,
 * clipping, resumable interrupted optimizer step, and inference checkpoint handoff.
 * All mathematical work delegates to the shared Bigram model reference backend.
 * This is not GPU performance simulation or full distributed training.
 */
(function(root,factory){'use strict';const a=factory(typeof require==='function'?require('./engineering_journey_engine.js'):root.SeetheTruthEngineeringEngine);if(typeof module==='object'&&module.exports)module.exports=a;root.SeetheTruthTrainingPipeline=a;})(typeof window!=='undefined'?window:{},function(E){'use strict';
if(!E)throw Error('Bigram numerical engine unavailable');
const SCHEMA='seethetruth-training-pipeline-v1',SOURCE='synthetic-24-next-token-pairs-v1';
const pairs=E.pairs(E.TRAIN),V=E.VOCAB.length;
const baseOptions=Object.freeze({seed:7,samplerSeed:19,lr:0.08,microBatchSize:2,accumulationSteps:4,weightDecay:0.01,warmupSteps:3,totalSteps:24,minLrScale:0.15,maxGradNorm:1});
function verify(ok,message){if(!ok)throw Error(message);}
function numberIn(x,lower,upper,name,integer=false){
 verify(Number.isFinite(x)&&x>=lower&&x<=upper&&(!integer||Number.isSafeInteger(x)),name+' invalid');
 return x;
}
function optionsOf(input={}){
 const c={...baseOptions,...input};
 numberIn(c.seed,1,2147483647,'seed',true);numberIn(c.samplerSeed,1,2147483647,'samplerSeed',true);
 numberIn(c.lr,0.000001,1,'lr');numberIn(c.microBatchSize,1,24,'microBatchSize',true);
 numberIn(c.accumulationSteps,1,24,'accumulationSteps',true);
 numberIn(c.microBatchSize*c.accumulationSteps,1,64,'effectiveBatch',true);
 numberIn(c.weightDecay,0,0.2,'weightDecay');numberIn(c.totalSteps,2,1000,'totalSteps',true);
 numberIn(c.warmupSteps,0,c.totalSteps-1,'warmupSteps',true);
 numberIn(c.minLrScale,0,1,'minLrScale');numberIn(c.maxGradNorm,0.00001,1000,'maxGradNorm');
 return c;
}
// Stable non-cryptographic dataset fingerprint: record identity, NOT a security attestation.
function fingerprint(){
 let h=2166136261>>>0;
 const string=[...E.VOCAB,'TRAIN',...E.TRAIN,'VALID',...E.VALID].join('\u0000');
 for(let i=0;i<string.length;i++){h^=string.charCodeAt(i);h=Math.imul(h,16777619)>>>0;}
 return h.toString(16).padStart(8,'0');
}
const DATA_FINGERPRINT=fingerprint();
function xorshift(seed){let s=seed>>>0;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
function shuffledIndices(seed,epoch){
 numberIn(seed,1,2147483647,'seed',true);numberIn(epoch,0,1000000,'epoch',true);
 // Seed mix depends on epoch and samplerSeed. Reconstructable without mutable RNG.
 const rand=xorshift((seed^Math.imul(epoch+1,0x9e3779b9))>>>0||1);
 const idx=Array.from({length:pairs.length},(_,i)=>i);
 for(let i=idx.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
 return idx;
}
function scheduledLr(c,step){
 const o=optionsOf(c);numberIn(step,1,1000001,'scheduler step',true);
 if(o.warmupSteps&&step<=o.warmupSteps)return o.lr*step/o.warmupSteps;
 const numerator=Math.min(Math.max(step-o.warmupSteps,0),o.totalSteps-o.warmupSteps);
 const progress=numerator/(o.totalSteps-o.warmupSteps);
 return o.lr*(o.minLrScale+(1-o.minLrScale)*0.5*(1+Math.cos(Math.PI*progress)));
}
function create(input={}){
 const config=optionsOf(input);
 const model=E.create({seed:config.seed,lr:config.lr,batchSize:config.microBatchSize*config.accumulationSteps,weightDecay:config.weightDecay});
 let epoch=0,cursor=0;
 function emptyPending(){return {microSteps:0,samples:0,lossSum:0,gradientSum:Array(V*V).fill(0),indices:[]};}
 let pending=emptyPending();
 function take(n,commit){
  let ep=epoch,pos=cursor;const out=[];
  for(let i=0;i<n;i++){
   if(pos===pairs.length){pos=0;ep++;}
   out.push({index:shuffledIndices(config.samplerSeed,ep)[pos],epoch:ep,position:pos});pos++;
  }
  if(commit){epoch=ep;cursor=pos;}
  return out;
 }
 function previewBatch(){const indices=take(config.microBatchSize,false);return {epoch,cursor,indices,items:indices.map(o=>({...pairs[o.index]}))};}
 function microStep(){
  verify(pending.microSteps<config.accumulationSteps,'accumulation buffer full; optimizer step required');
  // Compute on weights as of the current optimizer step. No parameter update here.
  const sample=take(config.microBatchSize,true),items=sample.map(o=>pairs[o.index]);
  const g=model.gradientForPairs(items);
  for(let i=0;i<pending.gradientSum.length;i++)pending.gradientSum[i]+=g.gradient[i]*items.length;
  pending.samples+=items.length;pending.lossSum+=g.loss*items.length;
  pending.microSteps++;pending.indices.push(...sample.map(o=>o.index));
  return {optimizerStep:model.step,microStep:pending.microSteps,requiredMicroSteps:config.accumulationSteps,
   sampleCount:pending.samples,indices:sample.map(x=>x.index),tokenPairs:items.map(x=>[E.VOCAB[x.x],E.VOCAB[x.y]]),
   microLoss:g.loss,gradNorm:g.gradNorm,parameterUnchanged:true,epoch,cursor,ready:pending.microSteps===config.accumulationSteps};
 }
 function optimizerStep(){
  verify(pending.microSteps===config.accumulationSteps,'must finish every microbatch before optimizer step');
  const raw=pending.gradientSum.map(v=>v/pending.samples),rawNorm=Math.hypot(...raw);
  const scale=Math.min(1,config.maxGradNorm/(rawNorm||1));
  const clipped=raw.map(v=>v*scale),lr=scheduledLr(config,model.step+1);
  const update=model.applyGradient(clipped,lr);
  const result={...update,microSteps:pending.microSteps,examples:pending.samples,
   batchLoss:pending.lossSum/pending.samples,rawGradNorm:rawNorm,clippedGradNorm:Math.hypot(...clipped),
   clipScale:scale,indices:pending.indices.slice(),epoch,cursor,
   train:model.evaluate('train'),validation:model.evaluate('validation')};
  pending=emptyPending();
  return result;
 }
 function train(n){
  numberIn(n,1,500,'training steps',true);let last;
  for(let step=0;step<n;step++){
   while(pending.microSteps<config.accumulationSteps)microStep();
   last=optimizerStep();
  }
  return last;
 }
 function evaluate(split='validation'){return model.evaluate(split);}
 function generate(opts){return model.generate(opts);}
 function checkpoint(){return {schema:SCHEMA,dataSource:SOURCE,dataFingerprint:DATA_FINGERPRINT,config:{...config},
  model:model.checkpoint(),sampler:{epoch,cursor},pending:{microSteps:pending.microSteps,samples:pending.samples,lossSum:pending.lossSum,
   gradientSum:pending.gradientSum.slice(),indices:pending.indices.slice()}};
 }
 function restore(snap){
  verify(snap&&snap.schema===SCHEMA&&snap.dataSource===SOURCE&&snap.dataFingerprint===DATA_FINGERPRINT,'pipeline dataset/source mismatch');
  verify(JSON.stringify(snap.config)===JSON.stringify(config),'pipeline configuration mismatch');
  const state=snap.sampler,b=snap.pending;
  verify(state&&Number.isSafeInteger(state.epoch)&&state.epoch>=0&&state.epoch<=1000000
    &&Number.isSafeInteger(state.cursor)&&state.cursor>=0&&state.cursor<=pairs.length,'invalid sampler state');
  verify(b&&Number.isSafeInteger(b.microSteps)&&b.microSteps>=0&&b.microSteps<=config.accumulationSteps
    &&Number.isSafeInteger(b.samples)&&b.samples===b.microSteps*config.microBatchSize,'invalid accumulation count');
  verify(Array.isArray(b.gradientSum)&&b.gradientSum.length===V*V&&b.gradientSum.every(Number.isFinite),'invalid accumulated gradient');
  verify(Number.isFinite(b.lossSum)&&b.lossSum>=0,'invalid accumulated loss');
  verify(Array.isArray(b.indices)&&b.indices.length===b.samples&&b.indices.every(x=>Number.isInteger(x)&&x>=0&&x<pairs.length),'invalid consumed indices');
  verify(b.microSteps!==0||(b.samples===0&&b.lossSum===0&&b.gradientSum.every(x=>x===0)),'dirty empty gradient buffer');
  // Sampler cursor must equal the consumed token-pair count modulo dataset length;
  // verify the full ordered replay since initial state, including epochs.
  const consumed=snap.model?.step*config.microBatchSize*config.accumulationSteps+b.samples;
  verify(Number.isSafeInteger(consumed)&&consumed<=10000000,'invalid progress counter');
  const expectedEpoch=Math.floor((consumed-1)/pairs.length),expectedCursor=consumed%pairs.length||pairs.length;
  const normalizedEpoch=consumed===0?0:expectedEpoch;
  const normalizedCursor=consumed===0?0:expectedCursor;
  verify(state.epoch===normalizedEpoch&&state.cursor===normalizedCursor,'sampler position inconsistent with training steps');
  if(b.indices.length){
   const offset=consumed-b.samples;
   const expected=Array.from({length:b.samples},(_,i)=>{
    const global=offset+i,ep=Math.floor(global/pairs.length),index=global%pairs.length;
    return shuffledIndices(config.samplerSeed,ep)[index];
   });
   verify(expected.every((x,i)=>x===b.indices[i]),'pending sampler lineage mismatch');
  }
  model.restore(snap.model);
  epoch=state.epoch;cursor=state.cursor;
  pending={microSteps:b.microSteps,samples:b.samples,lossSum:b.lossSum,gradientSum:b.gradientSum.slice(),indices:b.indices.slice()};
  return {restored:true,optimizerStep:model.step,pendingMicroSteps:pending.microSteps,epoch,cursor};
 }
 function diagnostic(){return {modelStep:model.step,epoch,cursor,pendingMicroSteps:pending.microSteps,
  effectiveBatch:config.microBatchSize*config.accumulationSteps,nextLearningRate:scheduledLr(config,model.step+1),
  dataFingerprint:DATA_FINGERPRINT,source:SOURCE,modelDigest:model.checkpoint().weights.reduce((acc,x,i)=>acc+x*(i+1),0)};
 }
 return {config:{...config},microStep,optimizerStep,train,evaluate,generate,checkpoint,restore,previewBatch,diagnostic,
  get modelStep(){return model.step;},get pendingMicroSteps(){return pending.microSteps;}};
}
return Object.freeze({VERSION:'2026.10.21',SCHEMA,SOURCE,DATA_FINGERPRINT,defaults:baseOptions,optionsOf,scheduledLr,shuffledIndices,create});
});