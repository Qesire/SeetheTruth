/* Educational analytical cost models. No measured latency or model-specific claims. */
(function(root,factory){
  'use strict';
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  root.SeetheTruthTradeoffs=api;
})(typeof window!=='undefined'?window:{},function(){
  'use strict';
  const integer=(v,name,min=1,max=1000000)=>{
    if(!Number.isSafeInteger(v)||v<min||v>max)throw new RangeError(name+' must be an integer in ['+min+', '+max+']');
    return v;
  };
  const positive=(v,name)=>{if(typeof v!=='number'||!Number.isFinite(v)||v<=0)throw new RangeError(name+' must be positive finite');return v;};
  function gqa({layers,heads,kvHeads,headDim,context,batch=1,bytes=2}){
    [layers,heads,kvHeads,headDim,context,batch,bytes].forEach((v,i)=>integer(v,['layers','heads','kvHeads','headDim','context','batch','bytes'][i]));
    if(heads%kvHeads!==0)throw new RangeError('query heads must be divisible by KV heads');
    const total=2*layers*batch*context*kvHeads*headDim*bytes;
    if(!Number.isSafeInteger(total))throw new RangeError('cache exceeds safe integer precision');
    const mha=2*layers*batch*context*heads*headDim*bytes;
    return {bytes:total,mhaBytes:mha,ratio:kvHeads/heads,saving:1-kvHeads/heads,groupSize:heads/kvHeads,
      assumptions:'K/V only, per-token cache storage; excludes allocator fragmentation, metadata and latency'};
  }
  function moe({tokens,hidden,intermediate,experts,topK,sharedExperts=0}){
    [tokens,hidden,intermediate,experts,topK].forEach((v,i)=>integer(v,['tokens','hidden','intermediate','experts','topK'][i]));
    integer(sharedExperts,'sharedExperts',0);
    if(topK>experts)throw new RangeError('topK cannot exceed experts');
    const perExpert=3*hidden*intermediate; // bias-free gated FFN: gate/up/down
    const total=(experts+sharedExperts)*perExpert;
    const active=(topK+sharedExperts)*perExpert;
    const flops=2*tokens*active;
    if(![total,active,flops].every(Number.isSafeInteger))throw new RangeError('count exceeds safe precision');
    return {totalExpertParameters:total,activeExpertParametersPerToken:active,
      activeFraction:active/total,approxExpertFlops:flops,
      assumptions:'bias-free SwiGLU-style experts; excludes router, communication, attention, imbalance and kernel utilization'};
  }
  function loop({width,intermediate,steps,untiedLayers=steps}){
    [width,intermediate,steps,untiedLayers].forEach((v,i)=>integer(v,['width','intermediate','steps','untiedLayers'][i]));
    // simplified decoder block: Q,K,V,O projections + gated FFN, ignoring norms and embedding
    const perBlock=4*width*width+3*width*intermediate;
    const shared=perBlock,untied=perBlock*untiedLayers;
    const iterativeFlops=steps*2*perBlock;
    if(![shared,untied,iterativeFlops].every(Number.isSafeInteger))throw new RangeError('count exceeds safe precision');
    return {sharedParameters:shared,untiedParameters:untied,parameterRatio:shared/untied,
      perTokenProjectionFlops:iterativeFlops,
      assumptions:'same core repeated with tied weights; token-local projection FLOPs only, excludes attention scores, cache, convergence and adaptive halting'};
  }
  return Object.freeze({gqa,moe,loop});
});
