/* Stateful, deterministic KV Cache teaching runner for GQA.
 * No model weights, tokenizer, GPU operations or token sampling.
 * Cache stores the projected K,V for every processed position, not just counters.
 */
(function(root,create){
 'use strict';
 const api=create(typeof module==='object'&&module.exports?require('./tensor_ir.js'):root.SeetheTruthTensorIR);
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.SeetheTruthCacheLab=api;
})(typeof window!=='undefined'?window:{},function(IR){
 'use strict';
 const MAX_TOKENS=12;
 function bad(msg){throw Error(msg);}
 function positive(v,max,label){if(!Number.isSafeInteger(v)||v<1||v>max)bad(label+' must be integer 1..'+max);return v;}
 function matrixFor(position,width,salt){
   return Array.from({length:width},(_,j)=>
     Math.sin((position+1)*(.37+j*.17)+salt*1.13)*.73+
     Math.cos((j+2)*.41+(position+1)*(.29+salt*.02))*.22);
 }
 function config(user={}){
   const prefix=positive(user.prefix??3,8,'prefix'),decode=positive(user.decode??4,MAX_TOKENS-prefix,'decode');
   const heads=positive(user.heads??2,8,'heads'),kvHeads=positive(user.kvHeads??1,8,'kvHeads');
   const headDim=positive(user.headDim??4,8,'headDim'),valueDim=positive(user.valueDim??4,8,'valueDim');
   if(heads%kvHeads!==0)bad('Hq must be divisible by Hkv');
   if(headDim%2)bad('RoPE headDim must be even');
   const rope=user.rope??true;
   if(typeof rope!=='boolean')bad('rope must be boolean');
   const total=prefix+decode;if(total>MAX_TOKENS)bad('total tokens exceed cache micro simulation limit');
   return Object.freeze({prefix,decode,heads,kvHeads,headDim,valueDim,rope,total,
     evidence:'stateful_deterministic_teaching_kv_cache_not_model_inference'});
 }
 function clone(x){return JSON.parse(JSON.stringify(x));}
 function Session(user={}){
   const settings=config(user);
   let phase='empty',cursor=0,keys=[],values=[],last=null;
   const history=[];
   function project(kind,position){
     const width=(kind==='q'?settings.heads:settings.kvHeads)*
       (kind==='v'?settings.valueDim:settings.headDim);
     return matrixFor(position,width,kind==='q'?1:kind==='k'?2:3);
   }
   function snapshot(){
     return clone({phase,cursor,config:settings,cache:{key:keys,value:values,length:keys.length,
       keyShape:[1,settings.kvHeads,keys.length,settings.headDim],
       valueShape:[1,settings.kvHeads,values.length,settings.valueDim]},
       last,history});
   }
   function advance(){
     if(cursor>=settings.total)bad('all configured positions already processed');
     const before=keys.length;
     let append,queryPositions;
     if(phase==='empty'){
       append=settings.prefix;queryPositions=Array.from({length:append},(_,i)=>i);phase='prefill';
     }else{
       append=1;queryPositions=[cursor];phase='decode';
     }
     const q=queryPositions.map(i=>project('q',i));
     const newK=queryPositions.map(i=>project('k',i));
     const newV=queryPositions.map(i=>project('v',i));
     // This is the actual cache mutation. Only NEW token projections are appended.
     keys.push(...newK);values.push(...newV);cursor+=append;
     if(keys.length!==cursor||values.length!==cursor)bad('KV cache length invariant violated');
     const result=IR.gqa(q,keys,values,{
       heads:settings.heads,kvHeads:settings.kvHeads,causal:true,rope:settings.rope
     });
     const accessed=keys.length;
     const oldWorkAvoided=phase==='decode'?(accessed-1):0;
     last={
       index:history.length,phase,queryPositions,appended:append,
       kAppendedPositions:queryPositions.slice(),cacheBefore:before,
       cacheAfter:keys.length,queryLength:q.length,keyLength:accessed,
       queryShape:[1,settings.heads,q.length,settings.headDim],
       keyShape:[1,settings.kvHeads,keys.length,settings.headDim],
       valueShape:[1,settings.kvHeads,values.length,settings.valueDim],
       probabilitiesShape:result.stages.probabilities.shape,
       mathMatmulFlops:result.knownMatmulFlops,
       avoidedOldProjectionTokens:oldWorkAvoided,
       selectedHead:0,
       output:result.output,
       attention:result.stages.probabilities,
       scores:result.stages.scores,
       positionedQuery:result.stages.q_positioned,
       positionedKeys:result.stages.k_positioned,
       mask:result.masks.allowed,
       projectedNewK:newK,projectedNewV:newV,
       note:phase==='prefill'?'Prefill computes all prefix Q,K,V once, appending all K/V to cache.':
            'Decode projects only the new token, appends K/V once, and attends to all cached positions.',
       cacheOwnership:'same_session_mutable_projected_kv_not_static_prefix_argument'
     };
     // History entries are snapshots; changing the active cache never alters older steps.
     history.push(clone(last));
     return snapshot();
   }
   function reset(){phase='empty';cursor=0;keys=[];values=[];last=null;history.length=0;return snapshot();}
   return Object.freeze({config:settings,advance,snapshot,reset,
     project:(kind,pos)=>{if(!['q','k','v'].includes(kind)||!Number.isSafeInteger(pos)||pos<0||pos>=settings.total)bad('invalid projection');return project(kind,pos);}
   });
 }
 function referenceFull(user={}){
   const s=Session(user), cfg=s.config,q=[],k=[],v=[];
   for(let i=0;i<cfg.total;i++){
     q.push(s.project('q',i));k.push(s.project('k',i));v.push(s.project('v',i));
   }
   return IR.gqa(q,k,v,{heads:cfg.heads,kvHeads:cfg.kvHeads,
     causal:true,rope:cfg.rope});
 }
 function countPlan(user={}){
   const cfg=config(user),ops=[],head=cfg.heads,di=cfg.headDim,dv=cfg.valueDim;
   let total=0,nonCachedTotal=0,projected=0;
   for(let i=0;i<=cfg.decode;i++){
     const isPrefill=i===0,q=i===0?cfg.prefix:1,k=cfg.prefix+i;
     const gemm=2*head*q*k*(di+dv);
     const newProjected=isPrefill?cfg.prefix:1;
     total+=gemm;projected+=newProjected;
     // Non-cached baseline recomputes the full prefix model forward as tokens grow.
     nonCachedTotal+=2*head*k*k*(di+dv);
     ops.push({phase:isPrefill?'prefill':'decode',queryTokens:q,cachedTokens:k,
       mathMatmulFlops:gemm,projectedNewTokens:newProjected});
   }
   return {steps:ops,attributedMatmulFlops:total,
     fullPrefixRecomputeMatmulFlops:nonCachedTotal,
     projectedPositionsWithCache:projected,
     fullPrefixRecomputeProjectedPositions:ops.reduce((sum,x)=>sum+x.cachedTokens,0),
     disclaimer:'only QK/PV GEMM; projections, softmax and memory traffic omitted; baseline is an explicit full-prefix recomputation teaching scenario'};
 }
 return Object.freeze({MAX_TOKENS,config,Session,referenceFull,countPlan});
});
