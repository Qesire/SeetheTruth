/* DeepSeek-inspired inference mathematics: CPU teaching contracts, not GPU kernels.
 * Top-K, sparse selected-index attention, physical-page KV lookups, greedy
 * speculative draft verification. Every public matrix stays bounded 2D.
 */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthInference=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const KINDS=Object.freeze(["inference_topk","inference_sparse_attention","inference_paged_kv_read","inference_greedy_verify"]);
 const fail=s=>{throw Error(s)};
 function bounded(x,name){
  if(!Array.isArray(x)||x.length!==2||x.some(v=>!Number.isSafeInteger(v)||v<1||v>16)||x[0]>8)
   fail(name+" must be a bounded 2D tensor");return x;
 }
 function smallInt(x,name,lower,upper){if(!Number.isSafeInteger(x)||x<lower||x>upper)fail(name+" must be integer in "+lower+".."+upper);return x;}
 function validate(kind,p){
  if(!KINDS.includes(kind))fail("unknown inference operator");
  if(kind==="inference_topk")smallInt(p.topk,"topk",1,8);
  if(kind==="inference_sparse_attention"){
   if(p.causal!==true&&p.causal!==false)fail("causal must be boolean");
   smallInt(p.queryOffset,"queryOffset",0,16);
  }
  if(kind==="inference_paged_kv_read"){
   smallInt(p.pageSize,"pageSize",1,8);smallInt(p.logicalLength,"logicalLength",1,16);
  }
 }
 function shape(kind,inputs,p){
  validate(kind,p);
  for(const [name,s] of Object.entries(inputs))bounded(s,name);
  let out;
  if(kind==="inference_topk"){
   const [t,v]=inputs.scores;if(p.topk>v)fail("topk must not exceed vocab width");
   out=[t,p.topk];
  }else if(kind==="inference_sparse_attention"){
   const {q,k,v,indices}=inputs;
   if(q[1]!==k[1]||k[0]!==v[0]||q[0]!==indices[0]||indices[1]>k[0])
    fail("sparse attention Q/K/V/indices incompatible");
   out=[q[0],v[1]];
  }else if(kind==="inference_paged_kv_read"){
   const {cache,blocks,positions}=inputs;
   if(blocks[1]!==1||positions[1]!==1||cache[0]%p.pageSize!==0||
      p.logicalLength>blocks[0]*p.pageSize)
    fail("invalid paged KV table, physical blocks or token positions");
   out=[positions[0],cache[1]];
  }else{
   const {draft,target}=inputs;
   if(draft[1]!==1||target[1]!==1||target[0]!==draft[0]+1)
    fail("greedy verification requires draft[K,1], target[K+1,1]");
   out=[draft[0],1];
  }
  if(out[0]>8||out[1]>16)fail("inference output exceeds teaching budget");
  return out;
 }
 function matrixShape(x,name){
  if(!Array.isArray(x)||!x.length||!Array.isArray(x[0])||!x[0].length||
   x.some(r=>!Array.isArray(r)||r.length!==x[0].length||
    r.some(v=>typeof v!=="number"||!Number.isFinite(v))))fail(name+" needs finite rectangular matrix");
  return [x.length,x[0].length];
 }
 function token(x,name){if(!Number.isSafeInteger(x)||x<0||x>65535)fail(name+" needs nonnegative integral token ids");return x;}
 function greedyMetadata(draft,target){
  if(draft.length+1!==target.length)fail("greedy target must include extra bonus token");
  let accepted=0;
  while(accepted<draft.length&&draft[accepted][0]===target[accepted][0])accepted++;
  return Object.freeze({accepted,nextToken:target[accepted][0],bonus:accepted===draft.length,
    attempted:draft.length,acceptedMask:draft.map((_,j)=>[j<accepted?1:0])});
 }
 function execute(kind,inputs,p){
  const dims=Object.fromEntries(Object.entries(inputs).map(([k,x])=>[k,matrixShape(x,k)]));
  shape(kind,dims,p);
  if(kind==="inference_topk"){
   return inputs.scores.map(row=>row.map((value,index)=>({value,index}))
    .sort((a,b)=>b.value-a.value||a.index-b.index).slice(0,p.topk).map(x=>x.index));
  }
  if(kind==="inference_sparse_attention"){
   const out=[];
   for(let t=0;t<inputs.q.length;t++){
    const inds=inputs.indices[t],seen=new Set(),scores=[];
    for(const ix of inds){
     smallInt(ix,"sparse index",0,inputs.k.length-1);
     if(seen.has(ix))fail("duplicate sparse index");
     if(p.causal&&ix>p.queryOffset+t)fail("sparse attention index points to future key");
     seen.add(ix);
     scores.push(inputs.q[t].reduce((sum,v,d)=>sum+v*inputs.k[ix][d],0)/Math.sqrt(inputs.q[t].length));
    }
    const max=Math.max(...scores),exp=scores.map(s=>Math.exp(s-max)),den=exp.reduce((a,b)=>a+b,0);
    out.push(Array.from({length:inputs.v[0].length},(_,d)=>
     inds.reduce((sum,ix,j)=>sum+exp[j]/den*inputs.v[ix][d],0)));
   }
   return out;
  }
  if(kind==="inference_paged_kv_read"){
   return inputs.positions.map(([position])=>{
    smallInt(position,"logical position",0,p.logicalLength-1);
    const logical=Math.floor(position/p.pageSize),offset=position%p.pageSize;
    const page=token(inputs.blocks[logical][0],"physical page");
    const physical=page*p.pageSize+offset;
    if(physical>=inputs.cache.length)fail("physical page outside allocated KV cache");
    return inputs.cache[physical].slice();
   });
  }
  inputs.draft.forEach(([id])=>token(id,"draft token"));
  inputs.target.forEach(([id])=>token(id,"target token"));
  return greedyMetadata(inputs.draft,inputs.target).acceptedMask;
 }
 return Object.freeze({KINDS,validate,shape,execute,greedyMetadata});
});
