/* Source-grounded CPU microcontracts for DeepInfer (2026-10-20).
 * Confidence scheduling matches the public Python reference algorithm in
 * deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py.
 * YoCo attention is an explicitly bounded scalar-model of a consumer SWA
 * mask/compact suffix and is NOT the full model's metadata implementation.
 */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthDeepInferRuntime=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const KINDS=Object.freeze(["deepinfer_mtp_confidence_schedule","deepinfer_yoco_swa_mask","deepinfer_yoco_swa_compact"]);
 const fail=s=>{throw Error(s);};
 const pos=(v,name,max=16)=>{if(!Number.isSafeInteger(v)||v<1||v>max)fail(name+" must be integer in 1.."+max);return v;};
 function validate(kind,p){
  if(!KINDS.includes(kind))fail("unknown DeepInfer runtime operator");
  if(kind==="deepinfer_mtp_confidence_schedule"){
   pos(p.totalQ,"totalQ",64);
  }else{
   pos(p.window,"window",16);pos(p.keepTokens,"keepTokens",16);
   if(!Number.isSafeInteger(p.backtracking)||p.backtracking<0||p.backtracking>16)fail("backtracking must be 0..16");
   if(p.keepTokens<p.window+p.backtracking)
    fail("DeepInfer YoCo requires keepTokens >= window + backtracking");
  }
 }
 function bounded(shape,name){
  if(!Array.isArray(shape)||shape.length!==2||shape.some(v=>!Number.isSafeInteger(v)||v<1||v>16)||shape[0]>8)
   fail(name+" must be a bounded two-axis matrix");
  return shape;
 }
 function shape(kind,inputs,p){
  validate(kind,p);Object.entries(inputs).forEach(([n,s])=>bounded(s,n));
  if(kind==="deepinfer_mtp_confidence_schedule"){
   const r=inputs.remap,confidence=inputs.prev_confidence;
   if(r[1]!==1||confidence[1]<1||p.totalQ<r[0]||p.totalQ>r[0]*(confidence[1]+1))
    fail("invalid MTP remap/nextn/totalQ budget");
   return [r[0],1];
  }
  const q=inputs.q,k=inputs.k,v=inputs.v;
  if(q[1]!==k[1]||k[0]!==v[0]||q[0]!==k[0])
   fail("YoCo sample requires aligned self-attention Q, K, V token axes");
  return [kind==="deepinfer_yoco_swa_mask"?q[0]:Math.min(q[0],p.keepTokens),v[1]];
 }
 function matrixShape(x,name){
  if(!Array.isArray(x)||x.length<1||!Array.isArray(x[0])||!x[0].length||
     x.some(row=>!Array.isArray(row)||row.length!==x[0].length||
      row.some(v=>typeof v!=="number"||!Number.isFinite(v))))fail(name+" requires finite rectangular matrix");
  return [x.length,x[0].length];
 }
 function schedule(remap,prev,totalQ){
  const rows=remap.length,nextn=prev[0].length;
  const ranks=[];
  for(let i=0;i<rows;i++){
   const original=remap[i][0];
   if(!Number.isSafeInteger(original)||original< -1||original>=prev.length)
    fail("remap must be -1 or valid prior sequence index");
   let prod=1;
   for(let j=0;j<nextn;j++){
    const v=original===-1?0:prev[original][j];
    if(v<0||v>1)fail("MTP confidence rates must be in [0,1]");
    prod*=v;
    ranks.push({row:i,col:j,value:prod,flat:i*nextn+j});
   }
  }
  ranks.sort((a,b)=>b.value-a.value||a.flat-b.flat);
  const output=Array.from({length:rows},()=>[1]);
  for(let idx=0;idx<totalQ-rows;idx++)output[ranks[idx].row][0]++;
  return output;
 }
 function yoco(q,k,v,p,mode){
  const n=q.length,keep=Math.min(n,p.keepTokens),cut=n-keep,rows=[];
  for(let t=(mode==="compact"?cut:0);t<n;t++){
   if(t<cut){rows.push(Array(v[0].length).fill(0));continue;}
   const start=Math.max(0,t-p.window+1),s=[];
   for(let j=start;j<=t;j++){
    let dot=0;
    for(let d=0;d<q[t].length;d++)dot+=q[t][d]*k[j][d];
    s.push({j,score:dot/Math.sqrt(q[t].length)});
   }
   const mx=Math.max(...s.map(x=>x.score)),weights=s.map(x=>Math.exp(x.score-mx));
   const den=weights.reduce((a,b)=>a+b,0);
   rows.push(Array.from({length:v[0].length},(_,d)=>
     s.reduce((a,x,i)=>a+weights[i]/den*v[x.j][d],0)));
  }
  return rows;
 }
 function report(qLength,p){
  validate("deepinfer_yoco_swa_mask",p);
  const cut=Math.max(0,qLength-p.keepTokens);
  const retained=qLength-cut;
  const eligiblePairs=Array.from({length:retained},(_,u)=>Math.min(p.window,cut+u+1)).reduce((a,b)=>a+b,0);
  return Object.freeze({totalQueryRows:qLength,retainedQueryRows:retained,removedQueryRows:cut,
    maskQuerySlots:qLength,compactQuerySlots:retained,eligibleQKPairs:eligiblePairs,
    note:"mask reserves all Q rows but masked prefix is not semantically reused; compact omits prefix rows; no measured GPU FLOPs"});
 }
 function execute(kind,inputs,p){
  const dims=Object.fromEntries(Object.entries(inputs).map(([name,x])=>[name,matrixShape(x,name)]));
  const expected=shape(kind,dims,p);let result;
  if(kind==="deepinfer_mtp_confidence_schedule")result=schedule(inputs.remap,inputs.prev_confidence,p.totalQ);
  else result=yoco(inputs.q,inputs.k,inputs.v,p,kind==="deepinfer_yoco_swa_mask"?"mask":"compact");
  if(result.length!==expected[0]||result.some(row=>row.length!==expected[1]||row.some(v=>!Number.isFinite(v))))
   fail("DeepInfer microcontract returned unexpected shape or non-finite output");
  return result;
 }
 return Object.freeze({KINDS,validate,shape,execute,schedule,report});
});
