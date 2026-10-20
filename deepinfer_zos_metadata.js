/* DeepInfer ZOS per-sequence CPU reference state, grounded in csrc/engine/zos.cpp,
 * compute_seqlens_k_zos.py and Worker._correct_seqlens_qk.
 * No GPU/pinned-memory or C++ scheduler is invoked; exact integer contracts only.
 */
(function(root,factory){"use strict";const a=factory(typeof module==="object"&&module.exports?require("./deepinfer_runtime_ops.js"):root.SeetheTruthDeepInferRuntime);if(typeof module==="object"&&module.exports)module.exports=a;root.SeetheTruthZOSMetadata=a;})(typeof window!=="undefined"?window:{},function(MTP){"use strict";
 const fail=s=>{throw Error(s);};
 const int=(x,name,min=0,max=1000000)=>{if(!Number.isSafeInteger(x)||x<min||x>max)fail(name+" integer out of bounds ["+min+","+max+"]");return x;};
 function seqs(items,name){
  if(!Array.isArray(items)||items.length>16)fail(name+" must be at most 16 sequences");
  const set=new Set();
  return items.map((v,i)=>{
   if(!v||typeof v!=="object"||typeof v.seqId!=="string"||!/^[A-Za-z0-9_-]{1,24}$/.test(v.seqId))fail(name+" invalid seqId at "+i);
   if(set.has(v.seqId))fail(name+" has duplicate sequence "+v.seqId);set.add(v.seqId);
   const isPlaceholder=v.isPlaceholder===true;
   if(v.isPlaceholder!==undefined&&typeof v.isPlaceholder!=="boolean")fail(name+" invalid isPlaceholder");
   return Object.freeze({seqId:v.seqId,isPlaceholder,
    seqlenQ:int(v.seqlenQ,name+" seqlenQ",1),seqlenK:int(v.seqlenK,name+" seqlenK",0)});
  });
 }
 function prepare(current,previous,{nextn=0,paddedLength=current.length}={}){
  int(nextn,"nextn",0,8);int(paddedLength,"paddedLength",current.length,32);
  const curr=seqs(current,"current"),prev=seqs(previous,"previous");
  const map=new Map(prev.filter(s=>!s.isPlaceholder).map((s,i)=>[s.seqId,prev.findIndex(p=>p.seqId===s.seqId)]));
  const sameOrder=curr.length===prev.length&&curr.every((s,i)=>!prev[i].isPlaceholder&&s.seqId===prev[i].seqId);
  const remap=curr.map((s,i)=>sameOrder?i:(map.get(s.seqId)??-1));
  const hasOverlap=remap.some(x=>x>=0);
  if(!hasOverlap)return null; // upstream returns four null/null/null/false fields, no GPU correction
  const seqlensQ=curr.map(s=>s.seqlenQ),seqlensK=curr.map(s=>s.seqlenK-(nextn>0?s.seqlenQ:0));
  if(seqlensK.some(v=>v<0))fail("initial seqlenK−seqlenQ must not be negative");
  while(remap.length<paddedLength){remap.push(-1);seqlensQ.push(1);seqlensK.push(1);}
  return Object.freeze({hasOverlap:true,sameOrder,activeCount:curr.length,paddedLength,nextn,
    remap:Object.freeze(remap),seqlensQ:Object.freeze(seqlensQ),seqlensKBase:Object.freeze(seqlensK),
    current:Object.freeze(curr),previous:Object.freeze(prev)});
 }
 function correct(prepared,{accepted,variableMtp=false,confidenceRates=null,totalQ=null}={}){
  if(!prepared||!prepared.hasOverlap)fail("correction requires previous/current overlap");
  if(typeof variableMtp!=="boolean")fail("variableMtp must be boolean");
  const {nextn,activeCount,remap,seqlensQ,seqlensKBase,previous}=prepared;
  if(!Array.isArray(accepted)||accepted.length!==previous.length)fail("accepted token counts require one per previous sequence");
  accepted.forEach((v,i)=>int(v,"accepted["+i+"]",0,nextn));
  let q=seqlensQ.slice();
  if(variableMtp){
   if(nextn<1)fail("varlen requires nextn>0");
   int(totalQ,"totalQ",activeCount,activeCount*(nextn+1));
   if(!Array.isArray(confidenceRates)||confidenceRates.length!==previous.length)
    fail("confidenceRates must cover previous batch");
   confidenceRates.forEach((row,i)=>{
    if(!Array.isArray(row)||row.length!==nextn)fail("confidenceRates row "+i+" width must equal nextn");
    for(const rate of row)if(typeof rate!=="number"||!Number.isFinite(rate)||rate<0||rate>1)
     fail("confidenceRates must be probabilities [0,1]");
   });
   // Exact CPU implementation of compute_seqlens_q_by_confidence_rates_ref
   const fresh=remap.slice(0,activeCount).map(v=>[v]);
   const picked=MTP.schedule(fresh,confidenceRates,totalQ).map(row=>row[0]);
   picked.forEach((v,i)=>q[i]=v);
  }
  const k=seqlensKBase.slice();
  for(let i=0;i<activeCount;i++){
   k[i]+=q[i];
   if(nextn>0&&remap[i]>=0)k[i]-=nextn-accepted[remap[i]];
   int(k[i],"corrected seqlenK["+i+"]",0);
  }
  return Object.freeze({remap:Object.freeze(remap.slice()),
   seqlensQ:Object.freeze(q),seqlensK:Object.freeze(k),
   acceptedPrevious:Object.freeze(accepted.slice()),variableMtp,totalQ:variableMtp?totalQ:null,
   nActive:activeCount,kind:"corrected GPU metadata modeled on CPU",
   // D2H occurs only on varlen; ordinary ZOS Q source remains unchanged on host.
   d2hSeqlensQ:variableMtp?Object.freeze(q.slice(0,activeCount)):null});
 }
 function samplingView(prepared,corrected,cachedTokens){
  // Mirrors ONLY C++ update_metadata_for_sampling_zos:
  // seqlen_q corrected; seqlen_k = seqlen_q + scheduler seq.cache.num_tokens.
  // refresh_for_sampling(seq,nextn) is NOT reconstructed.
  if(!Array.isArray(cachedTokens)||cachedTokens.length!==prepared.activeCount)fail("cache token counts required for current batch");
  return Object.freeze(Array.from({length:prepared.activeCount},(_,i)=>{
   int(cachedTokens[i],"scheduler cached tokens",0);
   return Object.freeze({seqId:prepared.current[i].seqId,
    remap:corrected.remap[i],seqlenQ:corrected.seqlensQ[i],
    seqlenK:corrected.seqlensQ[i]+cachedTokens[i],
    refreshForSamplingRequired:corrected.remap[i]>=0});
  }));
 }
 return Object.freeze({prepare,correct,samplingView});
});
