/* DeepInfer C++ scheduler.hpp NORMAL vote, _adjust_chunk_size, chunk prefill
 * bounded source-grounded math and decision subset. No actual KV allocation.
 */
(function(root,factory){"use strict";const o=factory();if(typeof module==="object"&&module.exports)module.exports=o;root.SeetheTruthPrefillPolicy=o;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const fail=s=>{throw Error(s)};
 const integer=(value,name,min=0,max=1024)=>{if(!Number.isSafeInteger(value)||value<min||value>max)fail(name+" must be integer in "+min+".."+max);return value;};
 const finite=(value,name)=>{if(typeof value!=="number"||!Number.isFinite(value)||value<0||value>1)fail(name+" must be in [0,1]");return value;};
 function vote({engine="NORMAL",decodeStep=0,allInterval=2,anyInterval=4,watermark=0.95,ranks=[]}={}){
  if(!["NORMAL","PROMPT","COMPLETION"].includes(engine))fail("unknown DeepInfer engine type");
  integer(decodeStep,"decodeStep",0);
  integer(allInterval,"allInterval",1);
  integer(anyInterval,"anyInterval",1);
  finite(watermark,"watermark");
  if(!Array.isArray(ranks)||!ranks.length||ranks.length>8)fail("ranks must have 1..8 entries");
  const wants=ranks.map((r,i)=>{
   if(!r||typeof r.waitingSchedulable!=="boolean"||typeof r.canActuallySchedule!=="boolean")fail("rank "+i+" flags invalid");
   return r.waitingSchedulable&&finite(r.cacheUsage,"rank "+i+" cacheUsage")<=watermark;
  });
  if(engine==="COMPLETION")
   return Object.freeze({point:"completion-only",step:"decode",perRank:ranks.map(()=>"decode"),wants,scheduledCount:0});
  if(engine==="PROMPT")
   return Object.freeze({point:"prompt-only",step:"prompt",perRank:ranks.map(()=> "prompt"),wants,scheduledCount:ranks.filter(r=>r.canActuallySchedule).length});
  const anyPoint=decodeStep!==0&&decodeStep%anyInterval===0;
  const allPoint=decodeStep%allInterval===0;
  const point=anyPoint?"ANY":allPoint?"ALL":"NONE";
  const selected=point==="ANY"?wants.some(Boolean):point==="ALL"?wants.every(Boolean):false;
  // A first positive vote is not sufficient: the second vote tests whether
  // any rank truly scheduled prefill after quotas and memory checks.
  const actual=selected?ranks.map((r,i)=>wants[i]&&r.canActuallySchedule):ranks.map(()=>false);
  const scheduledCount=actual.filter(Boolean).length;
  const step=scheduledCount>0?"prompt":"decode";
  return Object.freeze({point,selected,step,wants,actuallyScheduled:actual,
   perRank:actual.map(v=>step==="decode"?"decode":v?"prompt":"empty-prompt"),
   scheduledCount,nextDecodeStep:step==="prompt"?0:decodeStep+1,
   secondVoteFallback:selected&&scheduledCount===0});
 }
 function insidePad(tokens,boundary){
  if(boundary>=tokens.length||tokens[boundary]!=="PAD")return false;
  for(let i=boundary-1;i>=0;i--){if(tokens[i]==="IMG")return true;if(tokens[i]!=="PAD")return false;}
  return false;
 }
 function adjustImageChunk(tokens,cached,q,makeLarger){
  integer(cached,"cached",0,tokens.length);
  integer(q,"queryChunk",0,tokens.length-cached);
  if(typeof makeLarger!=="boolean")fail("makeLarger must be boolean");
  let boundary=cached+q;
  if(!insidePad(tokens,boundary))return Object.freeze({q,boundary,adjusted:false});
  if(makeLarger){
   while(boundary<tokens.length&&tokens[boundary]==="PAD")boundary++;
  }else{
   while(boundary>cached&&tokens[boundary-1]==="PAD")boundary--;
   if(boundary>cached&&tokens[boundary-1]==="IMG")boundary--;
  }
  return Object.freeze({q:boundary-cached,boundary,adjusted:true});
 }
 function chunk({tokens,cached=0,maxBatchedTokens=8,alreadyBatched=0,cpAlign=1,
  chunkedPrefill=true,lookahead=0,checkpoints=[],scheduledCount=0}={}){
  if(!Array.isArray(tokens)||tokens.length<1||tokens.length>128||
   tokens.some(t=>!["TEXT","IMG","PAD","AUDIO","VIDEO"].includes(t)))fail("bounded multimodal token types required");
  integer(cached,"cached",0,tokens.length);
  integer(maxBatchedTokens,"maxBatchedTokens",1,128);
  integer(alreadyBatched,"alreadyBatched",0,maxBatchedTokens);
  integer(cpAlign,"cpAlign",1,32);
  integer(lookahead,"lookahead",0,32);
  integer(scheduledCount,"scheduledCount",0,32);
  if(typeof chunkedPrefill!=="boolean")fail("chunkedPrefill boolean required");
  if(!Array.isArray(checkpoints)||checkpoints.length>32||checkpoints.some(v=>!Number.isSafeInteger(v)||v<1||v>=tokens.length))
   fail("invalid prefill chunk checkpoints");
  const workload=tokens.length-cached;
  if(workload<lookahead)fail("pending prompt workload shorter than lookahead");
  if(!workload)return Object.freeze({outcome:"complete",q:0,remaining:0,charged:0,status:"RUNNING"});
  const avail=maxBatchedTokens-alreadyBatched;
  const alignedAvail=Math.floor(avail/cpAlign)*cpAlign;
  let q=Math.min(workload,alignedAvail);
  const checkpoint=checkpoints.filter(x=>x>cached).sort((a,b)=>a-b)[0];
  if(checkpoint&&cached+q>=checkpoint)q=checkpoint-cached;
  const initialQ=q;
  let attempt=adjustImageChunk(tokens,cached,q,scheduledCount===0);
  // C++ extends only when the raw Q can still fit; if not, it shrinks.
  if(alreadyBatched+attempt.q>maxBatchedTokens)attempt=adjustImageChunk(tokens,cached,q,false);
  q=attempt.q;
  // Separate rounded CP allocation from physical source: if extending a
  // non-aligned image pushes the charged count beyond quota, conservatively
  // shrink instead of claiming the original C++ allowed that overrun.
  let charged=Math.ceil(q/cpAlign)*cpAlign;
  let alignmentFallback=false;
  if(alreadyBatched+charged>maxBatchedTokens){
   attempt=adjustImageChunk(tokens,cached,initialQ,false);
   q=attempt.q;charged=Math.ceil(q/cpAlign)*cpAlign;alignmentFallback=true;
  }
  if(q===0){
   if(scheduledCount===0)fail("source cannot schedule first prompt with zero Q after image boundary adjustment");
   return Object.freeze({outcome:"deferred",why:"zero Q after image boundary",q:0,remaining:workload,charged:0});
  }
  if(q<Math.max(lookahead,1))return Object.freeze({outcome:"deferred",why:"below lookahead",q,remaining:workload,charged:0});
  if(q<workload&&!chunkedPrefill)return Object.freeze({outcome:"deferred",why:"chunked prefill disabled",q,remaining:workload,charged:0});
  if(alreadyBatched+charged>maxBatchedTokens)fail("CP padded token budget violation");
  return Object.freeze({outcome:"scheduled",q,workload,chunked:q<workload,remaining:workload-q,
   newCached:cached+q,charged,checkpoint:checkpoint??null,
   imageAdjusted:attempt.adjusted,alignmentFallback,
   status:q===workload?"RUNNING":"WAITING",
   note:"pure source-derived chunk decision; no GPU allocation, MM embedding, MTP slot or distributed coordination"});
 }
 return Object.freeze({vote,insidePad,adjustImageChunk,chunk});
});
