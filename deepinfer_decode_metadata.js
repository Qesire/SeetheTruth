/* DeepInfer csrc/engine/zos.cpp::prepare_decode_zos_inputs and
 * csrc/engine/sequence.hpp::refresh_for_sampling, bounded CPU reference.
 * Does not model physical allocation, GPU pinned buffers, or token_input packing.
 */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthDecodeZOS=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const fail=s=>{throw Error(s)};
 const int=(x,n,min=0,max=1e6)=>{if(!Number.isSafeInteger(x)||x<min||x>max)fail(n+" must be integer in ["+min+","+max+"]");return x};
 const arr=(a,n,max=16)=>{if(!Array.isArray(a)||a.length>max)fail(n+" must be an array (length <= "+max+")");return a};
 function validate(input){
  if(!input||typeof input!=="object")fail("decode input object required");
  const {sequences,blockSize,maxNumBlocks,maxSeqlenQ,blockTableCompressRatio,
    needDefaultSeqlens}=input;
  arr(sequences,"sequences");
  int(blockSize,"blockSize",1,128);int(maxNumBlocks,"maxNumBlocks",1,128);
  int(maxSeqlenQ,"maxSeqlenQ",1,64);int(blockTableCompressRatio,"blockTableCompressRatio",1,128);
  if(blockSize%blockTableCompressRatio!==0)fail("blockSize must be divisible by blockTableCompressRatio");
  if(typeof needDefaultSeqlens!=="boolean")fail("needDefaultSeqlens must be boolean");
  const seen=new Set();
  return sequences.map((s,i)=>{
   if(!s||typeof s!=="object"||typeof s.seqId!=="string"||!/^[A-Za-z0-9_-]{1,24}$/.test(s.seqId))
    fail("invalid seqId "+i);
   if(seen.has(s.seqId))fail("duplicate seqId");seen.add(s.seqId);
   if(typeof s.isPlaceholder!=="boolean")fail("isPlaceholder must be boolean");
   int(s.seqlenQ,"seqlenQ",1,64);int(s.seqlenK,"seqlenK",0);
   int(s.stateId,"stateId",0);
   arr(s.blockTable,"blockTable",128).forEach((b,j)=>int(b,"blockTable["+i+"]["+j+"]",0));
   return {...s,blockTable:s.blockTable.slice()};
  });
 }
 function prepare(input){
  const ss=validate(input),{blockSize,maxNumBlocks,maxSeqlenQ,
   blockTableCompressRatio:ratio,needDefaultSeqlens}=input;
  const tokensPerBlock=blockSize/ratio;
  const numBlocks=[],offsets=[0],blocks=[],cuSeqlensQ=[],stateIds=[],placeholders=[];
  let totalQ=0;
  for(const s of ss){
   let count=1;
   if(!s.isPlaceholder){
    const maxK=s.seqlenK-s.seqlenQ+maxSeqlenQ;
    if(maxK<0)fail("maxK cannot be negative");
    const compressedK=Math.floor(maxK/ratio);
    count=Math.ceil(compressedK/tokensPerBlock);
   }
   if(count>maxNumBlocks||count>s.blockTable.length)
    fail("insufficient reserved blockTable entries for required compressed KV");
   numBlocks.push(count);blocks.push(...s.blockTable.slice(0,count));offsets.push(blocks.length);
   totalQ+=s.seqlenQ;cuSeqlensQ.push(totalQ);stateIds.push(s.stateId);
   placeholders.push(s.isPlaceholder);
  }
  return Object.freeze({
   sequences:Object.freeze(ss.map(s=>Object.freeze({...s,blockTable:Object.freeze(s.blockTable)}))),
   numBlocks:Object.freeze(numBlocks),blockTableOffsets:Object.freeze(offsets),
   blockTables:Object.freeze(blocks),cuSeqlensQ:Object.freeze(cuSeqlensQ),
   stateIds:Object.freeze(stateIds),isPlaceholder:Object.freeze(placeholders),
   seqlensQ:needDefaultSeqlens?Object.freeze(ss.map(s=>s.seqlenQ)):null,
   seqlensK:needDefaultSeqlens?Object.freeze(ss.map(s=>s.seqlenK)):null,
   totalQ,tokensPerBlock,ratio,
   note:"C++ integer layouts only; token input packing and GPU allocation omitted"});
 }
 function refreshForSampling(previous,sequence,{nextn=0,outputLen=0}={}){
  int(nextn,"nextn",0,8);int(outputLen,"outputLen",0);
  if(!previous||typeof previous!=="object"||!sequence||typeof sequence!=="object")
   fail("existing SequenceMetadata and Sequence states required");
  const mtp=sequence.mtp;
  if(!mtp||typeof mtp!=="object")fail("mtp state required");
  int(mtp.infoSlot,"mtp.infoSlot",0);
  if(typeof mtp.hasDraftInfo!=="boolean")fail("mtp.hasDraftInfo must be boolean");
  int(previous.mtpInfoWriteSlot,"previous.mtpInfoWriteSlot",0);
  int(mtp.lastOutputToken,"mtp.lastOutputToken",-1);
  const golden=mtp.goldenTokens;
  if(golden!==null&&golden!==undefined){
   arr(golden,"goldenTokens",256);
   golden.forEach((v,i)=>int(v,"goldenTokens["+i+"]",0,65535));
  }
  const keepWrite=mtp.infoSlot!==0?mtp.infoSlot:previous.mtpInfoWriteSlot;
  const readSlot=mtp.hasDraftInfo?mtp.infoSlot:0;
  const resultGolden=!golden||outputLen>=golden.length?null:
   Array.from({length:2*nextn+1},(_,i)=>golden[outputLen+i]??-1);
  return Object.freeze({
   metadata:Object.freeze({...previous,
    lastOutputToken:mtp.lastOutputToken,mtpInfoReadSlot:readSlot,
    mtpInfoWriteSlot:keepWrite,soStatus:sequence.soStatus,soMatcher:sequence.soMatcher,
    goldenToken:resultGolden===null?null:Object.freeze(resultGolden)}),
   nextSequence:Object.freeze({...sequence,mtp:Object.freeze({...mtp,hasDraftInfo:false})}),
   note:"MTP read slot is consumed; write slot changes only for nonzero infoSlot; structured output object is carried, not evaluated"});
 }
 function updateForSampling(prepared,corrected,states,{nextn=0}={}){
  if(!prepared||!corrected||!Array.isArray(states)||states.length!==prepared.sequences.length)
   fail("sampling requires packed metadata and matching Sequence states");
  int(nextn,"nextn",0,8);
  const output=[],statesNext=[];
  prepared.sequences.forEach((s,i)=>{
   const st=states[i];if(!st||st.seqId!==s.seqId)fail("scheduled state identity mismatch");
   int(st.cachedTokens,"cachedTokens",0);
   const q=corrected.seqlensQ[i],remap=corrected.remap[i];
   int(q,"correctedQ",1,64);int(remap,"remap",-1);
   let metadata={seqId:s.seqId,seqlenQ:q,seqlenK:q+st.cachedTokens,
    lastOutputToken:s.lastOutputToken??-1,mtpInfoReadSlot:s.mtpInfoReadSlot??0,
    mtpInfoWriteSlot:s.mtpInfoWriteSlot??0,soStatus:s.soStatus??0,
    soMatcher:s.soMatcher??null,goldenToken:s.goldenToken??null};
   let newState=st,refreshed=false;
   if(remap>=0){
    const res=refreshForSampling(metadata,st,{nextn,outputLen:st.outputLen??0});
    metadata=res.metadata;newState=res.nextSequence;refreshed=true;
   }
   output.push(Object.freeze({...metadata,refreshed}));statesNext.push(newState);
  });
  return Object.freeze({metadata:Object.freeze(output),states:Object.freeze(statesNext),
   note:"q/cache K and sampled MTP metadata updated; does not append tokens, allocate blocks, run matcher or process model outputs"});
 }
 return Object.freeze({validate,prepare,refreshForSampling,updateForSampling});
});
