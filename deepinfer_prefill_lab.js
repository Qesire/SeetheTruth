(function(){"use strict";
const P=window.SeetheTruthPrefillPolicy;
if(!P)throw Error("Prefill teaching source module unavailable");
const $=id=>document.getElementById(id);
const State=window.SeetheTruthStateMicroscope;
const stateView=window.SeetheTruthStateMicroscopeUI.mount($('stateMicroscope'));
const evidence={};
function sync(source){if(evidence.voteInput&&evidence.chunkInput){
 const frame=State.prefill(evidence,P);stateView.show(frame,source==='chunk'?1:0);
}}
const inputs={
 shrink:{tokens:["TEXT","IMG","PAD","PAD","PAD","TEXT","TEXT"],maxBatchedTokens:4,cpAlign:1},
 grow:{tokens:["TEXT","IMG","PAD","PAD","PAD","TEXT","TEXT"],maxBatchedTokens:5,cpAlign:1},
 busy:{tokens:["TEXT","IMG","PAD","PAD","PAD","TEXT","TEXT"],cached:1,
  maxBatchedTokens:2,scheduledCount:1,cpAlign:1},
 aligned:{tokens:["TEXT","TEXT","TEXT","TEXT","TEXT","TEXT","TEXT"],maxBatchedTokens:8,cpAlign:2}
};
const asRank=value=>({waitingSchedulable:value!=="no",canActuallySchedule:value==="yes",cacheUsage:.2});
function vote(){
 try{
  const voteInput={engine:"NORMAL",decodeStep:Number($("decodeStep").value),
   allInterval:Number($("allInterval").value),anyInterval:Number($("anyInterval").value),
   ranks:[asRank($("rank0").value),asRank($("rank1").value)]};
  const result=P.vote(voteInput);
  $("votePoint").textContent=result.point;
  $("voteStep").textContent=result.step==="prompt"?"Prompt":"Decode";
  $("voteCount").textContent=result.scheduledCount;
  $("voteDetail").textContent=[
   "Rank wants: "+JSON.stringify(result.wants),
   "实际调度: "+JSON.stringify(result.actuallyScheduled),
   "每 Rank 模式: "+result.perRank.join(" / "),
   "第二轮回退: "+(result.secondVoteFallback?"是":"否"),
   "下一次 decodeStep: "+result.nextDecodeStep].join("\n");
  window.SeetheTruthPrefillLast={...(window.SeetheTruthPrefillLast||{}),vote:result};
  evidence.voteInput=voteInput;evidence.voteResult=result;sync('vote');
 }catch(err){$("voteDetail").textContent="投票输入错误："+err.message}
}
function renderTokens(tokens,cached,q){
 const nodes=tokens.map((v,i)=>{
  const div=document.createElement("div");
  div.className="token "+(v==="IMG"?"img":v==="PAD"?"pad":"")+
   (i>=cached&&i<cached+q?" selected":"");
  div.textContent=i+" "+v;div.title="绝对位置 "+i;return div;
 });
 $("tokens").replaceChildren(...nodes);
}
function chunk(){
 const sample=inputs[$("scenario").value];
 if(!sample)return;
 try{
  const result=P.chunk(sample);
  $("chunkQ").textContent=String(result.q);
  $("chunkRemaining").textContent=String(result.remaining);
  $("chunkStatus").textContent=result.outcome==="scheduled"?result.status:"DEFER";
  $("chunkDetail").textContent=[
   "输入: "+JSON.stringify(sample),
   "调度结果: "+JSON.stringify(result),
   "蓝绿色轮廓是实际本次计算的 Q 范围",
   "不将逻辑 Token 数推断为 GPU 运行耗时"].join("\n");
  renderTokens(sample.tokens,sample.cached||0,result.q);
  window.SeetheTruthPrefillLast={...(window.SeetheTruthPrefillLast||{}),chunk:result};
  evidence.chunkInput={...sample};evidence.chunkResult=result;sync('chunk');
 }catch(err){$("chunkDetail").textContent="Chunk 输入被拒绝："+err.message}
}
$("evalVote").addEventListener("click",vote);
$("decodeStep").addEventListener("change",vote);
$("rank0").addEventListener("change",vote);
$("rank1").addEventListener("change",vote);
$("evalChunk").addEventListener("click",chunk);
$("scenario").addEventListener("change",chunk);
vote();chunk();
})();