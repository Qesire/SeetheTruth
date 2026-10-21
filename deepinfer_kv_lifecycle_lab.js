(function(){"use strict";const M=window.SeetheTruthKVLifecycle;if(!M)throw Error("KV lifecycle engine unavailable");
const $=id=>document.getElementById(id);
const scenarios={
 fork:{config:{numBlocks:7,blockSize:4},ops:[
  {op:"arrive",id:"A",tokenLen:8},{op:"allocate",id:"A"},
  {op:"arrive",id:"B",tokenLen:4},{op:"allocate",id:"B"},
  {op:"fork_pin",id:"C",from:"A",cachedTokens:5,tokenLen:8},
  {op:"fork_activate",id:"C"},{op:"finish",id:"A"},{op:"finish",id:"C"},{op:"finish",id:"B"}]},
 preempt:{config:{numBlocks:5,blockSize:4},ops:[
  {op:"arrive",id:"X",tokenLen:8},{op:"allocate",id:"X"},
  {op:"update_workload",id:"X",tokenLen:12,inflight:0,cachedTokens:8},{op:"append_slot",id:"X"},
  {op:"preempt",id:"X"},{op:"allocate",id:"X"},
  {op:"set_structured",id:"X",status:1},{op:"preempt",id:"X",stop:false}]},
 pressure:{config:{numBlocks:5,blockSize:4},ops:[
  {op:"arrive",id:"A",tokenLen:8},{op:"allocate",id:"A"},
  {op:"arrive",id:"B",tokenLen:8},{op:"allocate",id:"B"},
  {op:"preempt",id:"A"},{op:"finish",id:"B"},{op:"allocate",id:"A"}]}
};
let config=scenarios.fork.config,ops=scenarios.fork.ops.slice(),result=M.replay(config,ops),index=0;
const State=window.SeetheTruthStateMicroscope;
const stateView=window.SeetheTruthStateMicroscopeUI.mount(document.getElementById('stateMicroscope'));
let lastTimeline=null,lastFrame=null;
const status=(message,warning=false)=>{$("status").textContent=message;$("status").style.color=warning?"#f4bd82":"";};
function draw(){
 const state=result.timeline[index],last=state.events.at(-1);
 $("revision").textContent=String(index)+"/"+ops.length;
 $("freeCount").textContent=state.free.length;
 $("usedCount").textContent=String(state.numBlocks-1-state.free.length);
 $("poolLabel").textContent="容量 "+state.numBlocks+" (含保留块 0) · BlockSize "+state.blockSize;
 const blocks=[];
 for(let id=0;id<state.numBlocks;id++){
  const owners=Object.entries(state.requests).filter(([_,r])=>r.blocks.includes(id)).map(([name])=>name);
  const el=document.createElement("div");el.className="cell "+(id===0?"reserved":!state.refs[id]?"free":state.refs[id]>1?"shared":"owned");
  const head=document.createElement("span");head.textContent=String(id);
  const refs=document.createElement("small");refs.textContent=id===0?"SENTINEL":"ref="+state.refs[id];
  const own=document.createElement("small");own.textContent=owners.join(", ")||"—";
  el.append(head,refs,own);el.title=owners.length?"Owners: "+owners.join(", "):"No owners";
  blocks.push(el);
 }
 $("blocks").replaceChildren(...blocks);
 $("freeOrder").textContent="FIFO Free Queue（下次从最左侧分配）：["+state.free.join(", ")+"]";
 const trs=[];
 for(const [id,s] of Object.entries(state.requests)){
  const tr=document.createElement("tr");
  for(const val of [id,s.status,String(s.tokenLen),String(s.cachedTokens),String(s.recomputes),"["+s.blocks.join(", ")+"]"]){
   const td=document.createElement("td");td.textContent=val;tr.append(td);
  }trs.push(tr);
 }
 $("requests").replaceChildren(...trs);
 $("copies").textContent=state.copies.length?JSON.stringify(state.copies,null,2):"当前尚未产生 Partial Tail Clone 操作";
 const entries=result.timeline.map((entry,i)=>{
  const btn=document.createElement("button");btn.type="button";
  btn.className="entry"+(i===index?" active":"");
  const last=entry.events.at(-1);btn.textContent=(i?"#"+i+" · "+last.op+" · "+(last.id??""):"#0 · 初始化 Block Pool");
  btn.addEventListener("click",()=>{index=i;draw()});return btn;
 });
 $("timeline").replaceChildren(...entries);
 $("detail").textContent=index?JSON.stringify({operation:ops[index-1],event:last,
  capacity:state.numBlocks,free:state.free,refs:state.refs,copyActions:state.copies},null,2):
  "初始状态：物理块 0 永不分配；空闲块 FIFO=[1,...,N-1]；所有引用计数初始为 0。";
 $("prev").disabled=index===0;$("next").disabled=index===ops.length;
 window.SeetheTruthLastKVState=state;
 if(lastTimeline!==result){lastFrame=State.kv(result,M);lastTimeline=result;}
 stateView.show(lastFrame,index);
}
function setScenario(name){
 const s=scenarios[name];config={...s.config};ops=s.ops.map(x=>({...x}));
 result=M.replay(config,ops);index=0;
 $("operations").value=JSON.stringify({config,ops},null,2);
 status("已加载且验证："+name+"（"+ops.length+" 步）");draw();
}
$("scenario").addEventListener("change",()=>setScenario($("scenario").value));
$("prev").addEventListener("click",()=>{index=Math.max(0,index-1);draw()});
$("next").addEventListener("click",()=>{index=Math.min(ops.length,index+1);draw()});
$("reset").addEventListener("click",()=>setScenario($("scenario").value));
$("apply").addEventListener("click",()=>{
 try{
  const edited=JSON.parse($("operations").value);
  const next=M.replay(edited.config,edited.ops);
  config=edited.config;ops=edited.ops;result=next;index=ops.length;
  status("全部状态与引用计数验证通过："+ops.length+" 步");draw();
 }catch(e){status("输入被拒绝（已保留有效状态）："+e.message,true)}
});
$("download").addEventListener("click",()=>{
 const payload=JSON.stringify({source:"DeepInfer BlockManager/Scheduler bounded CPU semantics",
  config,operations:ops,timeline:result.timeline,scope:"no physical GPU memory allocation"},null,2);
 const url=URL.createObjectURL(new Blob([payload],{type:"application/json"}));
 const a=document.createElement("a");a.href=url;a.download="seethetruth-deepinfer-kv-lifecycle.json";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),500);
});
setScenario("fork");
})();