(function(){"use strict";const S=window.SeetheTruthSchedulerWorkflow;if(!S)throw Error("Scheduler workflow module missing");
const $=id=>document.getElementById(id),SVG="http://www.w3.org/2000/svg";
const scenarioForPolicy=name=>{
 const base=S.EXAMPLES[name];
 // A terminated preempted request must not be re-admitted by a later Prefill
 // fixture. Trim the pressure example into a valid terminal-only policy case.
 if(name==="pressure"&&$("stopPreempted").checked){
  const copy=JSON.parse(JSON.stringify(base));
  copy.cycles=[copy.cycles[0],
   {outputLens:{B:5,C:2},after:[{op:"finish",id:"C"}]}];
  return copy;
 }
 return base;
};
let spec=S.EXAMPLES.pressure,run=S.simulate(spec),index=0;
const htmlSafe=t=>String(t); // UI only uses textContent; no HTML interpolation
function el(t,attrs={},txt){const e=document.createElement(t);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));if(txt!==undefined)e.textContent=htmlSafe(txt);return e}
function svg(t,attrs={},txt){const e=document.createElementNS(SVG,t);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));if(txt!==undefined)e.textContent=htmlSafe(txt);return e}
function paint(){
 const state=run.snapshots[index],decision=index?run.decisions[index-1]:null;
 const scheduled=new Set(decision?.scheduled||[]);
 $("stepNum").textContent=index+"/"+run.cycles.length;
 $("preemptNum").textContent=run.decisions.slice(0,index).reduce((a,d)=>a+d.preempted.length,0);
 $("recomputeNum").textContent=run.decisions.slice(0,index).reduce((a,d)=>a+d.preempted.filter(p=>p.outcome==="recompute").length,0);
 $("freeNum").textContent=state.free.length;
 $("fifo").textContent="BlockManager FIFO 空闲顺序："+JSON.stringify(state.free)+"；块 0 保留，不允许分配";
 const cells=[];
 for(let i=0;i<state.numBlocks;i++){
  const refs=state.refs[i],own=Object.entries(state.requests).filter(([_,r])=>r.blocks.includes(i)).map(([name])=>name);
  const cell=el("div",{class:"block "+(i===0?"sentinel":refs===0?"free":refs>1?"shared":"used")});
  cell.append(el("strong",{},i),el("small",{},i===0?"SENTINEL":"refs="+refs),el("small",{},own.join(",")||"free"));cells.push(cell);
 }
 $("blocks").replaceChildren(...cells);
 const rows=[];
 for(const [id,r] of Object.entries(state.requests)){
  const tr=el("tr");
  for(const v of [id,r.status,r.tokenLen,JSON.stringify(r.blocks),r.recomputes,scheduled.has(id)?"YES":"—"])tr.append(el("td",{},v));
  rows.push(tr);
 }
 $("requestRows").replaceChildren(...rows);
 const table=[];
 run.decisions.forEach((d,j)=>{
  const tr=el("tr");for(const txt of [j,d.scheduled.join(", ")||"placeholder",
    d.preempted.map(p=>p.id+":"+p.outcome).join(" · ")||"无",
    d.waiting.join(", ")||"无",d.freeAfter.length])tr.append(el("td",{},txt));
  if(j===index-1)tr.style.background="#28504b";
  table.push(tr);
 });$("decisionRows").replaceChildren(...table);
 const trace=run.worker.events.filter(e=>e.step<Math.max(index,1)),n=run.worker.makespan||1;
 const w=1080,l=85,r=20,graph=svg("svg",{viewBox:"0 0 "+w+" 145",role:"img",
  "aria-label":"Worker CPU/GPU order derived from real KV scheduler decisions"});
 for(const [lane,y] of [["CPU",52],["GPU",97]]){
  graph.append(svg("text",{x:8,y},lane));graph.append(svg("line",{x1:l,y1:y+8,x2:w-r,y2:y+8,stroke:"#52708a"}));
 }
 for(const evt of trace){
  const y=evt.resource==="CPU"?33:78,x=l+(evt.start/n)*(w-l-r),ew=Math.max(2,(evt.duration/n)*(w-l-r));
  const rect=svg("rect",{x,y,width:ew,height:25,rx:4,fill:evt.resource==="CPU"?"#3487a6":"#736bc0",
   stroke:"#abcfe0","stroke-opacity":".4",tabindex:0,role:"button",
   "aria-label":evt.id+" at "+evt.start+".."+evt.end});
  rect.append(svg("title",{},evt.stage+" "+evt.id+" dependencies "+evt.dependsOn.join(", ")));
  rect.addEventListener("click",()=>{$("detail").textContent=JSON.stringify({decision,event:evt,requestStatus:state.requests},null,2)});
  graph.append(rect);
  if(ew>50)graph.append(svg("text",{x:x+4,y:y+17,"pointer-events":"none",fill:"#f6fbff"},evt.stage.slice(0,17)));
 }
 $("workerTimeline").replaceChildren(graph);
 $("detail").textContent=JSON.stringify(index?{
  cycle:index-1,scheduled:decision.scheduled,preempted:decision.preempted,
  blockedAdmissions:decision.blocked,waiting:decision.waiting,kvRevision:decision.kvRevision,
  actions:decision.actions,dispatchOwnershipAtForward:decision.dispatchBlocks,
  worker:run.worker.steps[index-1],workerForwardEvent:run.executionProof[index-1]?.forwardEvent
 }:{init:spec.setup,physicalKV:state.free,source:"scheduler.hpp::decode"},null,2);
 $("previous").disabled=index===0;$("next").disabled=index===run.cycles.length;
 window.SeetheTruthLastSchedulerState=state;
 window.SeetheTruthLastSchedulerRun=run;
}
const applyNew=(candidate)=>{
 const next=S.simulate(candidate,{stopPreempted:$("stopPreempted").checked});
 spec=candidate;run=next;index=0;paint();
};
$("scenario").addEventListener("change",()=>{
 try{applyNew(scenarioForPolicy($("scenario").value));$("editor").value=JSON.stringify(spec,null,2);
 $("status").textContent="已切换并完整验证资源与事件依赖";}
 catch(e){$("status").textContent="场景验证失败："+e.message}
});
$("stopPreempted").addEventListener("change",()=>{
 try{applyNew(scenarioForPolicy($("scenario").value))}
 catch(e){$("status").textContent="无法设置抢占模式："+e.message}
});
$("previous").addEventListener("click",()=>{index=Math.max(0,index-1);paint()});
$("next").addEventListener("click",()=>{index=Math.min(run.cycles.length,index+1);paint()});
$("apply").addEventListener("click",()=>{
 try{const candidate=JSON.parse($("editor").value);applyNew(candidate);
 $("status").textContent="验证通过（CPU KV 所有权 + Worker 因果依赖）"}
 catch(e){$("status").textContent="输入被拒绝，原有状态保留："+e.message}
});
$("download").addEventListener("click",()=>{
 const content=JSON.stringify(run,null,2),uri=URL.createObjectURL(new Blob([content],{type:"application/json"}));
 const a=el("a",{href:uri,download:"seethetruth-scheduler-kv-worker.json"});a.click();
 setTimeout(()=>URL.revokeObjectURL(uri),500);
});
$("editor").value=JSON.stringify(spec,null,2);paint();
})();