/* Shared read-only event/state microscope of the ACTIVE validated lab program.
 * Never use independent fixed examples: the exact last successful controller
 * input AND result must agree with a fresh execution of the unified adapter.
 */
(function(root){"use strict";
 const lab=root.SeetheTruthSpecializedLesson;
 const UI=root.SeetheTruthStateMicroscopeUI;
 if(!lab||!UI)throw Error("shared event microscope dependencies missing");
 const host=document.createElement("section");
 host.id="unifiedStateMicroscope";
 host.style.cssText="margin:24px auto 36px;max-width:1180px;padding:18px;border:1px solid #456379;border-radius:14px;background:#0d2435;color:#e5f6ff;";
 const title=document.createElement("h2");
 title.textContent="统一状态显微镜 / ACTIVE SOURCE-VALIDATED TRACE";
 const hint=document.createElement("p");
 hint.textContent="重新执行当前专项页面最后一次成功验证的输入，对照原控制器实际结果，再展示事件与状态。非法编辑不改变已验证快照。时间单位不是 GPU 实测延迟。";
 const button=document.createElement("button");
 button.id="runUnifiedState";
 button.type="button";
 button.textContent="核对当前实验的状态变化";
 button.style.cssText="padding:10px 13px;border-radius:8px;border:1px solid #63b9b3;background:#184b59;color:white;cursor:pointer;";
 const output=document.createElement("p");
 output.id="unifiedStateStatus";
 output.setAttribute("aria-live","polite");
 const canvas=document.createElement("div");
 canvas.id="unifiedStateSteps";
 host.append(title,hint,button,output,canvas);
 document.body.append(host);
 const view=UI.mount(canvas);
 let acceptedSnapshot=null;
 const equivalent=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 function run(){
  try{
   const current=root.SeetheTruthCurrentDomainRun;
   if(!current||current.id!==lab.id||typeof current.operation!=="string"||
      !current.input||typeof current.input!=="object"||!current.observed)
    throw Error("当前页面尚无可核验的成功执行记录");
   const result=lab.execute(current.operation,current.input);
   if(!equivalent(result.value,current.observed))
    throw Error("页面执行记录与统一执行内核结果不一致");
   if(!result.stateTrace)throw Error("执行域未提供受校验的事件/状态追踪");
   const index=Number.isInteger(current.selectedIndex)?current.selectedIndex:0;
   view.show(result.stateTrace,index);
   acceptedSnapshot=Object.freeze({id:current.id,operation:current.operation,
    input:JSON.parse(JSON.stringify(current.input)),evidence:result.evidence});
   output.textContent="当前实验结果一致 · "+result.stateTrace.steps.length+
    " 个已核验状态/事件 · "+result.stateTrace.evidenceTier;
   return result.stateTrace;
  }catch(error){
   acceptedSnapshot=null;
   view.clear();
   output.textContent="专项记录无法核验："+error.message;
   throw error;
  }
 }
 button.addEventListener("click",()=>{try{run()}catch(_){}});
 // All DeepInfer state laboratories use the same observation UI and the
 // exact accepted controller snapshot; no per-page replica of this adapter.
 if(root.SeetheTruthUnifiedObservationUI){
  root.SeetheTruthObservationPanel=root.SeetheTruthUnifiedObservationUI.mount({
   core:root.SeetheTruthUnifiedCore,
   snapshots:root.SeetheTruthObservationSnapshots,
   onCheckpoint:restored=>{
    // Match the existing state microscope's native frame, without
    // changing the controller, live allocations or any external state.
    view.show(restored.record.stateTrace,restored.checkpoint.index);
    acceptedSnapshot=Object.freeze({id:restored.record.id,
     operation:restored.record.operation,
     input:JSON.parse(JSON.stringify(restored.record.inputSnapshot)),
     evidence:restored.record.evidence});
    output.textContent="快照原生重放验证 · 教学状态步骤 "+
     restored.checkpoint.stepId+" · 仅恢复只读视图";
   },
   readAccepted:()=>{
    const current=root.SeetheTruthCurrentDomainRun;
    if(!current||current.id!==lab.id||!current.input||!current.observed)
     throw Error("当前页面没有可核验的成功执行记录");
    const result=lab.execute(current.operation,current.input);
    if(JSON.stringify(result.value)!==JSON.stringify(current.observed))
     throw Error("当前页面结果与原生执行不一致");
    return result;
   },
   title:"统一状态观测 · "+lab.id+" / 事件 / 回放差异"
  });
 }

 // Source-derived questions for the active state/event engine. The guide reads
 // the same native microscope frame and verifies each selected event via core.
 const lessonHost=document.createElement("section");
 lessonHost.id="stateLearningGuide";
 lessonHost.setAttribute("aria-label","状态机制引导");
 lessonHost.style.cssText="margin:0;width:100%;box-sizing:border-box;max-width:1180px;padding:18px;border:1px solid #497388;border-radius:14px;background:linear-gradient(125deg,#113c4a,#10283b);color:#e4f6ff;display:grid;gap:9px;";
 const lead=document.createElement("div"),stepLabel=document.createElement("small"),
   question=document.createElement("h2"),instruction=document.createElement("p"),
   observation=document.createElement("p"),actions=document.createElement("div");
 stepLabel.id="stateGuideStep";question.id="stateGuideQuestion";
 instruction.id="stateGuideInstruction";observation.id="stateGuideEvidence";
 observation.setAttribute("aria-live","polite");
 observation.style.cssText="font-size:12px;color:#a5f2d5;overflow-wrap:anywhere;";
 actions.style.cssText="display:flex;gap:8px;flex-wrap:wrap;align-items:center;";
 const go=document.createElement("button"),previous=document.createElement("button"),
   next=document.createElement("button");
 const guideButton=(button,id,value)=>{
   button.id=id;button.type="button";button.textContent=value;
   button.style.cssText="border:1px solid #669eaa;background:#16485a;color:#e8fbff;padding:9px 12px;border-radius:8px;cursor:pointer;";
   actions.append(button);
 };
 guideButton(previous,"stateGuideBack","上一步");
 guideButton(go,"stateGuideAction","核验当前运行");
 guideButton(next,"stateGuideNext","下一步");
 lead.append(stepLabel,question);lessonHost.append(lead,instruction,observation,actions);
 const labMain=document.querySelector("main");
 if(!labMain)throw Error("specialized learning guide requires main");
 labMain.insertBefore(lessonHost,labMain.children[1]||null);
 let guideRecord=null,guideFrame=null,guideKey=null,guideStages=[],guideIndex=0,
   focusedId=null,finished=false;
 const signature=current=>current?JSON.stringify({
   id:current.id,operation:current.operation,input:current.input,
   observed:current.observed
 }):null;
 const matches=()=>Boolean(guideKey&&
   signature(root.SeetheTruthCurrentDomainRun)===guideKey);
 const labels={
   deepinfer_kv:"KV BlockManager 的 FIFO、共享与引用计数如何保持一致？",
   deepinfer_scheduler:"请求何时拥有实际 KV 块并进入 Worker Forward？",
   deepinfer_worker:"CPU 和 GPU 事件为什么能够跨 Step 重叠？"
 };
 const cueFor=(entry,domain)=>{
  const changes=entry.changes||[],names=changes.map(x=>x.key);
  if(domain==="deepinfer_kv"){
   if(/fork_(pin|activate)/.test(entry.title))
    return "分支共享与尾块克隆如何改变物理块所有权？";
   if(names.includes("refs"))
    return "引用计数的变化如何对应当前请求持有的物理块？";
   if(names.includes("freeQueue"))
    return "FIFO 空闲块顺序如何决定接下来会分配哪个块？";
  }
  if(domain==="deepinfer_scheduler"){
   if(entry.after?.preempted?.length)
    return "这轮内存压力触发了哪些抢占，哪些请求仍被允许 Forward？";
   if(entry.after?.forwardEvent)
    return "调度批次和实际 Worker Forward 事件如何一一对应？";
  }
  if(domain==="deepinfer_worker"){
   if(entry.dependencies.length)
    return "为什么 "+entry.title+" 必须等待前面的事件完成？";
   if(entry.timing)
    return "这个 "+entry.timing.resource+" 事件占据了怎样的假设资源时间段？";
  }
  return "观察「"+entry.title+"」：哪些状态改变了，改变依赖什么？";
 };
 function chooseStages(frame){
  const events=frame.steps.filter(x=>x.changes.length);
  const available=events.length?events:frame.steps;
  // Selection is by available native states/events. A curated focus may move
  // within the executed frame, but can never fabricate a missing event.
  const selected=[],include=item=>{if(item&&!selected.includes(item))selected.push(item);};
  include(available[0]);
  if(lab.id==="deepinfer_kv")
   include(available.find(x=>/fork_(pin|activate)|preempt/.test(x.title)));
  else if(lab.id==="deepinfer_scheduler")
   include(available.find(x=>x.after?.preempted?.length));
  else if(lab.id==="deepinfer_worker")
   include(available.find(x=>x.dependencies.length>0&&x.timing));
  include(available[available.length-1]);
  return selected.slice(0,3).map(entry=>Object.freeze({
   id:entry.id,index:entry.index,title:entry.title,
   question:cueFor(entry,lab.id)
  }));
 }
 function activePoint(){return guideIndex?guideStages[guideIndex-1]:null;}
 function verified(){
  if(!matches()||!guideRecord||!guideFrame)return false;
  if(guideIndex===0)return true;
  const point=activePoint();
  return Boolean(point&&focusedId===point.id&&view.frame===guideFrame&&
   view.selected===point.index);
 }
 function invalidated(){
  const value=root.SeetheTruthCurrentDomainRun;
  if(!value||!matches()){
   guideRecord=null;guideFrame=null;guideKey=null;guideStages=[];guideIndex=0;focusedId=null;finished=false;
   view.clear();
   acceptedSnapshot=null;
   output.textContent="当前场景已改变；核验后可查看新的状态轨迹。";
  }
 }
 function paintGuide(){
  if(guideKey&&!matches())invalidated();
  const point=activePoint(),count=guideStages.length+1,complete=verified();
  stepLabel.textContent=(guideIndex+1)+" / "+count+" · "+lab.id;
  question.textContent=point?point.question:labels[lab.id]||"当前状态是如何由操作产生的？";
  instruction.textContent=point?
   "定位「"+point.title+"」，读取本次执行中的 BEFORE → AFTER 和因果依赖。":
   "当前页面已经计算了场景结果。点击「核验当前运行」，用统一内核重新执行并核对原始结果。";
  observation.textContent=!matches()?
   "当前场景尚未建立可回放证据；引导从本次输入重新开始。":
   point?complete?
    "已核验 "+point.id+" · "+guideFrame.steps[point.index].changes.length+
     " 个状态字段变化 · 来源 "+guideFrame.evidenceTier:
    "观测目标：定位 "+point.id+"，查看现有状态显微镜的变化明细。":
   "已复核 "+guideFrame.steps.length+" 个原生事件/状态 · "+
     guideFrame.evidenceTier+"；可继续逐项追踪。";
  go.textContent=point?"定位并核验 "+point.id:"核验当前运行";
  previous.disabled=guideIndex===0;
  next.disabled=!complete||finished;
  next.textContent=guideIndex>=guideStages.length?"完成本次探究":"下一步";
  go.disabled=false;
 }
 function start(){
  const current=root.SeetheTruthCurrentDomainRun;
  const frame=run(); // Native result parity; throws on mismatch or missing state trace.
  const record=lab.execute(current.operation,current.input);
  lab.replay(record);
  const stages=chooseStages(frame);
  guideRecord=record;guideFrame=frame;guideKey=signature(current);
  guideStages=stages;guideIndex=0;focusedId=null;finished=false;
  paintGuide();
 }
 function locate(){
  if(!matches()){invalidated();paintGuide();return;}
  const point=activePoint();
  if(!point)return;
  const proof=lab.inspectStep(guideRecord,{
   kind:"state_transition",stepId:point.id
  }); // Full replay happens before the source observation is accepted.
  if(proof.stepId!==point.id||!proof.replayVerified)
   throw Error("当前状态步骤无法由统一记录重放");
  view.show(guideFrame,point.index);
  focusedId=point.id;
  paintGuide();
 }
 go.addEventListener("click",()=>{
  try{if(guideIndex===0)start();else locate()}
  catch(error){observation.textContent="本次核验未完成："+error.message;invalidated();paintGuide();}
 });
 previous.addEventListener("click",()=>{if(guideIndex>0){guideIndex--;focusedId=null;finished=false;paintGuide();}});
 next.addEventListener("click",()=>{
  paintGuide();
  if(!verified())return;
  if(guideIndex<guideStages.length){guideIndex++;focusedId=null;paintGuide();}
  else{
   finished=true;
   paintGuide();
   observation.textContent="本次因果观察已完成；修改场景后可重新核验并比较状态变化。";
  }
 });
 const refreshAfterPageAction=()=>{
  // Native page controls dispatch synchronous user events. Refresh after
  // controller execution; invalid edits retain their previous valid record.
  queueMicrotask(paintGuide);
 };
 document.addEventListener("change",refreshAfterPageAction);
 document.addEventListener("input",event=>{
  if(event.target?.closest("#scenario,#stopPreempted,#forwardTime,#postTime,#flush"))
   refreshAfterPageAction();
 });
 document.addEventListener("click",event=>{
  if(event.target?.closest("#apply,#applySteps,#applyMetadata,#applyDecode,#reset,#next,#prev,#previous"))
   refreshAfterPageAction();
 });
 root.SeetheTruthStateLearningGuide=Object.freeze({
  get id(){return lab.id;},
  get step(){return guideIndex;},
  get count(){return guideStages.length;},
  get verified(){return verified();},
  get evidence(){return guideFrame?.evidenceTier||null;},
  get current(){return matches();},
  get selectedEvent(){return activePoint()?.id||null;}
 });
 paintGuide();

 root.SeetheTruthUnifiedStateLab=Object.freeze({
  run,
  get frame(){return view.frame;},
  get accepted(){return acceptedSnapshot;}
 });
})(typeof window!=="undefined"?window:{});