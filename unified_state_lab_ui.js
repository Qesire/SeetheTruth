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
 root.SeetheTruthUnifiedStateLab=Object.freeze({
  run,
  get frame(){return view.frame;},
  get accepted(){return acceptedSnapshot;}
 });
})(typeof window!=="undefined"?window:{});