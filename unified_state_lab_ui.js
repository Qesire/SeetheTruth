/* A shared read-only event microscope for independently executed state engines.
 * The source application keeps its own controls. These examples use the exact
 * same execution adapter as the curriculum and do not invent a tensor DAG.
 */
(function(root){"use strict";
 const lab=root.SeetheTruthSpecializedLesson;
 const UI=root.SeetheTruthStateMicroscopeUI;
 if(!lab||!UI)throw Error("shared event microscope dependencies missing");
 const worker=root.SeetheTruthDeepInferEvents;
 const scheduler=root.SeetheTruthSchedulerWorkflow;
 const fixtures={
  deepinfer_prefill:()=>({operation:"inspect_pair",input:{
   voteInput:{decodeStep:4,allInterval:2,anyInterval:4,ranks:[
    {waitingSchedulable:true,canActuallySchedule:true,cacheUsage:.2},
    {waitingSchedulable:false,canActuallySchedule:false,cacheUsage:.2}]},
   chunkInput:{tokens:["TEXT","TEXT","TEXT"],cached:0,maxBatchedTokens:2}}}),
  deepinfer_worker:()=>({operation:"simulate",input:{
   steps:worker.DEFAULT_STEPS.slice(0,2),options:{zeroOverhead:true}}}),
  deepinfer_kv:()=>({operation:"replay",input:{
   config:{numBlocks:7,blockSize:4},
   operations:[{op:"arrive",id:"A",tokenLen:8},
    {op:"allocate",id:"A"},{op:"arrive",id:"B",tokenLen:4},
    {op:"allocate",id:"B"},{op:"finish",id:"A"},{op:"finish",id:"B"}]}}),
  deepinfer_scheduler:()=>({operation:"simulate",input:{
   spec:scheduler.EXAMPLES.ample,options:{}}})
 };
 if(!fixtures[lab.id])throw Error("no shared microscope fixture for "+lab.id);
 const host=document.createElement("section");
 host.id="unifiedStateMicroscope";
 host.style.cssText="margin:24px auto 36px;max-width:1180px;padding:18px;border:1px solid #456379;border-radius:14px;background:#0d2435;color:#e5f6ff;";
 const title=document.createElement("h2");
 title.textContent="统一状态显微镜 / SOURCE-VALIDATED TRANSITIONS";
 const hint=document.createElement("p");
 hint.textContent="用同一份受检验的状态追踪格式查看本专项计算的每一步。时序是教学假设值；KV 状态是 CPU 参考运算；不推断生产环境性能。";
 const button=document.createElement("button");
 button.id="runUnifiedState";
 button.type="button";
 button.textContent="运行并检查专项状态变化";
 button.style.cssText="padding:10px 13px;border-radius:8px;border:1px solid #63b9b3;background:#184b59;color:white;cursor:pointer;";
 const output=document.createElement("p");
 output.id="unifiedStateStatus";
 output.setAttribute("aria-live","polite");
 const canvas=document.createElement("div");
 canvas.id="unifiedStateSteps";
 host.append(title,hint,button,output,canvas);
 document.body.append(host);
 const view=UI.mount(canvas);
 function run(){
  try{
   const data=fixtures[lab.id]();
   const result=lab.execute(data.operation,data.input);
   if(!result.stateTrace)throw Error("domain adapter returned no verified state trace");
   view.show(result.stateTrace);
   output.textContent="已核验 "+result.stateTrace.steps.length+" 个状态/事件 · "+result.stateTrace.evidenceTier;
   return result.stateTrace;
  }catch(error){
   view.clear();
   output.textContent="专项计算或状态校验失败："+error.message;
   throw error;
  }
 }
 button.addEventListener("click",()=>{try{run()}catch(_){}});
 root.SeetheTruthUnifiedStateLab=Object.freeze({run,get frame(){return view.frame;}});
})(typeof window!=="undefined"?window:{});
