/* Native Bigram microbatch explorer: a separate explicit teaching session
 * driven by user actions. Shares exact TrainingPipeline arithmetic and
 * UnifiedCore observation. Does not claim to mirror the six-lesson controller.
 */
(function(root){"use strict";
 const core=root.SeetheTruthUnifiedCore,P=root.SeetheTruthTrainingPipeline,
  UI=root.SeetheTruthUnifiedObservationUI,snapshots=root.SeetheTruthObservationSnapshots;
 if(!core||!P||!UI||!snapshots)throw Error("training timeline protocol unavailable");
 const host=document.createElement("section");host.id="nativeTrainingTimeline";
 host.style.cssText="max-width:1180px;margin:24px auto;padding:18px;border:1px solid #47758b;border-radius:14px;background:#112b3b;color:#e8f6ff";
 const h=document.createElement("h2");h.textContent="训练状态逐步恢复 · 原生 Microbatch / Optimizer";
 const explanation=document.createElement("p");
 explanation.textContent="这是独立的、用户逐步操作的 Bigram 教学会话，与上方六关训练任务分别记录状态；两个视图调用同一 CPU 参考引擎。每步检查点包括参数、动量、Sampler 与未完成的梯度累积。";
 const controls=document.createElement("div");controls.style.cssText="display:flex;gap:10px;flex-wrap:wrap";
 const button=(id,label)=>{
  const el=document.createElement("button");el.id=id;el.type="button";el.textContent=label;
  el.style.cssText="padding:9px 12px;border-radius:8px;background:#224c60;color:white;border:1px solid #6c9bad";
  controls.append(el);return el;
 };
 const restart=button("trainingTimelineStart","从当前参数开始新训练");
 const micro=button("trainingTimelineMicro","执行 Microbatch");
 const optimizer=button("trainingTimelineOptimizer","执行 Optimizer.step()");
 const status=document.createElement("p");status.id="trainingTimelineStatus";status.setAttribute("aria-live","polite");
 host.append(h,explanation,controls,status);document.body.append(host);
 let config=null,actions=[],accepted=null;
 const n=id=>Number(document.getElementById(id).value);
 function requestedConfig(){
  return {seed:n("modelSeed"),samplerSeed:n("samplerSeed"),lr:n("lr"),
   microBatchSize:n("microBatchSize"),accumulationSteps:n("accumulationSteps"),
   maxGradNorm:n("maxGradNorm")};
 }
 function redraw(){
  const last=accepted?.value?.snapshots?.at(-1),pending=last?.pending?.microSteps??0;
  micro.disabled=!config||actions.length>=24||pending>=config.accumulationSteps;
  optimizer.disabled=!config||actions.length>=24||pending!==config.accumulationSteps;
  status.textContent=accepted
   ?"已执行 "+actions.length+" 步 · optimizer.step="+last.model.step+
    " · 待累积微批="+pending+"/"+config.accumulationSteps+
    " · 原生模型/优化器检查点="+accepted.stateTrace.steps.length
   :config?"训练会话已初始化，请执行第一个 Microbatch。"
    :"尚未启动独立训练会话";
 }
 function start(){
  const chosen=requestedConfig();const probe=P.create(chosen);
  config=probe.config;actions=[];accepted=null;redraw();return config;
 }
 function push(action){
  if(!config)throw Error("尚未初始化训练会话");
  const next=actions.concat(action);
  const record=core.executeLab("engineering_pipeline","timeline",{config,actions:next});
  core.replay(record);
  accepted=record;actions=next;redraw();panel.refresh();
  return record;
 }
 function guard(fn){try{fn();}catch(error){status.textContent="源引擎拒绝："+error.message;}}
 const panel=UI.mount({core,snapshots,readAccepted:()=>accepted,
  title:"统一训练观测 · 原生权重 / 梯度 / 动量 / Sampler / 回放"});
 restart.addEventListener("click",()=>guard(start));
 micro.addEventListener("click",()=>guard(()=>push("micro")));
 optimizer.addEventListener("click",()=>guard(()=>push("optimizer")));
 redraw();
 root.SeetheTruthNativeTrainingTimeline=Object.freeze({
  start,push,get accepted(){return accepted;},get actions(){return actions.slice();},
  get config(){return config;},panel
 });
})(typeof window!=="undefined"?window:{});
