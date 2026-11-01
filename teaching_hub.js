(function(){"use strict";const C=window.SeetheTruthCurriculum,Core=window.SeetheTruthUnifiedCore;
if(!C||!Core)throw Error("shared teaching core unavailable");
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);
const teachingMode=params.get("edition")==="teaching"||location.hostname==="appassets.androidplatform.net";
$("editionLabel").textContent=teachingMode?"离线教学版 · 不生成模型产物":"跨平台教学版 · PC / Android";
$("version").textContent="课程版本 "+C.VERSION;
const courses=Core.inventory().courses;
$("lessonCount").textContent=courses.length+" 个专题";
const coverage=Core.inventory().counts;
$("coreCoverage").textContent="统一教学内核 · "+coverage.courses+" 门主题 / "+coverage.labs+" 个专项计算域";
let selected=C.lookup(params.get("focus"))?params.get("focus"):courses[0].id,filter="all";
const safeTarget=l=>{
 const u=new URL(l.href,location.href);
 if(teachingMode&&u.pathname.endsWith("/learn.html"))u.searchParams.set("edition","teaching");
 return u.pathname.split("/").at(-1)+u.search+u.hash;
};

/* Same curriculum identity, another projection: architecture / infrastructure
 * questions lead to existing real labs; atlas navigation never executes a lab. */
const Architecture=window.SeetheTruthArchitecturePerspective;
if(!Architecture)throw Error("shared architecture perspective unavailable");
const architecturePaths=Architecture.paths(C);
let architectureTrack=architecturePaths.find(p=>p.id===params.get("track"))||architecturePaths[0];
let architectureIndex=0;
const architectureEvidenceLabels={
 DESIGN_ONLY:"设计问题 · 尚未执行",
 ANALYTICAL_ONLY:"解析成本模型 · 非硬件测速",
 SOURCE_PINNED_SYNTHETIC:"源码约束 + 合成数值 · 非真实视频",
 SYNTHETIC_NUMERIC:"有界 CPU 数值 / 状态模拟",
 SCRIPTED_STATE_MACHINE:"确定性状态机 · 无真实工具调用",
 STATIC_GRAPH:"静态图结构与可组合算子",
 TOY_ANALYTIC:"解析目标函数 · 无模型参数更新",
 PROPOSED_REAL_EVALUATION:"真实评测尚待完成"
};
/* Granularity is orthogonal to architecture and the evidence plane.
 * Every link reuses a canonical course, specialist lab, or registered operator. */
const Grain=window.SeetheTruthGranularityLens;
const courseGrains=window.SeetheTruthGranularityLens.curriculumIndex(C);
const grainByCourse=new Map(courseGrains.map(x=>[x.id,x]));
const BlockCatalog=window.SeetheTruthBlockCatalog;
if(!Grain||!BlockCatalog)throw Error("shared granularity source/registry unavailable");
let grainId="system";
function renderGranularity(){
 const records=Grain.inspectTrack(architectureTrack.id,C,Core,{operators:BlockCatalog});
 const nav=Grain.navigation(architectureTrack.id,grainId,C,Core,{operators:BlockCatalog});
 $("granularitySteps").replaceChildren(...records.map(record=>{
  const button=document.createElement("button");button.type="button";
  button.dataset.grain=record.grain;button.dataset.source=record.sourceId;
  button.setAttribute("aria-selected",String(record.grain===grainId));
  const code=document.createElement("span");code.className="grain-code";code.textContent=record.code;
  const name=document.createElement("strong");name.textContent=Grain.GRAINS.find(g=>g.id===record.grain).name;
  const objective=document.createElement("small");objective.textContent=record.title;
  button.append(code,name,objective);
  button.addEventListener("click",()=>{grainId=record.grain;renderGranularity();});
  return button;
 }));
 const item=nav.current;
 $("granularityLabel").textContent=item.code+" · "+item.targetKind.toUpperCase()+" · "+item.evidence;
 $("granularityTitle").textContent=item.title;
 $("granularityObjective").textContent=item.objective;
 $("granularityIdentity").textContent="真实来源："+item.sourceId+" · 展开关系："+item.relation;
 $("granularityProof").textContent="父级覆盖："+item.containment+" · 数值等价："+nav.numericallyEquivalent;
 $("granularityLaunch").href=safeTarget({href:item.href});
 $("granularityLaunch").textContent="打开 "+Grain.GRAINS.find(g=>g.id===grainId).name+"对象";
}

function renderArchitecture(){
 $("architectureTracks").replaceChildren(...architecturePaths.map(path=>{
  const b=document.createElement("button");b.type="button";b.textContent=path.name;
  b.dataset.track=path.id;b.setAttribute("aria-pressed",String(architectureTrack.id===path.id));
  b.addEventListener("click",()=>{architectureTrack=path;architectureIndex=0;renderArchitecture();renderGranularity();});
  return b;
 }));
 $("architectureGoal").textContent="SYSTEM GOAL · "+architectureTrack.goal;
 $("architectureMetrics").textContent="关键验收："+architectureTrack.metrics;
 $("architectureSteps").replaceChildren(...architectureTrack.steps.map((step,i)=>{
  const b=document.createElement("button");b.type="button";b.className="arch-step";
  b.dataset.layer=step.level;b.dataset.course=step.course;
  b.setAttribute("aria-selected",String(i===architectureIndex));
  const level=Architecture.LEVELS.find(l=>l.id===step.level);
  const head=document.createElement("span");head.className="arch-id";head.textContent=level.en;
  const strong=document.createElement("strong");strong.textContent=level.name;
  const small=document.createElement("small");small.textContent=step.question;
  b.append(head,strong,small);
  b.addEventListener("click",()=>{architectureIndex=i;renderArchitecture();});
  return b;
 }));
 const selectedStep=architectureTrack.steps[architectureIndex];
 const layer=Architecture.LEVELS.find(l=>l.id===selectedStep.level);
 $("architectureSelectedLevel").textContent=layer.en+" / "+layer.name;
 $("architectureQuestion").textContent=selectedStep.question;
 $("architectureArtifact").textContent="探索对象："+selectedStep.artifact;
 $("architectureEvidence").textContent="当前证据："+(architectureEvidenceLabels[selectedStep.evidence]||selectedStep.evidence);
 $("architectureLaunch").href=safeTarget({href:selectedStep.href});
 $("architectureLaunch").textContent="打开 "+C.lookup(selectedStep.course).title;
}

function detail(){
 const l=Core.inspectCourse(selected);if(!l)return;
 $("detailTitle").textContent=l.title;$("level").textContent=l.level;
 $("theory").textContent=l.theory;$("why").textContent=l.why;
 $("question").textContent=l.check;$("answer").textContent=l.answer;
 $("answer").hidden=true;$("reveal").setAttribute("aria-expanded","false");
 $("reveal").textContent="显示解析";
 $("openLesson").href=safeTarget(l);
}
function render(){
 const query=$("search").value.toLocaleLowerCase().trim();
 const courseGrain=$("courseGrain").value;
 const lessons=courses.filter((l,i)=>{
  if(courseGrain!=="all"&&grainByCourse.get(l.id)?.grain!==courseGrain)return false;
  if(filter==="fundamentals"&&i>=4)return false;
  if(filter==="inference"&&(i<4||l.id.startsWith("engineering_")||l.id.startsWith("research_")))return false;
  if(filter==="engineering"&&!l.id.startsWith("engineering_"))return false;
  if(filter==="research"&&!l.id.startsWith("research_"))return false;
  return [l.title,l.theory,l.why,l.group].join(" ").toLocaleLowerCase().includes(query);
 });
 const cards=lessons.map(l=>{
  const b=document.createElement("button");b.className="card";b.type="button";
  b.dataset.lesson=l.id;b.setAttribute("aria-selected",String(l.id===selected));
  const group=document.createElement("div");group.className="topic";group.textContent=l.group;
  const h=document.createElement("h3");h.textContent=l.title;
  const p=document.createElement("p");p.textContent=l.theory;
  const gran=document.createElement("span");gran.className="course-grain-meta";
  const grainInfo=grainByCourse.get(l.id);
  gran.textContent=grainInfo.code+" · "+Grain.GRAINS.find(g=>g.id===grainInfo.grain).name;
  const footer=document.createElement("span");footer.className="open";footer.textContent="架构权衡 / 机制 / 反例 →";
  b.append(group,gran,h,p,footer);
  b.addEventListener("click",()=>{
   selected=l.id;detail();render();
   if(window.matchMedia("(max-width: 880px)").matches)$("deepDive").scrollIntoView({behavior:"smooth",block:"start"});
  });
  return b;
 });
 $("cards").replaceChildren(...cards);$("empty").hidden=!!cards.length;
}
for(const name of ["all","fundamentals","inference","engineering","research"]){
 $(name).addEventListener("click",()=>{filter=name;
  for(const n of ["all","fundamentals","inference","engineering","research"])$(n).setAttribute("aria-pressed",String(n===filter));
  render();
 });
}
$("search").addEventListener("input",render);
$("courseGrain").addEventListener("change",render);
$("reveal").addEventListener("click",()=>{
 const willReveal=$("answer").hidden;$("answer").hidden=!willReveal;
 $("reveal").textContent=willReveal?"收起解析":"显示解析";
 $("reveal").setAttribute("aria-expanded",String(willReveal));
});

/* One compact, executable micro-lab per research track. The numerical sources
 * are SeetheTruthResearchTracks + existing WanFFNTrace, not UI mock output. */
const Research=window.SeetheTruthResearchTracks;
if(!Research)throw Error("research exercises unavailable");
const fmt=v=>Number(v).toFixed(5);
const lastResearch={};
function updateResearch(){
 const video=Research.videoSparsity({
  keep:Number($("vKeep").value),scope:$("vScope").value,
  selection:$("vMetric").value,seed:Number($("vSeed").value)});
 lastResearch.video=video;
 $("videoResult").textContent=[
  "Dense → Sparse · [Token, Channel] = "+video.shape.join(" × "),
  "保留: "+video.kept.map((s,t)=>"t"+t+"{"+s.join(",")+"}").join("  "),
  "‖Ydense − Ysparse‖₂ = "+fmt(video.errorL2),
  "相对 FFN gated delta 误差 = "+fmt(100*video.relativeDeltaError)+"%",
  "组间误差² / 单通道误差²和 = "+fmt(video.cancellationRatio),
  "理想 FFN MAC 比例 = "+fmt(100*video.idealMacFraction)+"%",
  video.costNote].join("\n");
 const shift=Number($("pShift").value),param=Number($("pParam").value);
 if($("pObjective").value==="dpo"){
  const post=Research.dpo({chosen:-1+shift,rejected:-1.4,refChosen:-1,refRejected:-1.4,beta:param});
  lastResearch.post=post;
  $("postResult").textContent=[
   "DPO: −log σ(β [(logπθ+−logπref+)−(logπθ−−logπref−)])",
   "log-ratio margin = "+fmt(post.margin),
   "β·margin = "+fmt(post.logit)+" · loss = "+fmt(post.loss),
   "∂L/∂logπchosen = "+fmt(post.gradChosen),
   "∂L/∂logπrejected = "+fmt(post.gradRejected),
   "冻结参考模型；这是 logπ 上的导数，不是参数梯度。"].join("\n");
 }else{
  const post=Research.grpo({rewardGap:param,ratioShift:shift,epsilon:0.2});
  lastResearch.post=post;
  $("postResult").textContent=[
   "GRPO: A_i = (r_i − mean(r)) / std(r)",
   "rewards = ["+post.rewards.map(fmt).join(", ")+"]",
   "advantages = ["+post.terms.map(t=>fmt(t.advantage)).join(", ")+"]",
   "ratios = ["+post.terms.map(t=>fmt(t.ratio)).join(", ")+"]",
   "clip active = ["+post.terms.map(t=>t.clipActive?"yes":"no").join(", ")+"]",
   "−mean(min(ratio·A, clip(ratio)·A)) = "+fmt(post.loss),
   "优势视为 stop-gradient；演示没有 KL、Mask 和参数优化。"].join("\n");
 }
 const agent=Research.agent({architecture:$("aArchitecture").value,
  fault:$("aFault").value==="true",approved:$("aApproved").value==="true"});
 lastResearch.agent=agent;
 $("agentResult").textContent="模拟任务状态："+agent.status+
  "\n阶段数："+agent.events.length+" · 已核验："+(agent.verified?"是":"否")+
  "\n架构："+agent.architecture+
  "\n最终事件："+agent.events.at(-1).action;
 $("agentTrace").replaceChildren(...agent.events.map(e=>{
  const li=document.createElement("li");
  li.textContent=e.id+" ["+e.actor+"] "+e.action+" · "+e.before+" → "+e.after+
    (e.ok?"":" [阻断/失败]")+" · "+e.detail;
  return li;
 }));
 window.SeetheTruthLastResearch=lastResearch;
}
for(const id of ["vKeep","vScope","vMetric","vSeed","pObjective","pParam","pShift",
 "aArchitecture","aFault","aApproved"])$(id).addEventListener("change",updateResearch);
updateResearch();

const V=window.SeetheTruthPrefillPolicy;
if(!V)throw Error("shared DeepInfer math module not available");
const rank=(waitingSchedulable,canActuallySchedule=true)=>({waitingSchedulable,canActuallySchedule,cacheUsage:0.2});
const voteCases={
 all:{step:2,ranks:[rank(true),rank(true)],
  explain:"ALL 需要两个 Rank 均有工作；避免某个 Rank 仅仅为了同步而陪跑空 Prompt。"},
 any:{step:4,ranks:[rank(true),rank(false)],
  explain:"ANY 允许一个 Rank 触发 Prefill，其他 Rank 使用 empty-prompt 保持整个 DP 组的步型一致。"},
 fallback:{step:4,ranks:[rank(true,false),rank(false)],
  explain:"首次投票即使通过，仍须检查实际是否排出 Prefill；全组都排不出时回退 Decode，防止空 Prompt 循环。"}
};
function runVote(){
 const chosen=voteCases[$("voteCase").value];
 const out=Core.executeLab("deepinfer_prefill","vote",{
  decodeStep:chosen.step,allInterval:2,anyInterval:4,ranks:chosen.ranks}).value;
 $("voteResult").textContent="投票点 "+out.point+" · 全组执行 "+out.step+
  " · 真正执行 Prefill 的 Rank "+out.scheduledCount+
  " · 逐 Rank ["+out.perRank.join(", ")+"]"+
  (out.secondVoteFallback?" · 第二轮触发回退":"");
 $("voteExplanation").textContent=chosen.explain;
 window.SeetheTruthLastSharedVote=out;
}
$("voteCase").addEventListener("change",runVote);
runVote();
detail();render();renderArchitecture();renderGranularity();
window.SeetheTruthGranularityExplorer={select(track,grain){
 const t=architecturePaths.find(p=>p.id===track);if(!t)throw Error("unknown research track");
 if(!Grain.GRAINS.some(g=>g.id===grain))throw Error("unknown granularity");
 architectureTrack=t;grainId=grain;renderArchitecture();renderGranularity();
},current(){return Grain.resolve(architectureTrack.id,grainId,C,Core,{operators:BlockCatalog});}};
window.SeetheTruthArchitectureAtlas={paths:architecturePaths,select(track,level){
 const next=architecturePaths.find(p=>p.id===track);if(!next)throw Error("unknown architecture track");
 const index=next.steps.findIndex(s=>s.level===level);if(index<0)throw Error("unknown architecture level");
 architectureTrack=next;architectureIndex=index;renderArchitecture();renderGranularity();
},current(){return {track:architectureTrack.id,level:architectureTrack.steps[architectureIndex].level};}};
window.SeetheTruthTeachingHub={version:C.VERSION,lessons:courses.length,coreSchema:Core.SCHEMA,select(id){
 if(!C.lookup(id))throw Error("unknown lesson");selected=id;detail();render();
}};
})();