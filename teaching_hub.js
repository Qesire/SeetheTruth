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
 return u.pathname.split("/").at(-1)+u.search;
};
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
 const lessons=courses.filter((l,i)=>{
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
  const footer=document.createElement("span");footer.className="open";footer.textContent="展开数学 / 设计依据 →";
  b.append(group,h,p,footer);
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
detail();render();
window.SeetheTruthTeachingHub={version:C.VERSION,lessons:courses.length,coreSchema:Core.SCHEMA,select(id){
 if(!C.lookup(id))throw Error("unknown lesson");selected=id;detail();render();
}};
})();