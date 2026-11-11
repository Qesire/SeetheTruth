(function(){"use strict";const C=window.SeetheTruthCurriculum,Core=window.SeetheTruthUnifiedCore;
if(!C||!Core)throw Error("shared teaching core unavailable");
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);
const teachingMode=params.get("edition")==="teaching"||location.hostname==="appassets.androidplatform.net";
$("editionLabel").textContent=teachingMode?"离线教学版 · 不生成模型产物":"跨平台教学版 · PC / Android";
$("version").textContent="课程版本 "+C.VERSION;
const labNames=Object.freeze({
 source_atlas:"模型三维图谱",engineering_journey:"工程入门",engineering_pipeline:"训练状态流水线",
 tradeoffs:"架构成本权衡",paper:"论文机制变换",training:"训练回路",micro_training:"微型训练",
 cache:"KV Cache 时间线",semantic_state:"具名张量轴与状态",wan_ffn:"Wan FFN 源码显微镜",
 deepinfer_worker:"Worker 时序",deepinfer_kv:"KV Block 生命周期",
 deepinfer_scheduler:"Scheduler / KV / Worker",deepinfer_prefill:"Prefill 与多模态 Chunk",
 real_model:"真实模型证据",capabilities:"积木能力矩阵",telemetry:"张量与性能统计"
});
function labCategory(lab){
 if(["source_atlas","tradeoffs","paper","capabilities","semantic_state","wan_ffn"].includes(lab.id))return "架构与计算";
 if(["engineering_journey","engineering_pipeline","training","micro_training"].includes(lab.id))return "训练与工程";
 if(["cache","deepinfer_worker","deepinfer_kv","deepinfer_scheduler","deepinfer_prefill"].includes(lab.id))return "推理与缓存";
 return "来源与运行证据";
}
const labHome=$("allLabLinks"),groups=new Map();
const offlineExcluded=new Set(["index.html","real_model_lab.html","telemetry.html"]);
for(const lab of Core.labs){
 if(teachingMode&&offlineExcluded.has(lab.href))continue;
 const label=labCategory(lab);
 if(!groups.has(label))groups.set(label,[]);
 groups.get(label).push(lab);
}
for(const [group,labs] of groups){
 const heading=document.createElement("div");heading.className="lab-group-title";
 heading.textContent=group;labHome.append(heading);
 for(const lab of labs){
  const link=document.createElement("a");link.href=lab.href;
  link.textContent=labNames[lab.id]||lab.domain.replaceAll("_"," ");
  link.dataset.lab=lab.id;labHome.append(link);
 }
}
const courses=Core.inventory().courses;
$("lessonCount").textContent=courses.length+" 个专题";
const coverage=Core.inventory().counts;
$("coreCoverage").textContent=coverage.labs+" 个专项计算域";
// A course deep link resolves through the canonical shared curriculum.
const requested=params.get("course");
let selected=(requested&&C.lookup(requested)?requested:courses[0].id),filter="all";
const fundamentals=new Set(["building","papers","multimodal","vlm"]);
const safeTarget=l=>{
 const u=new URL(l.href,location.href);
 if(u.pathname.endsWith("/learn.html")&&l.target?.kind==="guided_graph")u.searchParams.set("mission",l.target.id);
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
 $("openLesson").textContent=l.target?.kind==="guided_graph"?"打开积木实验":"进入交互实验";
 $("vlmNavLink").setAttribute("aria-current",selected==="vlm"?"page":"false");
}
function render(){
 const query=$("search").value.toLocaleLowerCase().trim();
 const lessons=courses.filter(l=>{
  if(filter==="fundamentals"&&!fundamentals.has(l.id))return false;
  if(filter==="inference"&&(fundamentals.has(l.id)||l.id.startsWith("engineering_")))return false;
  if(filter==="engineering"&&!l.id.startsWith("engineering_"))return false;
  return [l.title,l.theory,l.why,l.group].join(" ").toLocaleLowerCase().includes(query);
 });
 if(lessons.length&&!lessons.some(l=>l.id===selected)){selected=lessons[0].id;detail();}
 const cards=lessons.map(l=>{
  const b=document.createElement("button");b.className="card";b.type="button";
  b.dataset.lesson=l.id;b.setAttribute("aria-selected",String(l.id===selected));
  b.setAttribute("aria-pressed",String(l.id===selected));
  const group=document.createElement("div");group.className="topic";group.textContent=l.group;
  const h=document.createElement("h3");h.textContent=l.title;
  const p=document.createElement("p");p.textContent=l.theory;
  b.append(group,h,p);
  b.addEventListener("click",()=>{
   selected=l.id;detail();render();
   const focused=[...document.querySelectorAll("#cards .card")].find(card=>card.dataset.lesson===l.id);
   focused?.focus({preventScroll:true});
   if(window.matchMedia("(max-width: 880px)").matches)$("deepDive").scrollIntoView({behavior:"smooth",block:"start"});
  });
  return b;
 });
 $("cards").replaceChildren(...cards);$("empty").hidden=!!cards.length;
}
for(const name of ["all","fundamentals","inference","engineering"]){
 $(name).addEventListener("click",()=>{filter=name;
  for(const n of ["all","fundamentals","inference","engineering"])$(n).setAttribute("aria-pressed",String(n===filter));
  render();
 });
}
$("search").addEventListener("input",render);
$("reveal").addEventListener("click",()=>{
 const willReveal=$("answer").hidden;$("answer").hidden=!willReveal;
 $("reveal").textContent=willReveal?"收起解析":"显示解析";
 $("reveal").setAttribute("aria-expanded",String(willReveal));
});
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