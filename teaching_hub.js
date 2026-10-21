(function(){"use strict";const C=window.SeetheTruthCurriculum;if(!C)throw Error("shared teaching curriculum unavailable");
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);
const teachingMode=params.get("edition")==="teaching"||location.hostname==="appassets.androidplatform.net";
$("editionLabel").textContent=teachingMode?"离线教学版 · 不生成模型产物":"跨平台教学版 · PC / Android";
$("version").textContent="课程版本 "+C.VERSION;
$("lessonCount").textContent=C.lessons.length+" 个专题";
let selected=C.lessons[0].id,filter="all";
const safeTarget=l=>{
 const u=new URL(l.href,location.href);
 if(teachingMode&&u.pathname.endsWith("/learn.html"))u.searchParams.set("edition","teaching");
 return u.pathname.split("/").at(-1)+u.search;
};
function detail(){
 const l=C.lookup(selected);if(!l)return;
 $("detailTitle").textContent=l.title;$("level").textContent=l.level;
 $("theory").textContent=l.theory;$("why").textContent=l.why;
 $("question").textContent=l.check;$("answer").textContent=l.answer;
 $("answer").hidden=true;$("reveal").setAttribute("aria-expanded","false");
 $("reveal").textContent="显示解析";
 $("openLesson").href=safeTarget(l);
}
function render(){
 const query=$("search").value.toLocaleLowerCase().trim();
 const lessons=C.lessons.filter((l,i)=>{
  if(filter==="fundamentals"&&i>=4)return false;
  if(filter==="inference"&&i<4)return false;
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
for(const name of ["all","fundamentals","inference"]){
 $(name).addEventListener("click",()=>{filter=name;
  for(const n of ["all","fundamentals","inference"])$(n).setAttribute("aria-pressed",String(n===filter));
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
 const out=V.vote({decodeStep:chosen.step,allInterval:2,anyInterval:4,ranks:chosen.ranks});
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
window.SeetheTruthTeachingHub={version:C.VERSION,lessons:C.lessons.length,select(id){
 if(!C.lookup(id))throw Error("unknown lesson");selected=id;detail();render();
}};
})();