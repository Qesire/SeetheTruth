(function(){"use strict";
const A=window.SeetheTruthSpecializedLesson.execute("audit",{}).value,
 M=window.SeetheTruthCapabilities.matrix(),$=id=>document.getElementById(id);
for(const [k,v] of Object.entries(A.counts))if($(k))$(k).textContent=v;
for(const category of [...new Set(A.rows.map(r=>r.category))].sort()){
 const el=document.createElement("option");el.value=category;el.textContent=category;$("category").append(el);
}
// Presentation taxonomy only. Every count is derived from currently installed
// source operator contracts or registered specialist engines. A domain may
// overlap others, unlike the exact disjoint native category inventory.
const domains=Object.freeze([
 {title:"语言模型与 Transformer",about:"Embedding、Attention、RoPE、FFN、MoE 与模型结构权衡。",
  match:/attention|gqa|rope|transformer|token|embedding|ffn|moe|mlp|norm|softmax|linear|expert|swiglu/i,
  labs:["tradeoffs"]},
 {title:"图像 · 音频 · 视频与时空计算",about:"视觉与音频编码、扩散视频、Wan FFN、具名五维时序张量。",
  match:/image|vision|patch|vae|latent|diffusion|video|tubelet|wan|audio|spectro|temporal|conv3d|multimodal/i,
  labs:["semantic_state","wan_ffn"]},
 {title:"训练 · 优化 · 后训练",about:"损失函数、反向传播、参数更新、RL 与可重放训练过程。",
  match:/loss|train|optimizer|adam|sgd|gradient|grpo|ppo|reinforce|reward|lora|mse|cross_entropy|backward/i,
  labs:["training","micro_training","engineering_journey","engineering_pipeline"]},
 {title:"推理引擎 · 缓存 · 调度",about:"Prefill/Decode、KV 生命周期、请求调度、Worker 时间与状态。",
  match:/cache|decode|prefill|schedule|paged|kv|mtp|batch|stream|infer/i,
  labs:["cache","deepinfer_prefill","deepinfer_kv","deepinfer_scheduler","deepinfer_worker"]},
 {title:"源代码 · 论文 · 运行证据",about:"源码结构、论文机制重写、能力审计与外部模型证据的边界。",
  match:/source|trace|inspect|telemetry|paper|patch|rewrite/i,
  labs:["source_atlas","paper","real_model","capabilities","telemetry"]}
]);
function renderCoverage(){
 const core=window.SeetheTruthUnifiedCore;
 if(!core||typeof core.inventory!=="function")throw Error("missing unified source lab inventory");
 const labs=core.inventory().labs,byId=new Map(labs.map(l=>[l.id,l]));
 const categories=[...new Set(A.rows.map(r=>r.category))].sort();
 const certified=new Map(M.rows.map(r=>[r.kind,r.certifiedCases.length]));
 const fragment=document.createDocumentFragment(),assigned=[];
 for(const d of domains){
  const rows=A.rows.filter(r=>d.match.test(r.kind+" "+r.title+" "+r.category));
  const card=document.createElement("article");card.className="coverage-card";
  card.setAttribute("data-domain",d.title);
  const title=document.createElement("h3");title.textContent=d.title;
  const about=document.createElement("p");about.textContent=d.about;
  const numbers=document.createElement("p");numbers.className="coverage-numbers";
  numbers.textContent="匹配积木 "+rows.length+" · NumPy 候选 "+
   rows.filter(r=>r.numpyEligible).length+" · PyTorch 候选 "+
   rows.filter(r=>r.torchEligible).length+" · 已有 CPU 案例 "+
   rows.filter(r=>certified.get(r.kind)>0).length+" 类";
  const links=document.createElement("div");links.className="coverage-labs";
  for(const id of d.labs){
   const lab=byId.get(id);if(!lab)throw Error("domain references unknown registered lab "+id);
   assigned.push(id);
   const a=document.createElement("a");a.href=lab.href;
   a.textContent=lab.id.replaceAll("_"," ")+" · 入口";a.title="专项运行类型："+lab.domain+
    "；来源："+lab.engine+"；声明证据："+lab.evidence;
   links.append(a);
  }
  const caption=document.createElement("p");caption.className="coverage-meta";
  caption.textContent="已注册专项实验室 "+d.labs.length+" · 点击入口查看真实教学引擎";
  const button=document.createElement("button");button.type="button";
  button.textContent="查看本领域匹配积木";
  button.addEventListener("click",()=>{
   $("category").value="";$("filter").value="all";$("search").value="";
   const selected=new Set(rows.map(r=>r.kind));
   renderWithPredicate(r=>selected.has(r.kind));
   $("count").scrollIntoView({block:"nearest"});
  });
  card.append(title,about,numbers,caption,links,button);fragment.append(card);
 }
 if(assigned.length!==labs.length||new Set(assigned).size!==labs.length)
  throw Error("coverage atlas must account for every distinct registered specialist lab");
 $("domainCoverage").replaceChildren(fragment);
 const categoriesFragment=document.createDocumentFragment();
 for(const name of categories){
  const button=document.createElement("button");button.type="button";
  const matching=A.rows.filter(r=>r.category===name);
  button.textContent=name+" · "+matching.length;
  button.addEventListener("click",()=>{
   $("search").value="";$("filter").value="all";$("category").value=name;render();
  });
  categoriesFragment.append(button);
 }
 $("categoryCoverage").replaceChildren(categoriesFragment);
 $("domainCoverageSummary").textContent="来自当前源码："+
  A.rows.length+" 种注册积木 · "+categories.length+
  " 个原生类别 · "+labs.length+" 个专项实验室 · "+
  M.certificateSummary.certifiedKinds+" 类具备限定 CPU 案例证书 · "+
  M.certificateSummary.fixtures+" 个独立证书测试图";
}
function inspect(kind){
 const r=M.rows.find(row=>row.kind===kind);
 if(!r)throw Error('未注册算子能力 '+kind);
 const summary=document.createElement('h2');summary.textContent=r.kind+' · '+r.title;
 const evidence=document.createElement('p');
 evidence.textContent='静态能力清单：'+M.evidence+'。可导出仅代表白名单资格；数值、VJP、模型一致性仍需实际测试。';
 const values=document.createElement('dl');
 for(const [name,proof] of Object.entries(r.capability)){
  const key=document.createElement('dt');key.textContent=name;
  const value=document.createElement('dd');
  value.textContent=proof.availability+' · '+proof.verification;
  values.append(key,value);
 }
 const cert=document.createElement('section');
 const records=r.certifiedCases;
 const caseTitle=document.createElement('h3');
 caseTitle.textContent=records.length?'已实际执行的限定测试案例 · '+records.length+' 个独立图':'尚无独立执行案例证书';
 const cases=document.createElement('ul');
 if(records.length){
  for(const item of records){
   const fixture=M.certificateSummary.fixtureIds.includes(item.fixtureId)?item.fixtureId:'未知案例';
   const li=document.createElement('li');
   li.textContent=fixture+' · PyTorch 2.6.0 CPU / float64 · 前向 '+
    item.forwardCoordinates+' 坐标 · Loss 反向 '+item.wholeLossAdjointCoordinates+
    ' 坐标 · 本案例前向 '+item.forward+' / 反向 '+item.backward;
   cases.append(li);
  }
 }else{
  const li=document.createElement('li');
  li.textContent='没有可展示的限定案例通过记录。登记或代码生成白名单不等于已经执行验证。';
  cases.append(li);
 }
 const scope=document.createElement('p');
 scope.textContent='证据仅覆盖这些实际运行的图及具体形状、坐标，不是此算子的全部形状、上下文或 GPU 认证。源码摘要 '+
 M.certificateSummary.sourceDigest.slice(0,16)+'…';
 cert.append(caseTitle,cases,scope);
 $('capabilityDetail').replaceChildren(summary,evidence,values,cert);
 return r;
}
function renderWithPredicate(predicate){
 const rows=A.rows.filter(predicate);
 const frag=document.createDocumentFragment();
 for(const r of rows){const tr=document.createElement("tr");
  for(const [i,value] of [r.kind,r.title,r.category,r.numpyEligible?"候选":"不支持",r.torchEligible?"候选":"不支持","需图级测试"].entries()){
   const td=document.createElement("td");
   if(i===0){const button=document.createElement("button");button.type="button";
    button.textContent=value;button.className="capabilityInspect";
    button.setAttribute("aria-label","查看 "+value+" 的能力证据");
    button.addEventListener("click",()=>inspect(r.kind));td.append(button);}
   else td.textContent=value;
   tr.append(td);
  }frag.append(tr);
 }
 $("rows").replaceChildren(frag);$("count").textContent=rows.length+" / "+A.rows.length;
}
function render(){
 const q=$("search").value.trim().toLocaleLowerCase(),c=$("category").value,m=$("filter").value;
 renderWithPredicate(r=>(!q||[r.kind,r.title,r.category].join(" ").toLocaleLowerCase().includes(q))&&
  (!c||r.category===c)&&(m==="all"||(m==="numpy"&&r.numpyEligible)||(m==="torch"&&r.torchEligible)||
  (m==="no-numpy"&&!r.numpyEligible)||(m==="no-torch"&&!r.torchEligible)||(m==="executed-case"&&M.rows.some(x=>x.kind===r.kind&&x.certifiedCases.length>0))));
}
for(const id of ["search","category","filter"])$(id).addEventListener(id==="search"?"input":"change",render);
render();
renderCoverage();
})();
