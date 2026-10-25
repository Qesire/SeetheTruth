(function(){"use strict";
const A=window.SeetheTruthSpecializedLesson.execute("audit",{}).value,
 M=window.SeetheTruthCapabilities.matrix(),$=id=>document.getElementById(id);
for(const [k,v] of Object.entries(A.counts))if($(k))$(k).textContent=v;
for(const category of [...new Set(A.rows.map(r=>r.category))].sort()){
 const el=document.createElement("option");el.value=category;el.textContent=category;$("category").append(el);
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
function render(){
 const q=$("search").value.trim().toLocaleLowerCase(),c=$("category").value,m=$("filter").value;
 const rows=A.rows.filter(r=>(!q||[r.kind,r.title,r.category].join(" ").toLocaleLowerCase().includes(q))&&
  (!c||r.category===c)&&(m==="all"||(m==="numpy"&&r.numpyEligible)||(m==="torch"&&r.torchEligible)||
  (m==="no-numpy"&&!r.numpyEligible)||(m==="no-torch"&&!r.torchEligible)||(m==="executed-case"&&M.rows.some(x=>x.kind===r.kind&&x.certifiedCases.length>0))));
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
for(const id of ["search","category","filter"])$(id).addEventListener(id==="search"?"input":"change",render);
render();
})();
