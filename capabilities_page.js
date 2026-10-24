(function(){"use strict";
const A=window.SeetheTruthSpecializedLesson.execute("audit",{}).value,$=id=>document.getElementById(id);
for(const [k,v] of Object.entries(A.counts))if($(k))$(k).textContent=v;
for(const category of [...new Set(A.rows.map(r=>r.category))].sort()){
 const el=document.createElement("option");el.value=category;el.textContent=category;$("category").append(el);
}
function render(){
 const q=$("search").value.trim().toLocaleLowerCase(),c=$("category").value,m=$("filter").value;
 const rows=A.rows.filter(r=>(!q||[r.kind,r.title,r.category].join(" ").toLocaleLowerCase().includes(q))&&
  (!c||r.category===c)&&(m==="all"||(m==="numpy"&&r.numpyEligible)||(m==="torch"&&r.torchEligible)||
  (m==="no-numpy"&&!r.numpyEligible)||(m==="no-torch"&&!r.torchEligible)));
 const frag=document.createDocumentFragment();
 for(const r of rows){const tr=document.createElement("tr");
  for(const value of [r.kind,r.title,r.category,r.numpyEligible?"候选":"不支持",r.torchEligible?"候选":"不支持","需图级测试"]){
   const td=document.createElement("td");td.textContent=value;tr.append(td);
  }frag.append(tr);
 }
 $("rows").replaceChildren(frag);$("count").textContent=rows.length+" / "+A.rows.length;
}
for(const id of ["search","category","filter"])$(id).addEventListener(id==="search"?"input":"change",render);
render();
})();
