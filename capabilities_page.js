(function(){"use strict";
const A=window.SeetheTruthCapabilities.audit(),$=id=>document.getElementById(id);
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
const Surface=window.SeetheTruthSurfaceAudit;
if(!Surface)throw Error('universal surface audit missing');
const statuses={tensor:'已接入 · 通用张量/算子',state:'已接入 · 通用事件/状态',specialized:'专用实现 · 尚待复用',navigation:'导航，无计算职责',desktop_only:'PC 只读 / 实测来源'};
const rows=document.createDocumentFragment();
for(const item of Surface.screens){
 const tr=document.createElement('tr'),td=document.createElement('td'),a=document.createElement('a');
 a.href=item.page;a.textContent=item.title;td.append(a);tr.append(td);
 for(const v of [statuses[item.status],item.note]){const cell=document.createElement('td');cell.textContent=v;tr.append(cell);}rows.append(tr);
}
$('surfaceAuditRows').replaceChildren(rows);
const total=Surface.screens;
$('coreTensorCount').textContent=total.filter(x=>x.status==='tensor').length;
$('coreStateCount').textContent=total.filter(x=>x.status==='state').length;
$('specializedCount').textContent=total.filter(x=>x.status==='specialized').length;
$('otherCount').textContent=total.filter(x=>['navigation','desktop_only'].includes(x.status)).length;
})();
