(function(){"use strict";
const u=new URL(location.href);
if(u.searchParams.get("edition")!=="teaching"&&location.hostname!=="appassets.androidplatform.net")return;
document.body.classList.add("teaching-edition");
const hidden=["exportPython","save","load","exportScope","pythonBackend","pythonExportPanel","traceReferenceFile"];
for(const id of hidden){const element=document.getElementById(id);if(element)element.hidden=true;}
const loadLabel=document.querySelector('label[for="load"]');if(loadLabel)loadLabel.hidden=true;
const disallowed=new Set(["exportPython","save","load"]);
document.addEventListener("click",event=>{
 const target=event.target.closest?.("button,label,a");
 if(target&&disallowed.has(target.id)||target?.matches?.('label[for="load"]')){
  event.preventDefault();event.stopImmediatePropagation();
 }
},true);
document.addEventListener("change",event=>{
 if(event.target?.id==="load"){event.preventDefault();event.stopImmediatePropagation();}
},true);
const removeResearchLinks=["real_model_lab.html","telemetry.html"];
document.querySelectorAll(".head-actions a").forEach(a=>{
 if(removeResearchLinks.some(f=>a.getAttribute("href")===f))a.hidden=true;
 if(a.getAttribute("href")==="index.html")a.hidden=true;
});
const stage=document.querySelector(".stage");
document.querySelectorAll(".mission").forEach(button=>{
 button.addEventListener("click",()=>{if(matchMedia("(max-width:700px)").matches)stage?.scrollIntoView({block:"start",behavior:"smooth"});});
});
})();