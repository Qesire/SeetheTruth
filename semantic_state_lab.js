/* Pure teaching renderer; all arithmetic is in semantic_state_examples.js. */
(function(){"use strict";
const Core=window.SeetheTruthUnifiedCore;
if(!Core)throw Error("unified semantic-state core unavailable");
const $=x=>document.getElementById(x);
function run(name){
 const result=Core.executeLab("semantic_state","scenario",{name}).value;
 $("caseTitle").textContent=result.title;
 $("explanation").textContent=result.explanation;
 $("prediction").textContent=result.prediction;
 $("status").textContent="数值合同通过 · "+name;
 $("trace").textContent=JSON.stringify(result.values,null,2);
 for(const b of document.querySelectorAll("[data-scenario]"))
  b.setAttribute("aria-pressed",String(b.dataset.scenario===name));
 return result;
}
for(const b of document.querySelectorAll("[data-scenario]"))
 b.addEventListener("click",()=>run(b.dataset.scenario));
run("stream");
window.SeetheTruthStateLab=Object.freeze({
 run,cases:window.SeetheTruthSemanticExamples.names
});
})();
