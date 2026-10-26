/* State experiment presentation: A/B views are projections of the single canonical executeLab result. */
(function(){"use strict";
const Core=window.SeetheTruthUnifiedCore;
if(!Core)throw Error("unified semantic-state core unavailable");
const $=id=>document.getElementById(id);
const lessons=Object.freeze({
 stream:{title:"状态分叉 → KV Cache 与时间依赖",reason:"在计算图课程中追踪相同前缀如何流入两条后续路径；比较 append 与 fork 的输出来源。",href:"learn.html?edition=teaching"},
 video:{title:"时间分块 → 五维张量计算图",reason:"沿 B/T/H/W/C 轴观察 slice 和 concat 的数据依赖，再把分块操作放进模型计算流。",href:"learn.html?edition=teaching"},
 axis:{title:"具名轴 → 广播与形状检查",reason:"把特征偏置的广播与不相容 Tk 轴放回积木连接过程，理解语义与形状校验。",href:"learn.html?edition=teaching"},
 reject:{title:"状态版本 → 流式计算依赖",reason:"在流状态积木课程里追踪 append 所要求的 nextPosition，理解为什么时间顺序是执行前提。",href:"learn.html?edition=teaching"}
});
function comparison(name,r){
 const v=r.values;
 switch(name){
 case "stream":return {a:["正常追加","前缀后续写入 5、6",{next:v.advanced.next,values:v.advanced.values}],b:["分叉追加","相同前缀后续写入 50、60",{next:v.fork.next,values:v.fork.values}],difference:"共享前缀保持一致，后缀独立变化；原状态的 nextPosition 不随分支修改。"};
 case "video":return {a:["原始视频","逐帧 5D 张量的首帧与末帧",{firstFrame:v.firstFrame,lastFrame:v.lastFrame}],b:["分块重组","沿 T 轴切片后重新拼接",{shape:v.shape,joined:v.joined,exactEqual:v.exactEqual}],difference:v.exactEqual?"每个坐标精确相等：分块与拼接保持张量值及轴语义。":"分块重组不等价：请检查时间轴和元素顺序。"};
 case "axis":return {a:["合法广播","特征轴 D 对齐",{axes:v.axes,shape:v.shape,values:v.values}],b:["错误轴名","试图将 Tk 与 T/D 对齐",{wrongAxisRejected:v.wrongAxisRejected}],difference:v.wrongAxisRejected?"具名轴校验拒绝不匹配的 Tk；维度大小相似也不意味着轴语义相同。":"错误轴未被拒绝，数学合同需检查。"};
 case "reject":return {a:["合法状态","原始 nextPosition 等于 6",{origin:v.origin,nextPosition:v.nextPosition,original:v.original}],b:["过期写入","尝试在位置 5 写入",{attemptedPosition:v.attemptedPosition,error:v.error}],difference:"过期写入被拒绝，原始状态保持不变；状态的时序不变量独立于张量形状。"};
 default:throw Error("unknown state comparison "+name);
 }
}
function run(name){
 const r=Core.executeLab("semantic_state","scenario",{name}).value;
 $("caseTitle").textContent=r.title;
 $("explanation").textContent=r.explanation;
 $("prediction").textContent=r.prediction;
 $("status").textContent="数值合同通过 · "+name;
 $("trace").textContent=JSON.stringify(r.values,null,2);
 const ab=comparison(name,r);
 for(const [prefix,record] of [["a",ab.a],["b",ab.b]]){
  $(prefix+"Title").textContent=record[0];
  $(prefix+"Description").textContent=record[1];
  $(prefix+"Values").textContent=JSON.stringify(record[2],null,2);
 }
 $("difference").textContent=ab.difference;
 const lesson=lessons[name];
 $("lessonTitle").textContent=lesson.title;
 $("lessonReason").textContent=lesson.reason;
 $("lessonLink").href=lesson.href;
 for(const b of document.querySelectorAll("[data-scenario]"))
  b.setAttribute("aria-pressed",String(b.dataset.scenario===name));
 return r;
}
for(const b of document.querySelectorAll("[data-scenario]"))
 b.addEventListener("click",()=>run(b.dataset.scenario));
run("stream");
window.SeetheTruthStateLab=Object.freeze({run,cases:window.SeetheTruthSemanticExamples.names,comparison});
})();
