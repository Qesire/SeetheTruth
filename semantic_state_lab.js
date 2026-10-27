/* State experiment presentation: A/B views are projections of the single canonical executeLab result. */
(function(){"use strict";
const Core=window.SeetheTruthUnifiedCore;
const Bridge=window.SeetheTruthStateBridge;
const CoordinateFocus=window.SeetheTruthCoordinateFocus;
if(!CoordinateFocus)throw Error("coordinate focus protocol unavailable");
const coordinateFocusSession=CoordinateFocus.store();
window.SeetheTruthCoordinateFocusSession=coordinateFocusSession;
if(!Core||!Bridge)throw Error("unified state core or evidence bridge unavailable");
const $=id=>document.getElementById(id);
const lessons=Object.freeze({
 stream:{title:"状态分叉 → KV Cache 与时间依赖",reason:"在计算图课程中追踪相同前缀如何流入两条后续路径；比较 append 与 fork 的输出来源。",href:"learn.html?edition=teaching&state_case=stream"},
 video:{title:"时间分块 → 五维张量计算图",reason:"沿 B/T/H/W/C 轴观察 slice 和 concat 的数据依赖，再把分块操作放进模型计算流。",href:"learn.html?edition=teaching&state_case=video"},
 axis:{title:"具名轴 → 广播与形状检查",reason:"把特征偏置的广播与不相容 Tk 轴放回积木连接过程，理解语义与形状校验。",href:"learn.html?edition=teaching&state_case=axis"},
 reject:{title:"状态版本 → 流式计算依赖",reason:"在流状态积木课程里追踪 append 所要求的 nextPosition，理解为什么时间顺序是执行前提。",href:"learn.html?edition=teaching&state_case=reject"}
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
function paintCoordinates(name,result){
 const groups=[];
 for(const branch of ["a","b"]){
  const root=$(branch+"CoordinateGrid"),indices=Bridge.coordinateList(name,result,branch);
  root.replaceChildren();
  $(branch+"CoordinateCount").textContent=indices.length?indices.length+" values":"拒绝执行";
  const buttons=[];
  if(!indices.length){
   const empty=document.createElement("p");
   empty.className="no-coordinate";
   empty.textContent="该反例在数学/状态校验阶段被拒绝，没有可追踪的数值输出；请查看错误条件与对应积木。";
   root.append(empty);
  }
  for(const i of indices){
   const item=Bridge.coordinate(name,result,branch,i);
   const button=document.createElement("button");button.type="button";
   button.dataset.coordinate=branch+":"+i;
   button.setAttribute("aria-pressed","false");
   button.setAttribute("aria-label",branch.toUpperCase()+" 实际输出坐标 ["+item.coordinate.join(",")+"] 值 "+item.value);
   button.textContent="#"+i+" · "+item.value;
   button.addEventListener("click",()=>{
    for(const {buttons:list} of groups)for(const control of list)
     control.setAttribute("aria-pressed","false");
    button.setAttribute("aria-pressed","true");
    const native=coordinateFocusSession.select(CoordinateFocus.fromState(item));
    const other=branch==="a"?"b":"a";
    const sibling=Bridge.coordinateList(name,result,other).includes(i)
     ?Bridge.coordinate(name,result,other,i):null;
    const delta=sibling?CoordinateFocus.compare(
      branch==="a"?native:CoordinateFocus.fromState(sibling),
      branch==="a"?CoordinateFocus.fromState(sibling):native).delta:null;
    const origin=item.origin===undefined?"":" · 绝对位置 "+item.origin;
    $("coordinateDetail").textContent=branch.toUpperCase()+" ["+item.coordinate.join(",")+"] = "+item.value+
     origin+"\n实际来源："+item.terms.map(t=>t.role+"["+t.index+"]="+t.value).join("；")+
     (delta===null?"\n对照分支没有可执行的同坐标输出":"\n同坐标 A−B = "+delta)+
     "\n"+item.note;
    $("coordinateGraphLink").href=Bridge.href(name,item.nodeId,branch,i);
    $("coordinateGraphLink").textContent="定位 "+item.nodeId+" 的上游计算路径 →";
   });
   root.append(button);buttons.push(button);
  }
  groups.push({branch,buttons});
 }
 const query=new URLSearchParams(window.location.search);
 const requestedBranch=query.get("state_branch"),requestedIndex=query.get("state_index");
 const requested=groups.find(g=>g.branch===requestedBranch)?.buttons.find(b=>
  b.dataset.coordinate===requestedBranch+":"+requestedIndex);
 const first=requested||groups[0].buttons[0];
 if(first)first.click();
}
function renderMechanismAnchors(name,result){
 const spec=Bridge.caseOf(name);
 const root=$("caseFlow");root.replaceChildren();
 for(const item of spec.anchors){
  const a=document.createElement("a");a.href=Bridge.href(name,item.id);
  a.textContent=item.label+" · "+item.kind;
  a.setAttribute("aria-label","在计算图课程中查看 "+item.label+" 积木");
  root.append(a);
 }
 const countA=Bridge.coordinateList(name,result,"a").length;
 const countB=Bridge.coordinateList(name,result,"b").length;
 $("caseQuickFacts").textContent="本次真实数值输出：A "+countA+" 坐标、B "+countB+
  " 坐标。逐坐标数据来自本次状态实验；以上链接定位已安装课程的对应积木。";
}
function run(name){
 const accepted=Core.executeLab("semantic_state","scenario",{name});
 const r=accepted.value;
 if(r.passed!==true)throw Error("本次状态实验数值合同失败");
 Core.replay?.(accepted);
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
 const primary=Bridge.caseOf(name).primary;
 $("aGraphLink").href=Bridge.href(name,primary.a);
 $("bGraphLink").href=Bridge.href(name,primary.b);
 $("aGraphLink").textContent=Bridge.caseOf(name).labels.a+" →";
 $("bGraphLink").textContent=Bridge.caseOf(name).labels.b+" →";
 paintCoordinates(name,r);
 renderMechanismAnchors(name,r);
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
const requested=new URLSearchParams(window.location.search).get("state_case");
run(Object.prototype.hasOwnProperty.call(lessons,requested)?requested:"stream");
window.SeetheTruthStateLab=Object.freeze({run,cases:window.SeetheTruthSemanticExamples.names,comparison,get coordinateFocus(){return coordinateFocusSession.current;}});
})();
