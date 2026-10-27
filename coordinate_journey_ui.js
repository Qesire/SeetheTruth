/* Compact, accessible native coordinate forward/backward explanation steps.
 * Rendering only; mathematical values come from coordinate_journey.js.
 */
(function(root,factory){"use strict";
 const api=factory(root?.SeetheTruthCoordinateJourney);
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthCoordinateJourneyUI=api;
})(typeof window!=="undefined"?window:null,function(Journey){"use strict";
 function mount(container,options={}){
  if(!container||typeof container.append!=="function")
   throw Error("coordinate journey UI requires a real host element");
  const doc=container.ownerDocument||document;
  const section=doc.createElement("section");
  section.className="coordinate-journey";section.id="sharedCoordinateJourney";
  section.hidden=true;section.setAttribute("aria-label","逐坐标来源与反向贡献时间线");
  const header=doc.createElement("div");header.className="coordinate-journey-head";
  const heading=doc.createElement("strong");heading.textContent="COORDINATE JOURNEY · 逐步取证";
  const counter=doc.createElement("span");counter.id="coordinateJourneyCount";
  header.append(heading,counter);
  const nav=doc.createElement("div");nav.className="coordinate-journey-nav";
  const previous=doc.createElement("button");previous.type="button";previous.id="coordinateJourneyPrevious";
  previous.textContent="上一环节";previous.setAttribute("aria-label","查看上一个真实计算环节");
  const slider=doc.createElement("input");slider.type="range";slider.id="coordinateJourneyPosition";
  slider.min="0";slider.step="1";slider.value="0";slider.max="0";
  slider.setAttribute("aria-label","跳转到计算依赖或训练状态时间线中的一个环节");
  const next=doc.createElement("button");next.type="button";next.id="coordinateJourneyNext";
  next.textContent="下一环节";next.setAttribute("aria-label","查看下一个真实计算环节");
  nav.append(previous,slider,next);
  const body=doc.createElement("div");body.className="coordinate-journey-body";
  body.setAttribute("role","status");body.setAttribute("aria-live","polite");
  const title=doc.createElement("strong");title.id="coordinateJourneyTitle";
  const detail=doc.createElement("p");detail.id="coordinateJourneyDetail";
  const metrics=doc.createElement("dl");metrics.className="coordinate-journey-metrics";
  body.append(title,detail,metrics);section.append(header,nav,body);container.append(section);
  let report=null,position=0;
  const format=n=>n===null||n===undefined?"未求值":
   typeof n==="number"?Number(n.toPrecision(10)).toString():String(n);
  function draw(){
   if(!report)return;
   const stages=report.steps,step=stages[position];
   counter.textContent=(position+1)+" / "+stages.length;
   previous.disabled=position<=0;next.disabled=position>=stages.length-1;
   slider.max=String(stages.length-1);slider.value=String(position);
   title.textContent=step.title||step.kind;
   detail.textContent=step.detail||"已验证原生计算环节";
   metrics.replaceChildren();
   const rows=[["节点",step.nodeId],["坐标",step.index?"["+step.index.join(",")+"]":null]];
   if(step.kind==="edge")rows.push(["来源 → 目标",step.sourceKey+" → "+step.targetKey],
    ["输入端口",step.port],["局部规则",step.rule],
    ["上游灵敏度",format(step.upstreamSensitivity)],
    ["边局部导数",format(step.localDerivative)],
    ["路径贡献",format(step.contribution)],
    ["该元素汇合后",format(step.accumulatedAfter)],
    ["累加方式",step.firstArrival?"第一条进入此元素的依赖":"与此前路径求和"]);
   else if(step.kind==="merge"||step.kind==="seed")rows.push(
    ["元素值",format(step.value)],["累计灵敏度",format(step.sensitivity)]);
   else if(step.kind==="optimizer_state"){
    rows.push(["优化器类型",report.focus.transition.optimizerKind]);
    for(const [name,pair]of [["一阶状态",step.moment1],["二阶状态",step.moment2]])
     if(pair)rows.push([name,format(pair.before)+" → "+format(pair.after)]);
   }else if(step.kind==="loss_transition")
    rows.push(["Loss 更新前",format(step.lossBefore)],["Loss 更新后",format(step.value)]);
   else if(step.kind==="parameter_after")
    rows.push(["参数更新后",format(step.value)],["参数位移",format(step.delta)]);
   else if(step.kind==="selected_path")
    rows.push(["已验证路径状态",step.status],["路径贡献",format(step.value)]);
   else if(step.value!==undefined)rows.push(["原生证据值",format(step.value)]);
   for(const [label,value]of rows){
    if(value===null||value===undefined||value==="")continue;
    const dt=doc.createElement("dt");dt.textContent=label;
    const dd=doc.createElement("dd");dd.textContent=value;
    metrics.append(dt,dd);
   }
   if(typeof options.onStep==="function")options.onStep(step,position,report);
  }
  const move=index=>{
   if(!report||!Number.isSafeInteger(index)||index<0||index>=report.steps.length)
    throw Error("coordinate journey position out of range");
   position=index;draw();
   return report.steps[position];
  };
  previous.addEventListener("click",()=>{if(position>0)move(position-1)});
  next.addEventListener("click",()=>{if(report&&position<report.steps.length-1)move(position+1)});
  slider.addEventListener("input",()=>move(Number(slider.value)));
  slider.addEventListener("change",()=>move(Number(slider.value)));
  return Object.freeze({
   show:source=>{
    if(!Journey||typeof Journey.fromDrilldown!=="function")
     throw Error("coordinate journey engine unavailable");
    const nextReport=Journey.fromDrilldown(source);
    report=nextReport;position=0;section.hidden=false;draw();
    return nextReport;
   },
   clear:()=>{report=null;position=0;section.hidden=true;metrics.replaceChildren()},
   move,get report(){return report},get position(){return position}
  });
 }
 return Object.freeze({mount});
});
