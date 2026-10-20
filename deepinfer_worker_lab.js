(function(){"use strict";
const E=window.SeetheTruthDeepInferEvents;
if(!E)throw Error("DeepInfer event simulator unavailable");
const $=id=>document.getElementById(id),svgNS="http://www.w3.org/2000/svg";
const scenarios={
 mixed:E.DEFAULT_STEPS.map(x=>({mode:x.mode,sequenceIds:[...x.sequenceIds]})),
 decode:[{mode:"decode",sequenceIds:["A"]},{mode:"decode",sequenceIds:["A"]},
         {mode:"decode",sequenceIds:["A"]},{mode:"decode",sequenceIds:["A"]}],
 nooverlap:[{mode:"decode",sequenceIds:["A"]},{mode:"decode",sequenceIds:["B"]},
            {mode:"decode",sequenceIds:["C"]},{mode:"decode",sequenceIds:["D"]}],
 placeholder:[{mode:"prompt",sequenceIds:["A"]},{mode:"decode",sequenceIds:["A"]},
              {mode:"placeholder",sequenceIds:[]},{mode:"decode",sequenceIds:["A"]}]
};
let steps=scenarios.mixed;
function svg(name,attrs,textContent){
 const e=document.createElementNS(svgNS,name);
 for(const [k,v] of Object.entries(attrs||{}))e.setAttribute(k,String(v));
 if(textContent!==undefined)e.textContent=textContent;
 return e;
}
const sourceDetail={
 schedule:"LLMEngine.step → Scheduler.schedule (C++ scheduling; only event boundary modelled)",
 launch_forward:"Worker.execute_model_step → execute_main_model call/launch",
 correct_zos_metadata:"Worker._correct_seqlens_qk: previous sampler, remap and accepted token metadata; D2H sync simplified",
 forward:"ModelRunner.execute_main_model (GPU compute; no actual NN layers executed)",
 post_prev:"Worker._post_process_step(previous state): synchronize prior sample completion",
 launch_sample:"ModelRunner.execute_sample metadata and launch; device timing is abstract",
 sample:"Sampler output + sample_end_evt.record",
 post_current:"Prompt/non-ZOS Worker._post_process_step(current step)",
 flush_pending:"Worker.flush_inflight_reqs"
};
function pretty(e,run){
 return ["Stage: "+e.stage+"  |  "+e.resource,
 "Step: "+e.step+"    logical ticks: "+e.start+" → "+e.end,
 "Dependencies: "+(e.dependsOn.length?e.dependsOn.join(", "):"none"),
 "Source: "+sourceDetail[e.stage],
 "Metadata: "+JSON.stringify(e.meta,null,2),
 "Run: "+(run.zeroOverhead?"ZOS":"Non-ZOS"),
 "Numbers are hypothetical per-resource durations, never measured latency."].join("\n");
}
function chart(run,container){
 const height=142,width=980,left=91,right=16,usable=width-left-right;
 const ticks=Math.max(1,run.makespan);
 const graph=svg("svg",{viewBox:"0 0 "+width+" "+height,role:"img",
  "aria-label":(run.zeroOverhead?"ZOS":"Non-ZOS")+" source-aware CPU and GPU event timeline"});
 for(let i=0;i<=5;i++){const x=left+i*usable/5;
   graph.append(svg("line",{x1:x,y1:21,x2:x,y2:116,class:"axis"}));
   graph.append(svg("text",{x:x,y:133,"text-anchor":"middle",class:"caption"},(ticks*i/5).toFixed(1)));
 }
 graph.append(svg("text",{x:12,y:58},"CPU"));
 graph.append(svg("text",{x:12,y:98},"GPU"));
 const colors={CPU:["#277c9e","#66bfdc"],GPU:["#57549d","#a89cfa"]};
 for(const e of run.events){
   const y=e.resource==="CPU"?37:77;
   const x=left+e.start/ticks*usable;
   const w=Math.max(e.duration/ticks*usable,1.4);
   const rect=svg("rect",{x,y,width:w,height:27,class:"event",
      fill:colors[e.resource][e.stage==="correct_zos_metadata"||e.stage==="post_prev"||e.stage==="flush_pending"?1:0],
      tabindex:0,role:"button","aria-label":"Step "+e.step+" "+e.stage});
   rect.append(svg("title",{},e.id+" ["+e.start+","+e.end+"]"));
   const show=()=>{$("details").textContent=pretty(e,run);};
   rect.addEventListener("click",show);
   rect.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();show();}});
   graph.append(rect);
   if(w>47)graph.append(svg("text",{x:x+5,y:y+18,"pointer-events":"none",style:"font-size:10px;fill:white"},
     e.stage.replace("launch_","↗").replace("correct_zos_metadata","correct").replace("post_","post ")));
 }
 container.replaceChildren(graph);
}
function render(){
 try{
   const durations={...E.DEFAULT_DURATIONS,forward:Number($("forwardTime").value),
     post:Number($("postTime").value)};
   const zos=E.simulate(steps,{zeroOverhead:true,flush:$("flush").checked,durations});
   const non=E.simulate(steps,{zeroOverhead:false,flush:$("flush").checked,durations});
   $("zosTime").textContent=zos.makespan;
   $("blockingTime").textContent=non.makespan;
   $("deltaTime").textContent=String(non.makespan-zos.makespan);
   $("forwardValue").textContent=durations.forward;
   $("postValue").textContent=durations.post;
   $("zosInfo").textContent=zos.events.length+" 个事件";
   $("blockingInfo").textContent=non.events.length+" 个事件";
   chart(zos,$("zosChart"));chart(non,$("blockingChart"));
   const rows=[];
   for(let i=0;i<steps.length;i++){
    const tr=document.createElement("tr");
    const at=(run)=>run.released.find(e=>e.step===i);
    for(const txt of [String(i),steps[i].mode,steps[i].sequenceIds.join(", ")||"—",
        at(zos)?"t="+at(zos).at+" ("+at(zos).via+")":"待 Flush",
        at(non)?"t="+at(non).at+" ("+at(non).via+")":"—"]){
      const td=document.createElement("td");td.textContent=txt;tr.append(td);
    }
    rows.push(tr);
   }
   $("results").replaceChildren(...rows);
   window.SeetheTruthLastWorkerComparison={zos,non};
   $("editorStatus").textContent="已执行合法事件图；时间为假设值。";
 }catch(err){$("editorStatus").textContent="输入被拒绝："+err.message;}
}
$("scenario").addEventListener("change",()=>{
 const selected=scenarios[$("scenario").value];
 if(!selected)return;
 steps=selected.map(x=>({mode:x.mode,sequenceIds:x.sequenceIds.slice()}));
 $("customSteps").value=JSON.stringify(steps,null,2);render();
});
$("forwardTime").addEventListener("input",render);
$("postTime").addEventListener("input",render);
$("flush").addEventListener("change",render);
$("applySteps").addEventListener("click",()=>{
 try{
   const raw=JSON.parse($("customSteps").value);
   E.validate(raw,{zeroOverhead:true});
   steps=raw.map(x=>({mode:x.mode,sequenceIds:[...x.sequenceIds]}));
   render();
 }catch(err){$("editorStatus").textContent="输入被拒绝："+err.message;}
});
$("download").addEventListener("click",()=>{
 const result=window.SeetheTruthLastWorkerComparison;
 if(!result)return;
 const payload=JSON.stringify({source:"SeetheTruth logical DeepInfer Worker model",steps,
   ...result},null,2);
 const blob=new Blob([payload],{type:"application/json"}),url=URL.createObjectURL(blob);
 const a=document.createElement("a");a.href=url;a.download="seethetruth-deepinfer-worker-events.json";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),500);
});
$("customSteps").value=JSON.stringify(steps,null,2);
render();
})();