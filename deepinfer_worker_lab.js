(function(){"use strict";
const E=window.SeetheTruthDeepInferEvents;
if(!E)throw Error("DeepInfer event simulator unavailable");
const $=id=>document.getElementById(id),svgNS="http://www.w3.org/2000/svg";
const State=window.SeetheTruthStateMicroscope;
const stateView=window.SeetheTruthStateMicroscopeUI.mount($('stateMicroscope'));
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
const seq=(seqId,seqlenQ,seqlenK)=>({seqId,seqlenQ,seqlenK});
const sampleFixture={
 nextn:2,variableMtp:true,records:[
  {sequences:[seq("A",1,8),seq("B",1,9)],accepted:[1,2],
   confidenceRates:[[0.9,0.8],[0.6,0.4]],totalQ:2,cachedTokens:[7,8]},
  {sequences:[seq("A",1,10),seq("B",1,12)],accepted:[1,2],
   confidenceRates:[[0.6,0.4],[0.9,0.8]],totalQ:2,cachedTokens:[9,11]},
  {sequences:[seq("B",1,13),seq("C",1,5)],accepted:[1,0],
   confidenceRates:[[0.8,0.7],[0.5,0.4]],totalQ:3,cachedTokens:[12,4],paddedLength:3},
  {sequences:[seq("B",1,14),seq("C",1,6)],accepted:[2,1],
   confidenceRates:[[0.9,0.8],[0.5,0.3]],totalQ:3,cachedTokens:[13,5]}
 ]
};
let metadataFixture=sampleFixture;
const sampleDecodeFixture={
 step:2,
 input:{blockSize:4,maxNumBlocks:6,maxSeqlenQ:3,blockTableCompressRatio:2,
  needDefaultSeqlens:true,sequences:[
   {seqId:"B",seqlenQ:1,seqlenK:13,blockTable:[7,4,8,9],stateId:21,isPlaceholder:false},
   {seqId:"C",seqlenQ:1,seqlenK:5,blockTable:[2,3],stateId:22,isPlaceholder:false}]},
 states:[
  {seqId:"B",cachedTokens:12,outputLen:1,soStatus:2,soMatcher:"example",
   mtp:{infoSlot:5,hasDraftInfo:true,lastOutputToken:88,goldenTokens:[10,11,12,13,14,15]}},
  {seqId:"C",cachedTokens:4,outputLen:0,soStatus:0,soMatcher:null,
   mtp:{infoSlot:0,hasDraftInfo:false,lastOutputToken:-1,goldenTokens:null}}
 ]
};
let decodeFixture=sampleDecodeFixture;

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
   const options={flush:$("flush").checked,durations};
   if($("numericMetadata").checked)options.metadataFixture=metadataFixture;
   if($("numericMetadata").checked&&$("decodeEnabled").checked)options.decodeFixture=decodeFixture;
   const zos=E.simulate(steps,{...options,zeroOverhead:true});
   const non=E.simulate(steps,{...options,zeroOverhead:false});
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
   const metadataRows=[];
   for(const trace of zos.metadataTrace){
    const tr=document.createElement("tr");
    const cols=[String(trace.step),String(trace.previousStep),
      JSON.stringify(trace.remap),JSON.stringify(trace.seqlensQ),
      JSON.stringify(trace.seqlensKBase),JSON.stringify(trace.seqlensK),
      trace.sampler?trace.sampler.map(x=>x.seqId+": "+x.seqlenQ+"/"+x.seqlenK).join(", "):"未提供 cache token fixture"];
    for(const cell of cols){const td=document.createElement("td");td.textContent=cell;tr.append(td);}
    metadataRows.push(tr);
   }
   $("metadataRows").replaceChildren(...metadataRows);
   const decodeRows=[];
   for(const trace of zos.metadataTrace){
    if(!trace.decode)continue;
    const packed=trace.decode.packed,state=trace.decode.sampling;
    const cells=[String(trace.step),JSON.stringify(packed.blockTables),
      JSON.stringify(packed.blockTableOffsets),JSON.stringify(packed.numBlocks),
      JSON.stringify(packed.cuSeqlensQ),
      state.metadata.map(x=>x.seqId+": "+x.seqlenQ+"/"+x.seqlenK).join(" · "),
      state.metadata.map(x=>x.seqId+": "+x.mtpInfoReadSlot+"/"+x.mtpInfoWriteSlot).join(" · "),
      state.metadata.map(x=>x.seqId+": "+JSON.stringify(x.goldenToken)).join(" · ")];
    const tr=document.createElement("tr");
    for(const value of cells){const td=document.createElement("td");td.textContent=value;tr.append(td);}
    decodeRows.push(tr);
   }
   $("decodeRows").replaceChildren(...decodeRows);
   $("decodeStatus").textContent=decodeRows.length+" 项压缩页表与采样状态计算已完成（非 GPU 内存分配）";
   $("metadataStatus").textContent=$("numericMetadata").checked?
     metadataRows.length+" 个跨步元数据修正已执行（合成采样数据）":
     "此场景未启用数值 Fixture；事件顺序仍可独立运行";
   window.SeetheTruthLastWorkerComparison={zos,non};
   stateView.show(State.worker(zos));
   $("editorStatus").textContent="已执行合法事件图；时间为假设值。";
 }catch(err){$("editorStatus").textContent="输入被拒绝："+err.message;}
}
$("scenario").addEventListener("change",()=>{
 const selected=scenarios[$("scenario").value];
 if(!selected)return;
 steps=selected.map(x=>({mode:x.mode,sequenceIds:x.sequenceIds.slice()}));
 $("numericMetadata").checked=$("scenario").value==="mixed";
 $("decodeEnabled").checked=$("scenario").value==="mixed";
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
   $("numericMetadata").checked=false; // custom steps need matching user-provided fixtures
   $("decodeEnabled").checked=false;
   render();
 }catch(err){$("editorStatus").textContent="输入被拒绝："+err.message;}
});
$("numericMetadata").addEventListener("change",render);
$("applyMetadata").addEventListener("click",()=>{
 try{
  const parsed=JSON.parse($("metadataJson").value);
  E.simulate(steps,{zeroOverhead:true,metadataFixture:parsed,
   ...($("decodeEnabled").checked?{decodeFixture}: {})});
  metadataFixture=parsed;$("numericMetadata").checked=true;render();
 }catch(err){$("metadataStatus").textContent="元数据 Fixture 被拒绝："+err.message;}
});
$("decodeEnabled").addEventListener("change",render);
$("applyDecode").addEventListener("click",()=>{
 try{
  const parsed=JSON.parse($("decodeJson").value);
  E.simulate(steps,{zeroOverhead:true,metadataFixture,decodeFixture:parsed});
  decodeFixture=parsed;$("numericMetadata").checked=true;$("decodeEnabled").checked=true;
  render();
 }catch(err){$("decodeStatus").textContent="Decode Fixture 被拒绝："+err.message;}
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
$("metadataJson").value=JSON.stringify(metadataFixture,null,2);
$("decodeJson").value=JSON.stringify(decodeFixture,null,2);
render();
})();