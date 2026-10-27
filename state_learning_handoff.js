/* Shared state-lab ↔ executable graph navigation.
 * A/B witnesses are projected from the accepted semantic-state execution.
 * The guided graphs use independent seeded teaching values: alignment is
 * operator/causality-level, not an assertion of coordinate equality.
 */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthStateBridge=api;
 if(root&&typeof document!=="undefined"&&root.SeetheTruthUnifiedWorkbench)
   api.mount(root,document);
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const cases=Object.freeze({
  stream:Object.freeze({
   title:"绝对位置与 KV 前缀分叉",preset:"stream_state_branches",
   summary:"两条分支共享初始化状态；各自的 Append 保留不同后缀。",
   challenge:"比较 append_a / append_b 的上游端口，再查看两个 Read 的输出。实验室以 origin=100、D=2；积木课程以 origin=10、D=3 独立执行。",
   primary:{a:"append_a",b:"append_b"},
   labels:{a:"定位 A 分支的 Append",b:"定位 B 分支的 Append"},
   anchors:[
    {id:"init",kind:"stream_state_init",label:"共同前缀",field:"initial.next"},
    {id:"append_a",kind:"stream_state_append",label:"A · 追加",field:"advanced.values"},
    {id:"append_b",kind:"stream_state_append",label:"B · 分叉追加",field:"fork.values"},
    {id:"read_a",kind:"stream_state_read",label:"A · 时间窗口",field:"readWindow.values"}
   ],
   edges:[["init","fork_a","state"],["init","fork_b","state"],["fork_a","append_a","state"],["fork_b","append_b","state"]]
  }),
  video:Object.freeze({
   title:"五维视频 T 轴切片与合并",preset:"extension_video_time_composition",
   summary:"原视频沿具名时间轴分段，再根据边的顺序拼接；检查输出坐标是否保持一致。",
   challenge:"查看 early、late 的 T 轴范围与 join 的两条输入边，再运行积木图。实验室使用整数 1…24；课程视频源使用独立的确定性正弦样本。",
   primary:{a:"video",b:"join"},
   labels:{a:"定位原始五维张量",b:"定位重组后的时间张量"},
   anchors:[
    {id:"video",kind:"semantic_video_source",label:"A · 原始视频",field:"shape"},
    {id:"early",kind:"semantic_video_slice",label:"第一段 T 切片",field:"firstFrame"},
    {id:"late",kind:"semantic_video_slice",label:"第二段 T 切片",field:"lastFrame"},
    {id:"join",kind:"semantic_video_concat",label:"B · 时间轴重组",field:"joined"}
   ],
   edges:[["video","early","x"],["video","late","x"],["early","join","a"],["late","join","b"]]
  }),
  axis:Object.freeze({
   title:"具名轴广播与轴身份",preset:"extension_named_axis_composition",
   summary:"按 D 广播是有效计算；Tk 错配来自实验室中的独立失败案例。",
   challenge:"追踪 x、bias → add 的 D 轴输入，在计算图里查看各通道增量。Tk 错误仍由状态实验室的反例执行，不会伪装成成功图节点。",
   primary:{a:"add",b:"bias"},
   labels:{a:"定位合法 D 轴广播",b:"检查合法偏置轴（Tk 错误见实验室）"},
   anchors:[
    {id:"x",kind:"axis_tensor_source",label:"T×D 数据源",field:"shape"},
    {id:"bias",kind:"axis_feature_bias",label:"B · 具名 D 偏置",field:"axes"},
    {id:"add",kind:"axis_named_add",label:"A · 合法广播",field:"values"}
   ],
   edges:[["x","add","x"],["bias","add","bias"]]
  }),
  reject:Object.freeze({
   title:"过期位置的状态写入",preset:"stream_state_branches",
   summary:"状态 Append 需要 position = nextPosition；过期写入使执行拒绝。",
   challenge:"当前积木图的合法位置为 12；把 append_a.position 改为 11，再运行并观察拒绝结果。实验室反例使用 nextPosition=6、attemptedPosition=5。",
   primary:{a:"init",b:"append_a"},
   labels:{a:"定位合法状态源",b:"定位 Append 并手动修改位置"},
   anchors:[
    {id:"init",kind:"stream_state_init",label:"A · 状态起点",field:"nextPosition"},
    {id:"append_a",kind:"stream_state_append",label:"B · 过期写入位置",field:"attemptedPosition"}
   ],
   edges:[["init","fork_a","state"],["fork_a","append_a","state"]]
  })
 });
 function caseOf(name){return Object.prototype.hasOwnProperty.call(cases,name)?cases[name]:null;}
 function readPath(values,path){
  return path.split(".").reduce((v,key)=>{
   if(v===null||typeof v!=="object"||!Object.prototype.hasOwnProperty.call(v,key))
    throw Error("实验结果缺少证据字段 "+path);
   return v[key];
  },values);
 }
 function anchorOf(name,id){
  const entry=caseOf(name);
  return entry?.anchors.find(a=>a.id===id)||null;
 }
 function href(name,id,branch=null,index=null){
  if(!anchorOf(name,id))throw Error("未知的计算图证据锚点");
  let url="learn.html?edition=teaching&state_case="+name+"&state_node="+id;
  if(branch!==null||index!==null){if(!["a","b"].includes(branch)||!Number.isSafeInteger(index)||index<0||index>4095)throw Error("无效坐标导航");url+="&state_branch="+branch+"&state_index="+index;}
  return url;
 }
 function validateGraph(name,graph){
  const entry=caseOf(name);
  if(!entry)throw Error("未知状态实验");
  if(!graph||!Array.isArray(graph.nodes)||!Array.isArray(graph.edges))
   throw Error("课程没有可检查的实际计算图");
  const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
  for(const a of entry.anchors){
   if(nodes.get(a.id)?.kind!==a.kind)throw Error("课程积木签名不符："+a.id+"/"+a.kind);
  }
  for(const [from,to,port] of entry.edges){
   if(!graph.edges.some(e=>e.from===from&&e.to===to&&e.port===port))
    throw Error("课程计算依赖已改变："+from+" → "+to+"."+port);
  }
  return entry;
 }
 function evidence(name,result){
  const entry=caseOf(name);
  if(!entry||!result||!result.values)throw Error("缺少统一实验执行结果");
  return entry.anchors.map(a=>Object.freeze({...a,sample:readPath(result.values,a.field)}));
 }
 function coordinateList(name,result,branch){
  const entry=caseOf(name);
  if(!entry||!["a","b"].includes(branch)||!result?.values)throw Error("无效坐标案例");
  if(branch==="b"&&["axis","reject"].includes(name))return Object.freeze([]);
  const v=result.values;
  const array=name==="stream"?(branch==="a"?v.advanced.values:v.fork.values):
   name==="video"?(branch==="a"?v.original:v.joined):
   name==="axis"?v.values:v.original;
  if(!Array.isArray(array)||!array.every(x=>typeof x==="number"&&Number.isFinite(x)))throw Error("数值坐标源缺失");
  const shape=name==="stream"?[1,3,2]:name==="video"?v.shape:
   name==="axis"?v.shape:[1,2,2];
  if(!Array.isArray(shape)||shape.reduce((a,b)=>a*b,1)!==array.length)throw Error("真实源张量形状不匹配");
  return Object.freeze(array.map((_,i)=>i));
 }
 function unravel(i,shape){
  const out=new Array(shape.length);
  for(let k=shape.length-1;k>=0;k--){out[k]=i%shape[k];i=Math.floor(i/shape[k]);}
  return out;
 }
 function coordinate(name,result,branch,index){
  const ids=coordinateList(name,result,branch);
  if(!Number.isSafeInteger(index)||index<0||index>=ids.length)throw Error("坐标超出执行张量范围");
  const v=result.values,shape=name==="stream"?[1,3,2]:
   name==="video"?v.shape:name==="axis"?v.shape:[1,2,2];
  const location=unravel(index,shape),axes=name==="video"?["B","T","H","W","C"]:
   name==="axis"?["T","D"]:["B","T","D"];
  let value,terms=[],traceNodes=[],related=null,origin;
  if(name==="stream"){
   value=(branch==="a"?v.advanced.values:v.fork.values)[index];
   const prefixLen=v.initial.prefix.length,fromPrefix=index<prefixLen;
   const source=fromPrefix?v.initial.prefix:v.appendChunks[branch];
   const sourceIndex=fromPrefix?index:index-prefixLen;
   const sourceValue=source[sourceIndex];
   if(value!==sourceValue)throw Error("stream append coordinate proof mismatch");
   terms=[{role:fromPrefix?"prefix":"append_chunk",index:sourceIndex,value:sourceValue}];
   const suffix=branch==="a"?"a":"b";
   traceNodes=fromPrefix?["prefix","init","fork_"+suffix,"append_"+suffix]:
    ["prefix","init","fork_"+suffix,"chunk_"+suffix,"append_"+suffix];
   origin=100+location[1];
  }else if(name==="video"){
   value=(branch==="a"?v.original:v.joined)[index];
   if(value!==v.original[index]||value!==v.joined[index])throw Error("video T composition coordinate mismatch");
   const time=location[1],span=shape.slice(2).reduce((a,b)=>a*b,1);
   terms=[{role:"original_video",index,value:v.original[index]},
    {role:time<2?"early_T_slice":"late_T_slice",index:time<2?index:index-2*span,value}];
   traceNodes=branch==="a"?["video"]:["video",time<2?"early":"late","join"];
  }else if(name==="axis"){
   value=v.values[index];
   const D=shape[1],d=location[1],x=v.input[index],bias=v.bias[d];
   if(x+bias!==value)throw Error("named-axis broadcast coordinate mismatch");
   terms=[{role:"input[T,D]",index,value:x},{role:"bias[D]",index:d,value:bias}];
   traceNodes=["x","bias","add"];
  }else{
   value=v.original[index];
   terms=[{role:"immutable_original",index,value}];
   traceNodes=["prefix","init"];
   origin=v.origin+location[1];
  }
  related=name==="video"?(branch==="a"?"video":"join"):
   name==="stream"?(branch==="a"?"append_a":"append_b"):
   name==="axis"?"add":"init";
  return Object.freeze({name,branch,index,coordinate:Object.freeze(location),axes:Object.freeze(axes),
   shape:Object.freeze(shape.slice()),value,terms:Object.freeze(terms),origin,
   nodeId:related,traceNodes:Object.freeze(traceNodes),
   note:"实验室中同一次确定性数值执行的坐标证明；课程使用独立样本，不声明跨页面数值相同"});
 }
 function upstreamPath(name,graph,nodeId){
  validateGraph(name,graph);
  if(!graph.nodes.some(n=>n.id===nodeId))throw Error("计算图不存在对应节点");
  const found=new Set(),work=[nodeId];
  while(work.length){
   const id=work.pop();
   if(found.has(id))continue;
   found.add(id);
   for(const e of graph.edges)if(e.to===id)work.push(e.from);
  }
  return Object.freeze(graph.nodes.filter(n=>found.has(n.id)).map(n=>n.id));
 }
 function coordinatePath(name,graph,proof){
  if(proof?.name!==name)throw Error("坐标案例不一致");
  const closure=upstreamPath(name,graph,proof.nodeId);
  if(!proof.traceNodes.every(id=>closure.includes(id)))throw Error("坐标依赖不属于实际计算图上游");
  const edges=graph.edges,selected=new Set(proof.traceNodes);
  for(const id of proof.traceNodes.filter(id=>id!==proof.nodeId)){
   if(id===proof.traceNodes[0]||!edges.some(e=>e.to===id))continue;
   if(!edges.some(e=>e.to===id&&selected.has(e.from)))
    throw Error("坐标依赖在课程中缺少连接："+id);
  }
  return Object.freeze(proof.traceNodes.slice());
 }
 function mount(host,doc){
  const query=new URLSearchParams(host.location.search);
  const name=query.get("state_case"),entry=caseOf(name);
  if(!entry)return null;
  const studio=host.SeetheTruthUnifiedWorkbench,Core=host.SeetheTruthUnifiedCore;
  const Focus=host.SeetheTruthCoordinateFocus;
  if(!Focus)throw Error("coordinate focus protocol unavailable");
  const focusSession=host.SeetheTruthCoordinateFocusSession||Focus.store();
  host.SeetheTruthCoordinateFocusSession=focusSession;
  const panel=doc.createElement("aside");panel.id="stateHandoff";panel.className="state-handoff";
  panel.setAttribute("aria-label","状态实验与计算图证据对应");
  const title=doc.createElement("strong");title.textContent="实验 → 计算图 · "+entry.title;
  const sub=doc.createElement("p");sub.className="state-handoff-caption";sub.textContent=entry.summary;
  const status=doc.createElement("p");status.setAttribute("role","status");
  const links=doc.createElement("div");links.className="state-handoff-links";
  const source=doc.createElement("p");source.className="state-handoff-provenance";
  source.textContent="机制同构 · 独立示例输入。下方坐标取自本次状态实验执行；画布结果须由当前积木图另行运行验证。";
  const challenge=doc.createElement("p");challenge.textContent=entry.challenge;
  const back=doc.createElement("a");back.href="semantic_state.html?state_case="+name;
  back.textContent="返回当前状态实验";
  let jump=null,trail=null;
  const close=doc.createElement("button");close.type="button";close.textContent="收起";
  close.className="state-handoff-close";close.addEventListener("click",()=>{panel.remove();jump?.remove();trail?.remove();});
  panel.append(title,sub,status,links,source,challenge,back,close);
  const inspector=doc.querySelector?.(".inspector");
  if(inspector&&typeof inspector.prepend==="function"){
   panel.classList.add("state-handoff-docked");
   inspector.prepend(panel);
   const stageTop=doc.querySelector?.(".stage-top");
   if(stageTop){
    jump=doc.createElement("button");jump.type="button";jump.id="stateHandoffJump";
    jump.className="state-handoff-jump";jump.textContent="实验 → 计算图 · 查看坐标证据";
    jump.addEventListener("click",()=>panel.scrollIntoView({block:"start",behavior:"auto"}));
    stageTop.append(jump);
    trail=doc.createElement("div");trail.id="stateHandoffTrail";trail.className="state-handoff-trail";
    trail.setAttribute("role","group");trail.setAttribute("aria-label","当前坐标的实际计算依赖路径");
    stageTop.append(trail);
   }
  }else doc.body.append(panel);
  try{
   const presetButton=entry.preset==="stream_state_branches"
    ?doc.getElementById("stateGraphExample")
    :doc.querySelector('[data-preset="'+entry.preset+'"]');
   if(!presetButton)throw Error("该积木课程尚未安装");
   presetButton.click();
   validateGraph(name,studio.inspect().graph);
   const accepted=Core.executeLab("semantic_state","scenario",{name});
   if(accepted.value.passed!==true)throw Error("本次状态实验条件未通过");
   Core.replay(accepted);
   const rows=evidence(name,accepted.value);
   const requested=query.get("state_node");
   const start=anchorOf(name,requested)?.id||entry.primary.a;
   const requestedBranch=query.get("state_branch"),requestedIndex=query.get("state_index");
   const controls=[];
   for(const item of rows){
    const button=doc.createElement("button");button.type="button";button.className="state-handoff-anchor";
    button.textContent=item.label+" · "+item.id;
    button.setAttribute("aria-label","定位积木 "+item.id);
    button.addEventListener("click",()=>{
     validateGraph(name,studio.inspect().graph);
     studio.focusNode(item.id);
     for(const b of controls)b.setAttribute("aria-pressed",String(b===button));
     status.textContent=item.label+" / "+item.field+" = "+JSON.stringify(item.sample)+
      "。当前数值来自状态实验，积木课程使用独立输入。";
     const closure=upstreamPath(name,studio.inspect().graph,item.id);
     for(const el of (typeof doc.querySelectorAll==="function"?doc.querySelectorAll("#blocks [data-node]"):[])){
      el.classList.toggle("state-evidence-path",closure.includes(el.dataset.node));
     }
    });
    controls.push(button);links.append(button);
   }
   controls[rows.findIndex(item=>item.id===start)].click();
   if(["a","b"].includes(requestedBranch)&&requestedIndex!==null&&/^(0|[1-9]\d*)$/.test(requestedIndex)){
    const proof=coordinate(name,accepted.value,requestedBranch,Number(requestedIndex));
    const actualPath=coordinatePath(name,studio.inspect().graph,proof);
    focusSession.select(Focus.fromState(proof));
    for(const el of (typeof doc.querySelectorAll==="function"?doc.querySelectorAll("#blocks [data-node]"):[])){
     el.classList.toggle("state-evidence-path",actualPath.includes(el.dataset.node));
    }
    if(jump)jump.textContent="实验坐标 ["+proof.coordinate.join(",")+"] · "+proof.value+" · 查看证据";
    if(trail){
     trail.replaceChildren();
     for(const [i,id] of actualPath.entries()){
      const chip=doc.createElement("button");chip.type="button";chip.className="state-handoff-path-chip";
      chip.textContent=(i?"→ ":"")+id;
      chip.setAttribute("aria-label","定位当前坐标路径上的 "+id+" 积木");
      chip.addEventListener("click",()=>{
       studio.focusNode(id);
       for(const e of doc.querySelectorAll("#blocks [data-node]"))
        e.classList.toggle("state-evidence-path",actualPath.includes(e.dataset.node));
       status.textContent="已定位 "+id+"；源实验坐标 ["+proof.coordinate.join(",")+"]，路径 "+actualPath.join(" → ")+"。";
      });
      trail.append(chip);
     }
    }
    status.textContent="实验坐标 ["+proof.coordinate.join(",")+"] = "+proof.value+
     " · 源 "+proof.terms.map(t=>t.role+"["+t.index+"]="+t.value).join(" + ")+
     "。本坐标对应的图路径："+actualPath.join(" → ")+"；课程输入独立，数值不直接等同。";
    back.href+="&state_branch="+requestedBranch+"&state_index="+proof.index;
   }
  }catch(err){status.textContent="课程关联待检查："+err.message;}
  return panel;
 }
 return Object.freeze({cases,caseOf,anchorOf,href,validateGraph,evidence,coordinateList,coordinate,upstreamPath,coordinatePath,mount});
});
