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
 function href(name,id){
  if(!anchorOf(name,id))throw Error("未知的计算图证据锚点");
  return "learn.html?edition=teaching&state_case="+name+"&state_node="+id;
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
 function mount(host,doc){
  const query=new URLSearchParams(host.location.search);
  const name=query.get("state_case"),entry=caseOf(name);
  if(!entry)return null;
  const studio=host.SeetheTruthUnifiedWorkbench,Core=host.SeetheTruthUnifiedCore;
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
  const close=doc.createElement("button");close.type="button";close.textContent="收起";
  close.className="state-handoff-close";close.addEventListener("click",()=>panel.remove());
  panel.append(title,sub,status,links,source,challenge,back,close);doc.body.append(panel);
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
    });
    controls.push(button);links.append(button);
   }
   controls[rows.findIndex(item=>item.id===start)].click();
  }catch(err){status.textContent="课程关联待检查："+err.message;}
  return panel;
 }
 return Object.freeze({cases,caseOf,anchorOf,href,validateGraph,evidence,mount});
});
