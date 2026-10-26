/* Scenario handoff: explicit allowlist, existing presets and existing node focus only. */
(function(){"use strict";
const cases=Object.freeze({
 stream:{preset:"stream_state_branches",kinds:["stream_state_fork","stream_state_append","stream_state_read"],title:"KV 前缀与状态分叉"},
 reject:{preset:"stream_state_branches",kinds:["stream_state_append"],title:"状态追加位置"},
 video:{preset:"video_causal_conv3d",kinds:["video_causal_conv3d"],title:"视频时间轴计算"},
 axis:{preset:"stream_state_branches",kinds:["nd_broadcast_add","stream_state_read"],title:"具名轴与流状态"}
});
const query=new URLSearchParams(window.location.search);
const scenario=query.get("state_case");
if(!Object.prototype.hasOwnProperty.call(cases,scenario))return;
const bridge=cases[scenario],studio=window.SeetheTruthUnifiedWorkbench;
const panel=document.createElement("aside");panel.id="stateHandoff";
panel.style.cssText="position:fixed;bottom:16px;left:16px;z-index:1000;max-width:min(410px,calc(100vw - 32px));padding:14px 16px;border:1px solid #77ddda;border-radius:12px;background:#102e40;color:#e6fbff;box-shadow:0 12px 42px #0009;font:13px/1.65 system-ui";
const title=document.createElement("strong");title.textContent="实验 → 计算图 · "+bridge.title;
const body=document.createElement("p");body.style.margin="6px 0";body.setAttribute("role","status");
const back=document.createElement("a");back.href="semantic_state.html?state_case="+scenario;back.textContent="返回原实验";back.style.color="#8ee8e7";
const close=document.createElement("button");close.type="button";close.textContent="关闭提示";close.style.cssText="margin-left:12px;background:transparent;color:inherit;border:1px solid #5c8797;border-radius:6px;padding:5px 9px";
close.addEventListener("click",()=>panel.remove());panel.append(title,body,back,close);document.body.append(panel);
try{
 const target=document.querySelector('[data-preset="'+bridge.preset+'"]');
 if(bridge.preset==="stream_state_branches")document.getElementById("stateGraphExample").click();
 else if(target)target.click();
 else throw Error("目标课程尚未安装");
 const graph=studio.inspect().inputSnapshot?.graph||studio.inspect().graph;
 const node=graph?.nodes?.find(n=>bridge.kinds.includes(n.kind));
 if(node){studio.focusNode(node.id);body.textContent="已加载对应课程并定位 "+node.kind+"。继续选中端口、逐坐标观察或修改输入。";}
 else body.textContent="已加载课程。请从画布上选择相关节点，沿计算依赖追踪。";
}catch(e){body.textContent="自动定位未完成："+e.message+"。可手动选择课程。";}
})();