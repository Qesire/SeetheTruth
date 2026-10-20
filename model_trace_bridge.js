/* Real-model trace ↔ Scratch block reference bridge.
 * Observations are NOT computational operands. Samples do not reconstruct
 * complete tensors, and imported JSON alone does not certify provenance.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthTraceBridge=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const Trace=typeof module==="object"&&module.exports?require("./model_trace.js"):root.SeetheTruthTrace;
const ALLOWED=Object.freeze({
 linear:"Linear 数学参考：nn.Linear 权重是 [out,in]；工作台的 Linear 使用固定教学参数",
 matmul:"矩阵乘法参考：nn.Linear 的 W[out,in] 在 x @ Wᵀ 中转置；不等于当前积木的权重",
 parameter:"参数参考：只有权重与梯度的统计和扁平化样本，不包含完整训练参数",
});
const escape=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function describe(trace,block){
 const result=Trace.validate(trace);
 if(!block||!Object.hasOwn(ALLOWED,block.kind)||typeof block.id!=="string")
   throw Error("请选择 Linear、MatMul 或 Parameter 积木关联 Linear 模型观测；其他算子不应误用这些证据");
 const p=trace.provenance;
 if(!/^[A-Za-z0-9_-]{1,36}$/.test(block.id))throw Error("invalid graph node identity");
 return Object.freeze({nodeId:block.id,nodeKind:block.kind,summary:result,
   modelId:p.modelId,revision:p.revision,modulePath:p.modulePath,role:ALLOWED[block.kind],
   observedWeightShape:trace.parameter.weight.shape.slice(),
   observedGradientShape:trace.parameter.gradient.shape.slice(),
   // This remains descriptive only. No fake full operands, no execution override.
   relation:"read_only_real_model_reference_not_a_graph_operand",
   evidenceTier:"REPLAY_REQUIRES_PINNED_MODEL_EXECUTION"});
}
function cells(sample,limit=16){
 if(!Array.isArray(sample)||sample.some(x=>!Number.isFinite(x)))throw Error("invalid sample");
 const data=sample.slice(0,limit),peak=Math.max(1e-12,...data.map(Math.abs));
 return '<div class="real-sample-values" role="img" aria-label="前 '+data.length+' 个有序扁平化采样值">'+
  data.map(v=>'<span class="real-sample-cell" title="'+escape(v)+'" style="background:hsla('+
  (v>=0?181:20)+',70%,45%,'+(0.14+0.7*Math.abs(v)/peak)+')">'+escape(Number(v).toPrecision(3))+'</span>').join("")+
  '</div>';
}
function render(trace,block){
 const info=describe(trace,block),w=info.summary.weight,g=info.summary.gradient;
 const stat=(name,item)=>'<div class="real-sample-metric"><span>'+escape(name)+'</span><b>'+
   escape(item.rms.toPrecision(5))+'</b><small>RMS · ‖·‖₂ '+escape(item.l2.toPrecision(5))+
   ' · zero '+escape((item.zeroFraction*100).toFixed(2))+'%</small></div>';
 const activation=(key,data)=>'<div class="real-sample-title">'+escape(key)+' · 已截断的真实观测样本（不是可重放操作数）</div>'+cells(data);
 return '<div class="real-sample-header"><strong>模型观测参考 · 只读</strong><span>CLAIMED REAL_CAPTURE / 待复跑来源</span></div>'+
  '<p class="real-sample-hint">关联节点 '+escape(info.nodeId)+' · '+escape(info.role)+'</p>'+
  '<p class="real-sample-source">'+escape(info.modelId)+' @ '+escape(info.revision)+'</p>'+
  '<p class="real-sample-source">源模块：'+escape(info.modulePath)+' · nn.Linear W['+escape(w.shape.join("×"))+'] · loss '+escape(info.summary.loss.toPrecision(6))+'</p>'+
  '<div class="real-sample-grid">'+stat("真实权重",w)+stat("真实梯度",g)+'</div>'+
  '<div class="real-sample-title">W 的前 '+Math.min(16,w.sample.length)+' 个扁平元素（全部 '+w.count+' 个）</div>'+cells(w.sample)+
  '<div class="real-sample-title">∂L/∂W 的前 '+Math.min(16,g.sample.length)+' 个扁平元素（全部 '+g.count+' 个）</div>'+cells(g.sample)+
  activation("输入激活",info.summary.activations.input)+activation("输出激活",info.summary.activations.output)+
  '<p class="real-sample-warning">这不是当前积木的实际权重或计算输出；不修改执行图、参数、优化器、导出代码。JSON 合同验证不能独立证明来源真实性，需按不可变 revision 复跑。</p>';
}
return Object.freeze({describe,render,supportedKinds:Object.freeze(Object.keys(ALLOWED))});
});
