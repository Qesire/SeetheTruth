/* Node-addressable backward inspection; does not alter the VJP or optimizer. */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?{L:require("./learning_lab.js"),Grad:require("./graph_autodiff.js"),IR:require("./execution_ir.js")}:
 {L:root.SeetheTruthLearning,Grad:root.SeetheTruthGraphAutodiff,IR:root.SeetheTruthExecutionIR});
 if(node)module.exports=api;
 if(root)root.SeetheTruthGradientInspection=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
const dims=a=>{const s=[];while(Array.isArray(a)){s.push(a.length);a=a[0];}return s;};
const stats=a=>{let count=0,square=0,max=0,nonzero=0;
 const walk=x=>{if(Array.isArray(x)){for(const v of x)walk(v);}
 else{if(typeof x!=="number"||!Number.isFinite(x))throw Error("nonfinite gradient");
 count++;square+=x*x;max=Math.max(max,Math.abs(x));if(x!==0)nonzero++;}};
 walk(a);return Object.freeze({elements:count,l2:Math.sqrt(square),maxAbs:max,nonzero});};
function inspect(raw,options={}){
 const graph=d.L.validateGraph(raw),forward=options.forward||d.L.evaluate(graph);
 const ir=d.IR.snapshot(graph,{trace:forward});
 const backward=d.Grad.backward(graph,forward,{targetId:options.targetId});
 const active=new Set(backward.pathIds);
 const nodes=ir.nodes.map(n=>{
  const gradient=backward.gradients[n.id];
  if(gradient===undefined)throw Error("missing VJP record "+n.id);
  const shape=dims(gradient);
  if(JSON.stringify(shape)!==JSON.stringify(n.shape))throw Error("gradient shape mismatch "+n.id);
  return Object.freeze({id:n.id,kind:n.kind,shape:n.shape,
   onLossPath:active.has(n.id),gradient:stats(gradient),
   localRule:backward.local[n.id]||null});
 });
 return Object.freeze({schema:"seethetruth-gradient-inspection-v1",evidence:"SYNTHETIC",
  targetId:backward.targetId,loss:backward.loss,
  nodes:Object.freeze(nodes),cutEdges:backward.cutEdges,forwardIR:ir});
}
return Object.freeze({inspect});
});
