/* UI-independent reusable arithmetic for shared-parameter, MoE and PPO lessons.
 * The isolated micro engine is the only source of numeric/VJP truth.
 */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?require("./micro_training_flows.js"):root.SeetheTruthMicroTrain);
 if(node)module.exports=api;
 if(root)root.SeetheTruthMicroScenarios=api;
})(typeof window!=="undefined"?window:null,function(T){"use strict";
if(!T)throw Error("micro training reference engine unavailable");
const names=Object.freeze(["shared","moe","ppo"]);
const read=(p,k)=>{const v=p[k];if(typeof v!=="number"||!Number.isFinite(v))throw Error("finite numerical field required: "+k);return v;};
function execute(name,parameters={}){
 if(!names.includes(name))throw Error("unknown micro training scenario "+name);
 if(!parameters||typeof parameters!=="object"||Array.isArray(parameters))throw Error("invalid micro training parameters");
 const v=k=>read(parameters,k);
 let result,summary,mechanism;
 if(name==="shared"){
  const tape=new T.Tape(),x=tape.input([[v("x1"),v("x2")]],"x");
  const weights=[[v("w1")],[v("w2")]];
  const w=tape.parameter("W",weights),a=tape.matmul(x,w);
  const b=tape.matmul(x,tape.parameter("W",weights));
  const sum=tape.add(a,b),loss=tape.meanSquare(sum,[[v("target")]]);
  tape.backward(loss);
  result={prediction:tape.value(sum),loss:tape.value(loss),gradients:tape.gradients(),events:tape.events};
  summary={loss:tape.value(loss)[0][0],dW1:tape.gradients().W[0][0],dW2:tape.gradients().W[1][0],sharedParameters:tape.parameters.size};
  mechanism=["前向：x → MatMul(W) 两次 → Add → MSE","反向：两个 MatMul 的 VJP 必须对同一 W 求和"];
 }else if(name==="moe"){
  const capacity=v("capacity");
  if(!Number.isSafeInteger(capacity)||capacity<1||capacity>2)throw Error("容量仅限 1 或 2");
  result=T.moeBackward([[1,2],[2,1]],[[[1],[0]],[[0],[1]]],[[v("r"),0],[0,5]],[[0],[0]],{capacity});
  summary={loss:result.loss,expert0:result.counts[0],expert1:result.counts[1],overflow:result.routed.filter(x=>!x.accepted).length};
  mechanism=["前向：router logits → Softmax → Top-1 dispatch → expert matmul","反向：每个专家局部 W 只接收实际 routed token 的梯度","router 梯度 = null，不能误称为零梯度"];
 }else{
  result=T.ppo([v("logit"),0],[0,0],0,v("adv"),v("eps"));
  summary={ratio:result.ratio,loss:result.loss,gradAction:result.gradLogits[0],clipped:!result.active};
  mechanism=["策略快照：old logits=[0,0]；action=0","L=-min(rA,clip(r,1-ε,1+ε)A)","仅在未裁剪分支保留 policy logits 梯度"];
 }
 return {scenario:name,result,summary,mechanism,contract:"bounded_cpu_numeric_not_gpu_trace"};
}
return Object.freeze({names,execute});
});
