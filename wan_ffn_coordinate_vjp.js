/* Pinned Wan FFN tiny-slice scalar VJP, using the same output coordinate
 * report contract as free-graph ElementLineage. This is source-aligned
 * synthetic Float64, not a model-checkpoint gradient or executable GPU VJP.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const api=factory(isNode?{
  Trace:require("./wan_ffn_trace.js"),Microscope:require("./compute_microscope.js"),
  Activation:require("./activation_math.js")
 }:{
  Trace:root.SeetheTruthWanFFNTrace,Microscope:root.SeetheTruthComputeMicroscope,
  Activation:root.SeetheTruthActivations
 });
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthWanFFNCoordinateVJP=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
const SCHEMA="seethetruth-coordinate-vjp-bridge-v1";
const fail=msg=>{throw Error("Wan source-pinned coordinate VJP: "+msg);};
const requireThat=(ok,msg)=>{if(!ok)fail(msg);};
const get=(a,width,t,c)=>a[t*width+c];
function inspect(data,frame,{stepId,coordinate,options}={}){
 requireThat(d.Trace&&d.Microscope&&d.Activation,"missing source-aligned dependencies");
 requireThat(options&&typeof options==="object","last accepted controller input required");
 const allowed=["tokens","dim","hidden","seed","gate","scale","shift"];
 requireThat(Object.keys(options).length===allowed.length&&allowed.every(k=>Object.hasOwn(options,k)),
  "unbound or incomplete controller input");
 // Re-run from the exact accepted controller input, not the displayed values.
 const original=d.Trace.trace(options);
 requireThat(JSON.stringify(original)===JSON.stringify(data),"controller replay does not match selected trace");
 const canonical=d.Microscope.wan(original);
 requireThat(JSON.stringify(canonical)===JSON.stringify(frame),"microscope does not match replayed pinned source slice");
 requireThat(canonical.evidenceTier===d.Microscope.TIERS.WAN&&
  canonical.origin==="source_aligned_micro_numeric_engine","invalid claim/evidence tier");
 const stage=d.Trace.STEPS.findIndex(s=>s.id===stepId);
 requireThat(stage>=0&&Array.isArray(coordinate)&&coordinate.length===2,"invalid source stage/coordinate");
 const c=data.config,T=c.tokens,C=c.dim,H=c.hidden;
 const [t,k]=coordinate,width=stage===0?6*C:stage===4||stage===5?H:C;
 requireThat(Number.isSafeInteger(t)&&t>=0&&t<T&&Number.isSafeInteger(k)&&k>=0&&k<width,
  "selected coordinate out of bounds");
 const g=Array.from({length:9},(_,i)=>new Array(T*data.steps[i].main.shape[1]).fill(0));
 g[stage][t*width+k]=1;
 const fields={afterSelf:new Array(T*C).fill(0),crossDelta:new Array(T*C).fill(0),
  e0:new Array(T*6*C).fill(0),modulation:new Array(6*C).fill(0),
  w1:new Array(H*C).fill(0),b1:new Array(H).fill(0),
  w2:new Array(C*H).fill(0),b2:new Array(C).fill(0),
  controlGate:[0],controlScale:[0],controlShift:[0]};
 const edges=[],stages=[];
 const link=(port,index,factor,rule)=>edges.push({port,source:port+"@"+index.join(","),localDerivative:factor,rule,term:""});
 const pushStage=(name,extra={})=>stages.push({stage:name,...extra});
 const s=data.steps.map(x=>x.main.values),e=data.e;
 // Reverse across exactly the first (stage+1) actual pinned forward operations.
 for(let i=stage;i>=0;i--){
  if(i===8)for(let j=0;j<T*C;j++){
   const v=g[8][j];g[1][j]+=v;g[7][j]+=v;
   if(j===t*C+k&&stage===8){
    link("cross",[t,k],1,"x residual bypass");
    link("gate",[t,k],1,"post-FFN delta");
    pushStage("wan_residual_add",crossValue:s[1][j],gatedFFN:s[7][j]);
   }
  }
  if(i===7)for(let j=0;j<T*C;j++){
   const v=g[7][j],gain=c.gate*e[5][j];
   g[6][j]+=v*gain;
   const token=Math.floor(j/C),chan=j%C;
   g[0][token*6*C+5*C+chan]+=v*s[6][j]*c.gate;
   fields.controlGate[0]+=v*s[6][j]*e[5][j];
   if(j===t*C+k&&stage===7){
    link("down",[t,k],gain,"down[t,c] × demoGate × e5[t,c]");
    link("e5",[t,k],s[6][j]*c.gate,"e5 is post-projection output gate");
    pushStage("wan_output_gate",down:s[6][j],e5:e[5][j],
     demoMultiplier:c.gate,localDownDerivative:gain);
   }
  }
  if(i===6)for(let j=0;j<T*C;j++){
   const v=g[6][j],token=Math.floor(j/C),out=j%C;
   fields.b2[out]+=v;
   for(let h=0;h<H;h++){
    const w=c.w2[out*H+h],activation=s[5][token*H+h];
    g[5][token*H+h]+=v*w;fields.w2[out*H+h]+=v*activation;
   }
   if(j===t*C+k&&stage===6){
    for(let h=0;h<H;h++){
     link("gelu",[t,h],c.w2[k*H+h],"Linear2 weight [out,hidden]");
     link("w2",[k,h],s[5][t*H+h],"∂Linear2/∂weight=hidden activation");
    }
    link("b2",[k],1,"Linear2 bias");
    pushStage("wan_linear2",inputWidth:H,outputChannel:k,bias:c.b2[k]);
   }
  }
  if(i===5)for(let j=0;j<T*H;j++){
   const upstream=s[4][j],slope=d.Activation.derivative("gelu_tanh",upstream);
   g[4][j]+=g[5][j]*slope;
   if(j===t*H+k&&stage===5){
    link("up",[t,k],slope,"GELU(tanh) exact analytic derivative");
    pushStage("wan_gelu_tanh",input:upstream,activation:s[5][j],derivative:slope);
   }
  }
  if(i===4)for(let j=0;j<T*H;j++){
   const v=g[4][j],token=Math.floor(j/H),h=j%H;
   fields.b1[h]+=v;
   for(let ch=0;ch<C;ch++){
    g[3][token*C+ch]+=v*c.w1[h*C+ch];
    fields.w1[h*C+ch]+=v*s[3][token*C+ch];
   }
   if(j===t*H+k&&stage===4){
    for(let ch=0;ch<C;ch++){
     link("affine",[t,ch],c.w1[k*C+ch],"Linear1 weight [hidden,input]");
     link("w1",[k,ch],s[3][t*C+ch],"∂Linear1/∂weight=conditioned activation");
    }
    link("b1",[k],1,"Linear1 bias");
    pushStage("wan_linear1",inputWidth:C,hiddenChannel:k,bias:c.b1[k]);
   }
  }
  if(i===3)for(let j=0;j<T*C;j++){
   const v=g[3][j],token=Math.floor(j/C),ch=j%C,z=s[2][j],
       gain=1+c.scale*e[4][j];
   g[2][j]+=v*gain;
   g[0][token*6*C+4*C+ch]+=v*z*c.scale;
   g[0][token*6*C+3*C+ch]+=v*c.shift;
   fields.controlScale[0]+=v*z*e[4][j];
   fields.controlShift[0]+=v*e[3][j];
   if(j===t*C+k&&stage===3){
    link("norm",[t,k],gain,"1 + scaleMultiplier × e4[t,c]");
    link("e4",[t,k],z*c.scale,"normalized × scaleMultiplier");
    link("e3",[t,k],c.shift,"shiftMultiplier × e3");
    pushStage("wan_conditional_affine",normalized:z,e3:e[3][j],e4:e[4][j],
     scaleMultiplier:c.scale,shiftMultiplier:c.shift);
   }
  }
  if(i===2)for(let token=0;token<T;token++){
   const vals=s[1].slice(token*C,(token+1)*C),
    mean=vals.reduce((a,b)=>a+b,0)/C,
    centered=vals.map(x=>x-mean),
    variance=centered.reduce((a,b)=>a+b*b,0)/C,
    inv=1/Math.sqrt(variance+d.Trace.SOURCE.eps),
    squaredInv=inv*inv,normalized=centered.map(v=>v*inv);
   const grads=g[2].slice(token*C,(token+1)*C),
    avg=grads.reduce((a,b)=>a+b,0)/C,
    avgGZ=grads.reduce((a,b,j)=>a+b*normalized[j],0)/C;
   for(let ch=0;ch<C;ch++)g[1][token*C+ch]+=inv*(grads[ch]-avg-normalized[ch]*avgGZ);
   if(token===t&&stage===2){
    for(let ch=0;ch<C;ch++){
     const factor=inv*((ch===k?1:0)-1/C-normalized[k]*normalized[ch]/C);
     link("cross",[t,ch],factor,"LayerNorm across C, including mean/variance covariance");
    }
    pushStage("wan_layernorm_not_rmsnorm",mean,variance,epsilon:d.Trace.SOURCE.eps,
     invStd:inv,normalizedSelected:normalized[k]);
   }
  }
  if(i===1)for(let j=0;j<T*C;j++){
   const v=g[1][j];fields.afterSelf[j]+=v;fields.crossDelta[j]+=v;
   if(j===t*C+k&&stage===1){
    link("afterSelf",[t,k],1,"output of self-attention branch");
    link("crossDelta",[t,k],1,"EXTERNAL cross-attention delta, not computed here");
    pushStage("wan_cross_attention_external_residual",base:c.afterSelf[j],externalDelta:c.crossDelta[j]);
   }
  }
  if(i===0)for(let token=0;token<T;token++)for(let group=0;group<6;group++)
   for(let ch=0;ch<C;ch++){
    const flat=token*6*C+group*C+ch,grad=g[0][flat];
    fields.e0[flat]+=grad;fields.modulation[group*C+ch]+=grad;
    if(stage===0&&token===t&&group*C+ch===k){
     link("e0",[t,group,ch],1,"per-token six-way conditioning");
     link("modulation",[group,ch],1,"shared modulation across tokens");
     pushStage("wan_six_chunk_condition",group,channel:ch,
      conditioning:e[group][t*C+ch]);
    }
   }
 }
 // Every source value is retained with the exact coordinate identity.
 const shapes={afterSelf:[T,C],crossDelta:[T,C],e0:[T,6,C],modulation:[6,C],
  w1:[H,C],b1:[H],w2:[C,H],b2:[C],controlGate:[1],controlScale:[1],controlShift:[1]};
 const flatIndex=(n,shape)=>{
  const out=new Array(shape.length);
  for(let axis=shape.length-1;axis>=0;axis--){out[axis]=n%shape[axis];n=Math.floor(n/shape[axis]);}
  return out;
 };
 const sources=Object.entries(fields).flatMap(([name,v])=>v.flatMap((derivative,i)=>{
  if(derivative===0)return [];
  requireThat(Number.isFinite(derivative),"nonfinite scalar sensitivity");
  return [{nodeId:name,kind:name.startsWith("control")?"demo_counterfactual_control":
   name==="crossDelta"?"external_upstream_value":"source_aligned_synthetic_operand",
   index:flatIndex(i,shapes[name]),sensitivity:derivative}];
 })).sort((a,b)=>Math.abs(b.sensitivity)-Math.abs(a.sensitivity)||
  a.nodeId.localeCompare(b.nodeId));
 const dot=d.Microscope.dot(frame,stepId,t,k);
 const derivativeWitness=Object.fromEntries(Object.entries(fields).map(([field,v])=>[field,v.slice()]));
 return Object.freeze({
  schema:SCHEMA,evidence:"PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64_NOT_MODEL_GRADIENT",
  source:{revision:d.Trace.SOURCE.commit,path:d.Trace.SOURCE.path,
   role:"SOURCE_ALIGNED_FFN_SUBGRAPH_ONLY"},
  selected:{nodeId:stepId,kind:stage===4||stage===6?"linear":stepId,index:[t,k],
   value:dot.value,forwardReconstructable:dot.reconstructable,
   forwardError:dot.reconstructable?dot.error:null},
  backward:{complete:true,truncated:false,boundaryNodes:[],nodeCount:
   g.reduce((sum,v)=>sum+v.filter(x=>x!==0).length,0),
   edgeCount:edges.length,localEdges:edges,sources,sourceCount:sources.length,
   internalStages:stages,discreteConditional:false,conditionalCuts:[],
   externalBoundary:"crossDelta is supplied, upstream cross-attention is outside this slice",
   note:"Unit-seeded analytic reverse-mode for the pinned synthetic FFN slice, including LayerNorm and e3/e4/e5; not real checkpoint gradients."},
  // The extra full source tensor is intended for independent VJP oracles and
  // exact per-coordinate consumers; it is never conflated with training loss.
  sensitivities:derivativeWitness,
  loss:{status:"NO_LOSS_OBJECTIVE"},
  codegen:{numpyOperatorEligible:false,torchOperatorEligible:false,
   wholeGraphExport:"NOT_VERIFIED",vjpExport:"NOT_CLAIMED"}
 });
}
return Object.freeze({SCHEMA,inspect});
});
