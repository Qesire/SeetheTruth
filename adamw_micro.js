/* Bounded AdamW optimizer state and one-step preview for explicit small matrices.
 * Stateful math is returned immutably; UI never silently modifies the forward graph.
 */
(function(root,factory){const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthAdamW=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
const validate=(a,name)=>{if(!Array.isArray(a)||!a.length||a.length>8||!Array.isArray(a[0])||!a[0].length||a[0].length>16||a.some(row=>!Array.isArray(row)||row.length!==a[0].length||row.some(x=>!Number.isFinite(x))))throw Error(name+" must be finite 2D teaching tensor");return a;};
const zeros=a=>a.map(row=>row.map(()=>0));
function step(parameter,gradient,previous=null,options={}){
 validate(parameter,"parameter");validate(gradient,"gradient");
 if(gradient.length!==parameter.length||gradient[0].length!==parameter[0].length)throw Error("gradient / parameter shape mismatch");
 const h={lr:0.01,beta1:0.9,beta2:0.999,eps:1e-8,weightDecay:0.01,...options};
 if(!Number.isFinite(h.lr)||h.lr<=0||h.lr>1||!Number.isFinite(h.beta1)||h.beta1<0||h.beta1>=1||!Number.isFinite(h.beta2)||h.beta2<0||h.beta2>=1||!Number.isFinite(h.eps)||h.eps<=0||h.eps>1||!Number.isFinite(h.weightDecay)||h.weightDecay<0||h.weightDecay>1)throw Error("invalid AdamW optimizer hyperparameters");
 const t=previous?.t||0;
 if(!Number.isSafeInteger(t)||t<0||t>1000000)throw Error("invalid AdamW step index");
 const m=previous?.m||zeros(parameter),v=previous?.v||zeros(parameter);validate(m,"first moment");validate(v,"second moment");
 if(m.length!==parameter.length||v.length!==parameter.length||m[0].length!==parameter[0].length||v[0].length!==parameter[0].length)throw Error("AdamW moment state shape mismatch");
 const nextM=zeros(parameter),nextV=zeros(parameter),nextParam=zeros(parameter),update=zeros(parameter);
 for(let i=0;i<parameter.length;i++)for(let j=0;j<parameter[0].length;j++){
   const g=gradient[i][j],w=parameter[i][j];
   const mt=h.beta1*m[i][j]+(1-h.beta1)*g;
   const vt=h.beta2*v[i][j]+(1-h.beta2)*g*g;
   const mHat=mt/(1-Math.pow(h.beta1,t+1)),vHat=vt/(1-Math.pow(h.beta2,t+1));
   const delta=h.lr*mHat/(Math.sqrt(vHat)+h.eps)+h.lr*h.weightDecay*w;
   nextM[i][j]=mt;nextV[i][j]=vt;update[i][j]=delta;nextParam[i][j]=w-delta;
   if(!Number.isFinite(nextParam[i][j]))throw Error("AdamW generated nonfinite update");
 }
 return {parameter:nextParam,state:{t:t+1,m:nextM,v:nextV},update,options:h,
  evidence:"deterministic decoupled AdamW optimizer math, detached teaching preview; no model checkpoint modification"};
}
return Object.freeze({step});
});
