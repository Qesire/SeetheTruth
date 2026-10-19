/* Four explicit educational optimizer strategies on a common immutable state:
 * {kind,t,m,v}. Legacy AdamW states with no kind are interpreted as AdamW.
 * No 2D/4D restriction except the existing catalog tensor and graph bounds.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthOptimizerRules=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const isNode=typeof module==="object"&&module.exports;
const W=isNode?require("./adamw_micro.js"):root.SeetheTruthAdamW;
const NDW=isNode?require("./nd_adamw_micro.js"):root.SeetheTruthNDAdamW;
const KINDS=Object.freeze(["adamw","sgd_momentum","rmsprop","lion"]);
const tensorShape=x=>{const sh=[];let v=x;while(Array.isArray(v)){sh.push(v.length);v=v[0];}return sh;};
const shapeOK=x=>{const sh=tensorShape(x);if(sh.length!==2&&sh.length!==4)throw Error("optimizer supports only 2D and 4D teaching tensors");if(sh.some(n=>!Number.isSafeInteger(n)||n<1))throw Error("invalid tensor dimensions");const all=[];const walk=(v,d)=>{if(d===sh.length){if(!Number.isFinite(v))throw Error("optimizer tensor must be finite");all.push(v);return;}if(!Array.isArray(v)||v.length!==sh[d])throw Error("ragged optimizer tensor");v.forEach(t=>walk(t,d+1));};walk(x,0);if(all.length>512)throw Error("optimizer tensor exceeds bounded allowance");return sh;};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
const map=(shape,fn)=>{const recur=(axis,idx)=>axis===shape.length?fn(idx):Array.from({length:shape[axis]},(_,i)=>recur(axis+1,[...idx,i]));return recur(0,[]);};
const at=(a,idx)=>idx.reduce((v,i)=>v[i],a);
function step(parameter,gradient,previous=null,options={}){
 const a=shapeOK(parameter),b=shapeOK(gradient);
 if(!same(a,b))throw Error("optimizer parameter / gradient shape mismatch");
 const kind=options.optimizer||"adamw";
 if(!KINDS.includes(kind))throw Error("unsupported optimizer strategy");
 const old=previous||{t:0,m:map(a,()=>0),v:map(a,()=>0)},previousKind=old.kind||"adamw";
 if(!KINDS.includes(previousKind))throw Error("unrecognized persisted optimizer state kind");
 if(previousKind!==kind&&old.t!==0)throw Error("optimizer changed with nonzero state; explicitly reset moments first");
 const ms=shapeOK(old.m),vs=shapeOK(old.v);
 if(!same(ms,a)||!same(vs,a))throw Error("optimizer moment dimensions mismatch");
 if(!Number.isSafeInteger(old.t)||old.t<0||old.t>=10000)throw Error("invalid optimizer state step");
 const h={lr:0.01,weightDecay:0.01,beta1:0.9,beta2:kind==="lion"?0.99:0.999,alpha:0.99,eps:1e-8,momentum:0.9,...options};
 if(!Number.isFinite(h.lr)||h.lr<=0||h.lr>1||!Number.isFinite(h.weightDecay)||h.weightDecay<0||h.weightDecay>1||
    !Number.isFinite(h.momentum)||h.momentum<0||h.momentum>=1||!Number.isFinite(h.alpha)||h.alpha<0||h.alpha>=1||
    !Number.isFinite(h.beta1)||h.beta1<0||h.beta1>=1||!Number.isFinite(h.beta2)||h.beta2<0||h.beta2>=1||
    !Number.isFinite(h.eps)||h.eps<=0||h.eps>1)throw Error("invalid optimizer hyperparameters");
 if(kind==="adamw"){
   const base=(a.length===4?NDW:W).step(parameter,gradient,old,h);
   return {...base,state:{...base.state,kind},optimizer:kind};
 }
 // Unlike AdamW, SGD and RMSprop use coupled L2 decay, as in PyTorch.
 // Lion uses decoupled weight decay and two different momentum mixtures.
 const update=map(a,idx=>{
   const w=at(parameter,idx),g=at(gradient,idx),m=at(old.m,idx),v=at(old.v,idx),effective=g+h.weightDecay*w;
   if(kind==="sgd_momentum"){
     const buffer=h.momentum*m+effective;
     return h.lr*buffer;
   }
   if(kind==="rmsprop"){
     const square=h.alpha*v+(1-h.alpha)*effective*effective;
     return h.lr*effective/(Math.sqrt(square)+h.eps);
   }
   const c=h.beta1*m+(1-h.beta1)*g;
   return h.lr*(Math.sign(c)+h.weightDecay*w);
 });
 const nextM=map(a,idx=>{
   const w=at(parameter,idx),g=at(gradient,idx),m=at(old.m,idx);
   if(kind==="sgd_momentum")return h.momentum*m+g+h.weightDecay*w;
   if(kind==="lion")return h.beta2*m+(1-h.beta2)*g;
   return m;
 });
 const nextV=map(a,idx=>{
   if(kind!=="rmsprop")return at(old.v,idx);
   const g=at(gradient,idx)+h.weightDecay*at(parameter,idx);
   return h.alpha*at(old.v,idx)+(1-h.alpha)*g*g;
 });
 const next=map(a,idx=>at(parameter,idx)-at(update,idx));
 shapeOK(next);
 return {parameter:next,update,state:{kind,t:old.t+1,m:nextM,v:nextV},options:h,optimizer:kind,
  evidence:"SGD/RMSprop coupled L2; Lion decoupled sign step; CPU-only immutable teaching state"};
}
return Object.freeze({step,KINDS});
});
