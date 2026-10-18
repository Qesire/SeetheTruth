/* Immutable, bounded N-dimensional AdamW for explicitly trainable tensor blocks.
 * Same bias correction and decoupled decay as the existing 2D AdamW.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthNDAdamW=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const ND=typeof module==="object"&&module.exports?require("./nd_tensor.js"):root.SeetheTruthND;
const same=(a,b)=>a.length===b.length&&a.every((n,i)=>n===b[i]);
const validate=(x,name)=>{const s=ND.shape(x);if(s.length!==4)throw Error(name+" must have four bounded axes");return s;};
const zero=sh=>ND.build(sh,()=>0);
function step(parameter,gradient,previous=null,options={}){
 const shape=validate(parameter,"N-D parameter");
 if(!same(shape,validate(gradient,"N-D gradient")))throw Error("N-D parameter/gradient shape mismatch");
 const h={lr:0.01,beta1:.9,beta2:.999,eps:1e-8,weightDecay:.01,...options};
 if(!Number.isFinite(h.lr)||h.lr<=0||h.lr>1||!Number.isFinite(h.beta1)||h.beta1<0||h.beta1>=1||
    !Number.isFinite(h.beta2)||h.beta2<0||h.beta2>=1||!Number.isFinite(h.eps)||h.eps<=0||h.eps>1||
    !Number.isFinite(h.weightDecay)||h.weightDecay<0||h.weightDecay>1)throw Error("invalid N-D AdamW hyperparameters");
 const t=previous?.t??0;
 if(!Number.isSafeInteger(t)||t<0||t>=10000)throw Error("N-D optimizer iteration limit");
 const m=previous?.m??zero(shape),v=previous?.v??zero(shape);
 if(!same(shape,validate(m,"first moment"))||!same(shape,validate(v,"second moment")))
   throw Error("N-D optimizer moment shape mismatch");
 if(ND.flatten(v).some(x=>x<0))throw Error("N-D second moment must be nonnegative");
 const b1=Math.pow(h.beta1,t+1),b2=Math.pow(h.beta2,t+1);
 const mt=ND.build(shape,i=>h.beta1*ND.at(m,i)+(1-h.beta1)*ND.at(gradient,i));
 const vt=ND.build(shape,i=>h.beta2*ND.at(v,i)+(1-h.beta2)*ND.at(gradient,i)**2);
 const update=ND.build(shape,i=>{
   const w=ND.at(parameter,i),mhat=ND.at(mt,i)/(1-b1),vhat=ND.at(vt,i)/(1-b2);
   return h.lr*mhat/(Math.sqrt(vhat)+h.eps)+h.lr*h.weightDecay*w;
 });
 const next=ND.build(shape,i=>ND.at(parameter,i)-ND.at(update,i));
 return {parameter:next,update,state:{t:t+1,m:mt,v:vt},options:h,
   evidence:"immutable N-D AdamW with bias correction and decoupled decay; not a real checkpoint update"};
}
return Object.freeze({step});
});
