/* Validate provenance on imported model traces; never infer authenticity from numeric plausibility. */
(function(root,fn){const api=fn();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthTrace=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
function validate(t){if(!t||t.schema!=="seethetruth-real-model-trace-v1")throw Error("unsupported trace schema");
const p=t.provenance;if(!p||p.kind!=="real_model_execution"||typeof p.modelId!=="string"||!p.modelId||!/^[a-f0-9]{40}$/i.test(p.revision||"")||p.device!=="cpu"||p.moduleType!=="torch.nn.Linear"||!p.modulePath)throw Error("missing immutable model provenance");
if(!Number.isFinite(t.loss)||!t.parameter||t.parameter.identity!==p.modulePath+".weight")throw Error("invalid model trace");
for(const k of ["weight","gradient"]){const o=t.parameter[k];if(!o||!Array.isArray(o.shape)||!o.shape.length||o.shape.some(n=>!Number.isInteger(n)||n<1)||!Number.isInteger(o.count)||o.count<1||!Array.isArray(o.sample)||o.sample.length<1||o.sample.length>256||o.sample.some(x=>!Number.isFinite(x))||["mean","rms","min","max","l2","zeroFraction"].some(v=>!Number.isFinite(o[v])))throw Error("invalid "+k+" statistics");}
const product=a=>a.reduce((x,y)=>x*y,1);
const parameter=t.parameter;
if(parameter.weight.shape.join(",")!==parameter.gradient.shape.join(","))
 throw Error("weight and gradient shapes must match the same parameter");
for(const key of ["weight","gradient"]){
 const o=parameter[key],tol=1e-4;
 if(product(o.shape)!==o.count||o.sample.length>o.count)
   throw Error("tensor element count does not agree with its shape");
 if(o.min>o.max||o.rms<0||o.l2<0||o.zeroFraction<0||o.zeroFraction>1||
    o.mean<o.min-tol||o.mean>o.max+tol||o.rms+tol<Math.abs(o.mean))
   throw Error("inconsistent tensor summary");
 if(o.sample.some(x=>x<o.min-tol||x>o.max+tol))
   throw Error("sample contradicts declared tensor range");
 if(Math.abs(o.l2-o.rms*Math.sqrt(o.count))>tol*Math.max(1,o.l2))
   throw Error("tensor L2 must match RMS and element count");
}
if(!/^[a-f0-9]{64}$/i.test(p.promptSha256||"")||!Array.isArray(p.tokenIds)||
  p.tokenIds.length<2||p.tokenIds.length>64||p.tokenIds.some(x=>!Number.isSafeInteger(x)||x<0))
 throw Error("missing hashed input and bounded real token IDs");
for(const k of ["input","output"]){const v=t.activation?.[k];if(!Array.isArray(v)||!v.length||v.length>256||v.some(x=>!Number.isFinite(x)))throw Error("invalid activation "+k);}
return {status:"provenance_claim_structurally_valid_not_independently_verified",source:p.modelId+"@"+p.revision,module:p.modulePath,loss:t.loss,weight:t.parameter.weight,gradient:t.parameter.gradient,activations:t.activation};
}
return Object.freeze({validate});
});
