/* One semantic register for bounded reduction/normalization operators.
 * Backend emitters retain their native NumPy/Torch implementations.
 * Shape and parameter validation comes from LearningLab before lowering.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const api=factory();
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthNormalizationRules=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
const descriptions=Object.freeze({
 softmax:Object.freeze({formula:"exp(x-max(x))/sum(exp(x-max(x)))",axis:-1,stable:true}),
 nd_softmax_last:Object.freeze({formula:"exp(x-max(x))/sum(exp(x-max(x)))",axis:-1,stable:true}),
 rmsnorm:Object.freeze({formula:"x/sqrt(mean(x^2)+epsilon)",axis:-1,epsilonPlacement:"inside_sqrt"}),
 qk_norm:Object.freeze({formula:"x/sqrt(mean(x^2)+epsilon)",axis:-1,epsilonPlacement:"inside_sqrt"}),
 layernorm:Object.freeze({formula:"(x-mean(x))/sqrt(mean((x-mean(x))^2)+epsilon)",axis:-1,epsilonPlacement:"inside_sqrt"}),
 attention:Object.freeze({formula:"softmax((q @ transpose(k))/sqrt(d_q)) @ v",axis:-1,scaleAxis:"q.last",stable:true}),
 log_softmax:Object.freeze({formula:"x-logsumexp(x)",axis:-1,stable:true}),
 row_logsumexp:Object.freeze({formula:"log(sum(exp(x)))",axis:-1,keepdims:true,stable:true})
});
const kinds=Object.freeze(Object.keys(descriptions).sort());
function emit(node,backend,port){
 if(!node||typeof node.kind!=="string"||typeof port!=="function"||!["numpy","torch"].includes(backend))
  throw Error("invalid normalization lowering request");
 const kind=node.kind,p=node.params||{};
 if(!Object.hasOwn(descriptions,kind))return null;
 const torch=backend==="torch";
 const x=()=>port("x");
 const sum=arg=>torch?"torch.logsumexp("+arg+",dim=-1,keepdim=True)":"np.logaddexp.reduce("+arg+",axis=-1,keepdims=True)";
 const mean=arg=>torch?"torch.mean("+arg+",dim=-1,keepdim=True)":"np.mean("+arg+",axis=-1,keepdims=True)";
 const sqrt=arg=>torch?"torch.sqrt("+arg+")":"np.sqrt("+arg+")";
 if(kind==="softmax"||kind==="nd_softmax_last")
  return torch?"torch.softmax("+x()+",dim=-1)":"_softmax("+x()+")";
 if(kind==="rmsnorm"||kind==="qk_norm")
  return "("+x()+"/"+sqrt(mean(x()+"**2")+"+"+p.epsilon)+")";
 if(kind==="layernorm"){
  if(torch)return "F.layer_norm("+x()+","+x()+".shape[-1:],eps="+p.epsilon+")";
  const centered="("+x()+"-"+mean(x())+")";
  return "("+centered+"/"+sqrt(mean(centered+"**2")+"+"+p.epsilon)+")";
 }
 if(kind==="attention"){
  const q=port("q"),k=port("k"),v=port("v");
  const score="("+q+" @ "+k+".T)";
  const scaled="("+score+"/"+(torch?"math.sqrt(":"np.sqrt(")+q+".shape[-1]))";
  const prob=torch?"torch.softmax("+scaled+",dim=-1)":"_softmax("+scaled+")";
  return "("+prob+" @ "+v+")";
 }
 if(kind==="log_softmax")
  return torch?"F.log_softmax("+x()+",dim=-1)":"("+x()+"-"+sum(x())+")";
 if(kind==="row_logsumexp")
  return sum(x());
 throw Error("unhandled normalization semantics "+kind);
}
return Object.freeze({schema:"seethetruth-normalization-semantics-v1",kinds,descriptions,emit});
});
