/* Exact shared arithmetic semantics for portable tensor brick lowering.
 * The expressions here only contain Python syntax valid for both NumPy
 * ndarrays and CPU torch.Tensor. Backend-specific operators remain separate.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const api=factory();
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthPortableOperatorRules=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
const OPS=Object.freeze({
 matmul:"@",nd_batched_matmul:"@",
 add:"+",mul:"*",subtract:"-",divide:"/",nd_broadcast_add:"+"
});
const KINDS=Object.freeze([...Object.keys(OPS),
 "scale","nd_scale","residual_scale","bias","square","temperature",
 "low_rank_project","row_bias_add","weighted_sum","slice_cols","slice_rows",
 "nd_cache_read","diffusion_ddim_step"
].sort());
const supported=new Set(KINDS);
function emit(node,port){
 if(!node||typeof node.kind!=="string"||typeof port!=="function")
  throw Error("invalid portable operator lowering request");
 const kind=node.kind,p=node.params||{};
 if(!supported.has(kind))return null;
 if(Object.hasOwn(OPS,kind))
  return "("+port("a")+" "+OPS[kind]+" "+port("b")+")";
 if(kind==="scale"||kind==="nd_scale"||kind==="residual_scale")
  return "("+port("x")+" * "+p.factor+")";
 if(kind==="bias")return "("+port("x")+" + "+p.bias+")";
 if(kind==="square")return "("+port("x")+" ** 2)";
 if(kind==="temperature")return "("+port("x")+" / "+p.temperature+")";
 if(kind==="low_rank_project")
  return "(("+port("x")+" @ "+port("down")+") @ "+port("up")+")";
 if(kind==="row_bias_add")return "("+port("scores")+" + "+port("bias")+")";
 if(kind==="weighted_sum")return "("+port("weights")+" @ "+port("values")+")";
 if(kind==="slice_cols")return port("x")+"[:,"+p.start+":"+(p.start+p.width)+"]";
 if(kind==="slice_rows")return port("x")+"["+p.start+":"+(p.start+p.length)+",:]";
 if(kind==="nd_cache_read")return port("x")+"[:,:, "+p.start+":"+(p.start+p.length)+",:]";
 if(kind==="diffusion_ddim_step"){
  const a=p.alphaT,b=p.alphaPrevious;
  return "("+JSON.stringify(Math.sqrt(b/a))+"*"+port("x")+" + "+
   JSON.stringify(Math.sqrt(1-b)-Math.sqrt(b*(1-a)/a))+"*"+port("epsilon")+")";
 }
 throw Error("unhandled portable operator "+kind);
}
return Object.freeze({kinds:KINDS,emit});
});
