/* Paper-to-block transformations: bounded executable patches, not model-scale kernels.
 * API contracts use matrices shaped [tokens, features], weights [input, output].
 */
(function(root,make){const api=make();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthPaperPatches=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
const papers={
 lora:{title:"LoRA",url:"https://arxiv.org/abs/2106.09685",kind:"parameterization",effect:"replace trainable dense delta with low-rank A·B; freeze W0"},
 dora:{title:"DoRA",url:"https://proceedings.mlr.press/v235/liu24bn.html",kind:"parameterization",effect:"separate magnitude and direction of updated weight"},
 flash2:{title:"FlashAttention-2",url:"https://arxiv.org/abs/2307.08691",kind:"execution_schedule",effect:"reorder tiled exact attention with online normalization, not approximate attention"},
 switch:{title:"Switch Transformers",url:"https://arxiv.org/abs/2101.03961",kind:"control_flow",effect:"top-1 expert dispatch, finite capacity and load auxiliary loss"},
 checkpoint:{title:"Activation Checkpointing",url:"https://arxiv.org/abs/1604.06174",kind:"memory_schedule",effect:"recompute selected forward intermediates on backward"}
};
const limit=16,finite=x=>typeof x==="number"&&Number.isFinite(x);
function valid(a){if(!Array.isArray(a)||a.length<1||a.length>limit||!Array.isArray(a[0])||a[0].length<1||a[0].length>limit||a.some(r=>!Array.isArray(r)||r.length!==a[0].length||r.some(x=>!finite(x))))throw Error("invalid 2-D finite bounded tensor");return a;}
function mm(a,b){valid(a);valid(b);if(a[0].length!==b.length)throw Error("contraction mismatch");return a.map(r=>b[0].map((_,j)=>r.reduce((s,v,k)=>s+v*b[k][j],0)));}
function add(a,b){valid(a);valid(b);if(a.length!==b.length||a[0].length!==b[0].length)throw Error("shape mismatch");return a.map((r,i)=>r.map((v,j)=>v+b[i][j]));}
function scale(a,s){return a.map(r=>r.map(v=>v*s));}
function stats(a){const v=valid(a).flat(),n=v.length;return {shape:[a.length,a[0].length],mean:v.reduce((x,y)=>x+y,0)/n,rms:Math.hypot(...v)/Math.sqrt(n),min:Math.min(...v),max:Math.max(...v),zeroFraction:v.filter(x=>x===0).length/n};}
function lora(x,w,a,b,alpha=1){valid(x);valid(w);valid(a);valid(b);
if(x[0].length!==w.length||a[0].length!==b.length||a.length!==w.length||b[0].length!==w[0].length||!finite(alpha))throw Error("invalid LoRA dimensions");
const base=mm(x,w),down=mm(x,a),up=mm(down,b),factor=alpha/a[0].length,update=scale(up,factor),output=add(base,update);
return {output,base,update,internals:{down,up,scaling:factor},statistics:{output:stats(output),update:stats(update)},
 contract:{frozenBase:true,rank:a[0].length,scaling:factor,trainable:["A","B"],paper:"lora"}};}
function dora(x,w,a,b,m,alpha=1){valid(x);valid(w);valid(a);valid(b);valid(m);
if(x[0].length!==w.length||a.length!==w.length||a[0].length!==b.length||b[0].length!==w[0].length||!finite(alpha)||m.length!==1||m[0].length!==w[0].length)throw Error("invalid DoRA shapes or scale");
const rawDelta=mm(a,b),factor=alpha/a[0].length,delta=scale(rawDelta,factor),v=add(w,delta),
 norms=v[0].map((_,j)=>Math.sqrt(v.reduce((s,r)=>s+r[j]*r[j],0)));
if(norms.some(n=>n<=1e-12||!finite(n)))throw Error("undefined zero direction");
const effective=v.map(r=>r.map((z,j)=>z*m[0][j]/norms[j])),output=mm(x,effective);
return {output,effectiveWeight:effective,directions:v,internals:{rawDelta,delta,norms:[norms],scale:factor},
 statistics:{output:stats(output),weight:stats(effective)},
 contract:{magnitudeAxis:"output_columns",baseFrozen:true,trainable:["A","B","m"],paper:"dora",note:"full differentiable DoRA training requires normalization VJP"}};}
function onlineAttention(q,k,v,chunk=2){valid(q);valid(k);valid(v);
if(q[0].length!==k[0].length||k.length!==v.length||!Number.isSafeInteger(chunk)||chunk<1||chunk>16)throw Error("attention dimensions");
const d=q[0].length,dim=v[0].length,output=[],rows=[],tileTrace=[];let tiles=0;
for(let qi=0;qi<q.length;qi++){
 const query=q[qi];let max=-Infinity,denom=0,acc=Array(dim).fill(0);
 for(let start=0;start<k.length;start+=chunk){
  tiles++;const keyTile=k.slice(start,start+chunk);
  const scores=keyTile.map(key=>query.reduce((z,x,t)=>z+x*key[t],0)/Math.sqrt(d));
  const previousMax=max,previousDenom=denom,previousAccumulator=acc.slice();
  const newMax=Math.max(max,...scores),corr=max===-Infinity?0:Math.exp(max-newMax),exps=scores.map(z=>Math.exp(z-newMax));
  denom=denom*corr+exps.reduce((z,x)=>z+x,0);
  acc=acc.map((a,j)=>a*corr+exps.reduce((z,e,t)=>z+e*v[start+t][j],0));max=newMax;
  tileTrace.push({query:qi,start,end:start+scores.length,scores,weights:exps,
   previousMax:Number.isFinite(previousMax)?previousMax:null,previousDenom,previousAccumulator,
   runningMax:max,normalizer:denom,accumulator:acc.slice(),correction:corr,partialOutput:acc.map(a=>a/denom)});
 }
 output.push(acc.map(a=>a/denom));rows.push({runningMax:max,normalizer:denom});
}
return {output,rows,tileTrace,statistics:stats(output),schedule:{tiles,chunk,storesFullAttention:false},
 contract:{exactMath:true,causal:false,dropout:false,backward:false,paper:"flash2",note:"educational online softmax, not a GPU FlashAttention-2 kernel"}};}
const variants=[
{id:"lora_linear",source:"lora",base:"linear",type:"rewrite",scope:"weight_parameterization",inputs:["X","W0","A","B"],state:["frozen_W0","trainable_A","trainable_B"],outputs:["Y"],formula:"Y=XW0+(α/r)XAB",changes:["parameter_storage","backward_parameter_paths"],preserves:["dense_equivalent_at_fixed_weights"],verify:["shape","numerical_equivalence","frozen_base"]},
{id:"dora_linear",source:"dora",base:"linear",type:"rewrite",scope:"weight_parameterization",inputs:["X","W0","A","B","m"],state:["frozen_W0","trainable_A","trainable_B","trainable_m"],outputs:["Y"],formula:"V=W0+(α/r)AB; W=m⊙V/||V||column",changes:["normalization_gradient","parameterization"],preserves:["linear_forward_given_effective_W"],verify:["direction_norm","zero_direction_rejection"]},
{id:"online_attention",source:"flash2",base:"attention",type:"schedule_rewrite",scope:"memory_lifetime",inputs:["Q","K","V"],state:["row_running_max","row_normalizer","row_accumulator"],outputs:["O"],formula:"streaming log-sum-exp accumulation",changes:["intermediate_storage","operator_schedule"],preserves:["exact_attention_output_without_mask"],verify:["dense_comparison","chunk_invariance"]},
{id:"switch_top1",source:"switch",base:"ffn",type:"control_flow_rewrite",scope:"routing",inputs:["tokens","router_logits","experts"],state:["capacity","dispatch_counts"],outputs:["expert_outputs"],formula:"e=argmax softmax(router(x)); y=expert_e(x)",changes:["active_parameter_paths","capacity_overflow"],preserves:[],verify:["routing_counts","overflow","gradient_mask"]},
{id:"activation_checkpoint",source:"checkpoint",base:"training_graph",type:"lifetime_rewrite",scope:"backward",inputs:["segment_inputs"],state:["rng_state","recompute_boundary"],outputs:["recomputed_activations"],formula:"recompute forward segment during backward",changes:["memory","forward_recomputation"],preserves:["gradients_when_recompute_deterministic"],verify:["gradient_parity","rng_restore"]}
];
function validateCatalog(){for(const v of variants){if(!papers[v.source]||!v.type||!v.scope||!v.inputs.length||!v.outputs.length||!v.verify.length)throw Error("incomplete paper patch "+v.id);}return {papers:Object.keys(papers).length,variants:variants.length};}
return Object.freeze({papers,variants,lora,dora,onlineAttention,stats,validateCatalog});
});
