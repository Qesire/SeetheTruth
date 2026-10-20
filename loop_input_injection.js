/* Source-referenced linear input reinjection for shared-depth Transformer.
 * Source: ifm-ai/xllm-loop xllm/models/looped.py, LinearInputInjection:
 *     state' = W @ concat(state, original_embeds) (+ optional bias)
 * Our educational path uses bias=False, two fixed recurrent steps, and a
 * single W_inj Parameter. It is NOT a Huginn full prelude/recurrent/coda model.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthLoopInjection=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const Loop=typeof module==="object"&&module.exports?require("./loop_transformer.js"):root.SeetheTruthLoopTransformer;
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const SOURCE="https://github.com/ifm-ai/xllm-loop/blob/main/xllm/models/looped.py";
const PAPER="https://arxiv.org/abs/2610.06833";
const title="任务 45 · Huginn-Linear 启发 · 共享循环输入重新注入";
function identityWeights(gain=0){
 if(typeof gain!=="number"||!Number.isFinite(gain)||Math.abs(gain)>1)
   throw Error("input injection gain must be finite in [-1,1]");
 return Array.from({length:8},(_,i)=>Array.from({length:4},(_,j)=>{
   if(i<4)return i===j?1:0;
   return i-4===j?gain:0;
 }));
}
function build(options={}){
 const gain=options.inputGain===undefined?0.15:options.inputGain;
 const values=identityWeights(gain),zeros=values.map(r=>r.map(()=>0));
 const graph=Loop.build({loops:2});
 const N=(id,kind,x,y,params={})=>graph.nodes.push({id,kind,x,y,params});
 const E=(from,to,port)=>graph.edges.push({from,to,port});
 N("Winject","parameter",3650,2280,{values,
    optimizer:{t:0,m:zeros.map(r=>r.slice()),v:zeros.map(r=>r.slice())}});
 // Explicitly add the SAME original embedding at BOTH recurrence entries.
 for(let r=0;r<2;r++){
   const previous=r===0?"lookup":"r0_state",tag="r"+r;
   const current="r"+r+"_attn_norm";
   const old=graph.edges.find(e=>e.to===current&&e.port==="x");
   if(!old||old.from!==previous)throw Error("loop input changed; cannot safely attach source injection");
   const cat=tag+"_injection_concat", projected=tag+"_injected_state";
   N(cat,"concat_cols",3950+r*320,2250,{});
   E(previous,cat,"a");E("lookup",cat,"b");
   N(projected,"matmul",4100+r*320,2360,{});
   E(cat,projected,"a");E("Winject",projected,"b");
   old.from=projected;
 }
 graph.title=title;
 const checked=L.validateGraph(graph);L.order(checked);
 if(checked.nodes.length>L.MAX_NODES||checked.edges.length>L.MAX_EDGES)
   throw Error("injection graph exceeds editor node/edge budget");
 return checked;
}
function inspect(raw){
 const g=L.validateGraph(raw),by=new Map(g.nodes.map(n=>[n.id,n]));
 if(g.title!==title)throw Error("not the fixed-loop input injection experiment");
 const edges=new Map(g.edges.map(e=>[e.to+":"+e.port,e.from]));
 const param=by.get("Winject");
 if(param?.kind!=="parameter"||param.params.values.length!==8||
    param.params.values.some(row=>row.length!==4))throw Error("injection projection must be a shared trainable [8,4] Parameter");
 for(let r=0;r<2;r++){
   const tag="r"+r,previous=r===0?"lookup":"r0_state";
   if(by.get(tag+"_injection_concat")?.kind!=="concat_cols"||
      by.get(tag+"_injected_state")?.kind!=="matmul"||
      edges.get(tag+"_injection_concat:a")!==previous||
      edges.get(tag+"_injection_concat:b")!=="lookup"||
      edges.get(tag+"_injected_state:a")!==tag+"_injection_concat"||
      edges.get(tag+"_injected_state:b")!=="Winject"||
      edges.get(tag+"_attn_norm:x")!==tag+"_injected_state")
     throw Error("injection recurrence wiring does not match fixed source-based two-pass semantics");
 }
 return Object.freeze({loops:2,operator:"linear_unbiased_W_concat_state_original",
  parameterId:"Winject",shape:[8,4],trueParameterAlias:true,
  source:SOURCE,paper:PAPER,
  cost:"2 extra 3x8 @ 8x4 matmuls, 2 concatenations, 32 physical weight elements",
  limitation:"bias-disabled math subset of xLLM LinearInputInjection; not a full Huginn checkpoint, no GPU scheduling"});
}
return Object.freeze({build,inspect,identityWeights,SOURCE,PAPER});
});
