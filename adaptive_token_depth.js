/* Controlled MoR-inspired per-token recurrent depth, NOT full MoR sparse dispatch.
 * Every step is still evaluated as a dense educational graph. A hard per-token
 * mask selects whether to adopt that step's candidate state.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthAdaptiveDepth=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const Loop=typeof module==="object"&&module.exports?require("./loop_transformer.js"):root.SeetheTruthLoopTransformer;
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const PAPER="https://arxiv.org/abs/2507.10524";
function build(opts={}){
 const seed=opts.seed===undefined?1:opts.seed;
 if(!Number.isSafeInteger(seed)||seed<0||seed>1000)throw Error("budget source seed out of range");
 const g=Loop.build({loops:2}),edge=(from,to,port)=>g.edges.push({from,to,port});
 const add=(id,kind,x,y,params={})=>g.nodes.push({id,kind,x,y,params});
 add("budget_ids","token_ids",25,2300,{rows:3,vocab:4,seed});
 const masks=[];
 for(let r=0;r<2;r++){
  const old=r===0?"lookup":"r0_adopted",candidate="r"+r+"_state",
        mask="r"+r+"_continue",joined="r"+r+"_adopted";
  add(mask,"token_depth_mask",900+r*1500,2350,{step:r+1,width:4});
  add(joined,"state_mask_blend",1150+r*1500,2350);
  edge("budget_ids",mask,"depths");
  edge(old,joined,"previous");
  edge(candidate,joined,"candidate");
  edge(mask,joined,"mask");
  // Redirect real downstream consumers (next recurrent layer and step CE)
  // to adopted H, not a raw candidate that should have exited.
  for(const e of g.edges){
   if(e.from===candidate&&!(e.to===joined&&e.port==="candidate"))e.from=joined;
  }
  masks.push(mask);
 }
 g.title="任务 44 · MoR 启发 · 逐 Token 硬深度预算与状态停止";
 const validated=L.validateGraph(g);
 if(validated.nodes.length>L.MAX_NODES||validated.edges.length>L.MAX_EDGES)throw Error("bounded adaptive-depth graph budget exceeded");
 return validated;
}
function inspect(raw){
 const g=L.validateGraph(raw),edges=new Map(g.edges.map(e=>[e.to+":"+e.port,e.from]));
 if(g.title!=="任务 44 · MoR 启发 · 逐 Token 硬深度预算与状态停止")throw Error("not an adaptive-depth lesson");
 for(let r=0;r<2;r++){
  const join="r"+r+"_adopted",mask="r"+r+"_continue",old=r?"r0_adopted":"lookup";
  if(edges.get(join+":previous")!==old||
     edges.get(join+":candidate")!=="r"+r+"_state"||
     edges.get(join+":mask")!==mask||
     edges.get(mask+":depths")!=="budget_ids")throw Error("adaptive-depth graph no longer has the claimed control topology");
 }
 return Object.freeze({steps:2,parameterIds:g.nodes.filter(n=>n.kind==="parameter").map(n=>n.id),
   semantic:"dense_evaluation_hard_per_token_state_selection",
   source:PAPER,warning:"Not sparse GPU execution; all Attention/FFN candidate matmuls still run"});
}
return Object.freeze({build,inspect,PAPER});
});