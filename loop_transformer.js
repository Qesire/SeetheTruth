/* Looped Transformer: graph-level recurrent-depth composition by static unrolling.
 * Every recurrent step contains real tensor computation, but all step instances
 * reference the SAME Parameter nodes, even in NumPy/Torch independent exports.
 * No graph cycles, hidden Python code, checkpoint import or GPU are required.
 *
 * Grounding: Ouro/LoopLM arXiv:2510.25741 (shared-depth, sandwich RMS, SwiGLU,
 * multi-step losses); Huginn arXiv:2502.05171 (distinct recurrent-depth family).
 * This bounded single-head implementation is a mathematical subset, NOT Ouro.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthLoopTransformer=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const PAPER="https://arxiv.org/abs/2510.25741";
const MAX_LOOPS=3; // with 80-node graph budget and explicit per-step attention/FFN
const matrix=(r,c,seed)=>Array.from({length:r},(_,i)=>Array.from({length:c},(_,j)=>
 Number((Math.sin((i+1)*0.47+(j+1)*0.31+seed)*0.18).toFixed(8))));
function build(opts={}){
 const loops=opts.loops===undefined?2:opts.loops;
 if(!Number.isSafeInteger(loops)||loops<1||loops>MAX_LOOPS)
   throw Error("loop count must be integer 1..3; never silently truncate recurrence");
 const chosen=opts.lossWeights===undefined?Array.from({length:loops},(_,i)=>i+1):opts.lossWeights;
 if(!Array.isArray(chosen)||chosen.length!==loops||chosen.some(v=>typeof v!=="number"||!Number.isFinite(v)||v<0)||
   chosen.reduce((s,v)=>s+v,0)<=0)throw Error("nonnegative finite depth-loss weights with positive total are required");
 const sum=chosen.reduce((s,v)=>s+v,0),alpha=chosen.map(v=>v/sum);
 const nodes=[],edges=[];
 const node=(id,kind,x,y,params={})=>{nodes.push({id,kind,x,y,params});return id;};
 const edge=(from,to,port)=>edges.push({from,to,port});
 const join=(kind,id,inputs,params={},x=0,y=0)=>{
  node(id,kind,x,y,params);
  for(const [port,from] of Object.entries(inputs))edge(from,id,port);
  return id;
 };
 const parameter=(id,r,c,seed,y)=>{
  const values=matrix(r,c,seed),zeros=values.map(row=>row.map(()=>0));
  return node(id,"parameter",30,y,{values,optimizer:{t:0,m:zeros,v:zeros}});
 };
 // One actual Parameter identity per matrix, shared by ALL iterations and
 // also between input Embedding lookup and tied LM output projection.
 node("tokens","token_ids",25,20,{rows:4,vocab:8,seed:1});
 join("slice_rows","prefix",{x:"tokens"},{start:0,length:3},240,20);
 join("slice_rows","labels",{x:"tokens"},{start:1,length:3},240,330);
 parameter("embedding",8,4,0.9,550);
 join("gather_rows","lookup",{x:"embedding",indices:"prefix"},{},430,20);
 parameter("Wq",4,4,1.2,735);
 parameter("Wk",4,4,1.9,905);
 parameter("Wv",4,4,2.5,1075);
 parameter("Wo",4,4,3.1,1245);
 parameter("Wup",4,6,3.8,1415);
 parameter("Wgate",4,6,4.5,1585);
 parameter("Wdown",6,4,5.2,1755);
 join("transpose2d","tied_head",{x:"embedding"},{},440,555);
 const states=[],logits=[],losses=[],probs=[],maskIds=[];
 let state="lookup";
 for(let step=0;step<loops;step++){
  const tag="r"+step+"_",offset=step*2150+750;
  let i=0;
  const f=(name,kind,inputs,params={})=>{
    const id=tag+name;
    join(kind,id,inputs,params,offset+(i%7)*165,100+Math.floor(i/7)*160);
    i++;return id;
  };
  // Shared weights, recurrent state (not a shared activation or KV cache).
  const norm=f("attn_norm","rmsnorm",{x:state},{epsilon:1e-5});
  const q=f("q","matmul",{a:norm,b:"Wq"});
  const k=f("k","matmul",{a:norm,b:"Wk"});
  const v=f("v","matmul",{a:norm,b:"Wv"});
  const kt=f("kt","transpose2d",{x:k});
  const scores=f("scores","matmul",{a:q,b:kt});
  const scaled=f("scaled","scale",{x:scores},{factor:0.5});
  const mask=f("mask","causal_mask",{x:scaled});
  const p=f("probs","masked_softmax",{logits:scaled,mask});
  const context=f("context","weighted_sum",{weights:p,values:v});
  const projected=f("proj","matmul",{a:context,b:"Wo"});
  const h=f("attn_res","add",{a:state,b:projected});
  const fNorm=f("ffn_norm","rmsnorm",{x:h},{epsilon:1e-5});
  const up=f("up","matmul",{a:fNorm,b:"Wup"});
  const gate=f("gate","matmul",{a:fNorm,b:"Wgate"});
  const activated=f("swiglu","swiglu",{up,gate});
  const down=f("down","matmul",{a:activated,b:"Wdown"});
  state=f("state","add",{a:h,b:down});
  const logit=f("logits","matmul",{a:state,b:"tied_head"});
  const loss=f("ce","cross_entropy",{logits:logit,labels:"labels"});
  states.push(state);logits.push(logit);losses.push(loss);probs.push(p);maskIds.push(mask);
 }
 const weighted=losses.map((loss,step)=>join("scale","weight"+step,{x:loss},{factor:alpha[step]},700+step*190,1960));
 let total=weighted[0];
 for(let i=1;i<weighted.length;i++)total=join("add","total"+i,{a:total,b:weighted[i]},{},1000+i*195,1990);
 // End with canonical target ID so existing train/backward UI and exports work.
 if(total!=="loss")total=join("scale","loss",{x:total},{factor:1},1550,2180);
 const graph=L.validateGraph({schema:L.SCHEMA,title:
  "任务 41 · Looped Transformer ×"+loops+" · 参数共享 + 多深度损失",
  nodes,edges});
 // For 3 steps the entire recurrent depth, supervision and source matrices
 // still fit the bounded editor's 80-node + 176-edge contracts.
 if(graph.nodes.length>L.MAX_NODES||graph.edges.length>L.MAX_EDGES)throw Error("loop graph exceeded bounded editor budget");
 return graph;
}
function inspect(raw){
 const graph=L.validateGraph(raw);
 const iterations=[0,1,2].filter(i=>graph.nodes.some(n=>n.id==="r"+i+"_state"));
 const shared=["embedding","Wq","Wk","Wv","Wo","Wup","Wgate","Wdown"];
 const parents=new Map(graph.edges.map(e=>[e.to+":"+e.port,e.from]));
 return Object.freeze({loops:iterations.length,
  parameterIds:shared.filter(id=>graph.nodes.some(n=>n.id===id)),
  parameterCount:shared.filter(id=>graph.nodes.some(n=>n.id===id)).length,
  // A genuine loop's next iteration must consume previous state:
  linked:iterations.every(i=>i===0?parents.get("r0_attn_norm:x")==="lookup":
    parents.get("r"+i+"_attn_norm:x")==="r"+(i-1)+"_state"),
  perIteration:iterations.map(i=>({index:i,finalStateId:"r"+i+"_state",logitsId:"r"+i+"_logits",
    lossId:"r"+i+"_ce",attentionIds:["r"+i+"_probs","r"+i+"_mask"]})),
  paper:PAPER,evidence:"unrolled_shared_parameter_identity_not_a_python_loop_or_adaptive_halting"});
}
function resize(raw,nextLoops){
 const source=L.validateGraph(raw),about=inspect(source);
 if(!about.linked||!about.loops)throw Error("not a connected Looped Transformer graph");
 const originalWeights=Array.from({length:about.loops},(_,i)=>source.nodes.find(n=>n.id==="weight"+i)?.params.factor);
 if(originalWeights.some(x=>typeof x!=="number"||!Number.isFinite(x)||x<0))throw Error("invalid existing depth loss factors");
 const total=originalWeights.reduce((s,v)=>s+v,0);
 if(!(total>0))throw Error("nonpositive existing depth loss mass");
 const canonical=build({loops:about.loops,lossWeights:originalWeights});
 // Changing loop depth is a structural rewrite, not a silent repair of
 // arbitrary user edits. Reject noncanonical edges and changed math operators.
 const signature=g=>JSON.stringify({
  nodes:g.nodes.map(n=>[n.id,n.kind,n.kind==="parameter"?
   n.params.values.map(row=>row.length):n.params]).sort((a,b)=>a[0].localeCompare(b[0])),
  edges:g.edges.map(e=>e.from+":"+e.to+":"+e.port).sort()
 });
 if(signature(source)!==signature(canonical))
  throw Error("custom rewiring/operator edits found; cannot change loop depth without discarding work");
 const target=build({loops:nextLoops});
 const sourceParams=new Map(source.nodes.filter(n=>n.kind==="parameter").map(n=>[n.id,n]));
 for(const n of target.nodes.filter(n=>n.kind==="parameter")){
  const old=sourceParams.get(n.id);
  if(!old)throw Error("missing shared parameter "+n.id);
  n.params=JSON.parse(JSON.stringify(old.params));
 }
 const valid=L.validateGraph(target);
 return {graph:valid,oldLoops:about.loops,newLoops:nextLoops,
  preservedParameterIds:[...sourceParams.keys()],
  oldObjectiveWeights:originalWeights,
  newObjectiveWeights:valid.nodes.filter(n=>/^weight[0-2]$/.test(n.id)).map(n=>n.params.factor),
  warning:"objective depth weights are rebuilt for the new depth; original parameter values and AdamW moments remain unchanged"};
}
return Object.freeze({build,inspect,resize,MAX_LOOPS,PAPER});
});
