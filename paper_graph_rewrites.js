/* Verified paper-to-ordinary-block graph rewrite: LoRA on a selected MatMul.
 * This is NOT a decorative module: resulting graph uses the standard executor,
 * forward/VJP/optimizer graph, save/import and Python exporters.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthPaperGraphRewrites=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const paper="https://arxiv.org/abs/2106.09685";
function lora(raw,outputId,options={}){
 const graph=L.validateGraph(raw),mat=graph.nodes.find(n=>n.id===outputId);
 if(!mat||mat.kind!=="matmul")throw Error("LoRA rewrite requires a selected 2D MatMul");
 if(graph.nodes.length+6>L.MAX_NODES||graph.edges.length+8>L.MAX_EDGES)
   throw Error("LoRA graph rewrite would exceed the bounded node/edge budget");
 const rank=options.rank??2,alpha=options.alpha??rank;
 if(!Number.isSafeInteger(rank)||rank<1||rank>8||typeof alpha!=="number"||!Number.isFinite(alpha)||alpha<=0||alpha>64)
   throw Error("LoRA rank must be 1–8 and alpha must be finite in (0,64]");
 const xEdge=graph.edges.find(e=>e.to===outputId&&e.port==="a"),
       wEdge=graph.edges.find(e=>e.to===outputId&&e.port==="b");
 if(!xEdge||!wEdge)throw Error("LoRA target MatMul requires both input edges");
 const base=graph.nodes.find(n=>n.id===wEdge.from);
 if(base.kind!=="input"&&base.kind!=="parameter")throw Error("LoRA frozen W0 must be an Input or explicitly frozen Parameter");
 const trace=L.evaluate(graph),byId=new Map(trace.steps.map(s=>[s.id,s]));
 const X=byId.get(xEdge.from),W=byId.get(wEdge.from);
 if(X.shape.length!==2||W.shape.length!==2||X.shape[1]!==W.shape[0])
   throw Error("LoRA requires aligned 2D X[T,I] and W0[I,O]");
 const inputDim=W.shape[0],outputDim=W.shape[1];
 if(rank>inputDim||rank>outputDim)throw Error("LoRA rank exceeds input/output channel dimensions");
 const existing=new Set(graph.nodes.map(n=>n.id));
 let seq=0;
 const unique=()=>{let id;do{id="lp"+(++seq)+"_"+outputId.slice(0,26);}while(existing.has(id));existing.add(id);return id;};
 const ids={base:unique(),a:unique(),b:unique(),xa:unique(),xab:unique(),scaled:unique()};
 const zeros=(r,c)=>Array.from({length:r},()=>Array(c).fill(0));
 const A=Array.from({length:inputDim},(_,i)=>Array.from({length:rank},(_,j)=>
   Math.sin((i+1)*17+(j+1)*13+outputId.length)*.05));
 const B=zeros(rank,outputDim);
 const parameter=(id,x,y,values)=>({id,kind:"parameter",x,y,params:{
   values,optimizer:{t:0,m:zeros(values.length,values[0].length),v:zeros(values.length,values[0].length)}
 }});
 const m=(id,kind,x,y,params={})=>({id,kind,x,y,params});
 const x=mat.x,y=mat.y;
 const added=[
   m(ids.base,"matmul",x-90,y-140),
   parameter(ids.a,x-670,y+120,A),parameter(ids.b,x-360,y+280,B),
   m(ids.xa,"matmul",x-360,y+90),m(ids.xab,"matmul",x-60,y+100),
   m(ids.scaled,"scale",x+240,y+100,{factor:alpha/rank})
 ];
 const next=JSON.parse(JSON.stringify(graph));
 const originalWeight=next.nodes.find(n=>n.id===wEdge.from);
 if(originalWeight.kind==="parameter")originalWeight.params.frozen=true;
 const target=next.nodes.find(n=>n.id===outputId);
 target.kind="add";target.params={};
 next.nodes.push(...added);
 next.edges=next.edges.filter(e=>e.to!==outputId).concat([
  {from:xEdge.from,to:ids.base,port:"a"},
  {from:wEdge.from,to:ids.base,port:"b"},
  {from:xEdge.from,to:ids.xa,port:"a"},
  {from:ids.a,to:ids.xa,port:"b"},
  {from:ids.xa,to:ids.xab,port:"a"},
  {from:ids.b,to:ids.xab,port:"b"},
  {from:ids.xab,to:ids.scaled,port:"x"},
  {from:ids.base,to:outputId,port:"a"},
  {from:ids.scaled,to:outputId,port:"b"}
 ]);
 // Incoming original two edges replaced by nine, net +7.
 next.title=(next.title+" · LoRA("+rank+", α="+alpha+")").slice(0,80);
 next.paperPatches=[...(graph.paperPatches||[]),{paper:"lora",outputId,baseId:ids.base,
    aId:ids.a,bId:ids.b,rank,alpha}];
 const verified=L.validateGraph(next),after=L.evaluate(verified),
       prev=byId.get(outputId).values,newValue=after.steps.find(n=>n.id===outputId).values;
 const delta=prev.flat().reduce((mx,z,i)=>Math.max(mx,Math.abs(z-newValue.flat()[i])),0);
 if(delta>1e-10)throw Error("LoRA zero-B initialization failed numerical parity: "+delta);
 return {graph:verified,patch:verified.paperPatches.at(-1),delta,source:paper,
   description:"W0 frozen Input or Parameter (value preserved), trainable A/B; output ID preserved; B initialized to zero"};
}
return Object.freeze({lora,papers:{lora:paper}});
});