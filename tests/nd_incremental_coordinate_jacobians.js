'use strict';
// Independent finite differences of the original bounded 4D CPU interpreter.
// The lineage engine is only the subject under test, never the oracle.
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),ND=require('../nd_tensor.js'),
 E=require('../element_lineage.js'),Bridge=require('../coordinate_vjp_bridge.js'),
 M=require('../compute_microscope.js'),Grad=require('../graph_autodiff.js');
const node=(id,kind,params={})=>({id,kind,params,x:0,y:0});
const edge=(from,to,port)=>({from,to,port});
const params=values=>({values,optimizer:{t:0,m:ND.build(ND.shape(values),()=>0),
 v:ND.build(ND.shape(values),()=>0)}});
const select=(graph,id,index,trace=L.evaluate(graph))=>
 ND.at(trace.steps.find(s=>s.id===id).values,index);
const swap=(graph,source)=>{
 const original=L.validateGraph(graph),copy=structuredClone(original);
 const values=L.evaluate(original).steps.find(s=>s.id===source).values;
 const n=copy.nodes.find(n=>n.id===source);
 assert.ok(n&&n.kind==='nd_input','convert only seeded ND input to same-valued parameter');
 n.kind='nd_parameter';n.params=params(values);
 return L.validateGraph(copy);
};
const perturb=(graph,id,idx,delta)=>{
 const g=structuredClone(graph),n=g.nodes.find(n=>n.id===id);
 assert.equal(n.kind,'nd_parameter');
 let row=n.params.values;for(const v of idx.slice(0,-1))row=row[v];
 row[idx.at(-1)]+=delta;return g;
};
const h=1e-6;
const finite=(g,source,srcIndex,target,targetIndex)=>{
 const a=select(perturb(g,source,srcIndex,h),target,targetIndex),
       b=select(perturb(g,source,srcIndex,-h),target,targetIndex);
 return (a-b)/(2*h);
};
function audit(g,target,outputIdx,sources,epsilon=3e-5){
 const trace=L.evaluate(g),rep=E.inspect(g,{targetId:target,outputIndex:outputIdx,
  trace,maxVertices:4096,maxEdges:16384});
 assert.equal(rep.complete,true,target+' frontier '+rep.frontiers.join(','));
 let checked=0;
 for(const v of rep.vertices.filter(v=>v.status==='source'&&sources.includes(v.id))){
  assert.notEqual(v.sensitivity,null,'missing scalar gradient '+v.key);
  const observed=finite(g,v.id,v.index,target,outputIdx);
  assert.ok(Math.abs(observed-v.sensitivity)<epsilon,
   v.key+' => '+target+outputIdx+' finite='+observed+' lineage='+v.sensitivity);
  checked++;
 }
 assert.ok(checked>=sources.length,'too few source visits '+target);
 return {rep,trace,checked};
}
// GQA and MQA each have 4 query heads; same KV head is reused 2 or 4
// times respectively. A *single* output must map to exactly one KV head.
for(const preset of ['nd_trainable_gqa','nd_trainable_mqa']){
 const g=L.preset(preset),trace=L.evaluate(g),dims=ND.shape(trace.steps.find(s=>s.id==='context').values);
 assert.deepEqual(dims,[1,4,3,2]);
 for(const [head,token,channel] of [[0,0,0],[1,1,1],[3,2,1]]){
  const target=[0,head,token,channel];
  const {rep,checked}=audit(g,'context',target,['query','key','value']);
  assert.ok(checked>=5);
  const nHead=preset==='nd_trainable_gqa'?Math.floor(head/2):0;
  assert.ok(rep.vertices.filter(v=>v.id==='key'&&v.status==='source')
   .every(v=>v.index[1]===nHead),'KV head selection must not leak to other head');
  const future=rep.vertices.filter(v=>v.id==='key'&&v.index[2]>token);
  assert.ok(future.every(v=>Math.abs(v.sensitivity)<1e-12),
   'future key must have exactly zero score-path gradient');
  const frame=M.learning(g,trace,L);
  const bridge=Bridge.inspect(g,trace,frame,{stepId:'context',coordinate:target});
  assert.equal(bridge.backward.complete,true);
  assert.equal(bridge.loss.status,'VERIFIED_GRAPH_LOSS_VJP');
  assert.equal(bridge.loss.targetId,'loss');
  assert.equal(bridge.backward.internalStages.length,0); // internal operations are separate bricks
 }
 // Verify an explicit combined loss backward, independent of every coordinate
 // path; GQA/MQA group sharing is not an ambiguous copy of gradient tensors.
 const gradients=Grad.backward(g,trace,{targetId:'loss'}).gradients;
 for(const src of ['query','key','value']){
  const sample=src==='query'?[0,3,1,0]:[0,0,1,1];
  const empirical=finite(g,src,sample,'loss',[0,0]);
  assert.ok(Math.abs(empirical-ND.at(gradients[src],sample))<3e-5,
   preset+' whole graph loss gradient '+src);
 }
}
// Rebuild the actual incremental cached GQA example with freely editable
// Q/K/V inputs. This preserves every original RoPE offset and cache read.
let cached=L.preset('nd_cached_gqa_decode');
for(const src of ['full_q','full_k','full_v'])cached=swap(cached,src);
const cachedTrace=L.evaluate(cached),cachedFrame=M.learning(cached,cachedTrace,L);
const {rep:ctx}=audit(cached,'context',[0,3,0,1],['full_q','full_k','full_v']);
assert.ok(ctx.vertices.some(v=>v.id==='q_rope'&&v.stages?.[0]?.stage==='adjacent_rope_pair'));
assert.ok(ctx.vertices.some(v=>v.id==='cached_k'&&v.stages?.[0]?.stage==='immutable_cache_append'));
assert.ok(ctx.vertices.some(v=>v.id==='q_current'&&v.stages?.[0]?.stage==='cache_window'));
assert.ok(ctx.vertices.some(v=>v.id==='causal'&&v.stages?.[0]?.stage==='absolute_causal_visibility'));
const frameReport=Bridge.inspect(cached,cachedTrace,cachedFrame,
 {stepId:'q_rope',coordinate:[0,3,0,1]});
assert.equal(frameReport.backward.complete,true);
assert.equal(frameReport.backward.internalStages[0].absolutePosition,2);
assert.equal(frameReport.backward.internalStages[0].stage,'adjacent_rope_pair');
const pairEdges=frameReport.backward.localEdges;
assert.equal(pairEdges.length,2);
assert.ok(Math.abs(pairEdges[0].localDerivative-Math.cos(2))<1e-12);
assert.ok(Math.abs(pairEdges[1].localDerivative-Math.sin(2))<1e-12);
// Cache prefix and appended token have different source ports; no causal
// backward dependency may cross from one local region to the other.
const prefix=E.inspect(cached,{targetId:'cached_k',outputIndex:[0,0,1,0],trace:cachedTrace});
const latest=E.inspect(cached,{targetId:'cached_k',outputIndex:[0,0,2,0],trace:cachedTrace});
assert.equal(prefix.complete,true);assert.equal(latest.complete,true);
assert.equal(prefix.edges.find(e=>e.from==='cached_k@0,0,1,0').port,'cache');
assert.equal(latest.edges.find(e=>e.from==='cached_k@0,0,2,0').port,'segment');
assert.equal(prefix.vertices.find(v=>v.id==='full_k'&&v.status==='source').index[2],1);
assert.equal(latest.vertices.find(v=>v.id==='full_k'&&v.status==='source').index[2],2);
// Causal blocked score is -1e9 *constant*, so its derivative is exactly 0,
// while a legal query/key pair retains identity. Offsets must be respected.
const cg=L.preset('nd_trainable_gqa'),ct=L.evaluate(cg);
const masked=E.inspect(cg,{targetId:'causal',outputIndex:[0,0,0,2],trace:ct});
assert.equal(masked.complete,true);
assert.equal(masked.vertices[0].stages[0].allowed,false);
assert.equal(masked.edges[0].localDerivative,0);
assert.equal(masked.vertices.find(v=>v.id==='query').sensitivity,0);
const allowed=E.inspect(cg,{targetId:'causal',outputIndex:[0,0,2,0],trace:ct});
assert.equal(allowed.complete,true);
assert.equal(allowed.vertices[0].stages[0].allowed,true);
assert.equal(allowed.edges[0].localDerivative,1);
// Static structure has valid RoPE linear dependencies; numerical cosine is
// determined by position, not by any imported model or fake training trace.
const structural=E.inspect(cached,{targetId:'q_rope',outputIndex:[0,0,0,1]});
assert.equal(structural.complete,true);
assert.ok(structural.vertices.some(v=>v.id==='full_q'&&v.status==='source'));
assert.equal(structural.vertices.find(v=>v.id==='full_q').sensitivity,null);
const stale=structuredClone(cachedTrace);stale.steps.find(s=>s.id==='q_rope').values[0][0][0][0]+=1;
assert.throws(()=>E.inspect(cached,{targetId:'context',outputIndex:[0,0,0,0],trace:stale}),/stale or altered/);
console.log('ND_INCREMENTAL_COORDINATE_VJP_PASS: GQA/MQA complete paths, cached RoPE absolute offset, causal zero, prefix/segment routing and independent finite differences');
