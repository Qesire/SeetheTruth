'use strict';
const assert=require('node:assert/strict'),ND=require('../nd_tensor.js'),L=require('../learning_lab.js'),
  C=require('../concept_flow.js'),R=require('../graph_readiness.js'),G=require('../graph_autodiff.js');
const close=(a,b,eps=3e-10)=>assert.ok(Math.abs(a-b)<=eps,'expected '+a+' ≈ '+b);
const same=(a,b,eps=3e-10)=>{assert.deepEqual(ND.shape(a),ND.shape(b));const x=ND.flatten(a),y=ND.flatten(b);x.forEach((v,i)=>close(v,y[i],eps));};
const rawK=ND.seedTensor([1,2,3,4],4),rawV=ND.seedTensor([1,2,3,4],6),rawQ=ND.seedTensor([1,4,3,4],2);
const allRot=ND.rope4(rawK,0,4),part1=ND.rope4(ND.cacheRead4(rawK,0,2),0,4),
      part2=ND.rope4(ND.cacheRead4(rawK,2,1),2,4);
same(ND.cacheAppend4(part1,part2),allRot);
same(ND.cacheAppend4(ND.cacheRead4(rawV,0,2),ND.cacheRead4(rawV,2,1)),rawV);
same(ND.rope4(rawQ,0,4),ND.cacheAppend4(ND.rope4(ND.cacheRead4(rawQ,0,2),0,4),ND.rope4(ND.cacheRead4(rawQ,2,1),2,4)));
ND.indices([1,2,3,4],idx=>{
 const x=ND.cacheRead4(allRot,idx[2],1)[idx[0]][idx[1]][0];
 close(x[idx[3]],allRot[idx[0]][idx[1]][idx[2]][idx[3]]);
});
const rope=ND.rope4(rawK,3,4);
for(let h=0;h<2;h++)for(let t=0;t<3;t++){
 const before=rawK[0][h][t].reduce((a,v)=>a+v*v,0),after=rope[0][h][t].reduce((a,v)=>a+v*v,0);
 close(before,after,1e-12);
}
assert.throws(()=>ND.rope4(rawK,0,3),/rotary/);
assert.throws(()=>ND.rope4(rawK,0,6),/rotary/);
assert.throws(()=>ND.cacheRead4(rawK,2,2),/outside/);
assert.throws(()=>ND.cacheAppend4(rawK,ND.seedTensor([1,4,1,4],1)),/matching/);
assert.throws(()=>ND.cacheAppend4(ND.seedTensor([1,2,5,4],1),ND.seedTensor([1,2,4,4],1)),/capacity/);
const graph=L.preset('nd_cached_gqa_decode');
assert.equal(graph.nodes.length,21);
assert.equal(graph.edges.length,22);
const concept=C.evaluate(graph),forward=L.evaluate(graph);
assert.deepEqual(concept.steps.map(s=>s.shape),forward.steps.map(s=>s.shape));
assert.equal(R.check(graph,{numeric:true}).executable,true);
const out=Object.fromEntries(forward.steps.map(s=>[s.id,s.values]));
same(out.cached_k,allRot);same(out.cached_v,rawV);
assert.deepEqual(ND.shape(out.scores),[1,4,1,3]);
assert.deepEqual(ND.shape(out.context),[1,4,1,4]);
const K=ND.repeatKVHeads(allRot,4),V=ND.repeatKVHeads(rawV,4);
const Q=ND.rope4(rawQ,0,4);
const fullScores=ND.matmul4(Q,ND.permute(K,[0,1,3,2]));
const fullProb=ND.softmaxLast(ND.causalMask4(ND.scale(fullScores,.5),0,0));
const fullContext=ND.matmul4(fullProb,V);
same(out.context,ND.cacheRead4(fullContext,2,1),1e-12);
same(out.probabilities,ND.cacheRead4(fullProb,2,1),1e-12);
assert.equal(forward.totalKnownFlops,96);
// Whole graph gradient includes distinct key/value prefix and current branches.
// Append ND-MSE and compare every source derivative with independent finite differences.
const supervised=JSON.parse(JSON.stringify(graph));
supervised.nodes.push({id:'target',kind:'nd_input',x:2820,y:820,params:{b:1,h:4,t:1,d:4,seed:11}});
supervised.nodes.push({id:'loss',kind:'nd_mse',x:3260,y:340,params:{}});
supervised.edges.push({from:'context',to:'loss',port:'prediction'},{from:'target',to:'loss',port:'target'});
const sf=L.evaluate(supervised),grad=G.backward(supervised,sf);
assert.equal(grad.targetId,'loss');
let count=0;
for(const source of ['full_q','full_k','full_v']){
 const n=supervised.nodes.find(n=>n.id===source),tensor=sf.steps.find(t=>t.id===source).values,
       actual=grad.gradients[source],shape=ND.shape(tensor);
 assert.deepEqual(ND.shape(actual),shape);
 ND.indices(shape,idx=>{
   const h=1e-6,delta=sign=>{
     const test=JSON.parse(JSON.stringify(supervised));
     const node=test.nodes.find(x=>x.id===source);
     // Replace source by an explicit trainable tensor without touching its dependent edges.
     node.kind='nd_parameter';node.params={values:ND.build(shape,i=>ND.at(tensor,i)+(i.every((v,j)=>v===idx[j])?sign*h:0)),
       optimizer:{t:0,m:ND.build(shape,()=>0),v:ND.build(shape,()=>0)}};
     return L.evaluate(test).steps.at(-1).values[0][0];
   };
   close(ND.at(actual,idx),(delta(1)-delta(-1))/(2*h),4e-6);
   count++;
 });
}
assert.equal(count,96);
assert.deepEqual(ND.shape(grad.gradients.full_k),[1,2,3,4]);
assert.ok(ND.flatten(grad.gradients.full_k).some(x=>Math.abs(x)>1e-8));
const broken=JSON.parse(JSON.stringify(graph));
broken.nodes.find(n=>n.id==='k_current').params.start=3;
assert.equal(R.check(broken).executable,false);
assert.equal(R.check(broken).nodes.find(n=>n.id==='k_current').status,'shape-error');
console.log('N-D RoPE+KV cache decode PASS: full-prefix numerical equivalence, 21 blocks, 96 full-graph finite differences, causal GQA head reuse');
