'use strict';
const assert=require('node:assert/strict');
const I=require('../inference_ops.js'),L=require('../learning_lab.js');
const C=require('../concept_flow.js'),R=require('../graph_readiness.js'),K=require('../block_knowledge.js');
assert.equal(I.KINDS.length,4);
for(const kind of I.KINDS){assert.ok(L.KINDS.includes(kind),kind);assert.ok(K.get(kind)?.refs.length,kind+' has official source card');}
const scores=[[1,4,4,-1],[9,9,0,-2]];
assert.deepEqual(I.shape('inference_topk',{scores:[2,4]},{topk:2}),[2,2]);
assert.deepEqual(I.execute('inference_topk',{scores},{topk:2}),[[1,2],[0,1]],'stable tie break lower index');
assert.throws(()=>I.shape('inference_topk',{scores:[2,4]},{topk:5}),/topk/);
const q=[[1,0]],k=[[1,0],[0,1]],v=[[3,5],[7,11]],indices=[[0,1]];
const output=I.execute('inference_sparse_attention',{q,k,v,indices},{causal:false,queryOffset:0});
const e0=Math.exp(1/Math.sqrt(2)),a=e0/(e0+1),b=1/(e0+1);
assert.ok(Math.abs(output[0][0]-(3*a+7*b))<1e-12);
assert.ok(Math.abs(output[0][1]-(5*a+11*b))<1e-12);
assert.deepEqual(I.execute('inference_sparse_attention',{q:[[1,0]],k,v,indices:[[1]]},{causal:false,queryOffset:0}),[[7,11]]);
assert.throws(()=>I.execute('inference_sparse_attention',{q,k,v,indices:[[1,1]]},{causal:false,queryOffset:0}),/duplicate/);
assert.throws(()=>I.execute('inference_sparse_attention',{q,k,v,indices:[[1]]},{causal:true,queryOffset:0}),/future/);
assert.throws(()=>I.shape('inference_sparse_attention',{q:[1,2],k:[2,3],v:[2,2],indices:[1,1]},{causal:false,queryOffset:0}),/incompatible/);
const cache=[[10,11],[20,21],[30,31],[40,41]],blocks=[[1],[0]],positions=[[0],[1],[2],[3]];
assert.deepEqual(I.execute('inference_paged_kv_read',{cache,blocks,positions},{pageSize:2,logicalLength:4}),
  [[30,31],[40,41],[10,11],[20,21]]);
assert.throws(()=>I.execute('inference_paged_kv_read',{cache,blocks:[[2],[0]],positions:[[0]]},{pageSize:2,logicalLength:4}),/outside/);
assert.throws(()=>I.execute('inference_paged_kv_read',{cache,blocks,positions:[[4]]},{pageSize:2,logicalLength:4}),/logical position/);
const draft=[[1],[2],[3]],target=[[1],[9],[3],[7]];
assert.deepEqual(I.execute('inference_greedy_verify',{draft,target},{}),[[1],[0],[0]]);
assert.deepEqual(I.greedyMetadata(draft,target),{accepted:1,nextToken:9,bonus:false,attempted:3,acceptedMask:[[1],[0],[0]]});
const all=[[1],[2],[3],[7]];
assert.deepEqual(I.greedyMetadata(draft,all),{accepted:3,nextToken:7,bonus:true,attempted:3,acceptedMask:[[1],[1],[1]]});
assert.throws(()=>I.shape('inference_greedy_verify',{draft:[3,1],target:[3,1]},{}),/K\+1/);
for(const name of ['inference_sparse_decode','inference_paged_kv','inference_speculative','inference_paged_sparse']){
 const graph=L.preset(name),numeric=L.evaluate(graph),symbolic=C.evaluate(graph),ready=R.check(graph);
 assert.equal(ready.executable,true,name+' shape readiness');
 assert.equal(numeric.steps.length,graph.nodes.length,name+' all connected nodes');
 for(const step of numeric.steps){
  assert.deepEqual(step.shape,symbolic.steps.find(n=>n.id===step.id).shape,name+'/'+step.id);
  assert.ok(step.values.flat(Infinity).every(Number.isFinite),name+'/'+step.id);
 }
}
const Cost=require('../graph_cost_report.js');
const paired=Cost.profile(L.preset('inference_sparse_decode'));
const sparseRow=paired.rows.find(row=>row.id==='sparse');
const denseRow=paired.rows.find(row=>row.id==='dense');
assert.equal(sparseRow.attentionMAC,2*2*(4+3),'selected K=2 MAC must use selected keys only');
assert.equal(denseRow.attentionMAC,2*4*(4+3),'dense MAC must use all Tk=4 keys');
assert.equal(paired.attentionMAC,84,'cost report counts both actual separate calls');
assert.ok(paired.unprofiledKinds.some(v=>v.includes('inference_sparse_attention')),
 'do not claim Softmax or Indexer hardware FLOPs');
console.log('INFERENCE_OPS_CONTRACTS_PASS topk ties, sparse softmax, page mapping, causal guard, greedy prefix, four lessons');
