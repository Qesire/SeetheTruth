'use strict';
const assert=require('node:assert/strict');
const B=require('../block_catalog.js'),L=require('../learning_lab.js'),K=require('../block_knowledge.js'),C=require('../concept_flow.js');
const run=(kind,inputs={},params={})=>B.execute({id:'paper_'+kind,kind,params},inputs);
const close=(x,y,e=1e-9)=>assert.ok(Math.abs(x-y)<=e,String(x)+' != '+String(y));
const lr=run('low_rank_project',{x:[[1,2]],down:[[1,0],[0,1]],up:[[2],[3]]});
assert.deepEqual(lr.values,[[8]]);assert.deepEqual(lr.internals.latent.values,[[1,2]]);
const r=run('partial_rope',{x:[[3,4,5,6],[3,4,5,6]]},{rotaryDim:2,offset:0});
assert.deepEqual(r.values[0],[3,4,5,6]);assert.deepEqual(r.values[1].slice(2),[5,6]);
close(Math.hypot(...r.values[1].slice(0,2)),5);
assert.deepEqual(run('row_bias_add',{scores:[[0,1],[1,0]],bias:[[2,-1]]}).values,[[2,0],[3,-1]]);
const group=run('group_topk_mask',{scores:[[9,0,4,4]]},{groups:2,keepGroups:1});
assert.deepEqual(group.values,[[1,1,0,0]]);
assert.deepEqual(run('masked_router_topk',{scores:[[9,0,4,4]],mask:[[0,0,1,1]]},{k:1}).values,[[2]]);
assert.throws(()=>run('masked_router_topk',{scores:[[1,2,3]],mask:[[0,0,1]]},{k:2}),/not enough/);
const logs=run('log_softmax',{x:[[1000,1001]]}).values[0];
close(logs.map(Math.exp).reduce((a,b)=>a+b,0),1);
const chosen=run('selected_logprob',{logits:[[1000,1001]],labels:[[1]]});
close(chosen.values[0][0],logs[1]);
assert.throws(()=>run('selected_logprob',{logits:[[1,2]],labels:[[2]]}),/out of range/);
const adv=run('group_advantage',{rewards:[[0,2],[3,3]]},{epsilon:1e-5}).values;
close(adv[0][0],-1/Math.sqrt(1+1e-5));close(adv[0][1],1/Math.sqrt(1+1e-5));
assert.deepEqual(adv[1],[0,0]);
const added=['low_rank_project','partial_rope','row_bias_add','group_topk_mask','masked_router_topk','log_softmax','selected_logprob','group_advantage'];
for(const kind of added){assert.ok(L.defs[kind],kind);assert.ok(K.get(kind)?.refs.length,kind);}
assert.equal(K.validate().cards,L.KINDS.length);
assert.deepEqual(C.analyze({id:'p',kind:'low_rank_project',params:{}},{x:[3,8],down:[8,2],up:[2,4]}).shape,[3,4]);

for(const name of ['mla_lite','moe_route','grpo_signals']){
  const graph=L.preset(name),numeric=L.evaluate(graph),concept=C.evaluate(graph);
  assert.equal(numeric.steps.length,graph.nodes.length,name);
  assert.deepEqual(numeric.steps.map(s=>s.shape),concept.steps.map(s=>s.shape),name);
}
const mla=L.evaluate(L.preset('mla_lite'));
assert.deepEqual(mla.steps.at(-1).shape,[3,4]);
const moe=L.evaluate(L.preset('moe_route'));
assert.deepEqual(moe.steps.at(-1).shape,[3,4]);
assert.deepEqual(L.evaluate(L.preset('grpo_signals')).steps.at(-1).shape,[2,4]);
console.log(added.length+' model-family primitives + 3 composed graph lessons PASS');
