'use strict';
const assert=require('node:assert/strict');
const B=require('../block_catalog.js'),L=require('../learning_lab.js'),C=require('../concept_flow.js'),K=require('../block_knowledge.js');
const run=(kind,inputs={},params={})=>B.execute({id:kind,kind,params},inputs);
const near=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<eps,String(a)+' != '+String(b));
const ids=[[0],[1],[2],[3]];
assert.deepEqual(run('response_token_reshape',{x:ids},{responses:2}).values,[[0,1],[2,3]]);
assert.throws(()=>run('response_token_reshape',{x:[[1,2]]},{responses:2}),/divisible/);
assert.deepEqual(run('response_advantage_broadcast',{advantage:[[1],[-1]],tokenGrid:[[2,3],[4,5]]}).values,[[1,1],[-1,-1]]);
assert.deepEqual(run('policy_importance_ratio',{current:[[0,Math.log(1.5)]],old:[[0,0]]}).values.map(row=>row.map(x=>+x.toFixed(10))),[[1,1.5]]);
assert.throws(()=>run('policy_importance_ratio',{current:[[50]],old:[[0]]}),/limit/);
const R=[[1.5,.5],[1.5,.5]],A=[[1,1],[-1,-1]];
const clipped=run('ppo_clipped_surrogate',{ratio:R,advantage:A},{epsilon:.2}).values;
near(clipped[0][0],1.2);near(clipped[0][1],.5);
near(clipped[1][0],-1.5);near(clipped[1][1],-.8);
assert.throws(()=>run('ppo_clipped_surrogate',{ratio:[[0]],advantage:[[1]]}),/positive/);
const kl=run('sampled_reference_kl',{current:[[0,0,0]],reference:[[0,.5,-.5]]}).values[0];
near(kl[0],0);near(kl[1],Math.expm1(.5)-.5);near(kl[2],Math.expm1(-.5)+.5);
assert.ok(kl.every(v=>v>=-1e-12));
const mask=run('completion_length_mask',{}, {responses:2,tokens:2,firstLength:2,secondLength:1}).values;
assert.deepEqual(mask,[[1,1],[1,0]]);
const objective=run('masked_grpo_objective',{
  surrogate:[[1,3],[2,100]],kl:[[.1,.1],[.2,100]],mask
},{beta:.5});
near(objective.values[0][0],((1+3)/2-.05+2-.1)/2);
assert.deepEqual(objective.internals.per_response.shape,[2,1]);
assert.throws(()=>run('masked_grpo_objective',{surrogate:[[1]],kl:[[0]],mask:[[0]]}),/empty response/);
const g=L.preset('grpo_objective'),trace=L.evaluate(g),abstract=C.evaluate(g);
assert.equal(g.nodes.length,20);
assert.deepEqual(trace.steps.map(t=>t.shape),abstract.steps.map(t=>t.shape));
const map=Object.fromEntries(trace.steps.map(t=>[t.id,t.values]));
const G=map.surrogate.length,T=map.surrogate[0].length,beta=.04;
let j=0;for(let response=0;response<G;response++){
 let count=0,partial=0;for(let t=0;t<T;t++)if(map.valid_mask[response][t]){count++;partial+=map.surrogate[response][t]-beta*map.reference_kl[response][t];}
 j+=partial/count/G;
}
near(map.objective[0][0],j);
near(map.loss[0][0],-j);
for(const id of ['response_token_reshape','response_advantage_broadcast','policy_importance_ratio','ppo_clipped_surrogate','sampled_reference_kl','completion_length_mask','masked_grpo_objective'])assert.ok(K.get(id)?.refs.length);
console.log('GRPO/PPO: ratio, signed clipping, reference KL, unequal completion masking, graph 20-node loss and shape contracts PASS');
