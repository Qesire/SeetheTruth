'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),T=require('../train_state.js'),G=require('../graph_autodiff.js'),C=require('../concept_flow.js');
const clone=x=>JSON.parse(JSON.stringify(x));
const close=(a,b,eps=8e-6)=>assert.ok(Math.abs(a-b)<eps,'expected '+a+' ≈ '+b);
const models=[
 ['trainable_attention',['wq','wk','wv'],10],
 ['trainable_swiglu',['w1','w3','w2'],10],
 ['trainable_moe',['w1','w2','w3'],24]
];
let checks=0;
for(const [preset,ids,nodeCount] of models){
 const graph=L.preset(preset),before=JSON.stringify(graph),a=L.evaluate(graph),
       grad=G.backward(graph,a),shape=C.evaluate(graph);
 assert.equal(graph.nodes.length,nodeCount);
 assert.deepEqual(a.steps.map(s=>s.shape),shape.steps.map(s=>s.shape));
 for(const id of ids){
   const n=graph.nodes.find(n=>n.id===id);
   assert.equal(n.kind,'parameter');assert.equal(n.params.optimizer.t,0);
   const tensor=n.params.values,g=grad.gradients[id];
   const positions=preset==='trainable_moe'?[[0,0],[3,1],[7,0]]:[[0,0],[0,1],[1,0],[1,1]];
   for(const [r,c] of positions){
     const plus=clone(graph),minus=clone(graph),h=1e-6;
     plus.nodes.find(n=>n.id===id).params.values[r][c]+=h;
     minus.nodes.find(n=>n.id===id).params.values[r][c]-=h;
     const x=L.evaluate(plus).steps.at(-1).values[0][0],y=L.evaluate(minus).steps.at(-1).values[0][0];
     close(g[r][c],(x-y)/(2*h));
     checks++;
   }
 }
 const two=T.steps(graph,{count:2,lr:.004,weightDecay:.005});
 assert.equal(two.history.length,2);assert.equal(two.finalLoss,two.history[1].afterLoss);
 assert.equal(JSON.stringify(graph),before,'multi-step transaction mutated original graph');
 assert.deepEqual(two.graph.nodes.map(n=>n.id),graph.nodes.map(n=>n.id));
 assert.deepEqual(two.graph.edges,graph.edges);
 for(const id of ids){
   const parameter=two.graph.nodes.find(n=>n.id===id);
   assert.equal(parameter.params.optimizer.t,2);
   assert.notDeepEqual(parameter.params.values,graph.nodes.find(n=>n.id===id).params.values);
 }
 const resumed=L.validateGraph(JSON.parse(JSON.stringify(two.graph)));
 const third=T.steps(resumed,{count:1,lr:.004,weightDecay:.005});
 const thirdDirect=T.step(two.graph,{lr:.004,weightDecay:.005});
 assert.deepEqual(third.graph,thirdDirect.graph,'restored state must reproduce next optimizer update');
 assert.ok(two.history.every(h=>Number.isFinite(h.beforeLoss)&&Number.isFinite(h.afterLoss)));
 console.log(preset+' PASS: 2 training steps + serializable AdamW moments, loss '+two.firstLoss+' -> '+two.finalLoss);
}
assert.equal(checks,33);
const blank=L.preset('trainable_attention');
assert.throws(()=>T.steps(blank,{count:0}),/integer from 1 to 8/);
assert.throws(()=>T.steps(blank,{count:9}),/integer from 1 to 8/);
assert.throws(()=>T.steps(blank,{count:1.5}),/integer from 1 to 8/);
assert.throws(()=>T.steps(blank,{count:1,lr:-.1}),/hyperparameters/);
console.log('Trainable QKV, SwiGLU and MoE parameterized graph contracts PASS: 33 full finite differences');
