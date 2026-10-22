'use strict';
const assert=require('node:assert/strict'),L=require('../learning_lab.js'),
 T=require('../training_session.js'),Base=require('../train_state.js');
const g=L.preset('trainable_core'),saved=JSON.stringify(g);
const out=T.step(g,{lr:0.01}),reference=Base.step(g,{lr:0.01});
assert.equal(out.schema,'seethetruth-training-session-v1');
assert.equal(out.evidence,'SYNTHETIC');
assert.equal(JSON.stringify(g),saved);
assert.deepEqual(out.graph,reference.graph);
assert.equal(out.beforeLoss,reference.beforeLoss);
assert.equal(out.afterLoss,reference.afterLoss);
assert.deepEqual(out.beforeIR.nodes.map(x=>x.id),out.afterIR.nodes.map(x=>x.id));
assert.ok(out.updates.length>0);
for(const row of out.updates){
 assert.ok(out.beforeIR.nodes.some(n=>n.id===row.id));
 assert.ok(out.afterIR.nodes.some(n=>n.id===row.id));
 assert.ok(Number.isFinite(row.gradientNorm)&&Number.isFinite(row.updateNorm));
 assert.equal(row.optimizerStep,1);
}
const next=T.step(out.graph,{lr:0.01});
assert.equal(next.updates[0].optimizerStep,2);
assert.throws(()=>T.step(L.preset('autodiff_core')),/trainable Parameter/);
console.log('TRAINING_SESSION_PASS immutable parameter and optimizer update IR');
