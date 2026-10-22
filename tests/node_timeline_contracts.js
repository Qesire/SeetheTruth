'use strict';
const assert=require('node:assert/strict'),L=require('../learning_lab.js');
const N=require('../node_timeline.js'),Train=require('../training_session.js');
const g=L.preset('trainable_core'),before=JSON.stringify(g);
const timeline=N.train(g,{lr:0.01}),reference=Train.step(g,{lr:0.01});
assert.equal(timeline.schema,'seethetruth-node-timeline-v1');
assert.equal(timeline.evidence,'SYNTHETIC');
assert.equal(JSON.stringify(g),before);
assert.deepEqual(timeline.graph,reference.graph);
assert.deepEqual(timeline.nodes.map(n=>n.id),reference.beforeIR.nodes.map(n=>n.id));
assert.equal(timeline.beforeLoss,reference.beforeLoss);
assert.equal(timeline.afterLoss,reference.afterLoss);
const weight=timeline.nodes.find(n=>n.id==='shared_weight');
assert.ok(weight.gradient.l2>0);
assert.equal(weight.optimizer.optimizerStep,1);
assert.deepEqual(weight.forwardShape,weight.updatedShape);
assert.equal(weight.optimizerState.kind,'adamw');
assert.equal(weight.optimizerState.stepBefore,0);
assert.equal(weight.optimizerState.stepAfter,1);
assert.deepEqual(weight.optimizerState.accumulators.map(x=>x.slot),['m','v']);
assert.ok(weight.optimizerState.accumulators.every(x=>x.before.elements===4&&x.after.elements===4));
assert.ok(weight.optimizerState.accumulators.some(x=>x.after.l2>0));
assert.ok(timeline.nodes.every(n=>n.optimizer===null ? n.optimizerState===null : n.optimizerState!==null));
assert.ok(timeline.nodes.some(n=>Object.values(n.inputSources).some(source=>source==='shared_weight')));

assert.ok(timeline.nodes.some(n=>n.optimizer===null));
const existing=Train.step(g,{lr:0.01});
const fromExisting=N.fromSession(g,existing);
assert.deepEqual(fromExisting.nodes,timeline.nodes);
assert.deepEqual(fromExisting.graph,existing.graph);
const first=fromExisting.nodes.find(n=>n.id==='shared_weight');
const outputBefore=existing.transaction.previous.steps.find(n=>n.id==='shared_weight').values;
const flat=outputBefore.flat(Infinity);
const mean=flat.reduce((a,b)=>a+b,0)/flat.length;
assert.ok(Math.abs(first.forwardValue.mean-mean)<1e-12);
assert.equal(first.forwardValue.elements,flat.length);
assert.equal(first.updatedValue.elements,flat.length);
const multi=Train.steps(g,{count:2,lr:0.01});
const second=N.fromSession(multi.sessions[0].graph,multi.sessions[1]);
assert.equal(second.nodes.find(n=>n.id==='shared_weight').optimizer.optimizerStep,2);
assert.deepEqual(second.graph,multi.graph);
assert.equal(second.nodes.find(n=>n.id==='shared_weight').optimizerState.stepBefore,1);
assert.equal(second.nodes.find(n=>n.id==='shared_weight').optimizerState.stepAfter,2);
for(const kind of ['sgd_momentum','rmsprop','adagrad','lion','adam']){
 const fresh=L.preset('trainable_core');
 const t=N.train(fresh,{optimizer:kind,lr:0.01});
 const record=t.nodes.find(n=>n.id==='shared_weight');
 const expected=kind==='sgd_momentum'||kind==='lion'?['m']:
    kind==='rmsprop'||kind==='adagrad'?['v']:['m','v'];
 assert.deepEqual(record.optimizerState.accumulators.map(x=>x.slot),expected);
 assert.equal(record.optimizerState.stepAfter,1);
}

console.log('NODE_TIMELINE_PASS forward VJP optimizer updated forward alignment');
