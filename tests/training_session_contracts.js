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
const two=T.steps(g,{count:2,lr:0.01});
const baseline=Base.steps(g,{count:2,lr:0.01});
assert.deepEqual(two.graph,baseline.graph);
assert.deepEqual(two.history,baseline.history);
assert.equal(two.sessions.length,2);
assert.equal(two.finalIR.nodes.length,two.next.steps.length);
assert.equal(two.sessions[0].updates[0].optimizerStep,1);
assert.equal(two.sessions[1].updates[0].optimizerStep,2);
assert.throws(()=>T.steps(g,{count:9}),/step count/);
const fs=require('node:fs'),path=require('node:path');
const page=fs.readFileSync(path.join(__dirname,'../learn.html'),'utf8');
assert.ok(page.includes('TrainingSession.steps(previous,options)'));
assert.ok(page.includes('<script src="training_session.js"></script>'));
assert.ok(page.indexOf('<script src="train_state.js"></script>')<page.indexOf('<script src="training_session.js"></script>'));
console.log('TRAINING_SESSION_PASS immutable parameter and optimizer update IR');
