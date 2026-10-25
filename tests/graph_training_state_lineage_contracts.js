'use strict';
const assert=require('node:assert/strict');
const Core=require('../learning_experience_core.js');
const L=require('../learning_lab.js'),Training=require('../training_session.js');
const Timeline=require('../node_timeline.js'),State=require('../state_microscope.js');
const Snap=require('../observation_snapshots.js');
const g=L.preset('trainable_core');
const original=JSON.stringify(g);
const options={lr:.01,targetId:'loss'};
const run=Core.executeLab('training','graph_step',{graph:g,options});
assert.equal(JSON.stringify(g),original,'graph source input remains immutable');
assert.equal(run.operation,'graph_step');
assert.equal(run.stateTrace.schema,State.SCHEMA);
assert.equal(run.stateTrace.evidenceTier,State.TIERS.GRAPH);
assert.equal(Core.describeActions('training').find(x=>x.name==='graph_step').observation,'state_transition');
State.validate(run.stateTrace);
const truth=Training.step(g,options);
const nodes=Timeline.fromSession(g,truth,options).nodes;
assert.deepEqual(run.value.session,truth);
assert.deepEqual(run.value.timeline.nodes,nodes);
assert.deepEqual(run.value.session.graph,truth.graph);
const graphLinks=run.stateTrace.graphLinks;
assert.deepEqual(Object.keys(graphLinks),nodes.map(x=>x.id));
assert.equal(run.stateTrace.steps.filter(x=>x.id.startsWith('graph:forward:')).length,nodes.length);
assert.equal(run.stateTrace.steps.filter(x=>x.id.startsWith('graph:updated:')).length,nodes.length);
assert.equal(run.stateTrace.steps.filter(x=>x.id.startsWith('graph:backward:')).length,nodes.filter(x=>x.onLossPath).length);
assert.equal(run.stateTrace.steps.filter(x=>x.id.startsWith('graph:optimizer:')).length,nodes.filter(x=>x.optimizer).length);
for(const node of nodes){
 const links=graphLinks[node.id];
 assert.ok(links.forward&&links.updated);
 assert.equal(Boolean(links.backward),node.onLossPath);
 assert.equal(Boolean(links.optimizer),Boolean(node.optimizer));
 const forward=run.stateTrace.steps.find(x=>x.id===links.forward);
 const after=run.stateTrace.steps.find(x=>x.id===links.updated);
 assert.equal(forward.after.nodeId,node.id);
 assert.equal(after.after.nodeId,node.id);
 assert.deepEqual(forward.after.inputSources,node.inputSources);
 assert.deepEqual(forward.after.output,node.forwardValue);
 assert.deepEqual(after.after.output,node.updatedValue);
 const incoming=Object.values(node.inputSources).filter(x=>x!==null);
 assert.deepEqual(forward.dependencies,incoming.map(x=>graphLinks[x].forward));
 assert.deepEqual(after.dependencies.slice(0,incoming.length),incoming.map(x=>graphLinks[x].updated));
 if(links.backward){
  const grad=run.stateTrace.steps.find(x=>x.id===links.backward);
  assert.deepEqual(grad.after.gradient,node.gradient);
  assert.ok(grad.dependencies.includes(links.forward));
 }
 if(links.optimizer){
  const optimizer=run.stateTrace.steps.find(x=>x.id===links.optimizer);
  assert.equal(optimizer.after.step,node.optimizerState.stepAfter);
  assert.equal(optimizer.before.step,node.optimizerState.stepBefore);
  assert.equal(optimizer.after.updateNorm,node.optimizer.updateNorm);
  assert.deepEqual(optimizer.after.moments.map(x=>x.slot),node.optimizerState.accumulators.map(x=>x.slot));
  assert.ok(optimizer.dependencies.includes(links.backward));
  assert.ok(after.dependencies.includes(links.optimizer));
 }
}
const param=nodes.find(x=>x.optimizer),paramSteps=graphLinks[param.id];
const detail=Core.inspectObservationStep(run,{kind:'state_transition',stepId:paramSteps.optimizer});
assert.equal(detail.replayVerified,true);
assert.equal(detail.detail.after.nodeId,param.id);
assert.equal(detail.detail.after.phase,'optimizer');
assert.equal(Core.replay(run).reproduced,true);
assert.equal(Core.compareRecords(run,run).valueEqual,true);
assert.deepEqual(Core.observe(run).sources[0].steps.find(x=>x.id===paramSteps.optimizer).dependencies,
 paramSteps.backward?[{from:paramSteps.backward,port:null}]:[]);
const another=Core.executeLab('training','graph_step',{graph:g,options:{...options,lr:.02}});
assert.equal(Core.compareRecords(run,another).inputEqual,false);
assert.ok(Core.compareRecords(run,another).changedSteps.some(x=>x.kind==='state_transition'&&x.stepId.startsWith('graph:optimizer:')));
const portable=Snap.create(Core,run);
assert.equal(Snap.restore(Core,portable).record.stateTrace.evidenceTier,State.TIERS.GRAPH);
const cursor=Snap.at(Core,portable,paramSteps.optimizer);
assert.equal(Snap.restore(Core,cursor).checkpoint.detail.after.nodeId,param.id);
const altered=structuredClone(cursor);altered.cursor.stateDigest='fnv1a32:deadbeef:8';
assert.throws(()=>Snap.restore(Core,altered),/state cursor replay differs/);
const tampered=structuredClone(run);tampered.stateTrace.steps.find(x=>x.id===paramSteps.optimizer).after.step=444;
assert.throws(()=>Core.replay(tampered),/replay mismatch: stateTrace/);
assert.throws(()=>Core.executeLab('training','graph_step',{graph:g,options:{...options,targetId:'unknown'}}));
assert.throws(()=>Core.executeLab('training','graph_step',{graph:g}),/requires a real graph and optimizer options/);
assert.throws(()=>Core.inspectObservationStep(run,{kind:'tensor_computation',stepId:paramSteps.optimizer,coordinate:[0]}),/requested observation trace unavailable/);
for(const optimizer of ['adam','sgd_momentum','rmsprop','adagrad','lion']){
 const r=Core.executeLab('training','graph_step',{graph:g,options:{...options,optimizer}});
 assert.deepEqual(r.value.session.graph,Training.step(g,{...options,optimizer}).graph);
 assert.ok(r.stateTrace.graphLinks[param.id].optimizer);
 assert.equal(Core.replay(r).reproduced,true);
}
console.log('GRAPH_TRAINING_STATE_LINEAGE_PASS: source graph, true ports, VJP, optimizer and updated forward, snapshot and replay');
