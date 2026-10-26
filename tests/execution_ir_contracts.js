'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js');
const IR=require('../execution_ir.js');
for(const name of ['residual','swiglu','stream_state_branches']){
 const g=L.preset(name),{trace,ir}=IR.evaluate(g);
 assert.equal(ir.schema,'seethetruth-execution-ir-v1');
 assert.equal(ir.evidence,'SYNTHETIC');
 assert.equal(ir.nodes.length,trace.steps.length);
 assert.deepEqual(ir.nodes.map(n=>n.id),L.order(g).map(n=>n.id));
 assert.ok(Object.isFrozen(ir)&&Object.isFrozen(ir.nodes));
 for(const n of ir.nodes){
  const ports=L.defs[n.kind].ports;
  assert.deepEqual(Object.keys(n.inputSources),ports);
  assert.ok(n.shape&&n.shape.every(Number.isInteger));
 }
 assert.throws(()=>IR.snapshot(g,{trace:{steps:trace.steps.slice(1)}}),/step count/);
 const swapped=trace.steps.slice();
 if(swapped.length>1){[swapped[0],swapped[1]]=[swapped[1],swapped[0]];assert.throws(()=>IR.snapshot(g,{trace:{steps:swapped}}),/identity/);}
}
// An observation envelope must reject a stale/malformed numerical payload even
// when every node id/kind happens to match the current graph.
{
 const graph=L.preset('residual'),forward=L.evaluate(graph);
 const corruptShape=structuredClone(forward);
 corruptShape.steps[0].shape=[1,1];
 assert.throws(()=>IR.snapshot(graph,{trace:corruptShape}),
   /numeric value\/shape mismatch/);
 const corruptValue=structuredClone(forward);
 corruptValue.steps[0].values[0][0]=Number.NaN;
 assert.throws(()=>IR.snapshot(graph,{trace:corruptValue}),
   /nonfinite or nonnumeric/);
 const missingCoordinate=structuredClone(forward);
 missingCoordinate.steps[0].values[0].pop();
 assert.throws(()=>IR.snapshot(graph,{trace:missingCoordinate}),
   /numeric value\/shape mismatch/);
 const wrongRank=structuredClone(forward);
 wrongRank.steps[0].shape=[1,1,1,1,1,1];
 assert.throws(()=>IR.snapshot(graph,{trace:wrongRank}),/invalid tensor shape/);
 assert.deepEqual(IR.snapshot(graph,{trace:forward}).nodes.map(n=>n.id),
   L.order(graph).map(n=>n.id));
}
console.log('EXECUTION_IR_PASS graph index / teaching values / stream state');
