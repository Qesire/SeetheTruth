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
console.log('EXECUTION_IR_PASS graph index / teaching values / stream state');
