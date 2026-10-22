'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js');
const C=require('../operator_contracts.js');
const A=require('../block_capabilities.js').audit();
const x=C.inventory();
assert.equal(x.schema,'seethetruth-operator-contracts-v1');
assert.equal(x.count,L.KINDS.length);
assert.equal(Object.keys(x.operators).length,L.KINDS.length);
for(const row of A.rows){
 const c=C.resolve(row.kind);
 assert.deepEqual(c.ports,L.defs[row.kind].ports);
 assert.equal(c.backends.numpyEligible,row.numpyEligible);
 assert.equal(c.backends.torchEligible,row.torchEligible);
 assert.equal(c.semantics.graphAutodiff,'requires_graph_test');
 assert.equal(c.semantics.modelParity,'not_claimed');
 assert.ok(Object.isFrozen(c)&&Object.isFrozen(c.ports));
}
assert.throws(()=>C.resolve('not_an_operator'),/unknown operator/);
assert.equal(C.resolve('stream_state_fork').ports.length,L.defs.stream_state_fork.ports.length);
console.log('OPERATOR_CONTRACT_INVENTORY_PASS '+x.count);
