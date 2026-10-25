'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js');
const C=require('../block_capabilities.js');
const N=require('../python_codegen.js'),T=require('../torch_codegen.js');
const base=C.audit(),matrix=C.matrix();
assert.equal(matrix.schema,'seethetruth-operator-capability-matrix-v1');
assert.equal(matrix.source,base.schema);
assert.equal(matrix.evidence,'DECLARATIVE_NOT_EXECUTED');
assert.equal(matrix.rows.length,L.KINDS.length);
assert.deepEqual(matrix.counts,base.counts);
const byKind=new Map(matrix.rows.map(x=>[x.kind,x]));
assert.equal(byKind.size,L.KINDS.length);
for(const kind of L.KINDS){
 const r=C.describe(kind),available=base.rows.find(x=>x.kind===kind);
 assert.deepEqual(r,byKind.get(kind),'describe projects canonical matrix');
 assert.deepEqual(Object.keys(r.capability).sort(),[
  'registered','staticShape','boundedForward','coordinateTrace',
  'localVjp','numpyExport','torchExport','modelParity','gpuKernel'].sort());
 assert.equal(r.capability.registered.verification,'registration_checked');
 assert.equal(r.capability.boundedForward.availability,'requires_valid_graph');
 assert.equal(r.capability.boundedForward.verification,'not_executed');
 assert.equal(r.capability.coordinateTrace.availability,'requires_numeric_trace');
 assert.equal(r.capability.localVjp.verification,'not_attested');
 assert.equal(r.capability.numpyExport.availability,available.numpyEligible?'eligible':'unsupported');
 assert.equal(r.capability.torchExport.availability,available.torchEligible?'eligible':'unsupported');
 assert.equal(N.supportedKinds.includes(kind),r.capability.numpyExport.availability==='eligible');
 assert.equal(T.supportedKinds.includes(kind),r.capability.torchExport.availability==='eligible');
 assert.equal(r.capability.modelParity.availability,'not_claimed');
 assert.equal(r.capability.gpuKernel.verification,'not_attested');
 for(const v of Object.values(r.capability))
  assert.ok(!['verified','passed','native_gpu'].includes(v.verification),'no unearned verification tier');
}
assert.throws(()=>C.describe('nonexistent_capability'),/unknown operator capability/);
const eligible=matrix.rows.filter(x=>x.capability.numpyExport.availability==='eligible').length;
assert.equal(eligible,base.counts.numpyEligible);
assert.equal(matrix.rows.filter(x=>x.capability.torchExport.availability==='eligible').length,
 base.counts.torchEligible);
console.log('OPERATOR_CAPABILITY_MATRIX_PASS '+JSON.stringify({
 kinds:matrix.rows.length,numpyEligible:base.counts.numpyEligible,
 torchEligible:base.counts.torchEligible,verifiedExecutionAtStaticAudit:0
}));
