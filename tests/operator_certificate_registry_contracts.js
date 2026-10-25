'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const crypto=require('node:crypto');
const L=require('../learning_lab.js'),T=require('../torch_codegen.js');
const R=require('../operator_certificate_registry.js'),C=require('../block_capabilities.js');
const root=path.resolve(__dirname,'..'),data=require('../operator_certificates_data.json');
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
assert.equal(R.SCHEMA,'seethetruth-executed-operator-case-certificates-v1');
assert.deepEqual(R.certificate,data,'browser/JSON evidence must be equal');
assert.equal(R.summary().sourceDigest,data.sourceDigest);
assert.equal(R.summary().certifiedKinds,data.cases.length);
assert.equal(R.summary().parameterCoordinates,8);
assert.equal(R.summary().forwardCoordinates,29);
assert.equal(R.summary().lossAdjointCoordinates,25);
assert.equal(R.summary().certifiedNodes,8);
const hashed=Object.fromEntries(Object.keys(data.sourceFiles).map(p=>
 [p,digest(fs.readFileSync(path.join(root,p)))]));
assert.deepEqual(hashed,data.sourceFiles,'even a single changed pinned source must invalidate certificate');
assert.equal(digest(JSON.stringify(hashed)),data.sourceDigest);
for(const entry of data.cases){
 const cert=R.forKind(entry.kind);
 assert.deepEqual(cert,entry);
 assert.ok(L.KINDS.includes(entry.kind));
 assert.ok(T.supportedKinds.includes(entry.kind));
 assert.equal(entry.fixtureId,data.fixture.id);
 assert.equal(entry.notGeneralized,true);
}
assert.equal(R.forKind('nd_cache_read'),null,'unattested kinds have no fabricated certificate');
const m=C.matrix();
assert.equal(m.evidence,'DECLARATIVE_NOT_EXECUTED','static support is not promoted globally');
assert.equal(m.certificateEvidence,'EXECUTED_PINNED_PYTORCH_CPU_GRAPH_CASE');
assert.equal(m.certificateSummary.sourceDigest,data.sourceDigest);
for(const row of m.rows){
 const expected=data.cases.find(c=>c.kind===row.kind);
 assert.deepEqual(row.certifiedCases,expected?[expected]:[]);
 assert.equal(row.capability.localVjp.verification,'not_attested',
  'one fixture may never certify local VJP for all cases');
 assert.equal(row.capability.torchExport.verification,'not_attested',
  'codegen eligibility alone is not per-shape verification');
}
const tamper=fn=>{const v=structuredClone(data);fn(v);assert.throws(()=>R.validate(v),/operator certificates:/);};
tamper(v=>{v.nodes[0].maxForwardErrorUpperBound=0.5;});
tamper(v=>{v.cases[0].caseIds.push('missing');});
tamper(v=>{v.cases[0].notGeneralized=false;});
tamper(v=>{v.nodes[0].shape=[2,3];});
tamper(v=>{v.environment.device='cuda';});
tamper(v=>{v.cases[0].wholeLossAdjointCoordinates+=1;});
tamper(v=>{v.nodes[0].kind='no_such_kind';});
const unchanged=R.forKind('matmul');
assert.ok(Object.isFrozen(unchanged)&&Object.isFrozen(R.certificate));
assert.equal(unchanged.forward,'passed_on_fixed_fixture');
console.log('OPERATOR_CERTIFICATE_REGISTRY_PASS '+JSON.stringify({
 kinds:data.cases.length,forward:R.summary().forwardCoordinates,
 backward:R.summary().lossAdjointCoordinates,
 parameterFD:R.summary().parameterCoordinates,sourceDigest:data.sourceDigest}));
