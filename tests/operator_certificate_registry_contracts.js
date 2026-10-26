'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const crypto=require('node:crypto');
const L=require('../learning_lab.js'),T=require('../torch_codegen.js');
const R=require('../operator_certificate_registry.js'),C=require('../block_capabilities.js');
const root=path.resolve(__dirname,'..'),data=require('../operator_certificates_data.json');
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
const S=R.summary();
assert.equal(R.SCHEMA,'seethetruth-executed-operator-case-certificates-v2');
assert.deepEqual(R.certificate,data,'browser/JSON evidence must be equal');
assert.equal(S.sourceDigest,data.sourceDigest);
assert.equal(S.fixtures,6);
assert.deepEqual(S.fixtureIds,['coordinate_training_graph','rectangular_relu_fanout',
 'shared_weight_two_matmuls','relu_exact_zero',
 'nd_token_reduce_head_broadcast','nd_batch_reduce_channel_broadcast']);
assert.ok(S.certifiedKinds>=13);
assert.ok(R.forKind('nd_broadcast_mul').length>=2);
assert.ok(R.forKind('nd_reduce_sum').length>=2);
assert.ok(R.forKind('nd_broadcast_add').length>=2);
assert.ok(R.forKind('nd_mse').length>=2);
assert.ok(R.forKind('nd_parameter').length>=2);
assert.ok(S.certifiedCases>S.certifiedKinds,'different real fixtures required');
assert.ok(S.forwardCoordinates>29&&S.lossAdjointCoordinates>25&&S.certifiedNodes>8);
assert.equal(S.excludedFiniteDifferences,1);
assert.ok(S.parameterCoordinates>8);
const hashed=Object.fromEntries(Object.keys(data.sourceFiles).map(p=>
 [p,digest(fs.readFileSync(path.join(root,p)))]));
assert.deepEqual(hashed,data.sourceFiles,'changed pinned source invalidates certificate');
assert.equal(digest(JSON.stringify(hashed)),data.sourceDigest);
for(const entry of data.cases){
 const cert=R.forKind(entry.kind);
 assert.ok(cert.some(c=>c.fixtureId===entry.fixtureId));
 assert.ok(L.KINDS.includes(entry.kind));
 assert.ok(T.supportedKinds.includes(entry.kind));
 assert.ok(S.fixtureIds.includes(entry.fixtureId));
 assert.equal(entry.notGeneralized,true);
}
assert.deepEqual(R.forKind('nd_cache_read'),[],'unattested kinds remain empty');
assert.ok(R.forKind('parameter').length>=4);
assert.ok(R.forKind('matmul').length>=3);
const m=C.matrix();
assert.equal(m.evidence,'DECLARATIVE_NOT_EXECUTED');
assert.equal(m.certificateEvidence,'EXECUTED_PINNED_PYTORCH_CPU_GRAPH_CASE');
assert.equal(m.certificateSummary.sourceDigest,data.sourceDigest);
for(const row of m.rows){
 const expected=data.cases.filter(c=>c.kind===row.kind);
 assert.deepEqual(row.certifiedCases,expected);
 assert.equal(row.capability.localVjp.verification,'not_attested');
 assert.equal(row.capability.torchExport.verification,'not_attested');
}
const tamper=fn=>{const v=structuredClone(data);fn(v);assert.throws(()=>R.validate(v),/operator certificates:/);};
tamper(v=>{v.nodes[0].maxForwardErrorUpperBound=0.5;});
tamper(v=>{v.cases[0].caseIds.push('missing');});
tamper(v=>{v.cases[0].notGeneralized=false;});
tamper(v=>{v.nodes[0].shape=[2,3];});
tamper(v=>{v.environment.device='cuda';});
tamper(v=>{v.cases[0].wholeLossAdjointCoordinates+=1;});
tamper(v=>{v.nodes[0].kind='no_such_kind';});
tamper(v=>{v.cases[0].fixtureId='not_a_fixture';});
tamper(v=>{v.fixtures[0].graphDigest='bad';});
tamper(v=>{v.fixtures[1].nodeCount+=1;});
tamper(v=>{v.nodes[1].fixtureId='relu_exact_zero';});
const unchanged=R.forKind('matmul');
assert.ok(Object.isFrozen(unchanged)&&Object.isFrozen(unchanged[0])&&Object.isFrozen(R.certificate));
console.log('OPERATOR_MULTI_FIXTURE_CERTIFICATE_PASS '+JSON.stringify({
 fixtures:S.fixtures,kinds:S.certifiedKinds,cases:S.certifiedCases,nodes:S.certifiedNodes,
 forward:S.forwardCoordinates,backward:S.lossAdjointCoordinates,parameterFD:S.parameterCoordinates,
 excludedKinks:S.excludedFiniteDifferences,sourceDigest:data.sourceDigest}));
