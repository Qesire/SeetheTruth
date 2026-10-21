'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const L=require('../learning_lab.js'),K=require('../block_knowledge.js'),
 NP=require('../python_codegen.js'),T=require('../torch_codegen.js'),A=require('../block_capabilities.js').audit();
assert.equal(A.schema,'seethetruth-capabilities-v1');
assert.equal(A.counts.registered,L.KINDS.length);
assert.equal(A.counts.knowledgeCards,L.KINDS.length);
assert.equal(A.counts.numpyEligible,NP.supportedKinds.length);
assert.equal(A.counts.torchEligible,T.supportedKinds.length);
assert.equal(A.counts.numpyMissing,A.counts.registered-A.counts.numpyEligible);
assert.equal(A.counts.torchMissing,A.counts.registered-A.counts.torchEligible);
assert.equal(new Set(A.rows.map(r=>r.kind)).size,L.KINDS.length);
for(const r of A.rows){
 assert.ok(L.defs[r.kind]&&K.get(r.kind));
 assert.equal(r.numpyEligible,NP.supportedKinds.includes(r.kind));
 assert.equal(r.torchEligible,T.supportedKinds.includes(r.kind));
 assert.equal(r.graphAutodiff,'requires_graph_test');
 assert.equal(r.modelParity,'not_claimed');
}
const gqa=A.rows.find(r=>r.kind==='gqa');
assert.equal(gqa.numpyEligible,false);assert.equal(gqa.torchEligible,false);
assert.equal(A.rows.find(r=>r.kind==='deepinfer_mtp_confidence_schedule').torchEligible,true);
const root=path.resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
for(const k of require('../stream_graph_blocks.js').kinds){
 assert.equal(A.rows.find(r=>r.kind===k).numpyEligible,true);
 assert.equal(A.rows.find(r=>r.kind===k).torchEligible,true);
}
for(const page of ['learn.html','capabilities.html']){
 const html=read(page);
 for(const dependency of ['semantic_state.js','stream_graph_blocks.js','learning_lab.js'])
  assert.ok(html.includes('src="'+dependency+'"'),page+' missing executable dependency '+dependency);
 assert.ok(html.includes('src="numpy_stream_lowering.js"'));
 assert.ok(html.indexOf('src="numpy_stream_lowering.js"')<html.indexOf('src="python_codegen.js"'));
 assert.ok(html.includes('src="torch_stream_lowering.js"'));
 assert.ok(html.indexOf('src="torch_stream_lowering.js"')<html.indexOf('src="torch_codegen.js"'));
 assert.ok(html.indexOf('src="semantic_state.js"')<html.indexOf('src="stream_graph_blocks.js"') &&
           html.indexOf('src="stream_graph_blocks.js"')<html.indexOf('src="learning_lab.js"'),
           page+' must load stream types before the Learning registry');
}
assert.ok(read('atlas_server.py').includes('"/capabilities.html":"capabilities.html"'));
assert.ok(read('teaching.html').includes('href="capabilities.html"'));
assert.ok(read('README.md').includes('node scripts/report_capabilities.js'));
console.log('CAPABILITY_AUDIT_PASS '+A.counts.registered+' blocks / '+A.counts.numpyEligible+' NumPy / '+A.counts.torchEligible+' PyTorch eligible');
