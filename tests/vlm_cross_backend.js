'use strict';
/* Execute original graph and independent Python (NumPy and PyTorch) oracles. */
const assert=require('node:assert/strict');
const path=require('node:path'),fs=require('node:fs'),crypto=require('node:crypto');
const {spawnSync}=require('node:child_process');
const L=require('../learning_lab.js');
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const sources=['vlm_block_pack.js','learning_lab.js','inference_ops.js',
 'tests/vlm_cross_backend.js','tests/python_vlm_reference.py'];
const digest=sha(JSON.stringify(sources.map(p=>[p,sha(fs.readFileSync(path.join(__dirname,'..',p)))])));
const template=L.preset('extension_vlm_query_patch_selection');
const arms=[2,4].map(topk=>{
 const graph=JSON.parse(JSON.stringify(template));
 graph.nodes.find(n=>n.id==='top_patches').params.topk=topk;
 const output=L.evaluate(graph);
 assert.equal(output.steps.length,graph.nodes.length);
 const t=Object.fromEntries(output.steps.map(step=>[step.id,step.values]));
 return {topk,gridWidth:2,q:t.text_embedding,k:t.visual_projector,
  v:t.visual_projector,js:{scores:t.scale_scores,indices:t.top_patches,
   coordinates:t.patch_coordinates,dense:t.dense_visual,sparse:t.sparse_visual,
   delta:t.readout_delta,mse:t.readout_error[0][0]}};
});
const done=spawnSync('python3',[path.join(__dirname,'python_vlm_reference.py')],{
 input:JSON.stringify(arms),encoding:'utf8',timeout:240000,maxBuffer:5*1024*1024
});
if(done.error)throw done.error;
if(done.status!==0)throw Error('VLM cross-backend failed:\n'+done.stderr+'\n'+done.stdout);
const reportLine=done.stdout.trim().split('\n').find(line=>line.startsWith('VLM_CROSS_BACKEND_PASS '));
assert.ok(reportLine,'independent reference report missing');
const report=JSON.parse(reportLine.slice('VLM_CROSS_BACKEND_PASS '.length));
assert.equal(report.cases.length,2);
assert.ok(report.cases.every(c=>c.maxAbsError<2e-11));
assert.ok(report.cases[0].droppedMass.some(m=>m>0));
assert.ok(report.cases[1].mse<1e-20);
console.log('VLM_GRAPH_CPU_PARITY_PASS '+JSON.stringify({
 digest,commit:process.env.GITHUB_SHA||'local-unpinned',
 cases:report.cases,scope:'actual JS graph vs independent NumPy/PyTorch CPU, no exporter claim'
}));
