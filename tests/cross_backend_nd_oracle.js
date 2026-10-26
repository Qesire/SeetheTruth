'use strict';
/* Heavy cross-backend experiment. This is separate from the fast source-math
 * test: it executes generated NumPy and PyTorch programs and asks an unrelated
 * scalar Python oracle to judge both, plus the original JS forward/VJP.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {spawnSync}=require('node:child_process');
const L=require('../learning_lab.js'),G=require('../graph_autodiff.js');
const NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const {cases,create}=require('./fixtures/nd_broadcast_reduction_graphs.js');
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const root=path.resolve(__dirname,'..');
const sources=['learning_lab.js','block_catalog.js','nd_tensor.js','nd_autodiff.js',
  'graph_autodiff.js','graph_lowering_plan.js','portable_operator_rules.js',
  'python_codegen.js','torch_codegen.js','tests/fixtures/nd_broadcast_reduction_graphs.js',
  'tests/python_cross_backend_reference.py','tests/cross_backend_nd_oracle.js'];
const sourceDigest=sha(JSON.stringify(Object.fromEntries(sources.map(file=>[
  file,sha(fs.readFileSync(path.join(root,file)))
]))));
const checks=[];
for(const spec of cases){
 const graph=create(spec.id),before=JSON.stringify(graph);
 const native=L.evaluate(graph),back=G.backward(graph,native,{targetId:'loss'});
 const nodes=Object.fromEntries(graph.nodes.map(n=>[n.id,n]));
 const steps=Object.fromEntries(native.steps.map(n=>[n.id,n.values]));
 for(const name of ['x','bias','gain','biased','gated','summed']){
  assert.ok(back.pathIds.includes(name),spec.id+' lacks loss path through '+name);
  assert.ok(back.gradients[name],spec.id+' lacks gradient at '+name);
 }
 const numpy=NP.compile(graph).source,torch=Torch.compile(graph,{lossId:'loss'}).source;
 assert.ok(!numpy.includes('import seethetruth')&&!torch.includes('import seethetruth'));
 checks.push({
  id:spec.id,shape:spec.shape,
  sources:{x:nodes.x.params.values,bias:nodes.bias.params.values,
    gain:nodes.gain.params.values,target:steps.target},
  js:{forward:steps,gradients:back.gradients},
  code:{numpy,torch}
 });
 assert.equal(JSON.stringify(graph),before,spec.id+' fixture changed');
}
assert.equal(checks.length,2);
const executable=path.join(__dirname,'python_cross_backend_reference.py');
const done=spawnSync('python3',[executable],{
 input:JSON.stringify(checks),encoding:'utf8',timeout:240000,maxBuffer:8*1024*1024
});
if(done.error)throw done.error;
if(done.status!==0)throw Error(
 'Independent cross-backend experiment failed (exit '+done.status+'):\n'+
 (done.stderr||'')+'\n'+(done.stdout||'').slice(-3000));
const report=done.stdout.trim().split('\n').find(line=>
 line.startsWith('CROSS_BACKEND_SCALAR_ORACLE_PASS '));
assert.ok(report,'independent Python backend report missing');
const evidence=JSON.parse(report.slice('CROSS_BACKEND_SCALAR_ORACLE_PASS '.length));
assert.equal(evidence.cases.length,2);
for(const row of evidence.cases){
 assert.ok(row.coordinates.numpy>0&&row.coordinates.torch>row.coordinates.numpy);
 assert.ok(row.coordinates.js>row.coordinates.numpy);
 assert.ok(row.maxAbsError.numpy<3e-9&&row.maxAbsError.torch<3e-9);
}
console.log('CROSS_BACKEND_EXECUTED_PASS '+JSON.stringify({
 sourceDigest,commit:process.env.GITHUB_SHA||'local-unpinned',
 fixtures:evidence.cases,claim:'exact graph coordinates and CPU Float64 only'
}));
