'use strict';
// Independently executed PyTorch CPU autograd from the existing generated
// standalone program; no JS GraphAutodiff or CoordinateBridge in Python.
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Torch=require('../torch_codegen.js');
const Proof=require('../training_coordinate_proof.js');
const {create}=require('./fixtures/coordinate_training_graph.js');
const graph=create(),compiled=Torch.compile(graph,{lossId:'loss'});
assert.equal(compiled.lossId,'loss');
assert.ok(!compiled.source.includes('import seethetruth'));
const code=compiled.source+'\n'+[
 'm=GraphRuntime()',
 'out=m.forward()',
 'loss=out["loss"]',
 'loss.backward()',
 'print(json.dumps({"loss":loss.detach().item(),',
 '"gradients":{key:par.grad.detach().tolist() for key,par in m.parameters.items() if par.grad is not None},',
 '"forward":{key:val.detach().tolist() for key,val in out.items()}},allow_nan=False))'
].join('\n');
const out=execFileSync('python3',['-c',code],{encoding:'utf8',maxBuffer:4*1024*1024,timeout:90000});
const py=JSON.parse(out.trim().split('\n').at(-1));
const js=L.evaluate(graph);
assert.ok(Math.abs(py.loss-js.steps.find(x=>x.id==='loss').values[0][0])<1e-11);
const options={targetId:'loss',lr:.01,weightDecay:.001};
for(const param of ['x','w']){
 for(let i=0;i<2;i++)for(let j=0;j<2;j++){
  const witness=Proof.inspect(graph,options,{parameterId:param,index:[i,j],
   viaNodeId:'project',viaIndex:[0,0]});
  const oracle=py.gradients[param][i][j];
  assert.ok(Number.isFinite(oracle)&&Math.abs(witness.wholeLossGradient-oracle)<2e-9,
   param+'['+[i,j]+'] JS whole-loss VJP='+witness.wholeLossGradient+' PyTorch='+oracle);
 }
}
console.log('INDEPENDENT_TORCH_TRAINING_COORDINATE_PASS: 8 native autograd elements, scalar loss and graph forward');
