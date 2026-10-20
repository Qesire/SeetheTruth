'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Py=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
function run(source){return JSON.parse(execFileSync('python3',['-c',source],
 {encoding:'utf8',timeout:120000,maxBuffer:12*1024*1024}).trim().split('\n').at(-1));}
function close(a,b,label){
 const x=a.flat(Infinity),y=b.flat(Infinity);assert.equal(x.length,y.length,label+' length');
 x.forEach((v,i)=>assert.ok(Number.isFinite(y[i])&&Math.abs(v-y[i])<3e-8+Math.abs(v)*1e-10,
   label+' index '+i+': '+v+' vs '+y[i]));
}
for(const name of ['inference_sparse_decode','inference_paged_kv','inference_speculative','inference_paged_sparse']){
 const graph=L.preset(name),expected=L.evaluate(graph).steps;
 const n=run(Py.compile(graph).source),t=run(Torch.compile(graph).source+
   '\nprint(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))\n');
 for(const step of expected){
  close(step.values,n[step.id],name+'/NumPy/'+step.id);
  close(step.values,t[step.id],name+'/Torch/'+step.id);
 }
 console.log('INFERENCE_CODEGEN_PARITY_PASS '+name+' nodes='+expected.length);
}
