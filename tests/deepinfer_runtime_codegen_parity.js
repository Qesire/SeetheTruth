'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),NP=require('../python_codegen.js'),PT=require('../torch_codegen.js');
function run(src){return JSON.parse(execFileSync('python3',['-c',src],
 {encoding:'utf8',timeout:120000,maxBuffer:12*1024*1024}).trim().split('\n').at(-1));}
function eq(a,b,context){
 const x=a.flat(Infinity),y=b.flat(Infinity);
 assert.equal(x.length,y.length,context+' size');
 x.forEach((v,i)=>assert.ok(Number.isFinite(y[i])&&Math.abs(v-y[i])<=3e-8+Math.abs(v)*1e-10,context+' '+i+' '+v+' vs '+y[i]));
}
for(const name of ['deepinfer_mtp_budget','deepinfer_yoco_replay']){
 const g=L.preset(name),expected=L.evaluate(g).steps;
 const np=NP.compile(g).source,pt=PT.compile(g).source;
 assert.ok(np.includes('def _deepinfer_mtp(')&&pt.includes('def _deepinfer_mtp('));
 const py=run(np),torch=run(pt+'\nprint(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))\n');
 for(const step of expected){eq(step.values,py[step.id],name+' NumPy '+step.id);
  eq(step.values,torch[step.id],name+' PyTorch '+step.id);}
 console.log('DEEPINFER_RUNTIME_PARITY_PASS '+name+' '+expected.length+' nodes');
}
