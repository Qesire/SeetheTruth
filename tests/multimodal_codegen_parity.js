'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Py=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
function run(code){return JSON.parse(execFileSync('python3',['-c',code],{
 encoding:'utf8',timeout:120000,maxBuffer:12*1024*1024
}).trim().split('\n').at(-1));}
function close(name,a,b,eps=2e-8){
 const u=a.flat(Infinity),v=b.flat(Infinity);
 assert.equal(u.length,v.length,name+' count');
 u.forEach((x,i)=>assert.ok(Number.isFinite(v[i])&&Math.abs(x-v[i])<eps+Math.abs(x)*1e-10,
  name+'['+i+'] '+x+' versus '+v[i]));
}
const presets=['image_vit_patch','audio_spectrogram','video_vivit_tubelet','multimodal_cross_attention'];
for(const name of presets){
 const graph=L.preset(name),trace=L.evaluate(graph);
 const np=Py.compile(graph),pt=Torch.compile(graph);
 const n=run(np.source),t=run(pt.source+'\nprint(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))\n');
 assert.equal(Object.keys(n).length,trace.steps.length);
 assert.equal(Object.keys(t).length,trace.steps.length);
 for(const step of trace.steps){close('NumPy '+name+'/'+step.id,step.values,n[step.id]);close('PyTorch '+name+'/'+step.id,step.values,t[step.id]);}
 assert.ok(np.source.includes('def run_graph(overrides=None):'));
 assert.ok(pt.source.includes('class GraphRuntime'));
 console.log('MULTIMODAL_PARITY_PASS '+name+' nodes='+trace.steps.length);
}
