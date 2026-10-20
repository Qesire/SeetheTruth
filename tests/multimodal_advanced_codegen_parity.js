'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Py=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
function run(code){return JSON.parse(execFileSync('python3',['-c',code],{encoding:'utf8',timeout:120000,maxBuffer:10*1024*1024}).trim().split('\n').at(-1));}
function compare(a,b,tag){
 const x=a.flat(Infinity),y=b.flat(Infinity);
 assert.equal(x.length,y.length,tag+' length');
 x.forEach((v,i)=>assert.ok(Number.isFinite(y[i])&&Math.abs(y[i]-v)<=2e-8+Math.abs(v)*1e-10,tag+': '+i+' '+v+' vs '+y[i]));
}
for(const name of ['vision_learned_conv','audio_mel_chain','diffusion_ddim_chain']){
 const graph=L.preset(name),steps=L.evaluate(graph).steps;
 const np=Py.compile(graph),pt=Torch.compile(graph);
 const outputs=run(np.source);
 const torch=run(pt.source+'\nprint(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))\n');
 for(const s of steps){
  compare(s.values,outputs[s.id],'numpy/'+name+'/'+s.id);
  compare(s.values,torch[s.id],'torch/'+name+'/'+s.id);
 }
 if(name==='vision_learned_conv'){
  const source=pt.source+[
   '', 'runtime=GraphRuntime()', 'outputs=runtime.forward()',
   'outputs["conv"].sum().backward()',
   'print(json.dumps(runtime.parameters["kernel"].grad.detach().tolist()))'
  ].join('\n')+'\n';
  const actual=run(source);
  const pix=steps.find(x=>x.id==='pixels').values;
  const gradient=Array.from({length:2},()=>Array.from({length:4},(_,j)=>{
   const dy=Math.floor(j/2),dx=j%2;let sum=0;
   for(const y of [0,2])for(const x of [0,2])sum+=pix[y+dy][x+dx];
   return sum;
  }));
  compare(gradient,actual,'actual torch autograd Conv2D weight gradient');
 }
 console.log('MULTIMODAL_ADVANCED_PARITY_PASS '+name+': '+steps.length+' graph nodes');
}
