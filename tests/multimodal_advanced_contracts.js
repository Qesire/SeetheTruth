'use strict';
const assert=require('node:assert/strict');
const A=require('../multimodal_advanced.js'),L=require('../learning_lab.js');
const C=require('../concept_flow.js'),B=require('../block_catalog.js'),K=require('../block_knowledge.js'),R=require('../graph_readiness.js');
const close=(x,y,e=1e-11)=>assert.ok(Math.abs(x-y)<=e,'expected '+x+' ≈ '+y);
assert.equal(A.KINDS.length,3);
for(const kind of A.KINDS){
 assert.ok(L.KINDS.includes(kind),kind+' registered');
 assert.ok(K.get(kind)?.refs.length,'source-backed knowledge '+kind);
}
const p={height:3,width:3,channels:1,kernelHeight:2,kernelWidth:2,stride:1,outChannels:1};
const x=[[1,2,3],[4,5,6],[7,8,9]],w=[[1,0,0,-1]];
assert.deepEqual(A.shape('vision_conv2d',{x:[3,3],kernel:[1,4]},p),[2,2]);
assert.deepEqual(A.execute('vision_conv2d',{x,kernel:w},p),[[-4,-4],[-4,-4]],
 'Conv2D is cross-correlation without filter reversal');
const p2={...p,height:4,width:4,stride:2,outChannels:2};
const x2=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i*4+j+1));
const k2=[[1,0,0,0],[0,0,0,1]];
assert.deepEqual(A.execute('vision_conv2d',{x:x2,kernel:k2},p2),[[1,6,3,8],[9,14,11,16]]);
assert.throws(()=>A.shape('vision_conv2d',{x:[3,3],kernel:[2,4]},p),/kernel/);
assert.throws(()=>A.shape('vision_conv2d',{x:[3,3],kernel:[1,4]},{...p,stride:0}),/stride/);
const mp={fftSize:4,bands:2,sampleRate:8000};
const bank=A.melWeights(mp);
assert.equal(bank.length,3);assert.equal(bank[0].length,2);
assert.ok(bank.flat().every(v=>v>=0&&v<=1));
const spec=[[0,1,0],[0,2,0]],mel=A.execute('audio_mel_filterbank',{x:spec},mp);
assert.equal(mel.length,2);assert.equal(mel[0].length,2);
for(let i=0;i<2;i++)for(let j=0;j<2;j++)close(mel[i][j],spec[i][1]*bank[1][j]);
assert.throws(()=>A.execute('audio_mel_filterbank',{x:[[-1,2,3]]},mp),/nonnegative/);
assert.throws(()=>A.shape('audio_mel_filterbank',{x:[3,4]},mp),/frequency bins/);
const dp={alphaT:0.36,alphaPrevious:0.64};
const step=A.execute('diffusion_ddim_step',{x:[[3,6]],epsilon:[[0,0]]},dp);
close(step[0][0],4);close(step[0][1],8);
const a=0.36,b=0.64,eps=0.25,xt=1.7;
const x0=(xt-Math.sqrt(1-a)*eps)/Math.sqrt(a);
close(A.execute('diffusion_ddim_step',{x:[[xt]],epsilon:[[eps]]},dp)[0][0],Math.sqrt(b)*x0+Math.sqrt(1-b)*eps);
assert.throws(()=>A.shape('diffusion_ddim_step',{x:[2,3],epsilon:[2,4]},dp),/shapes/);
assert.throws(()=>A.validate('diffusion_ddim_step',{alphaT:0.8,alphaPrevious:0.4}),/not decrease/);
const names=['vision_learned_conv','audio_mel_chain','diffusion_ddim_chain'];
for(const name of names){
 const graph=L.preset(name),numeric=L.evaluate(graph),concept=C.evaluate(graph),ready=R.check(graph);
 assert.equal(ready.executable,true,name+' graph readiness');
 assert.equal(numeric.steps.length,graph.nodes.length);
 for(const step of numeric.steps){
  assert.deepEqual(step.shape,concept.steps.find(s=>s.id===step.id)?.shape,name+' '+step.id);
  assert.ok(step.values.flat(Infinity).every(Number.isFinite));
 }
}
console.log('MULTIMODAL_ADVANCED_CONTRACTS_PASS: valid HWC Conv2D, HTK Mel, epsilon-DDIM, three real graphs');
