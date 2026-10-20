'use strict';
const assert=require('node:assert/strict');
const M=require('../multimodal_ops.js'),B=require('../block_catalog.js');
const L=require('../learning_lab.js'),C=require('../concept_flow.js');
const K=require('../block_knowledge.js'),R=require('../graph_readiness.js');
const flat=a=>a.flat(Infinity);
assert.equal(M.KINDS.length,4);
for(const kind of M.KINDS){
 assert.ok(B.KINDS.includes(kind)&&L.KINDS.includes(kind));
 assert.ok(K.get(kind)?.refs.length>0,'source-aware paper hover '+kind);
}
const pixels=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]];
const ip={height:4,width:4,channels:1,patchHeight:2,patchWidth:2};
assert.deepEqual(M.shape('vision_patchify',[4,4],ip),[4,4]);
assert.deepEqual(M.execute('vision_patchify',pixels,ip),[[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]]);
const video=Array.from({length:8},(_,i)=>[2*i+1,2*i+2]);
const vp={frames:4,height:2,width:2,channels:1,patchFrames:2,patchHeight:2,patchWidth:1};
assert.deepEqual(M.shape('video_tubelet',[8,2],vp),[4,4]);
assert.deepEqual(M.execute('video_tubelet',video,vp),[[1,3,5,7],[2,4,6,8],[9,11,13,15],[10,12,14,16]]);
const ap={fftSize:4,hop:2};
assert.deepEqual(M.shape('audio_stft_power',[1,8],ap),[3,3]);
const spec=M.execute('audio_stft_power',[Array(8).fill(1)],ap);
for(const row of spec)row.forEach((x,j)=>assert.ok(Math.abs(x-[4,1,0][j])<1e-12,'periodic Hann direct FFT oracle'));
const log=M.execute('audio_log_power',spec,{floor:1e-6});
assert.ok(Math.abs(log[0][0]-Math.log10(4))<1e-12);
assert.equal(log[0][1],0);assert.equal(log[0][2],-6);
assert.throws(()=>M.execute('audio_log_power',[[-1]],{floor:1e-6}),/nonnegative/);
assert.throws(()=>M.shape('vision_patchify',[4,4],{...ip,patchHeight:3}),/divisible/);
assert.throws(()=>M.shape('video_tubelet',[8,2],{...vp,frames:3}),/divisible/);
assert.throws(()=>M.shape('audio_stft_power',[1,8],{fftSize:6,hop:2}),/power-of-two/);
assert.throws(()=>M.shape('audio_stft_power',[1,3],ap),/waveform/);
assert.throws(()=>M.shape('video_tubelet',[8,2],{...vp,width:1}),/layout/);
const presets=['image_vit_patch','audio_spectrogram','video_vivit_tubelet','multimodal_cross_attention'];
for(const name of presets){
 const graph=L.preset(name);
 const math=L.evaluate(graph),shape=C.evaluate(graph),ready=R.check(graph);
 assert.equal(ready.executable,true,name+' should have a fully connected valid graph');
 assert.equal(math.steps.length,graph.nodes.length);
 assert.equal(shape.steps.length,graph.nodes.length);
 for(const step of math.steps){
  assert.deepEqual(step.shape,shape.steps.find(x=>x.id===step.id).shape,name+'/'+step.id+' math vs shape');
  assert.ok(flat(step.values).every(Number.isFinite));
 }
 if(name==='multimodal_cross_attention'){
  assert.equal(graph.nodes.filter(n=>n.kind==='attention').length,2);
  assert.deepEqual(math.steps.at(-1).shape,[4,4]);
 }
}
console.log('MULTIMODAL_CONTRACTS_PASS image patches, Hann STFT, tubelets, four graph presets and source coverage');
