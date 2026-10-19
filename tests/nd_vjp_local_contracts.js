'use strict';
const assert=require('node:assert/strict'),ND=require('../nd_tensor.js'),A=require('../nd_autodiff.js');
const approx=(x,y,e=5e-6)=>assert.ok(Math.abs(x-y)<=e,x+' ≠ '+y);
function perturb(x,at,delta){
 const sh=ND.shape(x);
 return ND.build(sh,idx=>ND.at(x,idx)+(at.every((v,i)=>v===idx[i])?delta:0));
}
function dot(a,b){
 assert.deepEqual(ND.shape(a),ND.shape(b));
 const flat=ND.flatten(a),weights=ND.flatten(b);
 return flat.reduce((s,v,i)=>s+v*weights[i],0);
}
const X=ND.seedTensor([1,2,3,2],1),Y=ND.seedTensor([1,1,3,2],5),K=ND.seedTensor([1,1,2,3],7);
const examples=[
 {kind:'nd_rope',input:{x:ND.seedTensor([1,2,3,4],2)},p:{offset:2,rotaryDim:4,base:10000},
   forward:d=>ND.rope4(d.x,2,4),ports:['x']},
 {kind:'nd_cache_read',input:{x:X},p:{start:1,length:2},
   forward:d=>ND.cacheRead4(d.x,1,2),ports:['x']},
 {kind:'nd_cache_append',input:{cache:ND.cacheRead4(X,0,2),segment:ND.cacheRead4(X,2,1)},p:{},
   forward:d=>ND.cacheAppend4(d.cache,d.segment),ports:['cache','segment']},
 {kind:'nd_permute',input:{x:X},p:{axis0:0,axis1:2,axis2:1,axis3:3},forward:d=>ND.permute(d.x,[0,2,1,3]),ports:['x']},
 {kind:'nd_reshape',input:{x:X},p:{b:1,h:3,t:2,d:2},forward:d=>ND.reshape(d.x,[1,3,2,2]),ports:['x']},
 {kind:'nd_flatten',input:{x:X},p:{},forward:d=>ND.reshape(d.x,[6,2]),ports:['x']},
 {kind:'nd_expand4d',input:{x:ND.seedTensor([3,2],5)},p:{},forward:d=>ND.reshape(d.x,[1,1,3,2]),ports:['x']},
 {kind:'nd_scale',input:{x:X},p:{factor:.37},forward:d=>ND.scale(d.x,.37),ports:['x']},
 {kind:'nd_causal_mask',input:{x:ND.seedTensor([1,2,3,3],2)},p:{queryStart:0,keyStart:0},
   forward:d=>ND.causalMask4(d.x),ports:['x']},
 {kind:'nd_softmax_last',input:{x:ND.seedTensor([1,2,3,3],2)},p:{},
   forward:d=>ND.softmaxLast(d.x),ports:['x']},
 {kind:'nd_broadcast_add',input:{a:X,b:Y},p:{},forward:d=>ND.broadcastBinary(d.a,d.b,(a,b)=>a+b),ports:['a','b']},
 {kind:'nd_reduce_sum',input:{x:X},p:{axis:2},forward:d=>ND.reduceSum(d.x,2,true),ports:['x']},
 {kind:'nd_batched_matmul',input:{a:X,b:K},p:{},forward:d=>ND.matmul4(d.a,d.b),ports:['a','b']},
 {kind:'nd_repeat_kv',input:{x:X},p:{queryHeads:4},forward:d=>ND.repeatKVHeads(d.x,4),ports:['x']},
 {kind:'nd_mse',input:{prediction:X,target:ND.seedTensor([1,2,3,2],8)},p:{},
   forward:d=>ND.meanSquaredError(d.prediction,d.target),ports:['prediction']}
];
let count=0;
for(const e of examples){
 const output=e.forward(e.input),upstream=ND.seedTensor(ND.shape(output),4),
       gradient=A.vjp(e.kind,e.input,e.p,upstream,output);
 for(const port of e.ports){
  const x=e.input[port],g=gradient[port];
  assert.deepEqual(ND.shape(x),ND.shape(g),e.kind+'.'+port+' shape');
  ND.indices(ND.shape(x),idx=>{
    const h=1e-6;
    const before=e.forward({...e.input,[port]:perturb(x,idx,-h)}),
          after=e.forward({...e.input,[port]:perturb(x,idx,h)});
    // Compare perturbation deltas before weighted reduction: masked score values
    // include -1e9 and would otherwise cause cancellation in the oracle.
    const delta=(trial)=>ND.build(ND.shape(output),at=>ND.at(trial,at)-ND.at(output,at));
    try{approx(ND.at(g,idx),(dot(delta(after),upstream)-dot(delta(before),upstream))/(2*h),5e-6);}
    catch(err){throw Error(e.kind+'.'+port+'['+idx.join(',')+']: '+err.message);}
    count++;
  });
 }
 if(e.kind==='nd_mse')assert.deepEqual(gradient.cuts,['target']);
}
console.log('N-D exact local VJPs PASS: '+examples.length+' operators and '+count+' independent finite-difference coordinates');
