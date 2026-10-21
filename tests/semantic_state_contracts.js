'use strict';
const assert=require('node:assert/strict');
const IR=require('../tensor_ir.js'),S=require('../semantic_state.js'),ND=require('../nd_tensor.js'),
 Learning=require('../learning_lab.js');
const F=(shape,axes,values)=>IR.make(shape,axes,values);
const fails=(fn,re)=>assert.throws(fn,re);
const input=F([2,3],['T','D'],[1,2,3,4,5,6]);
assert.deepEqual(S.reorder(input,['D','T']).data,[1,4,2,5,3,6]);
assert.deepEqual(S.align(input,['B','D','T']).shape,[1,3,2]);
assert.deepEqual(S.align(input,['B','D','T']).data,[1,4,2,5,3,6]);
fails(()=>S.align(input,['B','Tk','D']),/preserve every original named axis/);
fails(()=>S.reorder(input,['T','T']),/exact named-axis permutation/);
fails(()=>S.reorder(input,['T','C']),/exact named-axis permutation/);
assert.deepEqual(S.broadcast(F([3],['D'],[10,20,30]),['T','D'],[2,3]).data,[10,20,30,10,20,30]);
assert.deepEqual(S.binary(input,F([3],['D'],[10,20,30]),'add',['T','D']).data,[11,22,33,14,25,36]);
assert.deepEqual(S.binary(input,F([3],['D'],[2,3,4]),'multiply',['T','D']).data,[2,6,12,8,15,24]);
assert.deepEqual(S.binary(input,F([3],['D'],[1,2,3]),'subtract',['D','T']).data,[-0+0,3,0,3,0,3]);
fails(()=>S.binary(input,F([2],['D'],[7,8]),'add',['T','D']),/named-axis mismatch/);
fails(()=>S.binary(input,F([2],['Tk'],[7,8]),'add',['T','D']),/exact named-axis union/);
fails(()=>S.binary(input,input,'matmul',['T','D']),/operation must/);
fails(()=>S.broadcast(input,['T','D'],[3,3]),/singleton/);
assert.deepEqual(S.slice(input,'T',1,1).data,[4,5,6]);
assert.deepEqual(S.concat([S.slice(input,'T',0,1),S.slice(input,'T',1,1)],'T').data,input.data);
fails(()=>S.slice(input,'T',2,1),/out of bounds/);
fails(()=>S.slice(input,'T',1,2),/outside axis/);
fails(()=>S.concat([input,F([2,2],['T','D'],[1,2,3,4])],'T'),/non-time axis/);
fails(()=>S.concat([input,F([3,2],['D','T'],[1,2,3,4,5,6])],'T'),/identical named-axis/);
// Rank-four reference parity against independent ND numerical broadcast.
const nd=ND.seedTensor([1,2,3,4],7),b=ND.seedTensor([1,1,3,4],11);
const ax=['B','H','T','D'];
const ta=F([1,2,3,4],ax,ND.flatten(nd)),tb=F([1,1,3,4],ax,ND.flatten(b));
const sum=S.binary(ta,tb,'add',ax).data,ndSum=ND.flatten(ND.broadcastBinary(nd,b,(x,y)=>x+y));
assert.deepEqual(sum,ndSum);
const flipped=S.reorder(ta,['B','T','H','D']);
assert.deepEqual(flipped.data,ND.flatten(ND.permute(nd,[0,2,1,3])));
assert.deepEqual(S.reorder(flipped,ax).data,ta.data);
// Rank-five THWC-like tensor demonstrates actual temporal slice/concat calculation.
const video=F([1,3,2,2,2],['B','T','H','W','C'],Array.from({length:24},(_,i)=>i+1));
const frame0=S.slice(video,'T',0,1),last=S.slice(video,'T',2,1);
assert.deepEqual(frame0.shape,[1,1,2,2,2]);
assert.deepEqual(last.data,[17,18,19,20,21,22,23,24]);
const split=[S.slice(video,'T',0,1),S.slice(video,'T',1,1),last];
assert.deepEqual(S.concat(split,'T').data,video.data);
assert.deepEqual(S.reorder(S.reorder(video,['B','C','T','H','W']),video.axes).data,video.data);
const larger=F([1,5,2,2,2],['B','T','H','W','C'],Array.from({length:40},(_,i)=>i));
fails(()=>S.concat(Array(8).fill(larger),'T'),/bounded shape\/capacity/);
// Streaming: absolute position and capacity are semantics, not inferred from matching shapes.
const initial=F([1,2,2],['B','T','D'],[1,2,3,4]);
const one=F([1,1,2],['B','T','D'],[5,6]),alt=F([1,1,2],['B','T','D'],[50,60]);
const st=S.init(initial,{axis:'T',origin:100,capacity:5});
assert.equal(st.nextPosition,102);assert.equal(st.version,0);assert.equal(Object.isFrozen(st.tensor.data),true);
const newState=S.append(st,one,{position:102});
assert.equal(st.version,0);assert.equal(st.nextPosition,102);
assert.deepEqual(st.tensor.data,[1,2,3,4]);assert.equal(newState.version,1);
assert.deepEqual(newState.tensor.data,[1,2,3,4,5,6]);
assert.equal(newState.nextPosition,103);
const window=S.read(newState,101,2);
assert.deepEqual(window.tensor.data,[3,4,5,6]);assert.deepEqual([window.logicalStart,window.logicalEnd],[101,103]);
window.tensor.data[0]=777;assert.deepEqual(newState.tensor.data,[1,2,3,4,5,6]);
const other=S.append(S.fork(st),alt,{position:102});
assert.deepEqual(other.tensor.data,[1,2,3,4,50,60]);
assert.deepEqual(newState.tensor.data,[1,2,3,4,5,6]);
fails(()=>S.append(st,one),/explicit absolute position/);
fails(()=>S.append(newState,one,{position:102}),/out-of-order\/stale/);
fails(()=>S.read(st,101,2),/read beyond populated prefix/);
fails(()=>S.read(newState,99,1),/out of bounds/);
fails(()=>S.append(S.init(initial,{capacity:2}),one,{position:2}),/capacity/);
fails(()=>S.append(newState,F([1,1,3],['B','T','D'],[1,2,3]),{position:103}),/non-time axis mismatch/);
fails(()=>S.init(initial,{axis:'Time'}),/initial tensor must fit/);
fails(()=>S.validateState({...st,nextPosition:103}),/corrupted/);
fails(()=>S.verifyTensor({shape:[1,1],axes:['T','D'],data:[NaN],dtype:'float64'}),/finite/);
// Ordinary legacy graphs are unchanged, including ordered CPU teaching outputs.
for(const lesson of ['gqa','nd_attention_lesson','nd_cached_gqa_decode']){
 const g=Learning.preset(lesson);
 assert.deepEqual(Learning.evaluate(g),Learning.evaluate(JSON.parse(JSON.stringify(g))),lesson+' must retain v1 graph behavior');
}
console.log('SEMANTIC_STATE_PASS exact rank-1..5 named-axis math, ND parity, temporal split, immutable forked streams, invalid state guards, v1 graph compatibility');
