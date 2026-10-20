'use strict';
const assert=require('node:assert/strict'),Z=require('../deepinfer_zos_metadata.js'),
 E=require('../deepinfer_worker_events.js');
const S=(seqId,q,k,isPlaceholder=false)=>({seqId,seqlenQ:q,seqlenK:k,isPlaceholder});
const prev=[S('A',1,12),S('B',2,17),S('C',1,9)];
const curr=[S('B',1,18),S('D',1,5),S('A',1,13)];
const prepared=Z.prepare(curr,prev,{nextn:3,paddedLength:5});
assert.equal(prepared.hasOverlap,true);
assert.equal(prepared.sameOrder,false);
assert.deepEqual(prepared.remap,[1,-1,0,-1,-1]);
assert.deepEqual(prepared.seqlensQ,[1,1,1,1,1]);
assert.deepEqual(prepared.seqlensKBase,[17,4,12,1,1]);
const corrected=Z.correct(prepared,{accepted:[1,2,0]});
assert.deepEqual(corrected.seqlensQ,[1,1,1,1,1]);
assert.deepEqual(corrected.seqlensK,[17,5,11,1,1],'reference in-place adjustment');
assert.equal(corrected.d2hSeqlensQ,null,'fixed query mode has no varlen D2H');
assert.deepEqual(Z.samplingView(prepared,corrected,[17,4,12]),[
 {seqId:'B',remap:1,seqlenQ:1,seqlenK:18,refreshForSamplingRequired:true},
 {seqId:'D',remap:-1,seqlenQ:1,seqlenK:5,refreshForSamplingRequired:false},
 {seqId:'A',remap:0,seqlenQ:1,seqlenK:13,refreshForSamplingRequired:true}
]);
const same=Z.prepare([S('A',1,4),S('B',1,6)],[S('A',2,4),S('B',1,5)],
 {nextn:2,paddedLength:4});
assert.equal(same.sameOrder,true);
assert.deepEqual(same.remap,[0,1,-1,-1]);
assert.deepEqual(same.seqlensKBase,[3,5,1,1]);
assert.equal(Z.prepare([S('D',1,3)],[S('B',1,7)],{nextn:2}),null);
assert.equal(Z.prepare([S('A',1,3)],[S('A',1,5,true)],{nextn:2}),null,
 'prior placeholder must be excluded from mapping');
assert.deepEqual(Z.prepare([S('B',1,6),S('A',1,5)],[S('A',1,5),S('B',1,5)],
 {nextn:0}).remap,[1,0]);
const rate=Z.prepare([S('B',1,18),S('D',1,5),S('A',1,13)],prev,{nextn:2});
const varlen=Z.correct(rate,{
 accepted:[1,2,0],variableMtp:true,totalQ:5,
 confidenceRates:[[.9,.8],[.6,.7],[.5,.1]]
});
assert.deepEqual(varlen.seqlensQ,[2,1,2],
 'global cumulative confidence ranks: A=0.9, B=0.6 tie? order');
assert.deepEqual(varlen.seqlensK,[18,5,12]);
assert.deepEqual(varlen.d2hSeqlensQ,[2,1,2]);
const changed=Z.correct(rate,{accepted:[1,0,0]});
assert.notDeepEqual(changed.seqlensK,Z.correct(rate,{accepted:[1,2,0]}).seqlensK,
 'previous acceptance must genuinely change K');
assert.throws(()=>Z.prepare([S('A',1,6),S('A',1,8)],prev,{nextn:2}),/duplicate/);
assert.throws(()=>Z.prepare([S('A',2,1)],prev,{nextn:2}),/negative/);
assert.throws(()=>Z.prepare([S('A',1,6)],prev,{nextn:2,paddedLength:0}),/paddedLength/);
assert.throws(()=>Z.correct(rate,{accepted:[3,0,0]}),/accepted/);
assert.throws(()=>Z.correct(rate,{accepted:[1,2,0],variableMtp:true,totalQ:9,
 confidenceRates:[[.9,.8],[.7,.6],[.5,.4]]}),/totalQ/);
const steps=[{mode:'prompt',sequenceIds:['A','B']},
 {mode:'decode',sequenceIds:['A','B']},
 {mode:'decode',sequenceIds:['B','C']},
 {mode:'decode',sequenceIds:['B','C']}];
const fixture={nextn:2,variableMtp:true,records:[
 {sequences:[S('A',1,8),S('B',1,9)],accepted:[1,2],confidenceRates:[[.9,.8],[.6,.4]],totalQ:2,cachedTokens:[7,8]},
 {sequences:[S('A',1,10),S('B',1,12)],accepted:[1,2],confidenceRates:[[.6,.4],[.9,.8]],totalQ:2,cachedTokens:[9,11]},
 {sequences:[S('B',1,13),S('C',1,5)],accepted:[1,0],confidenceRates:[[.8,.7],[.5,.4]],totalQ:3,cachedTokens:[12,4],paddedLength:3},
 {sequences:[S('B',1,14),S('C',1,6)],accepted:[2,1],confidenceRates:[[.9,.8],[.5,.3]],totalQ:3,cachedTokens:[13,5]}
]};
const outcome=E.simulate(steps,{zeroOverhead:true,metadataFixture:fixture});
assert.equal(outcome.metadataTrace.length,2,'prompt immediately posts and has no pending state');
assert.deepEqual(outcome.metadataTrace[0].remap,[1,-1,-1]);
assert.deepEqual(outcome.metadataTrace[0].seqlensQ,[2,1,1]);
assert.deepEqual(outcome.metadataTrace[0].seqlensK,[14,5,1]);
assert.deepEqual(outcome.metadataTrace[0].previousAccepted,[1,2]);
assert.deepEqual(outcome.metadataTrace[0].sampler,[
 {seqId:'B',remap:1,seqlenQ:2,seqlenK:14,refreshForSamplingRequired:true},
 {seqId:'C',remap:-1,seqlenQ:1,seqlenK:5,refreshForSamplingRequired:false}
]);
const correction=outcome.events.find(x=>x.id==='s2-correct_zos_metadata');
assert.deepEqual(correction.meta.seqlensQ,[2,1,1]);
assert.equal(correction.meta.d2h,true);
const immediate=E.simulate(steps,{zeroOverhead:false,metadataFixture:fixture});
assert.equal(immediate.metadataTrace.length,0);
const without=E.simulate(steps,{zeroOverhead:true});
assert.equal(without.metadataTrace.length,0,'do not invent numerical metadata fixtures');
assert.throws(()=>E.simulate(steps,{zeroOverhead:true,metadataFixture:{
 ...fixture,records:fixture.records.slice(0,3)
}}),/one record per step/);
console.log('DEEPINFER_ZOS_METADATA_PASS C++ remap, padding, accepted-K, confidence Q, sampling view and event integration');
