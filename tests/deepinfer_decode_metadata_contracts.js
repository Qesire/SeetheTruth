'use strict';
const assert=require('node:assert/strict'),D=require('../deepinfer_decode_metadata.js');
const seq=(seqId,q,k,blocks,stateId=0,isPlaceholder=false)=>({
 seqId,seqlenQ:q,seqlenK:k,blockTable:blocks,stateId,isPlaceholder});
const input={blockSize:4,maxNumBlocks:4,maxSeqlenQ:3,blockTableCompressRatio:2,
 needDefaultSeqlens:true,sequences:[
 seq('B',2,8,[7,4,99],11),seq('C',1,3,[3,2],12),
 seq('PAD',1,1,[8],13,true)]};
const packed=D.prepare(input);
assert.deepEqual(packed.numBlocks,[2,1,1]);
assert.deepEqual(packed.blockTableOffsets,[0,2,3,4]);
assert.deepEqual(packed.blockTables,[7,4,3,8]);
assert.deepEqual(packed.cuSeqlensQ,[2,3,4]);
assert.deepEqual(packed.stateIds,[11,12,13]);
assert.deepEqual(packed.isPlaceholder,[false,false,true]);
assert.deepEqual(packed.seqlensQ,[2,1,1]);
assert.deepEqual(packed.seqlensK,[8,3,1]);
assert.equal(packed.totalQ,4);
assert.equal(packed.tokensPerBlock,2);
const noDefaults=D.prepare({...input,needDefaultSeqlens:false});
assert.equal(noDefaults.seqlensQ,null);assert.equal(noDefaults.seqlensK,null);
assert.deepEqual(noDefaults.blockTableOffsets,[0,2,3,4]);
const zero=D.prepare({blockSize:4,maxNumBlocks:2,maxSeqlenQ:1,
 blockTableCompressRatio:2,needDefaultSeqlens:true,sequences:[seq('Z',1,1,[])]});
assert.deepEqual(zero.numBlocks,[0],'upstream formula allows non-placeholder 0 compressed blocks');
assert.deepEqual(zero.blockTableOffsets,[0,0]);
assert.throws(()=>D.prepare({...input,blockTableCompressRatio:3}),/divisible/);
assert.throws(()=>D.prepare({...input,sequences:[seq('B',2,10,[7],11)]}),/insufficient/);
assert.throws(()=>D.prepare({...input,maxNumBlocks:1}),/insufficient/);
assert.throws(()=>D.prepare({...input,sequences:[seq('X',5,1,[0])]}),/maxK/);
assert.throws(()=>D.prepare({...input,sequences:[seq('A',1,3,[0]),seq('A',1,3,[1])]}),/duplicate/);
assert.deepEqual(input.sequences[0].blockTable,[7,4,99],'pack must not mutate input');
const previous={mtpInfoWriteSlot:9,soStatus:0,soMatcher:null};
const mtp={infoSlot:5,hasDraftInfo:true,lastOutputToken:77,
 goldenTokens:[11,12,13,14,15,16]};
const state={seqId:'B',cachedTokens:7,outputLen:1,soStatus:2,soMatcher:'opaque',
 mtp:{...mtp}};
const refresh=D.refreshForSampling(previous,state,{nextn:2,outputLen:1});
assert.equal(refresh.metadata.lastOutputToken,77);
assert.equal(refresh.metadata.mtpInfoReadSlot,5);
assert.equal(refresh.metadata.mtpInfoWriteSlot,5);
assert.deepEqual(refresh.metadata.goldenToken,[12,13,14,15,16]);
assert.equal(refresh.metadata.soStatus,2);
assert.equal(refresh.nextSequence.mtp.hasDraftInfo,false);
assert.equal(state.mtp.hasDraftInfo,true,'old slot ownership state must be immutable');
const again=D.refreshForSampling(refresh.metadata,refresh.nextSequence,{nextn:2,outputLen:2});
assert.equal(again.metadata.mtpInfoReadSlot,0,'draft info consumed exactly once');
assert.equal(again.metadata.mtpInfoWriteSlot,5,'write slot retained when active');
const noInfo=D.refreshForSampling({...previous,mtpInfoWriteSlot:9},{
 ...state,mtp:{...mtp,infoSlot:0,hasDraftInfo:false,goldenTokens:[11]}
},{nextn:2,outputLen:1});
assert.equal(noInfo.metadata.mtpInfoWriteSlot,9,'zero info slot does not clear previous write slot');
assert.equal(noInfo.metadata.goldenToken,null);
const maybe=D.updateForSampling(
 packed,{remap:[0,-1,-1],seqlensQ:[2,1,1]},
 [state,{seqId:'C',cachedTokens:4,mtp:{infoSlot:0,hasDraftInfo:false,
 lastOutputToken:-1},outputLen:0},
  {seqId:'PAD',cachedTokens:0,mtp:{infoSlot:0,hasDraftInfo:false,
 lastOutputToken:-1},outputLen:0}],{nextn:2});
assert.deepEqual(maybe.metadata.map(x=>[x.seqId,x.seqlenQ,x.seqlenK,x.refreshed]),[
 ['B',2,9,true],['C',1,5,false],['PAD',1,1,false]]);
assert.equal(maybe.metadata[0].mtpInfoReadSlot,5);
assert.equal(maybe.states[0].mtp.hasDraftInfo,false);
assert.equal(maybe.states[1].mtp.hasDraftInfo,false);
assert.throws(()=>D.updateForSampling(packed,{remap:[0,1,2],seqlensQ:[2,1,1]},
 [state,{seqId:'WRONG',cachedTokens:4},{}],{nextn:2}),/identity/);
const E=require('../deepinfer_worker_events.js');
const SS=(seqId,seqlenQ,seqlenK)=>({seqId,seqlenQ,seqlenK});
const steps=[{mode:'prompt',sequenceIds:['A','B']},
 {mode:'decode',sequenceIds:['A','B']},
 {mode:'decode',sequenceIds:['B','C']},
 {mode:'decode',sequenceIds:['B','C']}];
const fixture={nextn:2,variableMtp:true,records:[
 {sequences:[SS('A',1,8),SS('B',1,9)],accepted:[1,2],confidenceRates:[[.9,.8],[.6,.4]],totalQ:2,cachedTokens:[7,8]},
 {sequences:[SS('A',1,10),SS('B',1,12)],accepted:[1,2],confidenceRates:[[.6,.4],[.9,.8]],totalQ:2,cachedTokens:[9,11]},
 {sequences:[SS('B',1,13),SS('C',1,5)],accepted:[1,0],confidenceRates:[[.8,.7],[.5,.4]],totalQ:3,cachedTokens:[12,4],paddedLength:3},
 {sequences:[SS('B',1,14),SS('C',1,6)],accepted:[2,1],confidenceRates:[[.9,.8],[.5,.3]],totalQ:3,cachedTokens:[13,5]}
]};
const dx={step:2,input:{
 blockSize:4,maxNumBlocks:6,maxSeqlenQ:3,blockTableCompressRatio:2,
 needDefaultSeqlens:true,sequences:[
  seq('B',1,13,[7,4,8,9],21),seq('C',1,5,[2,3],22)
 ]},states:[
 {seqId:'B',cachedTokens:12,outputLen:1,soStatus:2,soMatcher:'example',mtp:{
  infoSlot:5,hasDraftInfo:true,lastOutputToken:88,goldenTokens:[10,11,12,13,14,15]}},
 {seqId:'C',cachedTokens:4,outputLen:0,mtp:{
  infoSlot:0,hasDraftInfo:false,lastOutputToken:-1,goldenTokens:null}}
]};
const ev=E.simulate(steps,{zeroOverhead:true,metadataFixture:fixture,decodeFixture:dx});
assert.equal(ev.metadataTrace.length,2);
assert.ok(!ev.metadataTrace[1].decode,'only requested Step decodes packed metadata');
const decoded=ev.metadataTrace[0].decode;
assert.deepEqual(decoded.packed.blockTableOffsets,[0,4,6]);
assert.deepEqual(decoded.packed.blockTables,[7,4,8,9,2,3]);
assert.deepEqual(decoded.packed.cuSeqlensQ,[1,2]);
assert.deepEqual(decoded.sampling.metadata.map(m=>[m.seqId,m.seqlenQ,m.seqlenK]),[
 ['B',2,14],['C',1,5]]);
assert.equal(decoded.sampling.metadata[0].mtpInfoReadSlot,5);
assert.equal(decoded.sampling.states[0].mtp.hasDraftInfo,false);
assert.deepEqual(decoded.sampling.metadata[0].goldenToken,[11,12,13,14,15]);
assert.deepEqual(dx.states[0].mtp.hasDraftInfo,true);
assert.equal(E.simulate(steps,{zeroOverhead:false,metadataFixture:fixture,decodeFixture:dx}).metadataTrace.length,0);
const broken=structuredClone(dx);broken.input.sequences[0].blockTable=[7,4];
assert.throws(()=>E.simulate(steps,{zeroOverhead:true,metadataFixture:fixture,decodeFixture:broken}),/insufficient/);
const incorrect=structuredClone(dx);incorrect.input.sequences[0].seqlenK=12;
assert.throws(()=>E.simulate(steps,{zeroOverhead:true,metadataFixture:fixture,decodeFixture:incorrect}),/exact current-step/);
console.log('DEEPINFER_DECODE_METADATA_PASS compressed offsets, q-prefix, slot lifecycle, golden tokens, validation');
