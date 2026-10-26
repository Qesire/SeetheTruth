'use strict';
/* Independently specified 5D coordinate mathematics. SemanticState is the
 * production teaching path, whereas this oracle uses flat scalar indexing.
 * No graph-codegen assertion is made for unexportable named-axis pack kinds.
 */
const assert=require('node:assert/strict');
const IR=require('../tensor_ir.js'),S=require('../semantic_state.js');
const shape=[1,3,2,2,2],axes=['B','T','H','W','C'],count=shape.reduce((a,b)=>a*b,1);
const video=Array.from({length:count},(_,i)=>0.23*Math.sin(0.71*(i+1))+0.03*(i%7));
const bias=[0.17,-0.24],gain=[0.8,1.1,-0.6];
const tensor=(s,a,d)=>IR.make(s,a,d);
const x=tensor(shape,axes,video);
const b=tensor([2],['C'],bias),g=tensor([3],['T'],gain);
const biased=S.binary(x,b,'add',axes);
const gated=S.binary(biased,g,'multiply',axes);
const slices=[S.slice(gated,'T',0,2),S.slice(gated,'T',2,1)];
const rejoined=S.concat(slices,'T');
const permuted=S.reorder(rejoined,['B','C','H','W','T']);
const restored=S.reorder(permuted,axes);
const ix=(t,h,w,c)=>(((t*2+h)*2+w)*2+c);
const scalarBiased=video.map((v,i)=>v+bias[i%2]);
const scalarGated=video.map((v,i)=>scalarBiased[i]*gain[Math.floor(i/8)]);
const loss=scalarGated.reduce((s,v)=>s+v*v,0)/count;
const gx=video.map((_,i)=>2*scalarGated[i]*gain[Math.floor(i/8)]/count);
const gb=bias.map((_,c)=>{
 let s=0;for(let t=0;t<3;t++)for(let h=0;h<2;h++)for(let w=0;w<2;w++)
  s+=gx[ix(t,h,w,c)];return s;
});
const gg=gain.map((_,t)=>{
 let s=0;for(let h=0;h<2;h++)for(let w=0;w<2;w++)for(let c=0;c<2;c++){
  const i=ix(t,h,w,c);s+=2*scalarGated[i]*scalarBiased[i]/count;
 }return s;
});
const close=(label,a,b)=>{
 assert.equal(a.length,b.length,label+' count');
 for(let i=0;i<a.length;i++)assert.ok(Math.abs(a[i]-b[i])<=1e-12,label+'['+i+']');
};
close('5D biased numeric',biased.data,scalarBiased);
close('5D gated numeric',gated.data,scalarGated);
close('5D split-concat',rejoined.data,scalarGated);
close('5D semantic permutation round trip',restored.data,scalarGated);
assert.deepEqual(permuted.shape,[1,2,2,2,3]);
assert.deepEqual(permuted.axes,['B','C','H','W','T']);
// Two same-sized axes cannot silently be substituted: named W and C both
// have extent 2 but yield different, internally legal numerical operations.
const wrong=S.binary(x,tensor([2],['W'],bias),'add',axes);
assert.ok(wrong.data.some((v,i)=>Math.abs(v-scalarBiased[i])>1e-3),
 'same numerical size is not equivalent named-axis semantics');
assert.throws(()=>S.binary(x,tensor([4],['C'],[0,1,2,3]),'add',axes),/named-axis mismatch/);
assert.throws(()=>S.slice(gated,'T',2,2),/outside axis/);
assert.throws(()=>S.concat([slices[0],S.reorder(slices[1],['B','H','T','W','C'])],'T'),
 /identical named-axis/);
assert.throws(()=>S.reorder(x,['B','C','C','W','T']),/exact named-axis permutation/);
assert.throws(()=>tensor(shape,axes,[...video.slice(0,-1),NaN]),/finite/);
const payload={schema:'seethetruth-5d-semantic-numerical-v1',shape,axes,
 video,bias,gain,
 native:{biased:biased.data,gated:gated.data,rejoined:rejoined.data,
  permuted:permuted.data,restored:restored.data},
 independent:{biased:scalarBiased,gated:scalarGated,loss,
  gradients:{video:gx,bias:gb,gain:gg}}};
module.exports=Object.freeze({payload});
if(require.main===module){
 console.log('SEMANTIC_5D_FAST_PASS '+JSON.stringify({
  shape,forwardCoordinates:count*3,gradientCoordinates:gx.length+gb.length+gg.length,
  negativeContracts:6,source:'SemanticState vs standalone scalar coordinates'
 }));
}
