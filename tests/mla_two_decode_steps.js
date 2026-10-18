'use strict';
const assert=require('node:assert/strict');
const B=require('../block_catalog.js');
const run=(kind,inputs,params={})=>B.execute({id:kind,kind,params},inputs).values;
const copy=x=>x.map(r=>r.slice()),mul=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0)));
const T=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const softmax=row=>{const mx=Math.max(...row),z=row.map(v=>Math.exp(v-mx)),sum=z.reduce((a,b)=>a+b,0);return z.map(v=>v/sum)};
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10, a+' != '+b);
const Wk=[[.8,-.3],[.2,.6]],Wv=[[.2,.7],[-.4,.5]];
let latent=[[1,.5],[-.2,.8]];
let keys=run('rope2d',{x:[[.6,.3],[-.1,.5]]},{offset:0});
for(const step of [
 {pos:2,c:[[.7,-.1]],rawKey:[[.5,.4]],q:[[.4,.8]],qpe:[[.6,-.5]]},
 {pos:3,c:[[.2,.9]],rawKey:[[-.3,.1]],q:[[-.3,.7]],qpe:[[.4,.2]]}
]){
 const oldC=copy(latent),oldP=copy(keys);
 const rotated=run('rope2d',{x:step.rawKey},{offset:step.pos});
 latent=run('cache_write_segment',{cache:latent,segment:step.c},{start:step.pos,capacity:6});
 keys=run('cache_write_segment',{cache:keys,segment:rotated},{start:step.pos,capacity:6});
 const C=run('cache_read_prefix',{cache:latent},{length:step.pos+1});
 const P=run('cache_read_prefix',{cache:keys},{length:step.pos+1});
 assert.deepEqual(oldC,latent.slice(0,step.pos),'cached latent history changed');
 assert.deepEqual(oldP,keys.slice(0,step.pos),'cached PE history changed');
 assert.deepEqual(P[step.pos],rotated[0],'PE of new token must use absolute decode offset');
 const Qpe=run('rope2d',{x:step.qpe},{offset:step.pos});
 const K=mul(C,Wk);
 const contentScores=mul(step.q,T(K))[0],qAbs=run('query_weight_absorb',{query:step.q,keyWeight:Wk});
 const absorbedScores=mul(qAbs,T(C))[0],positionScores=mul(Qpe,T(P))[0];
 const sDirect=softmax(contentScores.map((v,i)=>(v+positionScores[i])/2));
 const sAbs=softmax(absorbedScores.map((v,i)=>(v+positionScores[i])/2));
 const vExplicit=mul(C,Wv),outDirect=mul([sDirect],vExplicit)[0],outAbs=mul(mul([sAbs],C),Wv)[0];
 for(let i=0;i<step.pos+1;i++){close(contentScores[i],absorbedScores[i]);close(sDirect[i],sAbs[i]);}
 for(let d=0;d<2;d++)close(outDirect[d],outAbs[d]);
 // the previous stage remains an independent immutable snapshot
 assert.deepEqual(oldC,oldC.map(x=>x.slice()));
}
assert.deepEqual(latent.length,4);assert.deepEqual(keys.length,4);
assert.throws(()=>run('cache_read_prefix',{cache:latent},{length:5}),/committed prefix/);
console.log('Two decode steps MLA cache history, rotary positions and optimized/explicit attention parity PASS');
