'use strict';
const assert=require('node:assert/strict');
const ND=require('../nd_tensor.js'),L=require('../learning_lab.js'),C=require('../concept_flow.js'),R=require('../graph_readiness.js'),B=require('../block_catalog.js');
const close=(a,b,eps=1e-11)=>assert.ok(Math.abs(a-b)<eps, String(a)+" != "+String(b));
const source=ND.seedTensor([1,2,3,4],7);
assert.deepEqual(ND.shape(source),[1,2,3,4]);
const rotated=ND.permute(source,[0,2,1,3]),restored=ND.permute(rotated,[0,2,1,3]);
assert.deepEqual(ND.shape(rotated),[1,3,2,4]);assert.deepEqual(restored,source);
const flat=ND.reshape(source,[6,4]),roundtrip=ND.reshape(flat,[1,2,3,4]);
assert.deepEqual(roundtrip,source);
assert.deepEqual(ND.flatten(source),ND.flatten(roundtrip));
const bias=ND.seedTensor([1,1,3,4],11),summed=ND.broadcastBinary(source,bias,(a,b)=>a+b);
for(let h=0;h<2;h++)for(let t=0;t<3;t++)for(let d=0;d<4;d++)
 close(summed[0][h][t][d],source[0][h][t][d]+bias[0][0][t][d]);
const x=ND.reduceSum(source,2,true);
assert.deepEqual(ND.shape(x),[1,2,1,4]);
for(let h=0;h<2;h++)for(let d=0;d<4;d++)
 close(x[0][h][0][d],source[0][h].reduce((s,row)=>s+row[d],0));
const w=ND.seedTensor([1,2,4,3],5),scores=ND.matmul4(source,w);
assert.deepEqual(ND.shape(scores),[1,2,3,3]);
for(let h=0;h<2;h++)for(let t=0;t<3;t++)for(let o=0;o<3;o++){
 let reference=0;for(let k=0;k<4;k++)reference+=source[0][h][t][k]*w[0][h][k][o];
 close(scores[0][h][t][o],reference);
}
const shared=ND.seedTensor([1,1,4,3],2),broadcast=ND.matmul4(source,shared);
for(let h=0;h<2;h++)for(let t=0;t<3;t++)for(let o=0;o<3;o++){
 let reference=0;for(let k=0;k<4;k++)reference+=source[0][h][t][k]*shared[0][0][k][o];
 close(broadcast[0][h][t][o],reference);
}
assert.throws(()=>ND.broadcastBinary(source,ND.seedTensor([1,3,3,4],1),(a,b)=>a+b),/incompatible/);
assert.throws(()=>ND.matmul4(source,ND.seedTensor([1,2,3,4],1)),/dimensions mismatch/);
assert.throws(()=>ND.reshape(source,[1,2,4,4]),/count mismatch/);
assert.throws(()=>ND.permute(source,[0,0,1,3]),/permutation/);
assert.throws(()=>ND.shape([[1,2],[3]]),/ragged/);
assert.throws(()=>ND.seedTensor([8,8,8,8],1),/512/);
const recipe=L.preset("nd_attention_lesson");
assert.equal(recipe.nodes.length,9);
const concept=C.evaluate(recipe),numeric=L.evaluate(recipe);
assert.deepEqual(concept.steps.map(s=>s.shape),numeric.steps.map(s=>s.shape));
assert.equal(numeric.totalKnownFlops,144,"two-head batched GEMM count 2*B*H*M*N*K");
const shapes=Object.fromEntries(numeric.steps.map(s=>[s.id,s.shape]));
assert.deepEqual(shapes.batch_scores,[1,2,3,3]);
assert.deepEqual(shapes.biased,[1,2,3,3]);
assert.deepEqual(shapes.summed,[1,2,1,3]);
assert.deepEqual(shapes.flat,[2,3]);
assert.deepEqual(shapes.expand,[1,1,2,3]);
assert.deepEqual(shapes.perm,[1,2,1,3]);
const values=Object.fromEntries(numeric.steps.map(s=>[s.id,s.values]));
assert.deepEqual(values.perm,ND.permute(values.expand,[0,2,1,3]));
const ready=R.check(recipe,{numeric:true});
assert.equal(ready.executable,true);assert.equal(ready.numeric.steps,9);
const wrong=JSON.parse(JSON.stringify(recipe));
wrong.nodes.find(n=>n.id==='keys').params.t=3;
assert.equal(R.check(wrong).executable,false);
assert.equal(R.check(wrong).nodes.find(n=>n.id==='batch_scores').status,'shape-error');
const extra={schema:L.SCHEMA,title:'Old/new bridging',
 nodes:[{id:'old',kind:'input',x:0,y:0,params:{rows:3,width:4,seed:2}},
 {id:'new',kind:'nd_expand4d',x:220,y:0,params:{}},
 {id:'back',kind:'nd_flatten',x:440,y:0,params:{}},
 {id:'lossless',kind:'nd_reshape',x:660,y:0,params:{b:1,h:1,t:3,d:4}}],
 edges:[{from:'old',to:'new',port:'x'},{from:'new',to:'back',port:'x'},{from:'back',to:'lossless',port:'x'}]};
const evalOld=L.evaluate(extra),step=Object.fromEntries(evalOld.steps.map(s=>[s.id,s.values]));
assert.deepEqual(step.old,step.back);
assert.deepEqual(ND.flatten(step.new),ND.flatten(step.lossless));
console.log('N-D numerical acceptance PASS: 8 operators, exact 4D permute/broadcast/matmul/reduce, 9-node lesson, legacy 2D interchange and shape errors');
