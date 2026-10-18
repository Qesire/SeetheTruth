'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),ND=require('../nd_tensor.js'),AD=require('../graph_autodiff.js'),C=require('../concept_flow.js'),R=require('../graph_readiness.js');
const close=(x,y,eps=3e-6)=>assert.ok(Math.abs(x-y)<eps,x+' != '+y);
function manual(x,k,v,target,ratio){
 const rq=ND.shape(x)[1],rk=ND.shape(k)[1];assert.equal(rq/rk,ratio);
 const kg=ND.repeatKVHeads(k,rq),vg=ND.repeatKVHeads(v,rq);
 const kt=ND.permute(kg,[0,1,3,2]);
 const logits=ND.matmul4(x,kt);
 const scores=ND.scale(logits,Math.SQRT1_2);
 const p=ND.softmaxLast(ND.causalMask4(scores));
 const y=ND.matmul4(p,vg);
 return {loss:ND.meanSquaredError(y,target)[0][0],p,y};
}
for(const name of ['nd_gqa_attention','nd_mqa_attention']){
 const g=L.preset(name),f=L.evaluate(g),sym=C.evaluate(g),back=AD.backward(g,f);
 assert.equal(g.nodes.length,13);
 assert.deepEqual(sym.steps.map(s=>s.shape),f.steps.map(s=>s.shape));
 assert.equal(R.check(g,{numeric:true}).executable,true);
 assert.equal(back.targetId,'loss');
 const values=Object.fromEntries(f.steps.map(s=>[s.id,s.values]));
 const hkv=ND.shape(values.key)[1],ratio=4/hkv;
 const reference=manual(values.query,values.key,values.value,values.target,ratio);
 close(back.loss,reference.loss,1e-12);
 ND.flatten(values.context).forEach((x,i)=>close(x,ND.flatten(reference.y)[i],1e-12));
 for(let h=0;h<4;h++)for(let q=0;q<3;q++){
   const row=values.probabilities[0][h][q];
   close(row.reduce((a,b)=>a+b,0),1,1e-12);
   for(let k=q+1;k<3;k++)assert.equal(row[k],0);
 }
 for(const leaf of ['query','key','value']){
  const tensor=values[leaf],gradient=back.gradients[leaf],shape=ND.shape(tensor);
  assert.deepEqual(ND.shape(gradient),shape);
  let checks=0;
  ND.indices(shape,idx=>{
    const h=1e-6;
    const plus=ND.build(shape,pos=>ND.at(tensor,pos)+(idx.every((v,i)=>v===pos[i])?h:0));
    const minus=ND.build(shape,pos=>ND.at(tensor,pos)-(idx.every((v,i)=>v===pos[i])?h:0));
    const loss=(trial)=>manual(leaf==='query'?trial:values.query,leaf==='key'?trial:values.key,
       leaf==='value'?trial:values.value,values.target,ratio).loss;
    close(ND.at(gradient,idx),(loss(plus)-loss(minus))/(2*h),4e-6);
    checks++;
  });
  assert.equal(checks,ND.product(shape));
 }
 assert.ok(back.cutEdges.some(edge=>edge.port==='target'&&edge.nodeId==='loss'));
 assert.ok(ND.flatten(back.gradients.target).every(x=>x===0));
 // Grouped KV rows must collect all Query heads that share them.
 const expected=(head)=>Math.floor(head/ratio);
 for(let h=0;h<4;h++)assert.equal(expected(h),Math.floor(h/ratio));
 console.log(name+' PASS: full source VJPs finite differences, Hkv='+hkv+', scalar loss '+back.loss);
}
const x=ND.seedTensor([1,2,3,2],2),rep=ND.repeatKVHeads(x,4);
assert.deepEqual(rep[0][0],rep[0][1]);assert.deepEqual(rep[0][2],rep[0][3]);
assert.notDeepEqual(rep[0][0],rep[0][2]);
assert.throws(()=>ND.repeatKVHeads(x,3),/multiple/);
const mqa=ND.repeatKVHeads(ND.seedTensor([1,1,3,2],1),4);
assert.deepEqual(mqa[0][0],mqa[0][3]);
const wrong=L.preset('nd_gqa_attention');
wrong.nodes.find(n=>n.id==='key').params.h=3;
assert.equal(R.check(wrong).executable,false);
assert.equal(R.check(wrong).nodes.find(n=>n.id==='key_groups').status,'shape-error');
console.log('GQA/MQA end-to-end 4D backward PASS: 84 elementwise central finite differences, detach and group repeat');
