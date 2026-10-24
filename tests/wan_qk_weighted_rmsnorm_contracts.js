'use strict';
// Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff
// model.py WanRMSNorm 69-85, WanSelfAttention 126-155.
const assert=require('node:assert/strict');
const ND=require('../nd_tensor.js'),V=require('../nd_autodiff.js'),
 L=require('../learning_lab.js'),C=require('../concept_flow.js'),
 E=require('../element_lineage.js'),G=require('../graph_autodiff.js');
const same=(a,b,eps=2e-7,label='')=>assert.ok(Math.abs(a-b)<=eps,label+': '+a+' ≠ '+b);
const independent=(x,w,eps)=>{
 const sh=ND.shape(x),C=sh[3];
 return ND.build(sh,([b,h,t,c])=>{
  let sq=0;for(let j=0;j<C;j++)sq+=x[b][0][t][j]*x[b][0][t][j];
  return x[b][0][t][c]*w[0][0][0][c]/Math.sqrt(sq/C+eps);
 });
};
const change=(x,idx,delta)=>ND.build(ND.shape(x),at=>
 ND.at(x,at)+(at.every((v,i)=>v===idx[i])?delta:0));
const dot=(a,b)=>ND.flatten(a).reduce((s,x,i)=>s+x*ND.flatten(b)[i],0);
for(const dims of [[1,1,3,6],[2,1,4,8],[1,1,8,6]]){
 const X=ND.seedTensor(dims,4),W=ND.build([1,1,1,dims[3]],i=>
  1+ND.at(ND.seedTensor([1,1,1,dims[3]],7),i)*0.3),upstream=ND.seedTensor(dims,3);
 const eps=1e-6,actual=ND.wanQKWeightedRMSNorm4(X,W,eps),
  expected=independent(X,W,eps);
 ND.indices(dims,idx=>same(ND.at(actual,idx),ND.at(expected,idx),1e-13,'forward'));
 const v=V.vjp('nd_wan_qk_rmsnorm',{x:X,weight:W},{epsilon:eps},upstream,actual);
 for(const [port,tensor] of [['x',X],['weight',W]]){
  ND.indices(ND.shape(tensor),idx=>{
    const before=port==='x'?independent(change(X,idx,-1e-6),W,eps):independent(X,change(W,idx,-1e-6),eps);
    const after=port==='x'?independent(change(X,idx,1e-6),W,eps):independent(X,change(W,idx,1e-6),eps);
    same(ND.at(v[port],idx),(dot(after,upstream)-dot(before,upstream))/2e-6,3e-8,
     'independent coordinate '+port+idx);
  });
 }
 const analyzed=C.analyze({id:'norm',kind:'nd_wan_qk_rmsnorm',params:{epsilon:eps}},
  {x:dims,weight:[1,1,1,dims[3]]});
 assert.deepEqual(analyzed.shape,dims);
}
// Off-channel coupling: full width C=6 is NOT equivalent to per-head D=3.
// Distinct head-scale patterns ensure the two denominators disagree.
const x=[[[[1,2,3,4,5,6]]]],w=[[[[1,1,1,1,1,1]]]];
const full=ND.wanQKWeightedRMSNorm4(x,w);
const perHead=x[0][0][0].map((v,c)=>v/Math.sqrt(
 [0,1,2].reduce((s,j)=>s+x[0][0][0][Math.floor(c/3)*3+j]**2,0)/3+1e-6));
assert.ok(Math.abs(full[0][0][0][2]-perHead[2])>0.1,'head-local normalization is mathematically different');
const upstream=ND.seedTensor([1,1,1,6],2);
const backward=V.vjp('nd_wan_qk_rmsnorm',{x,weight:w},{epsilon:1e-6},upstream,full);
assert.ok(Math.abs(backward.x[0][0][0][1])>0.00001,'off-channel cross derivative');
for(const args of [
 [ND.seedTensor([1,2,3,6],1),w,1e-6],
 [ND.seedTensor([1,1,3,6],1),ND.seedTensor([1,1,2,6],1),1e-6],
 [ND.seedTensor([1,1,3,6],1),w,-1e-6]
])assert.throws(()=>ND.wanQKWeightedRMSNorm4(...args),/Wan QK RMSNorm/);
const graph=L.preset('nd_wan_qkv_fullwidth_attention'),trace=L.evaluate(graph),
 by=id=>trace.steps.find(s=>s.id===id).values;
assert.equal(graph.nodes.length,36);
assert.deepEqual(ND.shape(by('loss')),[1,1]);
for(const branch of ['q','k']){
 const proj=ND.matmul4(by('attention_input'),by('w_'+branch)),
  biased=ND.broadcastBinary(proj,by('b_'+branch),(a,b)=>a+b),
  normalized=independent(biased,by('gamma_'+branch),1e-6);
 ND.indices([1,1,8,6],idx=>same(ND.at(by(branch+'_norm'),idx),
  ND.at(normalized,idx),1e-12,'projection/norm '+branch));
 const viewed=ND.permute(ND.reshape(normalized,[1,8,1,6]),[0,2,1,3]);
 assert.deepEqual(viewed,by(branch+'_heads'));
}
const fullGraph=G.backward(graph,trace);
assert.equal(fullGraph.targetId,'loss');
for(const id of ['w_q','w_k','w_v','w_o','gamma_q','gamma_k'])
 assert.deepEqual(ND.shape(fullGraph.gradients[id]),ND.shape(by(id)));
const dGamma=fullGraph.gradients.gamma_q[0][0][0];
assert.ok(dGamma.some(v=>Math.abs(v)>1e-10),'learned gamma receives end-to-end loss gradients');
const p0=E.inspect(graph,{targetId:'q_norm',outputIndex:[0,0,2,1],trace,maxVertices:3000,maxEdges:13000});
assert.equal(p0.complete,true);
assert.equal(p0.vertices.find(v=>v.id==='q_norm').stages[0].stage,'wan_qk_fullwidth_rmsnorm');
assert.equal(p0.edges.filter(e=>e.from==='q_norm@0,0,2,1'&&e.port==='x').length,6);
assert.ok(p0.vertices.some(v=>v.id==='gamma_q'));
const sample=(id,index,sign)=>{
 const changed=structuredClone(graph),node=changed.nodes.find(n=>n.id===id),a=node.params.values;
 let arr=a;for(let i=0;i<index.length-1;i++)arr=arr[index[i]];
 arr[index[index.length-1]]+=sign*1e-5;
 return L.evaluate(changed).steps.find(s=>s.id==='loss').values[0][0];
};
for(const [id,index]of [
 ['gamma_q',[0,0,0,0]],['gamma_q',[0,0,0,4]],
 ['gamma_k',[0,0,0,1]],['w_q',[0,0,1,4]],['w_v',[0,0,2,3]],['w_o',[0,0,4,2]]
]){
 const fd=(sample(id,index,1)-sample(id,index,-1))/(2e-5);
 same(ND.at(fullGraph.gradients[id],index),fd,2e-6,'full-loss upstream '+id+index);
}
const stale=structuredClone(trace);stale.steps.find(s=>s.id==='q_norm').values[0][0][0][0]+=.3;
assert.throws(()=>E.inspect(graph,{targetId:'q_norm',outputIndex:[0,0,0,0],trace:stale}),/stale or altered/);
console.log('WAN_QK_WEIGHTED_RMSNORM_PASS: all 3-dimensional settings, independent local and full-loss gamma/weight FD, source ordering and wrong per-head counterexample');
