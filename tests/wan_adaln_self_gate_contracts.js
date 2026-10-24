'use strict';
// Official Wan2.2 model.py@1ea34ff48f87168174e12956e200b1d908b1c5ff
// WanAttentionBlock.forward lines 237–258. Actual math, independent oracle.
const assert=require('node:assert/strict');
const ND=require('../nd_tensor.js'),V=require('../nd_autodiff.js'),
 L=require('../learning_lab.js'),C=require('../concept_flow.js'),
 E=require('../element_lineage.js'),G=require('../graph_autodiff.js'),
 M=require('../compute_microscope.js'),Bridge=require('../coordinate_vjp_bridge.js');
function near(a,b,tol=3e-7,note=''){assert.ok(Math.abs(a-b)<=tol,
 note+' expected '+b+' actual '+a);}
const eq=(a,b,tol=1e-12)=>ND.indices(ND.shape(a),idx=>near(ND.at(a,idx),ND.at(b,idx),tol,idx.join(',')));
const plus=(a,b)=>ND.broadcastBinary(a,b,(x,y)=>x+y);
const times=(a,b)=>ND.broadcastBinary(a,b,(x,y)=>x*y);
const fullNorm=(x,eps)=>ND.build(ND.shape(x),([b,h,t,c])=>{
 const v=x[b][h][t],mean=v.reduce((a,b)=>a+b,0)/v.length,
 var0=v.reduce((sum,a)=>sum+(a-mean)**2,0)/v.length;
 return (v[c]-mean)/Math.sqrt(var0+eps);
});
const bump=(x,coord,d)=>ND.build(ND.shape(x),idx=>ND.at(x,idx)+
 (idx.every((v,i)=>v===coord[i])?d:0));
const dot=(x,g)=>ND.flatten(x).reduce((sum,a,i)=>sum+a*ND.flatten(g)[i],0);
for(const sh of [[1,1,8,6],[1,2,4,5],[2,1,3,6]]){
 const x=ND.seedTensor(sh,5),epsilon=1e-6,up=ND.seedTensor(sh,8),
    expected=fullNorm(x,epsilon),actual=ND.layerNormLast4(x,epsilon);
 eq(actual,expected,1e-13);
 const grads=V.vjp('nd_layernorm_last',{x},{epsilon},up,actual).x;
 ND.indices(sh,idx=>{
  const h=1e-6,
    fd=(dot(fullNorm(bump(x,idx,h),epsilon),up)-
      dot(fullNorm(bump(x,idx,-h),epsilon),up))/(2*h);
  near(ND.at(grads,idx),fd,2e-9,'LayerNorm derivative '+idx);
 });
 const n={id:'ln',kind:'nd_layernorm_last',params:{epsilon}},
  concept=C.analyze(n,{x:sh});
 assert.deepEqual(concept.shape,sh);
}
const a=ND.seedTensor([1,1,8,6],3),b=ND.seedTensor([1,1,1,6],4),
 up=ND.seedTensor([1,1,8,6],7),prod=times(a,b);
 const g=V.vjp('nd_broadcast_mul',{a,b},{},up,prod);
 for(const [name,value]of [['a',a],['b',b]])ND.indices(ND.shape(value),idx=>{
  const h=1e-6,
   f=(d)=>dot(name==='a'?times(bump(value,idx,d),b):times(a,bump(value,idx,d)),up);
  near(ND.at(g[name],idx),(f(h)-f(-h))/(2*h),3e-9,'broadcast VJP '+name+idx);
 });
assert.deepEqual(C.analyze({id:'mul',kind:'nd_broadcast_mul',params:{}},
  {a:[1,1,8,6],b:[1,1,1,6]}).shape,[1,1,8,6]);
const graph=L.preset('nd_wan_adaln_self_gated'),run=L.evaluate(graph),get=id=>
 run.steps.find(s=>s.id===id).values;
assert.equal(graph.nodes.length,51);
assert.equal(ND.shape(get('loss')).join(','),'1,1');
eq(get('norm1'),fullNorm(get('attention_input'),1e-6));
for(const i of [0,1,2])eq(get('e'+i+'_sum'),plus(get('e'+i),get('mod'+i)));
const x=get('attention_input'),norm=get('norm1'),scaled=get('e1_sum'),shift=get('e0_sum'),gate=get('e2_sum');
const independentCondition=plus(plus(norm,times(norm,scaled)),shift);
eq(get('conditioned'),independentCondition);
eq(get('gated_attention'),times(get('out'),gate));
eq(get('after_self'),plus(x,times(get('out'),gate)));
assert.notDeepEqual(get('conditioned'),norm); // real e0/e1 modulation not ignored
// Structural + numerical graph microscope uses the existing shared contract.
const frame=M.learning(graph,run,L);
for(const [id,index,stage]of [
 ['norm1',[0,0,0,1],'wan_norm1_lastchannel_layernorm'],
 ['norm_scale_product',[0,0,0,1],'broadcast_gate'],
 ['gated_attention',[0,0,0,1],'broadcast_gate']]){
 const p=E.inspect(graph,{targetId:id,outputIndex:index,trace:run,
  maxVertices:8192,maxEdges:32768});
 assert.equal(p.complete,true,id+' incomplete: '+p.frontiers.join(','));
 assert.equal(p.vertices.find(v=>v.id===id).stages[0].stage,stage);
 const b=Bridge.inspect(graph,run,frame,{stepId:id,coordinate:index});
 assert.equal(b.backward.complete,true);
 assert.equal(b.backward.internalStages[0].stage,stage);
}
const whole=G.backward(graph,run);
assert.equal(whole.targetId,'loss');
const params=['w_q','w_k','w_v','w_o','gamma_q','gamma_k','mod0','mod1','mod2'];
for(const id of params)assert.deepEqual(ND.shape(whole.gradients[id]),ND.shape(get(id)));
const fdLoss=(g,id,index,eps=1e-5)=>{
 const eval1=sign=>{
  const copy=structuredClone(g),node=copy.nodes.find(n=>n.id===id);
  let v=node.params.values;
  for(let i=0;i<index.length-1;i++)v=v[index[i]];
  v[index[index.length-1]]+=sign*eps;
  return L.evaluate(copy).steps.find(s=>s.id==='loss').values[0][0];
 };
 return (eval1(1)-eval1(-1))/(2*eps);
};
for(const [id,coord]of [
 ['mod0',[0,0,0,1]],['mod1',[0,0,0,4]],['mod2',[0,0,0,2]],
 ['w_q',[0,0,1,4]],['w_v',[0,0,2,3]],['gamma_q',[0,0,0,1]]
])near(ND.at(whole.gradients[id],coord),fdLoss(graph,id,coord),3e-6,
 'whole loss '+id+coord);
// Kill the exact e2 sum for ALL 8 tokens; the Attention computation remains
// numerical and enabled, but zero e2 makes the final Self-Attention update
// identically the raw x residual. The gate parameters can still receive grad.
const zero=structuredClone(graph),zeros=shape=>ND.build(shape,()=>0);
for(const id of ['e2','mod2']){
 const n=zero.nodes.find(n=>n.id===id),sh=id==='e2'?[1,1,8,6]:[1,1,1,6];
 n.kind='nd_parameter';n.params={values:zeros(sh),
  optimizer:{t:0,m:zeros(sh),v:zeros(sh)}};
}
const ztrace=L.evaluate(zero),zg=G.backward(zero,ztrace),zval=id=>
 ztrace.steps.find(s=>s.id===id).values;
eq(zval('after_self'),zval('attention_input'));
for(const id of ['w_q','w_k','w_v','w_o','gamma_q','gamma_k'])
 assert.ok(ND.flatten(zg.gradients[id]).every(v=>v===0),
  'Attention parameters must have zero loss gradient when e2==0: '+id);
assert.ok(ND.flatten(zg.gradients.attention_input).some(v=>Math.abs(v)>1e-7));
assert.ok(ND.flatten(zg.gradients.mod2).some(v=>Math.abs(v)>1e-7),
 'e2 gate itself must remain differentiable at zero');
const stale=structuredClone(run);stale.steps.find(s=>s.id==='norm1').values[0][0][0][0]+=.3;
assert.throws(()=>E.inspect(graph,{targetId:'norm1',outputIndex:[0,0,0,0],trace:stale}),/stale or altered/);
console.log('WAN_ADALN_SELF_GATE_PASS: channel LN & broadcast tensor FD, source-order e0/e1/e2, full-loss gamma/modulation FD, gate-zero residual and stale rejection');
