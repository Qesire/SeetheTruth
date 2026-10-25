'use strict';
// Wan I2V A14B source: Wan-Video/Wan2.2@1ea34ff;
// model.py WanCrossAttention (158-180), norm3 conditional (203-209),
// Self/Cross residual order (237-259), i2v_A14B.cross_attn_norm=True.
const assert=require('node:assert/strict'),ND=require('../nd_tensor.js'),
 L=require('../learning_lab.js'),G=require('../graph_autodiff.js'),
 E=require('../element_lineage.js'),M=require('../compute_microscope.js'),
 B=require('../coordinate_vjp_bridge.js'),Layout=require('../graph_layout.js');
const near=(a,b,tol=1e-9,msg='')=>assert.ok(Math.abs(a-b)<=tol,msg+': '+a+' vs '+b);
const eq=(a,b,tol=1e-9,msg='')=>{
 assert.deepEqual(ND.shape(a),ND.shape(b),'shape '+msg);
 ND.indices(ND.shape(a),idx=>near(ND.at(a,idx),ND.at(b,idx),tol,msg+':'+idx));
};
const matmul=(x,w)=>ND.build([1,1,ND.shape(x)[2],6],([b,h,t,c])=>{
 let out=0;for(let i=0;i<6;i++)out+=x[b][h][t][i]*w[0][0][i][c];
 return out;
});
const addBias=(x,b)=>ND.build(ND.shape(x),([n,h,t,c])=>x[n][h][t][c]+b[0][0][0][c]);
const independentLN=x=>ND.build(ND.shape(x),([b,h,t,c])=>{
 const row=x[b][h][t],avg=row.reduce((a,v)=>a+v,0)/row.length,
  variance=row.reduce((a,v)=>a+(v-avg)**2,0)/row.length;
 return (row[c]-avg)/Math.sqrt(variance+1e-6);
});
const rms=(x,w)=>ND.build(ND.shape(x),([b,h,t,c])=>{
 const row=x[b][h][t],sq=row.reduce((a,v)=>a+v*v,0)/row.length;
 return row[c]*w[0][0][0][c]/Math.sqrt(sq+1e-6);
});
const trainable=(node,values)=>{
 const sh=ND.shape(values);
 node.kind='nd_parameter';node.params={values,optimizer:{t:0,
  m:ND.build(sh,()=>0),v:ND.build(sh,()=>0)}};
};
const graph=L.preset('nd_wan_i2v_self_cross'),run=L.evaluate(graph),
 get=id=>run.steps.find(x=>x.id===id).values;
assert.equal(graph.nodes.length,89,'connected lesson cannot omit Cross path');
assert.deepEqual(ND.shape(get('after_self')),[1,1,8,6]);
assert.deepEqual(ND.shape(get('text_context')),[1,1,4,6]);
assert.deepEqual(ND.shape(get('cross_scores')),[1,1,8,4]);
assert.deepEqual(ND.shape(get('cross_probabilities')),[1,1,8,4]);
assert.deepEqual(ND.shape(get('after_cross')),[1,1,8,6]);
assert.deepEqual(ND.shape(get('loss')),[1,1]);
const afterSelf=get('after_self'),text=get('text_context'),
  ln=independentLN(afterSelf),gamma3=get('cross_norm3_gamma'),beta3=get('cross_norm3_beta');
const norm3=ND.build([1,1,8,6],([b,h,t,c])=>
  ln[b][h][t][c]*gamma3[0][0][0][c]+beta3[0][0][0][c]);
eq(norm3,get('cross_norm3'),1e-12,'norm3 affine');
const q=rms(addBias(matmul(norm3,get('cw_q')),get('cb_q')),get('cross_q_gamma'));
const k=rms(addBias(matmul(text,get('cw_k')),get('cb_k')),get('cross_k_gamma'));
const v=addBias(matmul(text,get('cw_v')),get('cb_v'));
eq(q,get('cross_q_heads'),1e-12,'cross Q from afterSelf');
eq(k,get('cross_k_heads'),1e-12,'cross K from text');
eq(v,get('cross_v_heads'),1e-12,'cross V from text');
const scores=ND.build([1,1,8,4],([b,h,t,s])=>{
 let sum=0;for(let c=0;c<6;c++)sum+=q[0][0][t][c]*k[0][0][s][c];
 return sum/Math.sqrt(6);
});
const prob=ND.build([1,1,8,4],([b,h,t,s])=>{
 const logits=scores[0][0][t],top=Math.max(...logits),
  exps=logits.map(x=>Math.exp(x-top));
 return exps[s]/exps.reduce((a,b)=>a+b,0);
});
eq(prob,get('cross_probabilities'),1e-12,'manual Cross probabilities');
const ctx=ND.build([1,1,8,6],([b,h,t,c])=>{
 let acc=0;for(let j=0;j<4;j++)acc+=prob[0][0][t][j]*v[0][0][j][c];
 return acc;
});
const projected=addBias(matmul(ctx,get('cw_o')),get('cb_o'));
const residual=ND.build([1,1,8,6],idx=>
 ND.at(afterSelf,idx)+ND.at(projected,idx));
eq(projected,get('cross_out_bias'),1e-12,'manual output projection');
eq(residual,get('after_cross'),1e-12,'Cross residual');
const nodes=new Map(graph.nodes.map(x=>[x.id,x]));
const ports=target=>graph.edges.filter(e=>e.to===target);
assert.equal(ports('cross_q_proj').find(e=>e.port==='a').from,'cross_norm3');
for(const p of ['cross_k_proj','cross_v_proj'])
 assert.equal(ports(p).find(e=>e.port==='a').from,'text_context');
assert.equal(ports('cross_norm3_ln').find(e=>e.port==='x').from,'after_self');
assert.equal(ports('loss').find(e=>e.port==='prediction').from,'after_cross');
assert.ok(!graph.nodes.some(n=>n.id.startsWith('cross_')&&n.kind==='nd_wan_rope3d'),
 'cross attention MUST NOT apply self-attention video RoPE to text keys');
for(let t=0;t<8;t++)near(prob[0][0][t].reduce((a,b)=>a+b,0),1,1e-13);
// Verify independent gradient of actual loss for learned parameters
// from cross, context and prior AdaLN: all routes use one connected DAG.
const grads=G.backward(graph,run);
assert.equal(grads.targetId,'loss');
const lossFor=(source,index,delta)=>{
 const edited=structuredClone(graph),n=edited.nodes.find(n=>n.id===source);
 if(n.kind==='nd_input')trainable(n,structuredClone(get(source)));
 let arr=n.params.values;
 for(const axis of index.slice(0,-1))arr=arr[axis];
 arr[index.at(-1)]+=delta;
 return L.evaluate(edited).steps.find(s=>s.id==='loss').values[0][0];
};
for(const [id,coord]of [
 ['cross_norm3_gamma',[0,0,0,1]],['cross_norm3_beta',[0,0,0,3]],
 ['cw_q',[0,0,2,5]],['cw_k',[0,0,1,3]],['cw_v',[0,0,4,2]],
 ['cw_o',[0,0,3,1]],['cross_q_gamma',[0,0,0,2]],
 ['cross_k_gamma',[0,0,0,4]],['cb_q',[0,0,0,0]],
 ['cb_k',[0,0,0,1]],['cb_v',[0,0,0,2]],['cb_o',[0,0,0,3]],
 ['text_context',[0,0,2,1]],['mod2',[0,0,0,3]]
]){
 const h=1e-5,fd=(lossFor(id,coord,h)-lossFor(id,coord,-h))/(2*h);
 assert.ok(grads.gradients[id],id+' gradient missing');
 near(ND.at(grads.gradients[id],coord),fd,4e-6,'loss derivative '+id+coord);
}
// Cross queries are video length 8; text keys/values length 4.
// Counterexample to incorrectly treating Cross as self-attention.
assert.notEqual(ND.shape(get('cross_scores'))[3],ND.shape(get('scores'))[3]);
const pin=E.inspect(graph,{targetId:'cross_k_norm',outputIndex:[0,0,0,1],
 trace:run,maxVertices:8192,maxEdges:32768});
assert.equal(pin.complete,true);
assert.ok(pin.vertices.some(x=>x.id==='text_context'&&x.status==='source'));
assert.ok(pin.vertices.some(x=>x.id==='cw_k'&&x.status==='source'));
assert.ok(!pin.vertices.some(x=>x.id==='after_self'));
const normWitness=E.inspect(graph,{targetId:'cross_norm3_ln',outputIndex:[0,0,1,2],
 trace:run,maxVertices:8192,maxEdges:32768});
assert.equal(normWitness.vertices.find(v=>v.id==='cross_norm3_ln').stages[0].stage,
 'wan_norm1_lastchannel_layernorm');
assert.equal(normWitness.vertices.find(v=>v.id==='cross_norm3_ln').status,'expanded');
const microscope=M.learning(graph,run,L);
assert.equal(microscope.steps.length,89);
const selected=B.inspect(graph,run,microscope,{stepId:'cross_k_norm',coordinate:[0,0,0,1]});
assert.equal(selected.backward.complete,true);
assert.equal(selected.backward.internalStages[0].stage,'wan_qk_fullwidth_rmsnorm');
// Reuse the exact same graph for layout; no rule duplication.
const laid=Layout.layout(graph);
assert.deepEqual(laid.edges,graph.edges);
assert.equal(Layout.metrics(laid).overlap.length,0);
eq(L.evaluate(laid).steps.find(s=>s.id==='after_cross').values,residual,1e-12);
const stale=structuredClone(run);
stale.steps.find(s=>s.id==='cross_k_norm').values[0][0][0][1]+=.15;
assert.throws(()=>E.inspect(graph,{targetId:'cross_k_norm',
 outputIndex:[0,0,0,1],trace:stale}),/stale or altered/);
console.log('WAN_I2V_CROSS_ATTENTION_PASS: norm3 true affine, independent 8x4 text attention, 14 end-to-end FD, no cross RoPE, context lineage, 89-node layout and stale trace');
