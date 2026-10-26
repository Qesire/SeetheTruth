'use strict';
/* Source-ordered full Wan2.2-I2V single-block micrograph. Reuse the real
 * native graph for Self/Cross; the independent oracle below reconstructs
 * ALL FFN coordinates separately from ND operators and rejects wrong order.
 */
const assert=require('node:assert/strict');
const ND=require('../nd_tensor.js'),L=require('../learning_lab.js'),
 Grad=require('../graph_autodiff.js'),Session=require('../graph_session.js'),
 IR=require('../execution_ir.js');
const graph=L.preset('nd_wan_i2v_full_block'),trace=L.evaluate(graph);
const steps=new Map(trace.steps.map(x=>[x.id,x.values])),v=id=>steps.get(id);
const rows=id=>v(id)[0][0],matrix=id=>v(id)[0][0][0];
const close=(name,a,b,eps=4e-12)=>{
 assert.equal(a.length,b.length,name+' length');
 for(let i=0;i<a.length;i++)assert.ok(Math.abs(a[i]-b[i])<=eps,
  name+'['+i+']='+a[i]+' vs '+b[i]);
};
const flatten=x=>x.flat(Infinity);
const edges=new Map(graph.edges.map(e=>[e.to+':'+e.port,e.from]));
assert.equal(edges.get('loss:prediction'),'after_ffn');
assert.equal(edges.get('norm2_ffn:x'),'after_cross');
assert.equal(edges.get('ffn_norm_scale_product:a'),'norm2_ffn');
assert.equal(edges.get('ffn_norm_scale_product:b'),'e4_sum');
assert.equal(edges.get('ffn_conditioned:b'),'e3_sum');
assert.equal(edges.get('ffn_gated:b'),'e5_sum');
assert.equal(edges.get('after_ffn:a'),'after_cross');
assert.equal(edges.get('after_ffn:b'),'ffn_gated');
assert.equal(edges.get('ffn_gelu_tanh:x'),'ffn_flat');
assert.deepEqual(ND.shape(v('after_cross')),[1,1,8,6]);
assert.deepEqual(ND.shape(v('after_ffn')),[1,1,8,6]);
assert.deepEqual(ND.shape(v('ffn_gelu_tanh')),[8,8]);
assert.deepEqual(ND.shape(v('ffn_act4d')),[1,1,8,8]);
assert.ok(graph.nodes.length>89&&graph.nodes.length<=128);
const epsilon=1e-6,x=rows('after_cross'),e3=rows('e3_sum'),
 e4=rows('e4_sum'),e5=rows('e5_sum'),
 w1=matrix('ffn_w1'),w2=matrix('ffn_w2'),
 b1=matrix('ffn_b1'),b2=matrix('ffn_b2');
const gelu=y=>0.5*y*(1+Math.tanh(Math.sqrt(2/Math.PI)*(y+0.044715*y*y*y)));
const norm=x.map(row=>{
 const mean=row.reduce((s,y)=>s+y,0)/row.length;
 const variance=row.reduce((s,y)=>s+(y-mean)*(y-mean),0)/row.length;
 return row.map(y=>(y-mean)/Math.sqrt(variance+epsilon));
});
const affine=norm.map((row,t)=>row.map((n,c)=>n*(1+e4[t][c])+e3[t][c]));
const up=affine.map(row=>Array.from({length:8},(_,h)=>
 b1[h]+row.reduce((s,val,c)=>s+val*w1[c][h],0)));
const act=up.map(row=>row.map(gelu));
const down=act.map(row=>Array.from({length:6},(_,c)=>
 b2[c]+row.reduce((s,val,h)=>s+val*w2[h][c],0)));
const gated=down.map((row,t)=>row.map((val,c)=>val*e5[t][c]));
const output=x.map((row,t)=>row.map((val,c)=>val+gated[t][c]));
for(const [id,expected] of [
 ['norm2_ffn',norm],['ffn_conditioned',affine],['ffn_up_bias',up],
 ['ffn_gelu_tanh',act],['ffn_down_bias',down],['ffn_gated',gated],
 ['after_ffn',output]
])close(id,flatten(v(id)),flatten(expected));
const target=flatten(v('target')),pred=flatten(output);
const loss=pred.reduce((s,val,i)=>s+(val-target[i])**2,0)/pred.length;
assert.ok(Math.abs(v('loss')[0][0]-loss)<1e-12,'full objective uses final block output');
const swapped=norm.map((row,t)=>row.map((n,c)=>(n+e3[t][c])*(1+e4[t][c])));
assert.ok(flatten(swapped).some((v,i)=>Math.abs(v-flatten(affine)[i])>1e-5),
 'counterfactual wrong AdaLN order must differ');
const beforeCross=L.preset('nd_wan_i2v_self_cross');
const prefix=new Map(beforeCross.nodes.map(n=>[n.id,n]));
for(const n of beforeCross.nodes)assert.deepEqual(
 graph.nodes.find(x=>x.id===n.id),n,'original Self/Cross node must remain unchanged');
const inspected=Session.inspect(graph,{numeric:true});
assert.equal(inspected.executable,true);
assert.deepEqual(inspected.ir,IR.snapshot(graph,{trace}));
assert.equal(inspected.ir.nodes.length,graph.nodes.length);
const gradients=Grad.backward(graph,trace);
assert.equal(gradients.targetId,'loss');
const sample=(id,coordinate,diff)=>{
 const raw=structuredClone(graph),param=raw.nodes.find(n=>n.id===id);
 if(!param||param.kind!=='nd_parameter')throw Error('not parameter '+id);
 let item=param.params.values;
 for(const axis of coordinate.slice(0,-1))item=item[axis];
 item[coordinate.at(-1)]+=diff;
 return L.evaluate(raw).steps.find(n=>n.id==='loss').values[0][0];
};
const verified=[];
for(const [id,idx] of [
 ['mod0',[0,0,0,2]],['cw_q',[0,0,3,2]],['cross_norm3_gamma',[0,0,0,4]],
 ['cw_v',[0,0,2,1]],['mod3',[0,0,0,1]],['mod4',[0,0,0,3]],
 ['mod5',[0,0,0,2]],['ffn_w1',[0,0,1,7]],['ffn_w2',[0,0,4,2]]
]){
 const h=1e-5,fd=(sample(id,idx,h)-sample(id,idx,-h))/(2*h);
 const exact=ND.at(gradients.gradients[id],idx);
 assert.ok(Number.isFinite(exact)&&Math.abs(exact-fd)<5e-6,
  id+'/'+idx+': complete block gradient '+exact+' vs FD '+fd);
 verified.push({id,index:idx,analytical:exact,fd});
}
// A local FFN-only witness would have no path to Self/Cross parameters.
// This exact full graph must retain nontrivial end-to-end sensitivity.
assert.ok(verified.some(x=>x.id==='mod0'&&Math.abs(x.analytical)>1e-14),
 'Self AdaLN gradient must flow through Cross and FFN');
assert.ok(verified.some(x=>x.id==='cross_norm3_gamma'&&Math.abs(x.analytical)>1e-14),
 'Cross norm3 gradient must reach end objective');
console.log('WAN_FULL_BLOCK_FAST_PASS '+JSON.stringify({
 sourceRevision:'1ea34ff48f87168174e12956e200b1d908b1c5ff',
 nodes:graph.nodes.length,shapes:{tokens:8,channels:6,hidden:8},
 reconstructedStages:7,fullLossFD:verified.length,
 coverage:'Self + Cross + norm2/e3-e4 + Linear1/GELU(tanh)/Linear2 + e5 residual'
}));
module.exports={graph,trace,verified};
