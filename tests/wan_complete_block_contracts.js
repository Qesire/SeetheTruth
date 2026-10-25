'use strict';
// Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff
// Complete post-cross FFN: norm2/e3,e4/Linear/GELU(tanh)/Linear/e5 residual.
const assert=require('node:assert/strict');
const ND=require('../nd_tensor.js'),L=require('../learning_lab.js'),
 G=require('../graph_autodiff.js'),A=require('../activation_math.js');
const graph=L.preset('nd_wan_i2v_complete_block'),run=L.evaluate(graph);
const get=id=>run.steps.find(s=>s.id===id).values;
const near=(a,b,t=1e-9)=>assert.ok(Math.abs(a-b)<t,a+' != '+b);
const eq=(a,b,t=1e-9)=>{assert.deepEqual(ND.shape(a),ND.shape(b));ND.indices(ND.shape(a),idx=>near(ND.at(a,idx),ND.at(b,idx),t));};
const prod=(a,b)=>ND.broadcastBinary(a,b,(x,y)=>x*y);
const sum=(a,b)=>ND.broadcastBinary(a,b,(x,y)=>x+y);
assert.ok(graph.nodes.length>89&&graph.nodes.length<=128);
assert.deepEqual(ND.shape(get('after_ffn')),[1,1,8,6]);
assert.deepEqual(ND.shape(get('ffn_linear1')),[1,1,8,8]);
for(let i=3;i<=5;i++)eq(get('e'+i+'_sum'),sum(get('e'+i),get('mod'+i)));
const x=get('after_cross'),n=ND.layerNormLast4(x,1e-6);
eq(get('ffn_norm2'),n);
const conditioned=sum(sum(n,prod(n,get('e4_sum'))),get('e3_sum'));
eq(get('ffn_conditioned'),conditioned);
const m1=ND.matmul4(conditioned,get('ffn_w1'));
eq(get('ffn_linear1'),sum(m1,get('ffn_b1')));
const gelu=ND.build(ND.shape(get('ffn_linear1')),idx=>A.value('gelu_tanh',ND.at(get('ffn_linear1'),idx)));
eq(get('ffn_gelu'),gelu);
const m2=ND.matmul4(gelu,get('ffn_w2'));
eq(get('ffn_linear2'),sum(m2,get('ffn_b2')));
eq(get('after_ffn'),sum(x,prod(get('ffn_linear2'),get('e5_sum'))));
const links=to=>graph.edges.filter(e=>e.to===to);
assert.equal(links('ffn_norm2').find(e=>e.port==='x').from,'after_cross');
assert.equal(links('loss').find(e=>e.port==='prediction').from,'after_ffn');
assert.equal(links('ffn_gated').find(e=>e.port==='b').from,'e5_sum');
const copy=structuredClone(graph),zero=ND.build([1,1,1,6],()=>0);
for(const id of ['e5','mod5']){const p=copy.nodes.find(n=>n.id===id);p.kind='nd_parameter';p.params={values:zero,optimizer:{t:0,m:zero,v:zero}};}
const z=L.evaluate(copy),at=id=>z.steps.find(s=>s.id===id).values;
eq(at('after_ffn'),at('after_cross'),0.00000000001);
const gradients=G.backward(graph,run);
for(const id of ['ffn_w1','ffn_w2','mod3','mod4','mod5','after_cross'])
 assert.ok(gradients.gradients[id],id+' missing full-block reverse path');
const fd=(id,index)=>{
 const calc=d=>{const g=structuredClone(graph),p=g.nodes.find(n=>n.id===id);let v=p.params.values;for(let i=0;i<index.length-1;i++)v=v[index[i]];v[index.at(-1)]+=d;return L.evaluate(g).steps.find(s=>s.id==='loss').values[0][0];};
 return (calc(1e-5)-calc(-1e-5))/2e-5;
};
for(const [id,index] of [['ffn_w1',[0,0,2,3]],['ffn_w2',[0,0,4,2]],['mod4',[0,0,0,1]],['mod5',[0,0,0,3]]])
 near(ND.at(gradients.gradients[id],index),fd(id,index),3e-6);
console.log('WAN_COMPLETE_BLOCK_PASS: connected FFN oracle, tanh GELU, gate-zero counterexample, 4 FD gradients');
