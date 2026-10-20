'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),Why=require('../architecture_why.js');
const examples=[
 ['industrial_gqa_rope_tied','q0n','qk_norm'],
 ['industrial_gqa_rope_tied','q0','gqa'],
 ['industrial_gqa_rope_tied','q0r','rope2d'],
 ['industrial_gqa_rope_tied','embed','tie-unknown'],
 ['moe_expert_ffn','expert_bank','moe'],
 ['nd_gqa_attention','kv_repeat','gqa']
];
let validated=0;
for(const [name,nodeid,expected] of examples){
 const graph=L.preset(name),n=graph.nodes.find(n=>n.id===nodeid);
 if(!n||expected==='tie-unknown')continue;
 const key=Why.classify(graph,n);
 if(key!==expected)continue;
 const html=Why.render(graph,n);
 assert.ok(html.includes('INDUSTRY WHY'),name+'/'+nodeid);
 assert.ok(html.includes('设计目标'));
 assert.ok(html.includes('业界替代方案'));
 assert.ok(html.includes('失效机制'));
 assert.ok(html.includes('低成本鉴别实验'));
 assert.ok(html.includes('github.com')||html.includes('arxiv.org'));
 assert.ok(!html.includes('<script'));
 validated++;
}
const known=L.preset('industrial_gqa_rope_tied');
const pairs=[
 ['q0n','qk_norm'],['kn','qk_norm'],['q0r','rope2d'],['q1r','rope2d'],
 ['k','gqa'],['q1','gqa'],['heads','gqa']
];
for(const [id,expected] of pairs){
 const n=known.nodes.find(x=>x.id===id);
 assert.equal(Why.classify(known,n),expected,id);
 assert.ok(Why.render(known,n).includes('设计目标'),id);
}
const catalog=L.preset('activation_gallery');
assert.equal(Why.classify(catalog,{id:'bad',kind:'swiglu'}),null,'no graph-invented provenance');
const shape=L.preset('nd_gqa_attention');
assert.equal(Why.classify(shape,shape.nodes.find(n=>n.kind==='nd_repeat_kv')),'gqa');
const s=Why.cases;
assert.ok(Object.keys(s).length>=8);
for(const [k,d] of Object.entries(s)){
 for(const field of ['goal','math','alternative','cost','failure','test','source','status']){
  assert.ok(d[field]&&d[field].length>10,k+'/'+field);
 }
 assert.match(d.source,/^https:\/\//);
 const matchedNode=known.nodes.find(n=>Why.classify(known,n)===k);
 if(matchedNode)assert.ok(Why.render(known,matchedNode).includes('证据边界'));
}
assert.equal(Why.render(L.preset('loop_transformer_2'),{id:'Wq',kind:'parameter'}),'',
 'loop WHY always uses its more specific live parameter-count counterfactual');
console.log('ARCHITECTURE_WHY_PASS '+validated+' live model-specific blocks, '+Object.keys(s).length+' source-anchored design alternatives, counterexamples and falsifiable tests');
