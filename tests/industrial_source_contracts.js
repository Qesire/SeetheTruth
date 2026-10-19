'use strict';
const assert=require('node:assert/strict'),L=require('../learning_lab.js'),Source=require('../industrial_source_inspector.js');
const g=L.preset('industrial_gqa_rope_tied');
const refs=Object.fromEntries(g.nodes.map(n=>[n.id,Source.kind(g,n.id)]));
for(const [id,tag] of [['qgain','qgain'],['kgain','qgain'],['q0n','qgain'],['q1n','qgain'],['q0r','rope'],['q1r','rope'],
['kr','rope'],['q0','gqa'],['q1','gqa'],['k','gqa'],['v','gqa'],['tied_head','tie'],['embed','tie'],
['mask','mask'],['prob0','mask'],['prob1','mask']]){
 assert.equal(refs[id],tag,id);
 const c=Source.card(tag),html=Source.render(tag);
 assert.ok(c&&c.papers.length>0);
 assert.match(c.papers[0].url,/https:\/\/github.com\//);
 assert.match(html,/源码拆解/);
 assert.match(html,/可证伪实验/);
 assert.ok(!html.includes('onerror='));
}
assert.match(Source.card('rope').trap,/相邻/);
assert.match(Source.card('tie').practice,/false/,'must not misrepresent Qwen3 untied default');
assert.match(Source.card('gqa').flow,/K/);
assert.equal(Source.kind(L.preset('causal_transformer_lm'),'qgain'),null,'no false Qwen3 reference on unrelated graph');
assert.equal(Source.kind(g,'not_a_node'),null);
for(const [lesson,loops] of [['loop_transformer_two',2],['loop_transformer_three',3]]){
 const graph=L.preset(lesson),source=JSON.stringify(graph);
 for(let i=1;i<=loops;i++){
  assert.equal(Source.kind(graph,'r'+i+'_norm'),'loop_shared');
  assert.equal(Source.kind(graph,'r'+i+'_qr'),'loop_causal');
  assert.equal(Source.kind(graph,'r'+i+'_prob'),'loop_causal');
  assert.equal(Source.kind(graph,'r'+i+'_state'),'loop_ffn');
 }
 assert.equal(Source.kind(graph,'loss'),'loop_adaptive');
 for(const key of ['loop_shared','loop_causal','loop_ffn','loop_adaptive']){
  const entry=Source.card(key);
  assert.ok(entry.papers.length>0);
  assert.match(Source.render(key),/源码拆解/);
  assert.match(entry.epistemic,/[A-Z_]+/);
 }
 assert.equal(JSON.stringify(graph),source,'paper inspector may not modify recurrence');
}
assert.match(Source.card('loop_adaptive').trap,/没有实际实现退出分布/);
assert.match(Source.card('loop_shared').trap,/不.*Ouro 完整 checkpoint/);
assert.equal(Source.render('unknown'),'');
const original=JSON.stringify(g);
for(const id of Object.keys(refs))Source.render(refs[id]);
assert.equal(JSON.stringify(g),original,'inspector is not a code execution or graph mutation');
console.log('INDUSTRIAL_SOURCE_INSPECTOR_PASS specific source-level contracts, no false model attribution, safe links and explicit configuration boundaries');
