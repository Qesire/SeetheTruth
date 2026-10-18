'use strict';
const assert=require('node:assert/strict'),L=require('../learning_lab.js'),G=require('../graph_layout.js'),C=require('../concept_flow.js');
const clone=x=>JSON.parse(JSON.stringify(x));
for(const kind of ['ffn','residual','gqa','mla_lite','moe_route','grpo_signals']){
 const graph=L.preset(kind),original=clone(graph),laid=G.layout(graph),metrics=G.metrics(laid);
 assert.equal(metrics.overlap.length,0,kind+' blocks overlap');
 assert.deepEqual(laid.nodes.map(n=>n.id),graph.nodes.map(n=>n.id));
 assert.deepEqual(laid.edges,graph.edges);
 assert.deepEqual(laid.nodes.map(n=>n.params),graph.nodes.map(n=>n.params));
 assert.deepEqual(graph,original,'layout mutated original graph');
 assert.deepEqual(G.layout(laid).nodes,laid.nodes,'layout not idempotent '+kind);
 for(const e of laid.edges){
  const a=laid.nodes.find(n=>n.id===e.from),b=laid.nodes.find(n=>n.id===e.to);
  assert.ok(a.x+220<b.x,'backward edge '+kind+':'+e.from+'->'+e.to);
 }
 const before=L.evaluate(graph),after=L.evaluate(laid),concept=C.evaluate(laid);
 assert.deepEqual(before.steps.map(s=>s.values),after.steps.map(s=>s.values),'numeric behavior changed '+kind);
 assert.deepEqual(after.steps.map(s=>s.shape),concept.steps.map(s=>s.shape),'concept changed '+kind);
 assert.ok(metrics.width>0&&metrics.height>0);
}
assert.throws(()=>G.layout({nodes:[{id:'a'},{id:'b'}],edges:[{from:'a',to:'b'},{from:'b',to:'a'}]}),/DAG/);
assert.throws(()=>G.layout({nodes:[{id:'a'}],edges:[{from:'a',to:'unknown'}]}),/edges/);
console.log('Graph layout: six complete numerical/qualitative comparisons, ID/edge preservation, nonoverlap and cycle checks PASS');
