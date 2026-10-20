'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js');
const X=require('../torch_export.js');
const graph=L.preset('ffn');
const result=X.compile(graph);
assert.equal(result.nodeCount,graph.nodes.length);
assert.match(result.code,/import torch/);
assert.match(result.code,/torch\.nn\.functional\.silu/);
assert.match(result.code,/_weight/);
assert.ok(!result.code.includes('eval('));
const broken=JSON.parse(JSON.stringify(graph));
broken.edges=broken.edges.filter(e=>!(e.to==='up'&&e.port==='x'));
assert.throws(()=>X.compile(broken),/Missing input port/);
const unsupported=JSON.parse(JSON.stringify(graph));
unsupported.nodes[1].kind='gqa';
assert.throws(()=>X.compile(unsupported),/unavailable/);
const addGraph=L.validateGraph({
 schema:L.SCHEMA,title:'branch',nodes:[
  {id:'a',kind:'input',x:0,y:0,params:{rows:2,width:4,seed:1}},
  {id:'b',kind:'input',x:0,y:150,params:{rows:2,width:4,seed:2}},
  {id:'sum',kind:'add',x:300,y:0,params:{}}
 ],edges:[{from:'a',to:'sum',port:'a'},{from:'b',to:'sum',port:'b'}]
});
assert.match(X.compile(addGraph).code,/values\["sum"\] = values\["a"\] \+ values\["b"\]/);
const invalid=JSON.parse(JSON.stringify(addGraph));invalid.nodes[0].params.width=3;
assert.throws(()=>X.compile(invalid));
console.log('PyTorch exporter contracts PASS');
