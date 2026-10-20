'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js');
const Cost=require('../graph_cost_report.js');
const ffn=Cost.profile(L.preset('ffn'));
assert.equal(ffn.nodeCount,4);
assert.equal(ffn.countedMatmulMAC,3*4*8+3*8*4);
assert.equal(ffn.countedMatmulFlopTerms,2*ffn.countedMatmulMAC);
assert.equal(ffn.physicalParameterElements,0);
assert.equal(ffn.implicitTeachingWeightElements,4*8+8+8*4+4);
assert.ok(ffn.unprofiledKinds.includes('silu'));
function n(id,kind,params,x=0,y=0){return {id,kind,params,x,y};}
const graph=L.validateGraph({schema:L.SCHEMA,title:'shared real Parameter',
 nodes:[n('x','input',{rows:2,width:2,seed:1}),n('weight','parameter',{values:[[1,2],[3,4]]}),
  n('a','matmul',{}),n('b','matmul',{},200,300),n('sum','add',{},400,200)],
 edges:[{from:'x',to:'a',port:'a'},{from:'weight',to:'a',port:'b'},
  {from:'x',to:'b',port:'a'},{from:'weight',to:'b',port:'b'},
  {from:'a',to:'sum',port:'a'},{from:'b',to:'sum',port:'b'}]});
const p=Cost.profile(graph);
assert.equal(p.parameterTensors,1);
assert.equal(p.physicalParameterElements,4);
assert.equal(p.parameterUsage[0].directConsumers,2);
assert.equal(p.countedMatmulNodes,2);
assert.equal(p.countedMatmulMAC,16,'two true 2x2 MatMul calls with same physical Parameter');
assert.equal(p.logicalOutputElements,20,'sum of graph output tensor sizes, not live memory');
assert.equal(p.rows.find(x=>x.id==='a').mac,8);
const broken=JSON.parse(JSON.stringify(graph));
broken.edges=broken.edges.filter(e=>!(e.to==='b'&&e.port==='a'));
assert.throws(()=>Cost.profile(broken),/Incomplete or invalid graph/);
const nd=Cost.profile(L.preset('nd_trainable_gqa'));
assert.ok(nd.nodeCount>5);
assert.ok(nd.physicalParameterElements>0);
assert.ok(nd.countedMatmulMAC>0);
assert.ok(nd.rows.some(r=>r.kind==='nd_batched_matmul'&&r.mac>0));
console.log('Graph cost analysis PASS: physical parameter aliases, exact known MAC, ND shape and fail-closed errors');
