/* Deterministic real-source projection: never synthesize operators or tensor shapes. */
'use strict';
const assert = require('node:assert/strict');
const view = require('../source_evidence_levels.js');
const graph = {
  nodes: [
    {id:'block',parent:null,label:'self.attn()',callee:'self.attn',source:{file:'m.py',line:4}},
    {id:'other',parent:null,label:'helper()',callee:'self.helper',source:{file:'m.py',line:5}},
    {id:'in',parent:'block',label:'x',certainty:'source_parameter',source:{file:'m.py',line:9}},
    {id:'mat',parent:'block',label:'MatMult',certainty:'source_operator',source:{file:'m.py',line:10}},
    {id:'view',parent:'block',label:'transpose()',callee:'x.transpose',source:{file:'m.py',line:11}},
    {id:'builtin',parent:'block',label:'q : Linear',module_constructor:'Linear',source:{file:'m.py',line:12}},
    {id:'custom',parent:'block',label:'Conv1D',module_constructor:'Conv1D',source:{file:'m.py',line:13}},
    {id:'nested',parent:'block',label:'child()',callee:'self.child',source:{file:'m.py',line:14}},
    {id:'gemm2',parent:'nested',label:'baddbmm()',callee:'torch.baddbmm',source:{file:'nested.py',line:21}},
    {id:'return',parent:'block',label:'Return',certainty:'source_return',source:{file:'m.py',line:15}},
  ],
  edges: [
    {from:'in',to:'mat',kind:'data'},
    {from:'mat',to:'view',kind:'data'},
    {from:'view',to:'builtin',kind:'data'},
    {from:'builtin',to:'return',kind:'data'},
    {from:'nested',to:'gemm2',kind:'hierarchy'},
  ],
};
assert.equal(view.classify(graph.nodes[3]),'matrix');
assert.equal(view.classify(graph.nodes[5]),'matrix');
assert.notEqual(view.classify(graph.nodes[6]),'matrix','custom Conv1D must not be inferred to be GEMM');
assert.equal(view.classify(graph.nodes[4]),'tensor');
const whole = view.project(graph,null,'macro');
assert.deepEqual(whole.nodes.map(n=>n.id),['block','other']);
const matrix = view.project(graph,'block','matrix');
assert.equal(matrix.matchCount,3,'two verified GEMM callsites plus explicit Linear');
assert.ok(matrix.nodes.some(n=>n.id==='gemm2'),'nested matrix site must be reachable');
assert.ok(matrix.nodes.some(n=>n.id==='mat'),'infix matmul must be reachable');
assert.ok(matrix.nodes.some(n=>n.id==='builtin'),'Linear is explicit evidence');
assert.ok(!matrix.matches.has('custom'),'custom class cannot be classified as matrix');
assert.strictEqual(matrix.nodes.find(n=>n.id==='gemm2'),graph.nodes[8],'original source provenance preserved');
const tensors=view.project(graph,'block','tensor');
assert.ok(tensors.nodes.some(n=>n.id==='view'));
assert.ok(tensors.nodes.some(n=>n.id==='mat'));
assert.ok(tensors.nodes.every(n=>n.source),'source locator never manufactured');
const onlyOne=view.project(graph,'block','matrix',1);
assert.equal(onlyOne.matchCount,3);
assert.equal(onlyOne.omitted,2);
assert.equal(onlyOne.nodes.length,1);
assert.throws(()=>view.project(graph,'missing','matrix'),/Unknown source scope/);
assert.throws(()=>view.project(graph,'block','fake'),/Invalid source level/);
console.log('Source evidence projections, explicit nested GEMMs and no fabricated shapes PASS');
