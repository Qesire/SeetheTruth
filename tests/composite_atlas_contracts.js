'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),C=require('../concept_flow.js'),A=require('../composite_atlas.js');
const info=A.validate();
assert.equal(info.modules,8);
assert.ok(info.stageCount>=26);
assert.equal(A.ids.length,8);
const deep=(a,b,label,tol=1e-9)=>{
 assert.deepEqual(a.map(r=>r.length),b.map(r=>r.length),label+' shape');
 for(let i=0;i<a.length;i++)for(let j=0;j<a[i].length;j++){
  const x=a[i][j],y=b[i][j];
  assert.ok(Math.abs(x-y)<tol,label+': ('+i+','+j+') '+x+' != '+y);
 }
};
for(const id of A.ids){
 const g=A.build(id),c=C.evaluate(g),micro=L.evaluate(g),m=A.get(id),stages=A.stages(id);
 assert.ok(m.title.length>15&&m.intro.length>35&&m.limits.length>35,id);
 assert.equal(g.nodes.length,c.steps.length,id);
 assert.equal(micro.steps.length,c.steps.length,id);
 assert.ok(micro.steps.every(s=>s.values.flat().every(Number.isFinite)),id);
 for(let i=0;i<c.steps.length;i++){
   assert.deepEqual(c.steps[i].shape,micro.steps[i].shape,id+':'+c.steps[i].id);
   assert.equal(c.steps[i].values,null,'never allocate numeric values in concept view');
 }
 assert.equal(stages.length,m.stages.length,id);
 for(const s of stages){
   assert.ok(s.detail.length>20,id);
   assert.ok(L.defs[s.nodeKind],id+': unknown leaf');
   assert.deepEqual(s.shape,c.steps.find(t=>t.id===s.node).shape);
   if(s.child)assert.ok(A.get(s.child),id+': absent child');
 }
}
const macro=C.evaluate(A.build('decoder'));
const final=macro.steps.find(s=>s.id==='res2');
assert.deepEqual(final.shape,[3,4]);
assert.deepEqual(macro.steps.find(s=>s.id==='attn').shape,[3,8]);
assert.deepEqual(macro.steps.find(s=>s.id==='sw').shape,[3,8]);
assert.deepEqual(macro.steps.find(s=>s.id==='res1').shape,[3,4]);
assert.ok(A.get('decoder').intro.includes('Attention'));
const sw=L.evaluate(A.build('ffn')).steps.at(-1).values;
const sx=L.evaluate(A.build('ffn_expanded')).steps.at(-1).values;
deep(sw,sx,'SwiGLU composed = primitive SiLU + Hadamard + projections');
// Independent one-head math composition must be numerically equivalent to the
// original single-head attention primitive with identical seeded projection IDs.
const parent=L.evaluate(L.preset('attention'));
const qk=L.evaluate(A.build('attention_math'));
deep(parent.steps.at(-1).values,qk.steps.at(-1).values,
  'Scaled attention = QK^T / sqrt(Dk) -> Softmax -> PV -> output projection');
// The mathematical sub-stages need actual independent dimensions and semantics.
const query=C.evaluate(A.build('attention_math'));
for(const [node,shape] of [['kt',[4,3]],['qk',[3,3]],['scale',[3,3]],['soft',[3,3]],['pv',[3,4]],['proj',[3,4]]]){
 assert.deepEqual(query.steps.find(t=>t.id===node).shape,shape,node);
}
assert.match(A.get('attention_math').limits,/不等价于完整 GQA/);
assert.match(A.get('decoder').limits,/不.*通用.*源码/);
assert.match(A.get('ffn_expanded').intro,/父级/);
assert.throws(()=>A.build('__prototype__'),/Unknown teaching module/);
assert.throws(()=>A.build('random'),/Unknown teaching module/);
// Ensure the module hierarchy cannot silently be rewritten into fake GPU ops.
for(const id of A.ids){
 const g=A.build(id);
 assert.ok(g.nodes.every(n=>L.KINDS.includes(n.kind)));
 assert.ok(!g.nodes.some(n=>/gpu|cuda|profiler|optimizer/.test(n.kind)));
}
console.log('Composite atlas: 8 verified hierarchical views; decoder 2 residual paths; exact FFN & QK/PV numeric composition oracles PASS');
