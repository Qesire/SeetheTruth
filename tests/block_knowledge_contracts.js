'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js');
const K=require('../block_knowledge.js');
const C=require('../consensus_knowledge.js');
const status=K.validate();
assert.equal(status.cards,45);
assert.ok(status.references>=18);
assert.deepEqual(Object.keys(K.sources).length,status.references);
const seen=new Set(),refs=new Set();
for(const kind of L.KINDS){
 const k=K.get(kind),d=L.defs[kind];
 assert.ok(k,"No knowledge for "+kind);
 assert.equal(k.kind,kind);
 assert.equal(k.title,d.title);
 assert.equal(k.formula,d.formula);
 for(const field of ['intuition','flow','practice','trap']){
  assert.ok(k[field]&&k[field].length>=18,kind+'.'+field);
  assert.ok(!/<script|javascript:|onerror=/i.test(k[field]),'untrusted content '+kind);
 }
 assert.match(k.epistemic,/不是指定模型/);
 assert.ok(k.refs.length>=1 && k.refs.length<=3);
 for(const name of k.refs){
   const ref=K.sources[name];refs.add(name);
   assert.ok(ref,kind+' source '+name);
   assert.ok(/^https:\/\/(docs\.pytorch\.org|arxiv\.org|aclanthology\.org|huggingface\.co)\//.test(ref[1]),'untrusted source '+ref[1]);
 }
 seen.add(kind);
}
assert.equal(seen.size,45);
for(const id of ['linear','rmsnorm','layernorm','attention','gqa','swiglu','geglu',
 'rope2d','masked_softmax','causal_mask','cross_entropy','kl_prob','dropout_demo']){
 const k=K.get(id);
 assert.ok(k.practice.length>=30,id+': missing actual engineering context');
 assert.ok(k.trap.length>22,id+': missing caveat');
}
assert.match(K.get('gelu_tanh').practice,/approximate=none/);
assert.match(K.get('rmsnorm').trap,/均值/);
assert.match(K.get('gqa').trap,/不是 Token 剪枝/);
assert.match(K.get('attention').practice,/FlashAttention/);
assert.match(K.get('dropout_demo').practice,/dropout_p/);
assert.match(K.get('cross_entropy').practice,/shift/);
assert.match(K.get('kl_prob').practice,/batchmean/);
assert.match(K.get('causal_mask').trap,/Tq=1/);
assert.match(K.get('masked_softmax').trap,/乘零/);
assert.equal(K.get('__proto__'),null);
assert.equal(K.get('<img onerror=alert(1)>'),null);
const checks=C.validate();
assert.ok(checks.entries>10,'Mechanism atlas still available');
console.log('Deep mechanism knowledge PASS: 45 distinct cards, grounded sources, model-evidence boundaries and named confusions');
