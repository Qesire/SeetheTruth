/* Evidence architecture and false-positive prevention tests; no browser needed. */
'use strict';
const assert = require('node:assert/strict');
const K = require('../consensus_knowledge.js');

assert.deepEqual(K.validate(),{entries:K.entries.length,refs:Object.keys(K.references).length});
assert.ok(K.entries.length >= 14);
assert.ok(Object.keys(K.references).length >= 10);
const all = new Set(K.entries.map(e=>e.id));
assert.equal(all.size,K.entries.length);
for(const e of K.entries){
  assert.ok(e.formula && e.mechanism && e.objective && e.assumptions && e.measure && e.contrast);
  assert.ok(e.chain.length>=2);
  assert.ok(e.refs.every(r=>K.references[r]?.url.startsWith('https://')));
}
function choose(opt){return K.resolve(opt)?.entry.id??null}
assert.equal(choose({id:'attention',preset:'gqa'}),'gqa');
assert.equal(choose({id:'attention',preset:'gpt'}),'scaled_attention');
assert.equal(choose({id:'norm1',preset:'gpt'}),'layernorm');
assert.equal(choose({id:'norm1',preset:'gqa'}),'rmsnorm');
assert.equal(choose({id:'ffn',preset:'gpt'}),'gelu_mlp');
assert.equal(choose({id:'ffn',preset:'gqa'}),'swiglu');
assert.equal(choose({id:'ffn',preset:'moe'}),'moe');
assert.equal(choose({id:'adaln',preset:'dit'}),'adaln_zero');
assert.equal(choose({id:'rl_advantage',workflow:'rl',rlAlgo:'ppo'}),'ppo');
assert.equal(choose({id:'rl_advantage',workflow:'rl',rlAlgo:'grpo'}),'grpo');
assert.equal(choose({id:'wf_loss',workflow:'train',preset:'gpt'}),'cross_entropy');
assert.equal(choose({id:'wf_loss',workflow:'train',preset:'dit'}),null); // diffusion loss ≠ language CE
const source={analysis:'static-python-ast',entry:'UnknownBlock.forward'};
assert.equal(choose({id:'ffn',preset:'gqa',node:{callee:'self.ffn'},graphMeta:source}),null);
assert.equal(choose({id:'attention',preset:'gqa',node:{callee:'self.qkv'},graphMeta:source}),null);
assert.equal(choose({id:'ffn',preset:'gpt',node:{callee:'self.gate',module_constructor:'Unrecognized'},graphMeta:source}),null);
assert.equal(choose({id:'mx_gmm',preset:'moe',node:{callee:'torch.matmul'},graphMeta:source}),null); // no assumption it's SwiGLU
assert.equal(choose({id:'foo',node:{callee:'torch.nn.functional.scaled_dot_product_attention'},graphMeta:source}),'scaled_attention');
assert.equal(choose({id:'node42',node:{callee:'self.down_proj',module_constructor:'Linear'},graphMeta:source}),'linear');
assert.equal(choose({id:'node42',node:{callee:'self.norm',module_constructor:'LayerNorm'},graphMeta:source}),'layernorm');
assert.equal(choose({id:'node42',node:{callee:'nn.functional.cross_entropy'},graphMeta:source}),'cross_entropy');
assert.equal(choose({id:'node42',node:{callee:'apply_rotary_pos_emb'},graphMeta:source}),'rope');
assert.equal(choose({id:'node42',node:{callee:'torch.topk'},graphMeta:source}),null);
assert.equal(choose({id:'node42',node:{callee:'torch.topk'},graphMeta:{analysis:'static-python-ast',entry:'MixtralSparseMoeBlock.forward'}}),'moe');
assert.equal(choose({id:'attention',preset:'gqa',node:{label:'Attention'},graphMeta:{analysis:'annotated-import'}}),null);
assert.equal(K.resolve({id:'ffn',preset:'gqa'}).applicationEvidence,'template_only_not_model_validated');
assert.equal(K.resolve({id:'node42',node:{callee:'self.linear',module_constructor:'Linear'},graphMeta:source}).applicationEvidence,'literature_only_not_model_validated');
console.log('Mechanism atlas catalog, source guard, provenance and instructional mapping PASS');
