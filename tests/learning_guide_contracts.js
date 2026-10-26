'use strict';
// Real graph recipes -> grounded guided questions -> source-local proof nodes.
// Every guided node is required to exist in the exact current graph.
const assert=require('node:assert/strict'),fs=require('node:fs');
const L=require('../learning_lab.js'),P=require('../graph_learning_path.js');
const page=fs.readFileSync(require('node:path').resolve(__dirname,'../learn.html'),'utf8');
const ids=[...page.matchAll(/data-preset="([^"]+)"/g)].map(m=>m[1]);
assert.equal(ids.length,new Set(ids).size,'guided identity must come from real buttons');
let observed=0;
for(const id of ids){
 const graph=L.preset(id),path=P.build(graph),route=P.plan(graph,{id});
 const byId=new Map(path.steps.map(s=>[s.id,s]));
 assert.equal(route.schema,'seethetruth-learning-guide-v1');
 assert.equal(route.id,id);
 assert.equal(route.source,'graph_learning_path');
 assert.ok(route.steps.length>=2&&route.steps.length<=4,id+' guided step count');
 assert.equal(route.steps[0].kind,'select',id);
 assert.equal(route.steps[1].kind,'numeric',id);
 for(const stage of route.steps){
  assert.equal(stage.ordinal,route.steps.indexOf(stage)+1,id);
  assert.ok(typeof stage.question==='string'&&stage.question.trim().length>8,id);
  assert.ok(typeof stage.instruction==='string'&&stage.instruction.trim(),id);
  if(stage.node){
   assert.ok(byId.has(stage.node),id+': nonexistent guided node '+stage.node);
   assert.equal(stage.proof.nodeId,stage.node,id);
   assert.equal(stage.proof.kind,byId.get(stage.node).kind,id);
   assert.deepEqual(stage.proof.shape,byId.get(stage.node).shape,id);
   assert.deepEqual(stage.proof.ports.map(p=>p.name),byId.get(stage.node).ports.map(p=>p.port),id);
  }
  if(stage.kind==='numeric')assert.equal(stage.proof,null,id);
 }
 observed++;
}
const focus={
 ffn:['up','act','down'],
 attention:['q','attn','proj'],
 nd_wan_i2v_full_block:['after_self','ffn_gelu_tanh','after_ffn'],
 image_vit_patch:['patches','image_embedding','image_attend'],
 audio_spectrogram:['spectrum','log_power','audio_attend'],
 video_causal_conv3d:['causal_conv','activate','temporal_attend'],
 nd_gqa_attention:['key_groups','probabilities','context']
};
for(const [id,nodes] of Object.entries(focus)){
 const stage=P.plan(L.preset(id),{id});
 assert.deepEqual(stage.steps.filter(s=>s.node).map(s=>s.node),nodes,id);
}
const changed=L.preset('ffn');
const revised=JSON.parse(JSON.stringify(changed));
revised.nodes.find(n=>n.id==='act').id='new_activation';
for(const e of revised.edges){if(e.from==='act')e.from='new_activation';if(e.to==='act')e.to='new_activation';}
const derived=P.plan(revised,{id:'ffn'});
assert.ok(derived.steps.filter(s=>s.node).every(s=>revised.nodes.some(n=>n.id===s.node)),
 'changed graph must construct a new route without phantom authored nodes');
const blank=P.plan(L.preset('blank'),{id:'blank'});
assert.equal(blank.steps.length,2,'an Input-only graph offers one locator and one numerical observation');
console.log('LEARNING_GUIDE_PASS '+observed+' real source-declared graph lessons, scoped dynamic proof nodes and fallback');
