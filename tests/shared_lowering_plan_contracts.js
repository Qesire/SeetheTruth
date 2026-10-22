'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),P=require('../graph_lowering_plan.js');
const NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const Caps=require('../block_capabilities.js');
for(const name of ['ffn','trainable_core','stream_state_branches']){
 const graph=L.preset(name);
 for(const [backend,codegen] of [['numpy',NP],['torch',Torch]]){
  const plan=P.prepare(graph,{supportedKinds:codegen.supportedKinds});
  const original=L.evaluate(graph),index=L.indexGraph(L.validateGraph(graph));
  assert.deepEqual(plan.sorted.map(n=>n.id),index.ordered.map(n=>n.id),backend+' '+name);
  assert.deepEqual(plan.forward,original,backend+' '+name);
  assert.deepEqual([...plan.edges], [...index.incomingByPort],backend+' '+name);
  assert.deepEqual(plan.sorted.map(n=>plan.steps.get(n.id)?.shape),original.steps.map(s=>s.shape));
  const exported=codegen.compile(graph);
  assert.deepEqual(exported.nodes.map(n=>n.id),plan.sorted.map(n=>n.id),backend+' '+name);
  assert.ok(exported.source.length>100);
 }
}
const base=L.preset('ffn');
const dangling=L.validateGraph({...base,nodes:[...base.nodes,{id:'isolated_gqa',kind:'gqa',x:0,y:600,params:{}}]});
for(const [backend,codegen,marker] of [
 ['numpy',NP,'无法导出可执行 Python：不支持'],
 ['torch',Torch,'PyTorch 不支持的积木']
]){
 assert.throws(()=>codegen.compile(dangling),e=>e.message.includes(marker)&&e.message.includes('isolated_gqa(gqa)'),backend+' must fail closed before numeric interpretation');
 assert.throws(()=>P.prepare(dangling,{supportedKinds:codegen.supportedKinds,unsupportedMessage:bad=>'rejected:'+bad[0].kind}),/rejected:gqa/);
 const plan=P.prepare(dangling,{targetId:'down',supportedKinds:codegen.supportedKinds});
 assert.ok(!plan.sorted.some(n=>n.id==='isolated_gqa'));
 assert.equal(plan.steps.size,4);
 assert.deepEqual(codegen.compile(dangling,{targetId:'down'}).nodes.map(n=>n.id),plan.sorted.map(n=>n.id));
}
const missing=L.validateGraph({...base,edges:base.edges.filter(e=>e.to!=='act')});
assert.throws(()=>P.prepare(missing,{supportedKinds:NP.supportedKinds}),/还没有连接|缺少/);
assert.throws(()=>NP.compile(missing),/还没有连接|缺少/);
assert.throws(()=>Torch.compile(missing),/还没有连接|缺少/);
assert.throws(()=>P.prepare(base,{}),/supported kinds/);
const audit=Caps.audit();
for(const row of audit.rows){
 assert.equal(row.numpyEligible,NP.supportedKinds.includes(row.kind));
 assert.equal(row.torchEligible,Torch.supportedKinds.includes(row.kind));
}
console.log('SHARED_LOWERING_PLAN_PASS deterministic DAG, ports, trace, backend fail-closed, target branch, registry');
