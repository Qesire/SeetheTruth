'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),P=require('../graph_lowering_plan.js');
const IR=require('../execution_ir.js'),Session=require('../graph_session.js');
const NDcases=require('./fixtures/nd_broadcast_reduction_graphs.js');
const NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const Caps=require('../block_capabilities.js');
const fixtureGraphs=['ffn','trainable_core','stream_state_branches',
  ...NDcases.cases.map(x=>x.id)];
for(const name of fixtureGraphs){
 const graph=NDcases.cases.some(c=>c.id===name)?NDcases.create(name):L.preset(name);
 for(const [backend,codegen] of [['numpy',NP],['torch',Torch]]){
  const plan=P.prepare(graph,{supportedKinds:codegen.supportedKinds});
  const original=L.evaluate(graph),index=L.indexGraph(L.validateGraph(graph));
  assert.deepEqual(plan.sorted.map(n=>n.id),index.ordered.map(n=>n.id),backend+' '+name);
  assert.deepEqual(plan.forward,original,backend+' '+name);
  assert.deepEqual([...plan.edges], [...index.incomingByPort],backend+' '+name);
  assert.deepEqual(plan.sorted.map(n=>plan.steps.get(n.id)?.shape),original.steps.map(s=>s.shape));
  // Backend compilation must share the exact same source/port/shape
  // observation envelope as a user-facing numeric GraphSession.
  const independentIR=IR.snapshot(graph,{trace:original});
  const inspected=Session.inspect(graph,{numeric:true});
  assert.deepEqual(plan.executionIR,independentIR,backend+' '+name+' canonical IR');
  assert.deepEqual(plan.executionIR,inspected.ir,backend+' '+name+' session/plan IR');
  assert.deepEqual(plan.executionIR.nodes.map(x=>x.id),plan.sorted.map(x=>x.id));
  for(const node of plan.executionIR.nodes)
   for(const [port,source] of Object.entries(node.inputSources))
    assert.equal(source,plan.edges.get(node.id+':'+port)||null,
      backend+' '+name+' '+node.id+'.'+port+' source parity');
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
 assert.deepEqual(plan.executionIR.nodes.map(n=>n.id),plan.sorted.map(n=>n.id));
 assert.deepEqual(plan.executionIR,Session.inspect(dangling,{numeric:true,targetId:'down'}).ir);
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
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const assets=new Set(read('android/teaching-assets.txt').split(/\r?\n/).map(x=>x.trim()));
for(const script of ['graph_lowering_plan.js','shape_index_rules.js','execution_ir.js']){
 assert.ok(assets.has(script),'Android missing canonical graph dependency '+script);
 assert.ok(read('atlas_server.py').includes('"/'+script+'":"'+script+'"'),
   'server missing canonical graph dependency '+script);
}
for(const page of ['learn.html','capabilities.html']){
 const html=read(page),scripts=[...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map(m=>m[1]);
 const dependency=scripts.indexOf('graph_lowering_plan.js');
 assert.ok(dependency>scripts.indexOf('branch_execution.js'),page+' must load graph branch utilities');
 assert.ok(scripts.indexOf('shape_index_rules.js')<scripts.indexOf('execution_ir.js'),
   page+' shape mathematics must precede canonical execution IR');
 assert.ok(scripts.indexOf('execution_ir.js')<dependency,
   page+' must load graph identity IR before the exporter planner');
 assert.equal(scripts.filter(s=>s==='execution_ir.js').length,1,
   page+' must install the canonical IR once');
 assert.ok(dependency<scripts.indexOf('python_codegen.js'),page+' must load planner before NumPy');
 assert.ok(dependency<scripts.indexOf('torch_codegen.js'),page+' must load planner before Torch');
}
console.log('SHARED_LOWERING_PLAN_PASS deterministic DAG, ports, trace, backend fail-closed, target branch, registry');
