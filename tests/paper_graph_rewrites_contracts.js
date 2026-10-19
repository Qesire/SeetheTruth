'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),Rewrite=require('../paper_graph_rewrites.js');
const G=require('../graph_autodiff.js'),T=require('../train_state.js');
const Py=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const before=L.preset('autodiff_core'),snapshot=JSON.stringify(before);
const old=L.evaluate(before).steps;
const result=Rewrite.lora(before,'project',{rank:2,alpha:2});
const graph=result.graph,after=L.evaluate(graph).steps;
assert.equal(JSON.stringify(before),snapshot,'paper rewrite must not mutate original');
assert.equal(result.delta,0,'B=zeros should yield exact before/after forward equality');
assert.equal(graph.nodes.length,before.nodes.length+6);
assert.equal(graph.edges.length,before.edges.length+7);
assert.equal(graph.nodes.find(n=>n.id==='project').kind,'add','original output node ID preserved');
assert.equal(graph.nodes.filter(n=>n.kind==='parameter').length,2);
assert.deepEqual(graph.paperPatches,[result.patch]);
assert.ok(result.source.includes('arxiv.org/abs/2106.09685'));
for(const oldNode of old.filter(n=>n.id!=='project'&&n.id!=='shared_weight')){
 const current=after.find(s=>s.id===oldNode.id);
 assert.ok(current,'original downstream '+oldNode.id+' preserved');
 const a=oldNode.values.flat(Infinity),b=current.values.flat(Infinity);
 a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-10,oldNode.id+' forward changed at '+i));
}
const checked=L.validateGraph(JSON.parse(JSON.stringify(graph)));
assert.deepEqual(checked,graph,'patch metadata must survive saved JSON roundtrip');
const grad=G.backward(graph,L.evaluate(graph));
assert.ok(grad.gradients[result.patch.bId].flat().some(x=>Math.abs(x)>1e-8),'LoRA B initially receives real gradient');
assert.ok(grad.gradients[result.patch.aId].flat().every(x=>Math.abs(x)<1e-12),'LoRA A initially has zero gradient if B=0');
const trained=T.steps(graph,{count:2,lr:.004,weightDecay:.005});
assert.equal(trained.history.length,2);
for(const id of [result.patch.aId,result.patch.bId]){
 const n=trained.graph.nodes.find(n=>n.id===id);
 assert.equal(n.params.optimizer.t,2);
}
assert.deepEqual(trained.graph.nodes.find(n=>n.id==='shared_weight'),graph.nodes.find(n=>n.id==='shared_weight'),'frozen original base source did not update');
assert.equal(Py.compile(graph).nodes.length,graph.nodes.length);
assert.equal(Torch.compile(graph).nodes.length,graph.nodes.length);
assert.throws(()=>Rewrite.lora(before,'normalized'),/selected 2D MatMul/);
assert.throws(()=>Rewrite.lora(before,'project',{rank:9}),/rank/);
assert.throws(()=>Rewrite.lora(before,'project',{rank:2,alpha:Infinity}),/alpha/);
assert.throws(()=>Rewrite.lora(before,'project',{rank:8}),/rank exceeds/);
assert.throws(()=>Rewrite.lora(L.preset('trainable_attention'),'q_proj'),/nontrainable Input/);
const broken=JSON.parse(JSON.stringify(graph));broken.paperPatches[0].aId='not_present';
assert.throws(()=>L.validateGraph(broken),/invalid LoRA rewrite lineage/);
console.log('LoRA source-paper DAG rewrite PASS: frozen W0, 2 trainable low-rank params, exact forward parity, VJP/AdamW, JSON, NumPy/Torch export');
