'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),T=require('../train_state.js');
const G=require('../graph_autodiff.js'),Torch=require('../torch_codegen.js');
function python(program, extra){
 const code=program.source+"\n"+extra+"\n";
 const out=execFileSync('python3',['-c',code],{encoding:'utf8',timeout:90000,maxBuffer:5*1024*1024});
 return JSON.parse(out.trim().split('\n').at(-1));
}
const close=(a,b,where,eps=4e-7)=>{
 const u=a.flat(Infinity),v=b.flat(Infinity);
 assert.equal(u.length,v.length,where+' item count');
 u.forEach((n,i)=>assert.ok(Number.isFinite(v[i])&&Math.abs(n-v[i])<=eps+Math.abs(n)*1e-10,where+'['+i+']: '+n+' vs '+v[i]));
};
let checked=0;
for(const name of ['trainable_core','trainable_attention','trainable_swiglu','nd_trainable_gqa','nd_trainable_mqa','nd_cached_gqa_decode']){
 const graph=L.preset(name),program=Torch.compile(graph);
 const py=python(program,'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
 const js=L.evaluate(graph).steps;
 assert.equal(Object.keys(py).length,js.length,name+' count');
 for(const n of js){close(n.values,py[n.id],name+'/'+n.id);checked++;}
 assert.ok(program.source.includes('torch.nn.Parameter')||program.trainable.length===0);
 assert.ok(!program.source.includes('from seethetruth'),'no runtime dependency');
 console.log('Torch forward parity PASS '+name+': '+js.length+' nodes');
}
for(const name of ['trainable_core','trainable_attention','trainable_swiglu','nd_trainable_gqa','nd_trainable_mqa']){
 const graph=L.preset(name),program=Torch.compile(graph),g=G.backward(graph,L.evaluate(graph));
 const lr=.004,weightDecay=.005;
 const py=python(program,"r=GraphRuntime()\nr.forward()[r.loss_id].sum().backward()\nfirst={k:p.grad.detach().tolist() for k,p in r.parameters.items() if p.grad is not None}\nhistory=r.train(steps=2,lr=.004,weight_decay=.005)\nafter={k:p.detach().tolist() for k,p in r.parameters.items()}\nimport tempfile, os\nwith tempfile.TemporaryDirectory() as temp:\n    path=os.path.join(temp,\"test-graph-checkpoint.pt\")\n    r.save_checkpoint(path)\n    restored=GraphRuntime()\n    restored.load_checkpoint(path)\n    r.train(steps=1,lr=.004,weight_decay=.005)\n    restored.train(steps=1,lr=.004,weight_decay=.005)\n    resume={k:p.detach().tolist() for k,p in restored.parameters.items()}\n    uninterrupted={k:p.detach().tolist() for k,p in r.parameters.items()}\nprint(json.dumps({\"first\":first,\"after\":after,\"history\":history,\n                  \"resume\":resume,\"uninterrupted\":uninterrupted}))");
 const js=T.steps(graph,{count:2,lr,weightDecay});
 for(const [id,val] of Object.entries(py.first))close(val,g.gradients[id],name+'/'+id+'/gradient',8e-6);
 for(const [id,val] of Object.entries(py.after)){
  close(val,js.graph.nodes.find(n=>n.id===id).params.values,name+'/'+id+'/2-step',6e-6);
  close(py.resume[id],py.uninterrupted[id],name+'/'+id+'/checkpoint',1e-10);
 }
 assert.equal(py.history.length,2);
 assert.ok(Number.isFinite(py.history[0].beforeLoss));
 assert.ok(Number.isFinite(py.history[1].afterLoss));
 console.log('Torch backward+state parity PASS '+name+', 2 steps + exact checkpoint resume');
}
const Rewrite=require('../paper_graph_rewrites.js');
const adapter=Rewrite.lora(L.preset('trainable_attention'),'q_proj').graph;
const exported=Torch.compile(adapter);
assert.ok(!exported.trainable.includes('wq'),'frozen pretrained W0 must not be torch.nn.Parameter');
const adaptedOutput=python(exported,'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
for(const n of L.evaluate(adapter).steps)close(n.values,adaptedOutput[n.id],'LoRA/'+n.id);
const frozenCode=[
 'g=GraphRuntime()',
 'original=g.inputs["wq"].clone()',
 'history=g.train(steps=2,lr=.004,weight_decay=.005)',
 'print(json.dumps({"w":g.inputs["wq"].tolist(),"original":original.tolist(),"history":history}))'
].join("\n");
const frozenRun=python(exported,frozenCode);
assert.deepEqual(frozenRun.w,frozenRun.original,'Torch optimizer mutated frozen base W0');
assert.equal(frozenRun.history.length,2);
const clone=x=>JSON.parse(JSON.stringify(x));
const graph=L.preset('trainable_swiglu');
assert.throws(()=>Torch.compile(L.preset('trainable_moe')),/不支持的积木/);
assert.throws(()=>Torch.compile(L.preset('grpo_objective')),/不支持的积木/);
const partial=clone(graph);partial.nodes.push({id:'loose',kind:'softmax',x:20,y:20,params:{}});
assert.throws(()=>Torch.compile(partial),/没有连接|还没有连接/);
const scoped=Torch.compile(partial,{targetId:'w1'});
assert.equal(scoped.nodes.length,1);
const sample=python(scoped,'print(json.dumps({k:v.detach().tolist() for k,v in run_graph().items()}))');
assert.ok(sample.w1);
const hacked=clone(graph);hacked.nodes[0].kind='evil';
assert.throws(()=>Torch.compile(hacked),/未知积木/);
assert.equal(new Set(Torch.supportedKinds).size,Torch.supportedKinds.length);
console.log('Torch CPU executable training export PASS: '+checked+' JS-vs-Torch intermediate tensors, 5 forward/backward/AdamW/checkpoint oracles');
