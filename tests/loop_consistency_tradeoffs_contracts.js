'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Loop=require('../loop_transformer.js'),
 Consistency=require('../loop_consistency.js'),Why=require('../loop_design_tradeoffs.js'),
 NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js'),
 Obj=require('../training_objective.js'),Grad=require('../graph_autodiff.js'),
 Train=require('../train_state.js'),Concept=require('../concept_flow.js');
function python(code,extra=''){
 const out=execFileSync('python3',['-c',code+'\n'+extra],{encoding:'utf8',timeout:120000,maxBuffer:10*1024*1024});
 return JSON.parse(out.trim().split('\n').at(-1));
}
function near(label,a,b,eps=1e-7){
 const x=a.flat(Infinity),y=b.flat(Infinity);
 assert.equal(x.length,y.length,label+' element count');
 x.forEach((z,i)=>assert.ok(Number.isFinite(z)&&Math.abs(z-y[i])<eps+1e-10*Math.abs(y[i]),
 label+'['+i+'] '+z+' != '+y[i]));
}
const snapshots=Why.comparisons();
assert.equal(snapshots.length,3);
for(let i=0;i<snapshots.length;i++){
 const g=Loop.build({loops:i+1}),actual=Why.profile(g),r=snapshots[i];
 assert.deepEqual(actual,r,'why surface numbers must derive from same real graph');
 assert.equal(actual.physicalParameterElements,168,'8 actual shared Parameter tensors/168 elements');
 assert.equal(actual.coreParameterElements,136);
 assert.equal(actual.nonRecurrentParameterElements,32);
 assert.equal(actual.depthMatchedUntiedParameterElements,32+(i+1)*136);
 assert.equal(actual.parameterTensors,8);
 if(i>0){
  assert.ok(actual.matmulMAC>snapshots[i-1].matmulMAC,'reused loops must actually add GEMM compute');
  assert.equal(actual.physicalParameterElements,snapshots[i-1].physicalParameterElements,'loop depth cannot clone parameters');
 }
}
assert.equal(snapshots[2].graphNodes,79);
const graph=L.preset('loopformer_consistency'),save=JSON.stringify(graph),contract=Consistency.inspect(graph);
const trace=L.evaluate(graph),step=Object.fromEntries(trace.steps.map(s=>[s.id,s]));
assert.equal(graph.nodes.length,61,'two full shared recurrent blocks + four real supervision nodes');
assert.equal(Concept.evaluate(graph).steps.length,61);
assert.equal(contract.short,'r0_state');assert.equal(contract.long,'r1_state');
assert.equal(contract.teacher,'deep_teacher');assert.equal(contract.coefficient,0.1);
assert.equal(graph.nodes.filter(n=>n.kind==='parameter').length,8);
assert.equal(Obj.analyze(graph).preferred,'loss');
assert.ok(Obj.analyze(graph).isLoss('loss'));
const lambda=0.1;
const expression=step.base_loss.values[0][0]+lambda*step.short_consistency.values[0][0];
assert.ok(Math.abs(step.loss.values[0][0]-expression)<1e-12);
assert.ok(step.short_consistency.values[0][0]>0,'initial teacher and student are not identical');
near('teacher equals deep state numerically',step.deep_teacher.values,step.r1_state.values);
const np=NP.compile(graph),torch=Torch.compile(graph),jsNames=torch.trainable.sort();
assert.deepEqual(jsNames,['Wdown','Wgate','Wk','Wo','Wq','Wup','Wv','embedding'].sort());
const n=python(np.source),t=python(torch.source,
 'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
for(const s of trace.steps){
 near('NP/'+s.id,n[s.id],s.values);
 near('Torch/'+s.id,t[s.id],s.values);
}
const back=Grad.backward(graph,trace);
assert.ok(back.cutEdges.some(x=>x.nodeId==='short_consistency'&&x.port==='target'),
'teacher path must not receive gradient through consistency MSE target');
const script=[
 'g=GraphRuntime()',
 'g.forward()[g.loss_id].sum().backward()',
 'grad={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
 'history=g.train(steps=2,lr=.004,weight_decay=.005)',
 'updated={k:p.detach().tolist() for k,p in g.parameters.items()}',
 'import os,tempfile',
 'with tempfile.TemporaryDirectory() as work:',
 '  path=os.path.join(work,"loop_consistency.pt")',
 '  g.save_checkpoint(path)',
 '  restored=GraphRuntime()',
 '  restored.load_checkpoint(path)',
 '  g.train(steps=1,lr=.004,weight_decay=.005)',
 '  restored.train(steps=1,lr=.004,weight_decay=.005)',
 '  continuous={k:p.detach().tolist() for k,p in g.parameters.items()}',
 '  resume={k:p.detach().tolist() for k,p in restored.parameters.items()}',
 'print(json.dumps({"grad":grad,"updated":updated,"continuous":continuous,"resume":resume,"history":history}))'
].join('\n');
const p=python(torch.source,script),js=Train.steps(graph,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
for(const id of jsNames){
 near('BPTT/'+id,p.grad[id],back.gradients[id],7e-7);
 near('AdamW/'+id,p.updated[id],js.graph.nodes.find(n=>n.id===id).params.values,1e-5);
 near('resume/'+id,p.continuous[id],p.resume[id],1e-11);
}
assert.equal(p.history.length,2);
const baseline=Consistency.build({coefficient:0});
const baselineValues=L.evaluate(baseline).steps;
const baselineLoss=baselineValues.find(x=>x.id==='loss').values[0][0];
const ce=baselineValues.find(x=>x.id==='base_loss').values[0][0];
assert.ok(Math.abs(baselineLoss-ce)<1e-12,'λ=0 exactly recovers baseline supervised objective');
assert.equal(L.preset('loop_transformer_2').nodes.filter(n=>n.kind==='parameter').length,
graph.nodes.filter(n=>n.kind==='parameter').length);
assert.throws(()=>Consistency.build({coefficient:-1}),/consistency coefficient/);
assert.throws(()=>Consistency.build({coefficient:NaN}),/consistency coefficient/);
assert.throws(()=>Consistency.inspect(L.preset('loop_transformer_2')),/not a valid/);
assert.throws(()=>Why.profile(L.preset('token_lm_trainable')),/loop profiling requires/);
assert.equal(Why.keyFor(graph,'short_consistency'),'consistency');
assert.equal(Why.keyFor(graph,'Wq'),'sharing');
assert.match(Why.render(graph,'short_consistency'),/LoopFormer/);
assert.match(Why.render(graph,'Wq'),/MatMul MAC/);
assert.match(Why.render(graph,'r1_ce'),/每一轮/);
assert.equal(Why.render(L.preset('ffn'),'x'),'');
assert.equal(JSON.stringify(graph),save,'no inspection/training/export mutation');
console.log('LOOP_CONSISTENCY_PASS 61 real tensor intermediates×3 engines, deep stop-gradient, weighted CE+MSE, 8 shared BPTT gradients, AdamW, checkpoint, λ=0 ablation');
console.log('LOOP_DESIGN_WHY_PASS actual Parameter=168 vs untied=168/304/440, GEMM count increases, evidence and design falsification');
