'use strict';
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Loop=require('../loop_transformer.js'),
 Why=require('../loop_design_tradeoffs.js'),Grad=require('../graph_autodiff.js'),
 Train=require('../train_state.js'),NP=require('../python_codegen.js'),
 Torch=require('../torch_codegen.js');
function python(src,extra){
 const run=execFileSync('python3',['-c',src+'\n'+extra],{encoding:'utf8',timeout:120000,maxBuffer:7*1024*1024});
 return JSON.parse(run.trim().split('\n').at(-1));
}
function close(label,a,b,eps=7e-8){
 const flat=x=>x.flat(Infinity),x=flat(a),y=flat(b);
 assert.equal(x.length,y.length,label+' size');
 x.forEach((v,i)=>assert.ok(Number.isFinite(v)&&Math.abs(v-y[i])<=eps+Math.abs(y[i])*1e-10,
   label+'['+i+']: '+v+' vs '+y[i]));
}
const name='loop_residual_scaling',defaultGraph=L.preset(name);
assert.equal(defaultGraph.nodes.length,61,'explicit 57-node 2-loop graph plus four branch scales');
assert.equal(Loop.inspectResidualScaling(defaultGraph).rule,'inverse_loops');
assert.equal(Why.keyFor(defaultGraph,'r0_attn_delta'),'residual_scaling');
const Industry=require('../industrial_source_inspector.js');
assert.equal(Industry.kind(defaultGraph,'r0_attn_delta'),'loop_residual');
assert.match(Industry.render('loop_residual'),/DeepLoop/);
assert.match(Industry.render('loop_residual'),/Pre-Norm/);
assert.equal(Industry.kind(L.preset('causal_transformer_lm'),'r0_attn_delta'),null);

assert.match(Why.render(defaultGraph,'r0_ffn_delta'),/为什么共享循环 Transformer 的残差/);
assert.match(Why.render(defaultGraph,'loss'),/状态变化 RMS/);
const rows=Why.residualComparisons();
assert.deepEqual(rows.map(x=>x.rule),['unit','inverse_sqrt','inverse_loops']);
assert.deepEqual(rows.map(x=>x.epsilon),[1,1/Math.sqrt(2),.5]);
assert.equal(new Set(rows.map(x=>x.paramElements)).size,1,'matched parameter storage count');
assert.equal(new Set(rows.map(x=>x.matmulMAC)).size,1,'matched actual MatMul workload');
assert.ok(new Set(rows.map(x=>x.loss.toFixed(13))).size>1,'real loop scaling must alter actual outputs');
assert.ok(new Set(rows.map(x=>x.wqGradientL2.toFixed(13))).size>1,'scaling must alter actual gradients');
console.log('LOOP_RESIDUAL_ABLATION_NUMERIC_EVIDENCE '+JSON.stringify(rows));

let tensors=0;
for(const rule of Loop.RESIDUAL_RULES){
 const graph=Loop.buildResidualScaling({rule}),info=Loop.inspectResidualScaling(graph);
 const original=JSON.stringify(graph),oracle=L.evaluate(graph).steps;
 assert.equal(info.nodes,61);
 assert.equal(info.parameters.length,8);
 assert.equal(graph.edges.length,Loop.build({loops:2}).edges.length+4);
 const np=NP.compile(graph),torch=Torch.compile(graph);
 assert.deepEqual([...torch.trainable].sort(),[...info.parameters].sort());
 const outNP=python(np.source,"print(json.dumps({k:v.tolist() for k,v in run_graph().items()}))");
 const outT=python(torch.source,"print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))");
 for(const v of oracle){
   close(rule+'/numpy/'+v.id,outNP[v.id],v.values);
   close(rule+'/torch/'+v.id,outT[v.id],v.values);
   tensors+=2;
 }
 const grads=Grad.backward(graph,L.evaluate(graph));
 for(const id of info.parameters)assert.ok(grads.gradients[id],rule+' parameter '+id+' missing BPTT');
 assert.equal(JSON.stringify(graph),original,'compiler mutated graph');
 const output=new Map(oracle.map(s=>[s.id,s]));
 for(let r=0;r<2;r++){
   const mask=output.get('r'+r+'_mask').values;
   assert.deepEqual(mask,[[1,0,0],[1,1,0],[1,1,1]]);
 }
}
const graph=Loop.buildResidualScaling({rule:'inverse_loops'}),torch=Torch.compile(graph),
  jsGrad=Grad.backward(graph,L.evaluate(graph)),jsTrain=Train.steps(graph,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
const py=python(torch.source,[
'g=GraphRuntime()',
'g.forward()[g.loss_id].sum().backward()',
'grad={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
'history=g.train(steps=2,lr=.004,weight_decay=.005)',
'after={k:p.detach().tolist() for k,p in g.parameters.items()}',
'import tempfile,os',
'with tempfile.TemporaryDirectory() as d:',
'    p=os.path.join(d,"residual-scaling.pt")',
'    g.save_checkpoint(p)',
'    restored=GraphRuntime()',
'    restored.load_checkpoint(p)',
'    g.train(steps=1,lr=.004,weight_decay=.005)',
'    restored.train(steps=1,lr=.004,weight_decay=.005)',
'    full={k:p.detach().tolist() for k,p in g.parameters.items()}',
'    resumed={k:p.detach().tolist() for k,p in restored.parameters.items()}',
'print(json.dumps({"grad":grad,"after":after,"full":full,"resumed":resumed,"history":history}))'
].join('\n'));
for(const id of torch.trainable){
 close('inverse BPTT '+id,py.grad[id],jsGrad.gradients[id],5e-7);
 close('inverse AdamW '+id,py.after[id],jsTrain.graph.nodes.find(n=>n.id===id).params.values,6e-6);
 close('inverse resume '+id,py.full[id],py.resumed[id],1e-10);
}
assert.equal(py.history.length,2);
for(const [id,row,col] of [['Wq',0,1],['Wgate',0,1],['embedding',0,0]]){
 const eps=1e-5;
 const get=sign=>{
  const modified=JSON.parse(JSON.stringify(graph));
  modified.nodes.find(n=>n.id===id).params.values[row][col]+=sign*eps;
  return L.evaluate(modified).steps.find(n=>n.id==='loss').values[0][0];
 };
 const estimate=(get(1)-get(-1))/(2*eps),analytical=jsGrad.gradients[id][row][col];
 assert.ok(Math.abs(estimate-analytical)<1e-6,'finite difference '+id+': '+estimate+' != '+analytical);
}
const resumed=JSON.parse(JSON.stringify(jsTrain.graph)),before=Object.fromEntries(resumed.nodes.filter(n=>n.kind==='parameter').map(n=>[n.id,JSON.stringify(n.params)]));
for(const node of resumed.nodes.filter(n=>/^r[01]_(attn_delta|ffn_delta)$/.test(n.id)))node.params.factor=1/Math.sqrt(2);
resumed.title='任务 43 · Loop Residual Scaling ×2 · inverse_sqrt';
const valid=L.validateGraph(resumed);
assert.equal(Loop.inspectResidualScaling(valid).rule,'inverse_sqrt');
assert.deepEqual(before,Object.fromEntries(valid.nodes.filter(n=>n.kind==='parameter').map(n=>[n.id,JSON.stringify(n.params)])),
 'switching residual coefficient must not reset weights or optimizer moments');
assert.throws(()=>Loop.buildResidualScaling({loops:3}),/exactly two loops/);
assert.throws(()=>Loop.buildResidualScaling({rule:'made-up'}),/unknown/);
const broken=JSON.parse(JSON.stringify(defaultGraph));broken.edges.find(e=>e.to==='r0_state'&&e.port==='b').from='r0_down';
assert.throws(()=>Loop.inspectResidualScaling(broken),/no longer connected/);
const rewire=JSON.parse(JSON.stringify(defaultGraph));
rewire.edges.find(e=>e.to==='r1_q'&&e.port==='a').from='r0_attn_norm';
assert.throws(()=>Loop.inspectResidualScaling(rewire),/no longer a matched residual-scaling experiment/,
  'changing intermediate attention math must invalidate the comparison label');
assert.equal(Industry.kind(rewire,'r0_attn_delta'),null,'modified graph must not retain false paper identity');
const changedData=JSON.parse(JSON.stringify(defaultGraph));
changedData.nodes.find(n=>n.id==='tokens').params.seed=5;
assert.throws(()=>Loop.inspectResidualScaling(changedData),/no longer a matched residual-scaling experiment/,
  'changing the data source invalidates controlled initial-state evidence');
const unequal=JSON.parse(JSON.stringify(defaultGraph));unequal.nodes.find(n=>n.id==='r1_attn_delta').params.factor=.6;
assert.throws(()=>Loop.inspectResidualScaling(unequal),/diverged/);
console.log('LOOP_RESIDUAL_SCALING_PASS 3 controlled rules, '+tensors+' JS/NumPy/Torch intermediate checks, tied gradients, finite differences, 2-step AdamW/checkpoint, evidence-labelled design tradeoffs');
