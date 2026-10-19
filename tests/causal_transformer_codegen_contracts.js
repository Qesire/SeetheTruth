'use strict';
// Independent full-graph CPU execution and causal information-flow oracle.
// The source is a small, explicit trainable single-head Transformer, not a pretrained LLM.
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),N=require('../python_codegen.js'),
  Torch=require('../torch_codegen.js'),G=require('../graph_autodiff.js'),
  Training=require('../train_state.js'),Concept=require('../concept_flow.js');
function invoke(source,extra=''){
 const bytes=execFileSync('python3',['-c',source+'\n'+extra],{encoding:'utf8',timeout:120000,maxBuffer:6*1024*1024});
 return JSON.parse(bytes.trim().split('\n').at(-1));
}
const flat=x=>x.flat(Infinity),shape=x=>Array.isArray(x)?[x.length,...shape(x[0])]:[];
function same(label,a,b,eps=3e-8){
 assert.deepEqual(shape(a),shape(b),label+' shape');
 const xs=flat(a),ys=flat(b);
 xs.forEach((v,i)=>assert.ok(Number.isFinite(v)&&Math.abs(v-ys[i])<=eps+Math.abs(ys[i])*1e-10,
  label+'['+i+'] '+v+' != '+ys[i]));
}
const graph=L.preset('causal_transformer_lm');
const source=JSON.stringify(graph),initial=L.evaluate(graph),oracle=new Map(initial.steps.map(x=>[x.id,x]));
assert.equal(graph.nodes.length,23);
assert.equal(Concept.evaluate(graph).steps.length,23,'full concept-first flow must infer every shape');
const seq=oracle.get('token_ids').values.map(row=>row[0]);
assert.deepEqual(oracle.get('prefix').values.map(row=>row[0]),seq.slice(0,3));
assert.deepEqual(oracle.get('labels').values.map(row=>row[0]),seq.slice(1,4));
const probs=oracle.get('probs').values,mask=oracle.get('mask').values;
assert.deepEqual(mask,[[1,0,0],[1,1,0],[1,1,1]]);
for(let i=0;i<3;i++){
 let total=0;
 for(let j=0;j<3;j++){
  if(j>i)assert.equal(probs[i][j],0,'future token leaks through causal mask');
  total+=probs[i][j];
 }
 assert.ok(Math.abs(total-1)<1e-12,'masked softmax row normalization');
}
const np=N.compile(graph),torch=Torch.compile(graph);
assert.deepEqual([...torch.trainable].sort(),['embedding_table','lm_head','wk','wq','wv']);
assert.equal(torch.lossId,'loss');
const numpy=invoke(np.source),pt=invoke(torch.source,
 'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
for(const step of initial.steps){
 same('NumPy/'+step.id,numpy[step.id],step.values);
 same('Torch/'+step.id,pt[step.id],step.values);
}
assert.equal(JSON.stringify(graph),source,'compilers must never mutate editable graph');
// Change a future observed token while keeping earlier prefix tokens fixed:
// earlier hidden states/logits must be numerically invariant.
const changed=JSON.parse(JSON.stringify(oracle.get('token_ids').values));
changed[2][0]=(changed[2][0]+1)%8;
const arg=JSON.stringify(JSON.stringify(changed));
const futureNP=invoke(np.source,'v=run_graph({"token_ids":json.loads('+arg+')})\nprint(json.dumps({k:x.tolist() for k,x in v.items()}))');
const futurePT=invoke(torch.source,'v=GraphRuntime({"token_ids":json.loads('+arg+')}).forward()\nprint(json.dumps({k:x.detach().tolist() for k,x in v.items()}))');
same('NumPy/Torch changed context',futureNP.context,futurePT.context);
same('NumPy/Torch changed logits',futureNP.logits,futurePT.logits);
for(const id of ['context','logits','probs']){
 same('causal invariance/'+id+'/prefix',futureNP[id].slice(0,2),numpy[id].slice(0,2),1e-12);
}
assert.notDeepEqual(futureNP.context[2],numpy.context[2],
 'changing third prefix token must affect its own context');
assert.notDeepEqual(futureNP.logits[2],numpy.logits[2],
 'changing third prefix token must affect its own prediction');
// Only the structural mask connection is non-differentiable, while Q/K/V remain trainable.
const backward=G.backward(graph,initial);
assert.ok(backward.cutEdges.some(x=>x.nodeId==='probs'&&x.port==='mask'));
assert.ok(backward.pathIds.includes('q')&&backward.pathIds.includes('k')&&backward.pathIds.includes('v'));
const fitted=Training.steps(graph,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
const py=invoke(torch.source,[
 'g=GraphRuntime()',
 'g.forward()[g.loss_id].sum().backward()',
 'grad={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
 'history=g.train(steps=2,lr=.004,weight_decay=.005)',
 'updated={k:p.detach().tolist() for k,p in g.parameters.items()}',
 'import os,tempfile',
 'with tempfile.TemporaryDirectory() as folder:',
 '  path=os.path.join(folder,"causal.pt")',
 '  g.save_checkpoint(path)',
 '  replay=GraphRuntime()',
 '  replay.load_checkpoint(path)',
 '  g.train(steps=1,lr=.004,weight_decay=.005)',
 '  replay.train(steps=1,lr=.004,weight_decay=.005)',
 '  continuous={k:p.detach().tolist() for k,p in g.parameters.items()}',
 '  restored={k:p.detach().tolist() for k,p in replay.parameters.items()}',
 'print(json.dumps({"grad":grad,"updated":updated,"history":history,"continuous":continuous,"restored":restored}))'
].join('\n'));
for(const id of torch.trainable){
 same('causal VJP/'+id,py.grad[id],backward.gradients[id],3e-7);
 same('causal AdamW/'+id,py.updated[id],fitted.graph.nodes.find(n=>n.id===id).params.values,3e-6);
 same('causal checkpoint/'+id,py.continuous[id],py.restored[id],1e-10);
 assert.ok(flat(py.grad[id]).some(v=>Math.abs(v)>1e-12),'no backward signal for '+id);
}
assert.equal(py.history.length,2);
const eps=1e-5;
for(const [id,i,j] of [['wq',0,1],['wk',1,1],['wv',2,0],['embedding_table',seq[0],0]]){
 const perturbed=(dir)=>{
  const g=JSON.parse(source);g.nodes.find(n=>n.id===id).params.values[i][j]+=dir*eps;
  return L.evaluate(g).steps.find(n=>n.id==='loss').values[0][0];
 };
 const fd=(perturbed(1)-perturbed(-1))/(2*eps),analytic=backward.gradients[id][i][j];
 assert.ok(Math.abs(fd-analytic)<1e-6,'finite difference gradient mismatch '+id+': '+fd+' vs '+analytic);
}
// Test shape and data guards directly, without generating fake valid exported code.
const invalid=JSON.parse(source);
invalid.nodes.find(n=>n.id==='mask').kind='input';
invalid.nodes.find(n=>n.id==='mask').params={rows:2,width:2,seed:1};
assert.throws(()=>L.evaluate(invalid),/逐元素操作|Mask|形状|相同|same/i);
console.log('CAUSAL_TRANSFORMER_TRAINING_PASS 23 real tensor nodes, exact causal invariance, masked VJP, 5 trainable params, finite difference, NumPy/Torch parity, 2 AdamW + checkpoint replay');
