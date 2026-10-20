'use strict';
/* Real recurrent-depth attention/FFN. SAME Parameter IDs across every step.
 * Independent NumPy/Torch subprocesses, true grad accumulation and BPTT.
 */
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),N=require('../python_codegen.js'),
  Torch=require('../torch_codegen.js'),Grad=require('../graph_autodiff.js'),
  Train=require('../train_state.js'),Concept=require('../concept_flow.js');
const py=(source,extra='')=>{
 const out=execFileSync('python3',['-c',source+'\n'+extra],{
   timeout:120000,encoding:'utf8',maxBuffer:7*1024*1024}).trim();
 return JSON.parse(out.split('\n').at(-1));
};
const shape=x=>Array.isArray(x)?[x.length,...shape(x[0])]:[];
function close(label,a,b,epsilon=4e-8){
 assert.deepEqual(shape(a),shape(b),label+' tensor shape');
 const x=a.flat(Infinity),y=b.flat(Infinity);
 for(let i=0;i<x.length;i++)assert.ok(Number.isFinite(x[i])&&
   Math.abs(x[i]-y[i])<=epsilon+Math.abs(y[i])*1e-10,
   label+'['+i+'] '+x[i]+' !== '+y[i]);
}
const params=['embed','wq','wk','wv','wo','wgate','wup','wdown'].sort();
const cases=[['loop_transformer_two',2,56],['loop_transformer_three',3,75]];
let total=0;
for(const [recipe,count,wantNodes] of cases){
 const graph=L.preset(recipe),snapshot=JSON.stringify(graph),observed=L.evaluate(graph);
 const steps=Object.fromEntries(observed.steps.map(x=>[x.id,x]));
 assert.equal(graph.nodes.length,wantNodes,'all recurrent ops visible, not hidden in a loop black box');
 assert.equal(Concept.evaluate(graph).steps.length,wantNodes,'all recurrent ops have a conceptual shape path');
 assert.equal(graph.nodes.filter(n=>n.kind==='parameter').length,8,'no duplicate Parameter identity');
 const torch=Torch.compile(graph),np=N.compile(graph);
 assert.deepEqual([...torch.trainable].sort(),params,'loop depth must NOT increase parameter count');
 assert.equal(torch.lossId,'loss');
 assert.equal(torch.nodes.length,graph.nodes.length);
 const n=py(np.source),t=py(torch.source,
   'print(json.dumps({k:x.detach().tolist() for k,x in GraphRuntime().forward().items()}))');
 for(const node of observed.steps){
   close(recipe+' NumPy/'+node.id,n[node.id],node.values);
   close(recipe+' Torch/'+node.id,t[node.id],node.values);
   total+=2;
 }
 const ids=steps.ids.values.map(r=>r[0]);
 assert.deepEqual(steps.prefix.values.map(x=>x[0]),ids.slice(0,3));
 assert.deepEqual(steps.labels.values.map(x=>x[0]),ids.slice(1,4));
 const mask=steps.mask.values;
 assert.deepEqual(mask,[[1,0,0],[1,1,0],[1,1,1]]);
 for(let i=1;i<=count;i++){
   const prefix='r'+i+'_';
   const p=steps[prefix+'prob'].values;
   for(let row=0;row<3;row++){
     for(let col=row+1;col<3;col++)assert.equal(p[row][col],0,prefix+' future token attention probability');
     assert.ok(Math.abs(p[row].reduce((s,v)=>s+v,0)-1)<1e-12);
   }
   assert.equal(steps[prefix+'state'].shape[0],3);
   assert.equal(steps[prefix+'state'].shape[1],4);
 }
 assert.notDeepEqual(steps.r1_state.values,steps['r'+count+'_state'].values,
  'new loop steps must perform actual hidden state refinement');
 const changed=JSON.parse(JSON.stringify(steps.ids.values));
 changed[2][0]=(changed[2][0]+1)%8;
 const arg=JSON.stringify(JSON.stringify(changed));
 const pn=py(np.source,'r=run_graph({"ids":json.loads('+arg+')})\nprint(json.dumps({k:v.tolist() for k,v in r.items()}))');
 const pt=py(torch.source,'r=GraphRuntime({"ids":json.loads('+arg+')}).forward()\nprint(json.dumps({k:v.detach().tolist() for k,v in r.items()}))');
 close(recipe+' changed NumPy↔Torch',pn.logits,pt.logits);
 for(const i of Array.from({length:count},(_,j)=>j+1)){
  for(const key of ['r'+i+'_prob','r'+i+'_state'])
   close(recipe+' causal invariance '+key,pn[key].slice(0,2),n[key].slice(0,2),1e-12);
 }
 close(recipe+' output causal invariance',pn.logits.slice(0,2),n.logits.slice(0,2),1e-12);
 assert.notDeepEqual(pn.logits[2],n.logits[2],'own token must change logits');

 const backward=Grad.backward(graph,observed);
 const js=Train.steps(graph,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
 const real=py(torch.source,[
  'g=GraphRuntime()',
  'g.forward()[g.loss_id].sum().backward()',
  'grad={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
  'history=g.train(steps=2,lr=.004,weight_decay=.005)',
  'after={k:p.detach().tolist() for k,p in g.parameters.items()}',
  'import tempfile,os',
  'with tempfile.TemporaryDirectory() as d:',
  '    path=os.path.join(d,"recurrent.pt")',
  '    g.save_checkpoint(path)',
  '    restored=GraphRuntime()',
  '    restored.load_checkpoint(path)',
  '    g.train(steps=1,lr=.004,weight_decay=.005)',
  '    restored.train(steps=1,lr=.004,weight_decay=.005)',
  '    original={k:p.detach().tolist() for k,p in g.parameters.items()}',
  '    resumed={k:p.detach().tolist() for k,p in restored.parameters.items()}',
  'print(json.dumps({"grad":grad,"after":after,"original":original,"resumed":resumed,"history":history}))'
 ].join('\n'));
 for(const id of params){
  assert.ok(backward.gradients[id],id+' did not get BPTT gradient');
  close(recipe+' accumulated BPTT '+id,real.grad[id],backward.gradients[id],5e-7);
  close(recipe+' AdamW '+id,real.after[id],js.graph.nodes.find(n=>n.id===id).params.values,6e-6);
  close(recipe+' checkpoint '+id,real.original[id],real.resumed[id],1e-10);
 }
 assert.equal(real.history.length,2);
 assert.equal(JSON.stringify(graph),snapshot,'compiled/grad graph must be immutable');
 const epsilon=1e-5;
 for(const [id,i,j] of [['wq',0,1],['wk',1,1],['wgate',0,0],['wdown',1,1],['embed',ids[0],0]]){
  const loss=delta=>{
   const g=JSON.parse(snapshot);
   g.nodes.find(n=>n.id===id).params.values[i][j]+=delta*epsilon;
   return L.evaluate(g).steps.find(n=>n.id==='loss').values[0][0];
  };
  const fd=(loss(1)-loss(-1))/(2*epsilon);
  const grad=backward.gradients[id][i][j];
  assert.ok(Math.abs(fd-grad)<1e-6,recipe+' finite diff '+id+' '+fd+' vs '+grad);
 }
 console.log('LOOP_REAL_TRAINING_PASS '+recipe+' loops='+count+' nodes='+wantNodes+
  ' 8 physical parameter identities BPTT and 2-step AdamW checkpoint replay');
}
// A DAG displays depth recurrence by unrolling; actual feedback cycles must be rejected.
const cycle=JSON.parse(JSON.stringify(L.preset('loop_transformer_two')));
cycle.edges.find(e=>e.to==='r1_norm'&&e.port==='x').from='r2_state';
assert.throws(()=>L.order(L.validateGraph(cycle)),/循环计算依赖/);
console.log('LOOP_TRANSFORMER_CODEGEN_PASS '+total+' independent node tensor comparisons, 2 and 3 loops; causal, shared gradients, finite differences, checkpoint exact');
