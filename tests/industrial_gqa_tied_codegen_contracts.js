'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),NP=require('../python_codegen.js'),
 T=require('../torch_codegen.js'),G=require('../graph_autodiff.js'),
 Train=require('../train_state.js'),Concept=require('../concept_flow.js');
const py=(source,extra='')=>JSON.parse(execFileSync('python3',['-c',source+'\n'+extra],{
 encoding:'utf8',timeout:120000,maxBuffer:6*1024*1024}).trim().split('\n').at(-1));
const values=a=>a.flat(Infinity);
function close(name,a,b,eps=6e-8){
 const x=values(a),y=values(b);
 assert.equal(x.length,y.length,name+' length');
 x.forEach((v,i)=>assert.ok(Number.isFinite(v)&&Math.abs(v-y[i])<eps+Math.abs(y[i])*1e-10,
   name+'['+i+'] '+v+' != '+y[i]));
}
const graph=L.preset('industrial_gqa_rope_tied'),before=JSON.stringify(graph),
  trace=L.evaluate(graph),steps=new Map(trace.steps.map(s=>[s.id,s]));
assert.equal(graph.nodes.length,45);
assert.equal(Concept.evaluate(graph).steps.length,45,'all industrial graph nodes need shape contracts');
const masks=steps.get('mask').values;
assert.deepEqual(masks,[[1,0,0],[1,1,0],[1,1,1]]);
for(const id of ['prob0','prob1']){
 for(let t=0;t<3;t++){
  const row=steps.get(id).values[t];
  assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);
  for(let j=t+1;j<3;j++)assert.equal(row[j],0,'causal leakage '+id+' '+t+'/'+j);
 }
}
assert.equal(steps.get('q').shape[1],4,'query width = head count times head_dim');
for(const k of ['k','v','q0','q1','kn','kr','q0r','q1r'])assert.equal(steps.get(k).shape[1],2,k+' per-head width');
close('2Q1KV shared heads',steps.get('heads').values,
 steps.get('out0').values.map((row,i)=>row.concat(steps.get('out1').values[i])));
const embedding=steps.get('embed').values;
close('tied head is the SAME table transposed',steps.get('tied_head').values,
 embedding[0].map((_,col)=>embedding.map(row=>row[col])));
const numpy=NP.compile(graph),torch=T.compile(graph);
assert.deepEqual([...torch.trainable].sort(),['embed','kgain','qgain','wk','wo','wq','wv']);
assert.equal(torch.lossId,'loss');
const outN=py(numpy.source),outT=py(torch.source,
 'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
for(const s of trace.steps){
 close('NumPy/'+s.id,outN[s.id],s.values);
 close('Torch/'+s.id,outT[s.id],s.values);
}
assert.equal(JSON.stringify(graph),before,'compilers must not modify graph');
// Source token changed at t=2 must not affect output logits at t=0,1.
const other=JSON.parse(JSON.stringify(steps.get('ids').values));
other[2][0]=(other[2][0]+1)%8;
const literal=JSON.stringify(JSON.stringify(other));
const changedN=py(numpy.source,
 'v=run_graph({"ids":json.loads('+literal+')})\nprint(json.dumps({k:x.tolist() for k,x in v.items()}))');
const changedT=py(torch.source,
 'v=GraphRuntime({"ids":json.loads('+literal+')}).forward()\nprint(json.dumps({k:x.detach().tolist() for k,x in v.items()}))');
close('override consistent',changedN.logits,changedT.logits);
for(const k of ['out0','out1','logits'])close('no future leakage '+k,changedN[k].slice(0,2),outN[k].slice(0,2),1e-12);
assert.notDeepEqual(changedN.logits[2],outN.logits[2],'own-token change must change output');
const backward=G.backward(graph,trace);
assert.ok(backward.pathIds.includes('q0r')&&backward.pathIds.includes('q1r')&&backward.pathIds.includes('kr'));
for(const id of ['embed','qgain','kgain','wq','wk','wv','wo'])assert.ok(values(backward.gradients[id]).some(v=>Math.abs(v)>1e-12),'missing gradient '+id);
const js=Train.steps(graph,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
const external=py(torch.source,[
'g=GraphRuntime()',
'g.forward()[g.loss_id].sum().backward()',
'grad={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
'history=g.train(steps=2,lr=.004,weight_decay=.005)',
'after={k:p.detach().tolist() for k,p in g.parameters.items()}',
'import tempfile,os',
'with tempfile.TemporaryDirectory() as d:',
'  p=os.path.join(d,"gqa.pt")',
'  g.save_checkpoint(p)',
'  restored=GraphRuntime()',
'  restored.load_checkpoint(p)',
'  g.train(steps=1,lr=.004,weight_decay=.005)',
'  restored.train(steps=1,lr=.004,weight_decay=.005)',
'  full={k:p.detach().tolist() for k,p in g.parameters.items()}',
'  replay={k:p.detach().tolist() for k,p in restored.parameters.items()}',
'print(json.dumps({"grad":grad,"after":after,"history":history,"full":full,"replay":replay}))'
].join('\n'));
for(const id of torch.trainable){
 close('GQA VJP/'+id,external.grad[id],backward.gradients[id],8e-7);
 close('AdamW/'+id,external.after[id],js.graph.nodes.find(n=>n.id===id).params.values,5e-6);
 close('checkpoint/'+id,external.full[id],external.replay[id],1e-10);
}
assert.equal(external.history.length,2);
for(const [id,i,j] of [['embed',0,0],['qgain',0,0],['kgain',0,1],['wq',0,1],['wk',1,1],['wo',1,2]]){
 const eps=1e-5;
 const loss=(step)=>{
  const g=JSON.parse(before);g.nodes.find(n=>n.id===id).params.values[i][j]+=step*eps;
  return L.evaluate(g).steps.find(n=>n.id==='loss').values[0][0];
 };
 const fd=(loss(1)-loss(-1))/(2*eps),analytical=backward.gradients[id][i][j];
 assert.ok(Math.abs(fd-analytical)<1e-6,'finite difference '+id+' '+fd+' '+analytical);
}
console.log('INDUSTRIAL_GQA_TIED_PASS 45 compositional ops, 2 query heads/1 shared KV, per-head QK RMS+RoPE, true parameter alias, no future leaks, 7 parameter gradients, AdamW/checkpoint');
