'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),NP=require('../python_codegen.js'),
  Torch=require('../torch_codegen.js'),Grad=require('../graph_autodiff.js'),
  Training=require('../train_state.js');
function invoke(script,extra=''){
 const out=execFileSync('python3',['-c',script+'\n'+extra],{encoding:'utf8',timeout:120000,maxBuffer:6*1024*1024});
 return JSON.parse(out.trim().split('\n').at(-1));
}
function approx(label,a,b,eps=1.2e-8){
 const xs=a.flat(Infinity),ys=b.flat(Infinity);
 assert.equal(xs.length,ys.length,label+' count');
 xs.forEach((x,i)=>assert.ok(Number.isFinite(x)&&Math.abs(x-ys[i])<=eps+Math.abs(ys[i])*2e-10,
  label+'['+i+'] '+x+' versus '+ys[i]));
}
function checkGraph(name){
 const g=L.preset(name),initial=JSON.stringify(g),oracle=L.evaluate(g).steps;
 const numpy=NP.compile(g),torch=Torch.compile(g);
 const pn=invoke(numpy.source);
 const pt=invoke(torch.source,'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
 for(const v of oracle){
  approx(name+'/NumPy/'+v.id,pn[v.id],v.values);
  approx(name+'/Torch/'+v.id,pt[v.id],v.values,2e-8);
 }
 assert.equal(JSON.stringify(g),initial,'exporters must not mutate editable graph');
 console.log('TOKEN_CE_FORWARD_PASS '+name+' '+oracle.length+' independent intermediates per backend');
 return {g,numpy,torch};
}
checkGraph('embedding');checkGraph('classifier');
const {g,numpy,torch}=checkGraph('token_lm_trainable');
assert.equal(torch.lossId,'loss');
assert.deepEqual([...torch.trainable].sort(),['embedding_table','lm_head']);
const oracle=L.evaluate(g);
const seq=oracle.steps.find(n=>n.id==='token_ids').values.map(x=>x[0]);
assert.deepEqual(oracle.steps.find(n=>n.id==='input_tokens').values.map(x=>x[0]),seq.slice(0,3),'shifted LM inputs');
assert.deepEqual(oracle.steps.find(n=>n.id==='labels').values.map(x=>x[0]),seq.slice(1,4),'labels must be exact next tokens');
const gradient=Grad.backward(g,oracle);
const training=Training.steps(g,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
const result=invoke(torch.source,[
'r=GraphRuntime()',
'r.forward()[r.loss_id].sum().backward()',
'grads={k:v.grad.detach().tolist() for k,v in r.parameters.items()}',
'history=r.train(steps=2,lr=.004,weight_decay=.005)',
'updated={k:v.detach().tolist() for k,v in r.parameters.items()}',
'import tempfile,os',
'with tempfile.TemporaryDirectory() as folder:',
'  path=os.path.join(folder,"next-token.pt")',
'  r.save_checkpoint(path)',
'  restored=GraphRuntime()',
'  restored.load_checkpoint(path)',
'  r.train(steps=1,lr=.004,weight_decay=.005)',
'  restored.train(steps=1,lr=.004,weight_decay=.005)',
'  resumed={k:v.detach().tolist() for k,v in restored.parameters.items()}',
'  continuous={k:v.detach().tolist() for k,v in r.parameters.items()}',
'print(json.dumps({"grads":grads,"history":history,"updated":updated,"resumed":resumed,"continuous":continuous}))'
].join('\n'));
for(const id of ['embedding_table','lm_head']){
 approx(id+'/VJP',result.grads[id],gradient.gradients[id],2e-7);
 approx(id+'/AdamW',result.updated[id],training.graph.nodes.find(n=>n.id===id).params.values,6e-7);
 approx(id+'/checkpoint',result.resumed[id],result.continuous[id],1e-11);
}
assert.equal(result.history.length,2);
const grad=gradient.gradients.embedding_table;
const selected=new Set(oracle.steps.find(n=>n.id==='token_ids').values.map(x=>x[0]));
assert.ok(grad.some((row,i)=>selected.has(i)&&row.some(v=>Math.abs(v)>1e-9)),'used embedding receives CE gradient');
assert.ok(grad.every((row,i)=>selected.has(i)||row.every(v=>v===0)),'unselected embedding rows have zero gradient');
const changed=JSON.parse(JSON.stringify(oracle.steps.find(n=>n.id==='token_ids').values));
changed[0][0]=(changed[0][0]+1)%8;
const overrides=JSON.stringify(JSON.stringify(changed));
const changedN=invoke(numpy.source,
 'const_tokens=json.loads('+overrides+')\n'+
 'r=run_graph({"token_ids":const_tokens})\n'+
 'print(json.dumps({k:v.tolist() for k,v in r.items()}))');
const changedT=invoke(torch.source,
 'r=GraphRuntime({"token_ids":json.loads('+overrides+')}).forward()\n'+
 'print(json.dumps({k:v.detach().tolist() for k,v in r.items()}))');
approx('token override NumPy',changedN.token_ids,changed);
approx('token override Torch',changedT.token_ids,changed);
approx('override logits consistent',changedN.logits,changedT.logits);
assert.notDeepEqual(changedN.lookup,oracle.steps.find(n=>n.id==='lookup').values);
assert.notDeepEqual(changedN.loss,oracle.steps.find(n=>n.id==='loss').values);
const bad=invoke(torch.source,[
'try:',
'  _cross_entropy(torch.zeros((2,8),dtype=DTYPE),torch.tensor([[0.5],[2.]],dtype=DTYPE))',
'except ValueError as e:',
'  print(json.dumps({"error":str(e)}))'
].join('\n'));
assert.match(bad.error,/integral labels/);
// Force duplicate token IDs in the actual editable graph, testing scatter-add gradient.
const dup=JSON.parse(JSON.stringify(g));
dup.nodes.find(n=>n.id==='token_ids').params.vocab=2;
const dg=L.validateGraph(dup),trace=L.evaluate(dg),gradJS=Grad.backward(dg,trace);
const compiled=Torch.compile(dg);
const check=invoke(compiled.source,
 'g=GraphRuntime()\ng.forward()[g.loss_id].sum().backward()\n'+
 'print(json.dumps({"gradient":g.parameters["embedding_table"].grad.detach().tolist(),"ids":g.inputs["token_ids"].tolist()}))');
approx('repeated ID scatter-add',check.gradient,gradJS.gradients.embedding_table,2e-7);
assert.ok(new Set(check.ids.map(x=>x[0])).size<check.ids.length,'duplicate IDs must be exercised');
const epsilon=1e-5,base=dg.nodes.find(n=>n.id==='embedding_table').params.values;
for(const [row,col] of [[0,0],[1,2]]){
 const hi=JSON.parse(JSON.stringify(dg)),lo=JSON.parse(JSON.stringify(dg));
 hi.nodes.find(n=>n.id==='embedding_table').params.values[row][col]+=epsilon;
 lo.nodes.find(n=>n.id==='embedding_table').params.values[row][col]-=epsilon;
 const estimate=(L.evaluate(hi).steps.at(-1).values[0][0]-L.evaluate(lo).steps.at(-1).values[0][0])/(2*epsilon);
 assert.ok(Math.abs(estimate-gradJS.gradients.embedding_table[row][col])<1e-6,'embedding finite-difference gradient '+row+','+col);
}
console.log('TOKEN_CE_TRAINING_PASS JS↔NumPy↔Torch, 2 parameters, CE derivative, repeated row scatter, finite differences, 2 AdamW steps, exact checkpoint resume, input overrides');
