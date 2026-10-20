'use strict';
/* End-to-end oracle for source-aligned W[state;initial_emb] re-injection.
 * JS forward/VJP/AdamW, independent Python NumPy/Torch and checkpoint
 * are compared against actual tensors, not string presence.
 */
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Injection=require('../loop_input_injection.js'),
 Loop=require('../loop_transformer.js'),N=require('../python_codegen.js'),
 Torch=require('../torch_codegen.js'),Grad=require('../graph_autodiff.js'),
 Train=require('../train_state.js'),Concept=require('../concept_flow.js');
const execute=(src,extra='')=>JSON.parse(execFileSync('python3',['-c',src+'\n'+extra],{
 encoding:'utf8',timeout:120000,maxBuffer:8*1024*1024}).trim().split('\n').at(-1));
const flat=x=>x.flat(Infinity);
function same(label,a,b,tol=9e-8){
 const x=flat(a),y=flat(b);assert.equal(x.length,y.length,label+" shape/element mismatch");
 x.forEach((v,i)=>assert.ok(Number.isFinite(v)&&Math.abs(v-y[i])<=tol+Math.abs(y[i])*1e-10,label+'['+i+']: '+v+' != '+y[i]));
}
const g=Injection.build(),snap=JSON.stringify(g),how=Injection.inspect(g),
 baseline=Loop.build({loops:2});
assert.equal(how.loops,2);
assert.equal(how.parameterId,'Winject');
assert.deepEqual(how.shape,[8,4]);
assert.match(how.source,/xllm\/models\/looped.py/);
assert.equal(g.nodes.length,baseline.nodes.length+5);
assert.equal(g.nodes.filter(n=>n.kind==='parameter').length,
 baseline.nodes.filter(n=>n.kind==='parameter').length+1);
assert.equal(Concept.evaluate(g).steps.length,g.nodes.length);
const observed=L.evaluate(g);
const outputs=Object.fromEntries(observed.steps.map(v=>[v.id,v.values]));
const np=N.compile(g),torch=Torch.compile(g);
assert.equal(torch.lossId,'loss');
assert.ok(torch.trainable.includes('Winject'));
assert.equal(torch.trainable.length,baseline.nodes.filter(n=>n.kind==='parameter').length+1);
const outN=execute(np.source),outT=execute(torch.source,
 'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
for(const step of observed.steps){
 same('NumPy/'+step.id,outN[step.id],step.values);
 same('Torch/'+step.id,outT[step.id],step.values);
}
assert.equal(JSON.stringify(g),snap,'compiler mutated editable graph');
// Strong function-preserving initialization: [I;0] MUST be exactly
// equivalent to the original no-injection recurrent graph.
const control=Injection.build({inputGain:0}),truth=L.evaluate(baseline);
const reference=Object.fromEntries(truth.steps.map(v=>[v.id,v.values]));
const identity=Object.fromEntries(L.evaluate(control).steps.map(v=>[v.id,v.values]));
for(const id of ['r0_state','r1_state','r0_ce','r1_ce','loss']){
 same('zero-gain function-preserving '+id,identity[id],reference[id],3e-12);
}
assert.notDeepEqual(outputs.r1_state,reference.r1_state,'nonzero input path is actually executed');
// Both loops genuinely use the unique trained injection weight.
// Modifying one shared element must change both rounds.
const mutated=JSON.parse(snap);
mutated.nodes.find(n=>n.id==='Winject').params.values[4][0]+=.2;
const after=Object.fromEntries(L.evaluate(mutated).steps.map(v=>[v.id,v.values]));
assert.notDeepEqual(after.r0_state,outputs.r0_state);
assert.notDeepEqual(after.r1_state,outputs.r1_state);
// Causal: changing future context token does not change earlier positions.
const token=JSON.parse(JSON.stringify(outputs.tokens));
token[2][0]=(token[2][0]+1)%8;
const j=JSON.stringify(JSON.stringify(token));
const changedN=execute(np.source,
 'o=run_graph({"tokens":json.loads('+j+')})\nprint(json.dumps({k:v.tolist() for k,v in o.items()}))');
const changedT=execute(torch.source,
 'o=GraphRuntime({"tokens":json.loads('+j+')}).forward()\n'+
 'print(json.dumps({k:v.detach().tolist() for k,v in o.items()}))');
for(const id of ['r0_state','r1_state','r0_logits','r1_logits']){
 same('no future leak '+id,changedN[id].slice(0,2),outputs[id].slice(0,2),1e-12);
 same('override NumPy/Torch '+id,changedN[id],changedT[id]);
}
assert.notDeepEqual(changedN.r1_state[2],outputs.r1_state[2]);
// Adjoint consistency and persistent optimizer identity across steps.
const grad=Grad.backward(g,observed);
const js=Train.steps(g,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
const trained=execute(torch.source,[
'r=GraphRuntime()',
'r.forward()[r.loss_id].sum().backward()',
'grads={k:p.grad.detach().tolist() for k,p in r.parameters.items()}',
'history=r.train(steps=2,lr=.004,weight_decay=.005)',
'updated={k:p.detach().tolist() for k,p in r.parameters.items()}',
'import tempfile,os',
'with tempfile.TemporaryDirectory() as folder:',
'  filename=os.path.join(folder,"injection.pt")',
'  r.save_checkpoint(filename)',
'  clone=GraphRuntime()',
'  clone.load_checkpoint(filename)',
'  r.train(steps=1,lr=.004,weight_decay=.005)',
'  clone.train(steps=1,lr=.004,weight_decay=.005)',
'  continued={k:p.detach().tolist() for k,p in r.parameters.items()}',
'  restored={k:p.detach().tolist() for k,p in clone.parameters.items()}',
'print(json.dumps({"grads":grads,"updated":updated,"history":history,"continued":continued,"restored":restored}))'
].join('\n'));
for(const id of torch.trainable){
 same('BPTT '+id,trained.grads[id],grad.gradients[id],8e-7);
 same('AdamW '+id,trained.updated[id],js.graph.nodes.find(n=>n.id===id).params.values,4e-6);
 same('checkpoint '+id,trained.restored[id],trained.continued[id],1e-10);
}
assert.equal(trained.history.length,2);
for(const [r,c] of [[0,0],[4,0],[5,2]]){
 const eps=1e-5,probe=direction=>{
  const copy=JSON.parse(snap);copy.nodes.find(n=>n.id==='Winject').params.values[r][c]+=direction*eps;
  return L.evaluate(copy).steps.find(s=>s.id==='loss').values[0][0];
 };
 const finite=(probe(1)-probe(-1))/(2*eps);
 assert.ok(Math.abs(finite-grad.gradients.Winject[r][c])<2e-6,
  'injection gradient finite difference '+r+'/'+c+': '+finite+' vs '+grad.gradients.Winject[r][c]);
}
assert.throws(()=>Injection.build({inputGain:Infinity}),/finite/);
assert.throws(()=>Injection.build({inputGain:1.2}),/\[-1,1\]/);
assert.throws(()=>Injection.inspect(baseline),/not the fixed-loop/);
const broken=JSON.parse(snap);broken.edges.find(e=>e.to==='r1_injection_concat'&&e.port==='b').from='r0_state';
assert.throws(()=>Injection.inspect(broken),/injection recurrence wiring/);
console.log('LOOP_INPUT_INJECTION_PASS 2× real shared W[state;original], 2 independent CPU backends, function preservation, BPTT, causal, AdamW, checkpoint and finite differences');
