'use strict';
/* Real CPU proof of Ouro-inspired depth-supervised shared-weight recurrence.
 * No claims of learned halting, truncated BPTT, actual Ouro weights or KV-cache.
 */
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Loop=require('../loop_transformer.js'),
  Obj=require('../training_objective.js'),G=require('../graph_autodiff.js'),
  Train=require('../train_state.js'),Concept=require('../concept_flow.js'),
  NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const py=(source,extra='')=>{
 const result=execFileSync('python3',['-c',source+'\n'+extra],{
   encoding:'utf8',timeout:120000,maxBuffer:9*1024*1024});
 return JSON.parse(result.trim().split('\n').at(-1));
};
function close(name,a,b,epsilon=5e-8){
 const flat=x=>x.flat(Infinity),x=flat(a),y=flat(b);
 assert.equal(x.length,y.length,name+' shape/count');
 x.forEach((v,i)=>assert.ok(Number.isFinite(v)&&Math.abs(v-y[i])<=epsilon+Math.abs(y[i])*1e-10,
   name+' ['+i+']: '+v+' vs '+y[i]));
}
const names=["embedding","Wq","Wk","Wv","Wo","Wup","Wgate","Wdown"].sort();
const cases=[["loop_transformer_1",1,35],["loop_transformer_2",2,57],["loop_transformer_3",3,79]];
let checked=0;
for(const [preset,depth,nodeCount] of cases){
 const graph=L.preset(preset),saved=JSON.stringify(graph),rec=Loop.inspect(graph),
       observed=L.evaluate(graph),by=new Map(observed.steps.map(s=>[s.id,s]));
 assert.equal(rec.loops,depth);assert.ok(rec.linked,'each next state must follow prior loop');
 assert.equal(graph.nodes.length,nodeCount,'entire attention+SwiGLU+CE must be visible');
 assert.equal(Concept.evaluate(graph).steps.length,nodeCount);
 assert.equal(graph.nodes.filter(n=>n.kind==="parameter").length,8,'no parameter cloning');
 assert.deepEqual([...rec.parameterIds].sort(),names);
 const proof=Obj.analyze(graph);
 assert.equal(proof.preferred,"loss",'multi-step objective is the actual training target');
 assert.ok(proof.isLoss("loss"));
 assert.equal(proof.choices.filter(s=>s.endsWith("_ce")).length,depth);
 const alpha=Array.from({length:depth},(_,i)=>(i+1)/(depth*(depth+1)/2));
 const objective=alpha.reduce((v,a,i)=>v+a*by.get("r"+i+"_ce").values[0][0],0);
 assert.ok(Math.abs(objective-by.get("loss").values[0][0])<1e-12,'weighted per-depth scalar total');
 const ids=by.get("tokens").values.map(row=>row[0]);
 assert.deepEqual(by.get("prefix").values.map(row=>row[0]),ids.slice(0,3));
 assert.deepEqual(by.get("labels").values.map(row=>row[0]),ids.slice(1,4));
 for(let i=0;i<depth;i++){
  const mask=by.get("r"+i+"_mask").values;
  assert.deepEqual(mask,[[1,0,0],[1,1,0],[1,1,1]]);
  const prob=by.get("r"+i+"_probs").values;
  for(let t=0;t<3;t++){
   assert.ok(Math.abs(prob[t].reduce((a,b)=>a+b,0)-1)<1e-12);
   for(let j=t+1;j<3;j++)assert.equal(prob[t][j],0,'future token leakage');
  }
 }
 const np=NP.compile(graph),torch=Torch.compile(graph);
 assert.equal(torch.lossId,"loss",'Torch default must optimize the aggregate objective');
 assert.deepEqual([...torch.trainable].sort(),names);
 const numeric=py(np.source),real=py(torch.source,
  'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
 for(const step of observed.steps){
  close(preset+'/numpy/'+step.id,numeric[step.id],step.values);
  close(preset+'/torch/'+step.id,real[step.id],step.values,2e-7);
  checked+=2;
 }
 // Cross-loop gradients must accumulate on the same trainable Parameter.
 const reverse=G.backward(graph,observed);
 assert.equal(reverse.targetId,"loss");
 const external=py(torch.source,[
  'g=GraphRuntime()',
  'g.forward()[g.loss_id].sum().backward()',
  'grad={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
  'history=g.train(steps=2,lr=.004,weight_decay=.005)',
  'updated={k:p.detach().tolist() for k,p in g.parameters.items()}',
  'import tempfile,os',
  'with tempfile.TemporaryDirectory() as d:',
  '  path=os.path.join(d,"rec-depth.pt")',
  '  g.save_checkpoint(path)',
  '  restored=GraphRuntime()',
  '  restored.load_checkpoint(path)',
  '  g.train(steps=1,lr=.004,weight_decay=.005)',
  '  restored.train(steps=1,lr=.004,weight_decay=.005)',
  '  continuous={k:p.detach().tolist() for k,p in g.parameters.items()}',
  '  replay={k:p.detach().tolist() for k,p in restored.parameters.items()}',
  'print(json.dumps({"grad":grad,"history":history,"updated":updated,"continuous":continuous,"replay":replay}))'
 ].join('\n'));
 const js=Train.steps(graph,{targetId:"loss",count:2,lr:.004,weightDecay:.005});
 assert.equal(external.history.length,2);
 for(const id of names){
  close(preset+'/BPTT/'+id,external.grad[id],reverse.gradients[id],8e-7);
  close(preset+'/AdamW/'+id,external.updated[id],js.graph.nodes.find(n=>n.id===id).params.values,8e-6);
  close(preset+'/resume/'+id,external.continuous[id],external.replay[id],1e-10);
 }
 // Definite test of cross-step causality.
 const changed=JSON.parse(JSON.stringify(by.get("tokens").values));
 changed[2][0]=(changed[2][0]+1)%8;
 const literal=JSON.stringify(JSON.stringify(changed));
 const changedN=py(np.source,
  'v=run_graph({"tokens":json.loads('+literal+')})\nprint(json.dumps({k:x.tolist() for k,x in v.items()}))');
 const changedT=py(torch.source,
  'v=GraphRuntime({"tokens":json.loads('+literal+')}).forward()\nprint(json.dumps({k:x.detach().tolist() for k,x in v.items()}))');
 for(let i=0;i<depth;i++){
  close('causal unchanged prior logits step '+i,
        changedN["r"+i+"_logits"].slice(0,2),numeric["r"+i+"_logits"].slice(0,2),1e-12);
 }
 close('overridden source recomputes python',changedN.loss,changedT.loss);
 assert.notDeepEqual(changedN["r"+(depth-1)+"_state"][2],numeric["r"+(depth-1)+"_state"][2]);
 // Differentiable sum should agree with finite differences of shared weight.
 const eps=1e-5,id="Wq",i=0,j=1,objectiveAt=dir=>{
  const mutant=JSON.parse(saved);
  mutant.nodes.find(n=>n.id===id).params.values[i][j]+=dir*eps;
  return L.evaluate(mutant).steps.find(s=>s.id==="loss").values[0][0];
 };
 const finite=(objectiveAt(1)-objectiveAt(-1))/(2*eps);
 assert.ok(Math.abs(finite-reverse.gradients[id][i][j])<2e-6,
  preset+' multidepth objective finite diff '+finite+' vs '+reverse.gradients[id][i][j]);
 assert.equal(JSON.stringify(graph),saved,'no hidden rewriting while compiling/testing');
 console.log('LOOP_DEPTH_SUPERVISION_PASS '+depth+' loops; '+nodeCount+' nodes; 8 shared parameters; JS NumPy Torch VJP/AdamW/replay');
}
let source=L.preset("loop_transformer_2");
let trained=Train.steps(source,{targetId:"loss",count:1,lr:.004,weightDecay:.005}).graph;
const snapshot=Object.fromEntries(trained.nodes.filter(n=>n.kind==="parameter").map(n=>[n.id,JSON.stringify(n.params)]));
let bigger=Loop.resize(trained,3).graph;
assert.equal(Loop.inspect(bigger).loops,3);
for(const n of bigger.nodes.filter(n=>n.kind==="parameter"))assert.equal(JSON.stringify(n.params),snapshot[n.id],
 'resize must preserve trained values and optimizer moments');
let smaller=Loop.resize(bigger,1).graph;
assert.equal(Loop.inspect(smaller).loops,1);
for(const n of smaller.nodes.filter(n=>n.kind==="parameter"))assert.equal(JSON.stringify(n.params),snapshot[n.id]);
assert.throws(()=>Loop.build({loops:4}),/1..3/);
assert.throws(()=>Loop.build({loops:2,lossWeights:[0,0]}),/weights/);
assert.throws(()=>Loop.build({loops:2,lossWeights:[1,-1]}),/weights/);
const illegally=JSON.parse(JSON.stringify(trained));
illegally.edges.find(e=>e.to==="r1_attn_norm").from="tokens";
assert.throws(()=>Loop.resize(illegally,3),/custom rewiring|not a connected/);
const unrelated=L.preset("trainable_swiglu");
assert.throws(()=>Loop.resize(unrelated,3),/not a connected/);
const arbitrary={
 ...L.preset("loop_transformer_1"),
 nodes:L.preset("loop_transformer_1").nodes.concat([
  {id:"extra",kind:"input",x:200,y:100,params:{rows:2,width:2,seed:1}}])
};
assert.throws(()=>Loop.resize(arbitrary,2),/custom rewiring/);
const truncated=JSON.parse(JSON.stringify(source));
truncated.edges.push({from:"r1_state",to:"r0_attn_norm",port:"x"});
assert.throws(()=>L.order(L.validateGraph(truncated)),/循环计算依赖|重复|端口/);
console.log('LOOP_DEPTH_SUPERVISION_CODEGEN_PASS '+checked+' real independent tensor checks and exact trained-parameter-preserving depth changes');
