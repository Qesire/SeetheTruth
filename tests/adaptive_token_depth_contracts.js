'use strict';
// MoR-inspired *semantic* token-depth halt; all recurrent candidate matmuls run.
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),MoR=require('../adaptive_token_depth.js'),
  Loop=require('../loop_transformer.js'),NP=require('../python_codegen.js'),
  Torch=require('../torch_codegen.js'),Grad=require('../graph_autodiff.js'),
  Train=require('../train_state.js'),Concept=require('../concept_flow.js');
const python=(source,extra='')=>{
 const result=execFileSync('python3',['-c',source+'\n'+extra],{
   encoding:'utf8',timeout:120000,maxBuffer:9*1024*1024});
 return JSON.parse(result.trim().split('\n').at(-1));
};
function close(label,a,b,eps=5e-8){
 const av=a.flat(Infinity),bv=b.flat(Infinity);
 assert.equal(av.length,bv.length,label+' count');
 av.forEach((x,i)=>assert.ok(Number.isFinite(x)&&Math.abs(x-bv[i])<eps+Math.abs(bv[i])*1e-10,
  label+'['+i+'] '+x+' vs '+bv[i]));
}
const graph=L.preset('mor_token_depth_budget'),before=JSON.stringify(graph);
const info=MoR.inspect(graph),regular=Loop.build({loops:2});
assert.equal(info.steps,2);
assert.equal(info.semantic,'dense_evaluation_hard_per_token_state_selection');
assert.equal(graph.nodes.length,regular.nodes.length+5);
assert.equal(graph.edges.length,regular.edges.length+8);
assert.deepEqual(info.parameterIds,regular.nodes.filter(n=>n.kind==='parameter').map(n=>n.id));
assert.equal(Concept.evaluate(graph).steps.length,graph.nodes.length);
const f=L.evaluate(graph),byId=Object.fromEntries(f.steps.map(s=>[s.id,s.values]));
const budget=byId.budget_ids.map(r=>r[0]);
assert.deepEqual(budget,[2,1,0],'fixed seed gives all three depth states');
const mask1=byId.r0_continue,mask2=byId.r1_continue;
assert.deepEqual(mask1.map(r=>r[0]),[1,1,0]);
assert.deepEqual(mask2.map(r=>r[0]),[1,0,0]);
for(let t=0;t<3;t++){
 const chosen0=budget[t]>=1?byId.r0_state[t]:byId.lookup[t],
       chosen1=budget[t]>=2?byId.r1_state[t]:byId.r0_adopted[t];
 close('first adoption row '+t,byId.r0_adopted[t],chosen0);
 close('second adoption row '+t,byId.r1_adopted[t],chosen1);
}
const np=NP.compile(graph),pt=Torch.compile(graph);
const outNP=python(np.source),outPT=python(pt.source,
 'print(json.dumps({k:t.detach().tolist() for k,t in GraphRuntime().forward().items()}))');
for(const s of f.steps){
 close('JS/NumPy/'+s.id,outNP[s.id],s.values);
 close('JS/Torch/'+s.id,outPT[s.id],s.values);
}
assert.deepEqual(pt.trainable.sort(),info.parameterIds.sort(),'depth mask adds no trainable weights');
assert.equal(JSON.stringify(graph),before,'compilers must not alter user graph');
const adj=Grad.backward(graph,f);
for(let r=0;r<2;r++){
 const id='r'+r+'_state',mask=byId['r'+r+'_continue'];
 const dv=adj.gradients[id];
 for(let t=0;t<3;t++)if(mask[t][0]===0)close('inactive candidate gradient '+id+'/'+t,dv[t],[0,0,0,0],1e-12);
}
assert.ok(adj.cutEdges.some(x=>x.nodeId==='r0_adopted'&&x.port==='mask'));
assert.ok(!adj.pathIds.includes('r0_continue')&&!adj.pathIds.includes('r1_continue'),'hard control graph must not enter differentiable VJP path');
// Per-token budgets can be changed via real runtime source overrides.
const forced=[[0],[0],[0]],maximal=[[2],[2],[2]],json=x=>JSON.stringify(JSON.stringify(x));
for(const [id,depths,expected] of [['none',forced,'lookup'],['all',maximal,'r1_state']]){
 const changed=python(np.source,
  'v=run_graph({"budget_ids":json.loads('+json(depths)+')})\n'+
  'print(json.dumps({k:x.tolist() for k,x in v.items()}))');
 close('override '+id,changed.r1_adopted,changed[expected],1e-12);
 const changeTorch=python(pt.source,
  'v=GraphRuntime({"budget_ids":json.loads('+json(depths)+')}).forward()\n'+
  'print(json.dumps({k:x.detach().tolist() for k,x in v.items()}))');
 close('torch override '+id,changeTorch.r1_adopted,changed.r1_adopted);
}
const trained=Train.steps(graph,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
const t=python(pt.source,[
 'g=GraphRuntime()',
 'g.forward()[g.loss_id].sum().backward()',
 'grads={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
 'history=g.train(steps=2,lr=.004,weight_decay=.005)',
 'updated={k:p.detach().tolist() for k,p in g.parameters.items()}',
 'import os,tempfile',
 'with tempfile.TemporaryDirectory() as tmp:',
 '  path=os.path.join(tmp,"adapt.pt")',
 '  g.save_checkpoint(path)',
 '  clone=GraphRuntime()',
 '  clone.load_checkpoint(path)',
 '  g.train(steps=1,lr=.004,weight_decay=.005)',
 '  clone.train(steps=1,lr=.004,weight_decay=.005)',
 '  resumed={k:p.detach().tolist() for k,p in clone.parameters.items()}',
 '  uninterrupted={k:p.detach().tolist() for k,p in g.parameters.items()}',
 'print(json.dumps({"grads":grads,"updated":updated,"history":history,"resumed":resumed,"uninterrupted":uninterrupted}))'
].join('\n'));
for(const id of info.parameterIds){
 close('shared param VJP '+id,t.grads[id],adj.gradients[id],6e-7);
 close('shared param AdamW '+id,t.updated[id],trained.graph.nodes.find(n=>n.id===id).params.values,6e-6);
 close('AdamW checkpoint '+id,t.resumed[id],t.uninterrupted[id],1e-10);
}
assert.equal(t.history.length,2);
// Finite difference on shared parameters.
const eps=1e-5,source=JSON.stringify(graph);
for(const [id,i,j] of [['Wq',0,0],['Wv',1,2],['embedding',2,0]]){
 const perturb=sign=>{const x=JSON.parse(source);x.nodes.find(n=>n.id===id).params.values[i][j]+=sign*eps;return L.evaluate(x).steps.at(-1).values[0][0];};
 const fd=(perturb(1)-perturb(-1))/(2*eps),exact=adj.gradients[id][i][j];
 assert.ok(Math.abs(fd-exact)<2e-6,'finite-difference '+id+': '+fd+' vs '+exact);
}
// Fail closed for malformed budget or mask; never silently clamp.
const bad=python(pt.source,[
 'try:',
 '  GraphRuntime({"budget_ids":[[4],[1],[0]]}).forward()',
 'except ValueError as e:',
 '  print(json.dumps({"error":str(e)}))'
].join('\n'));
assert.match(bad.error,/budgets/);
console.log('MOR_HARD_BUDGET_PASS graph nodes='+graph.nodes.length+' all intermediary NumPy/Torch parity, 3 per-token depths, mask VJP, shared 8 parameters, AdamW checkpoint, no fake sparse speed');
