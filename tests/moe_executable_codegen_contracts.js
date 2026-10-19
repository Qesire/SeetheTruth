'use strict';
// Exact MoE route, parameter-sharing and bounded AdamW contract across JS, standalone NumPy, Torch.
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),N=require('../python_codegen.js'),
      T=require('../torch_codegen.js'),G=require('../graph_autodiff.js'),
      Training=require('../train_state.js');
function run(source,output){
 const extra=output==='torch'?
   '\nprint(json.dumps({k:t.detach().tolist() for k,t in GraphRuntime().forward().items()}))\n':'';
 const result=execFileSync('python3',['-c',source+extra],{timeout:120000,encoding:'utf8',maxBuffer:6*1024*1024});
 return JSON.parse(result.trim().split('\n').at(-1));
}
function shape(x){return Array.isArray(x)?[x.length,...shape(x[0])]:[];}
function close(a,b,label,tol=1e-8){
 assert.deepEqual(shape(a),shape(b),label+' shape');
 const x=a.flat(Infinity),y=b.flat(Infinity);
 x.forEach((v,i)=>assert.ok(Number.isFinite(v)&&Math.abs(v-y[i])<=tol+Math.abs(y[i])*1e-10,label+'['+i+'] '+v+' != '+y[i]));
}
let count=0;
for(const name of ['moe_route','moe_expert_ffn','trainable_moe']){
 const graph=L.preset(name),before=JSON.stringify(graph);
 const outputs=L.evaluate(graph).steps;
 const numpy=N.compile(graph),torch=T.compile(graph);
 const py=run(numpy.source,'numpy'),pt=run(torch.source,'torch');
 assert.equal(numpy.nodes.length,graph.nodes.length);
 assert.equal(torch.nodes.length,graph.nodes.length);
 for(const st of outputs){
   close(py[st.id],st.values,name+'/'+st.id+'/NumPy');
   close(pt[st.id],st.values,name+'/'+st.id+'/Torch');
   count+=2;
 }
 assert.equal(JSON.stringify(graph),before,'codegen altered original graph');
 assert.match(numpy.source,/_expert_linear_bank/);
 assert.match(torch.source,/torch.einsum/);
 console.log('MoE real independent executable parity PASS '+name+' '+outputs.length+' nodes');
}
const graph=L.preset('trainable_moe'),compiled=T.compile(graph);
const original=L.evaluate(graph),grads=G.backward(graph,original);
const extra=[
 'g=GraphRuntime()',
 'g.forward()[g.loss_id].sum().backward()',
 'grad={k:p.grad.detach().tolist() for k,p in g.parameters.items()}',
 'history=g.train(steps=2,lr=.004,weight_decay=.005)',
 'after={k:p.detach().tolist() for k,p in g.parameters.items()}',
 'import tempfile,os',
 'with tempfile.TemporaryDirectory() as folder:',
 '    path=os.path.join(folder,"moe.pt")',
 '    g.save_checkpoint(path)',
 '    resumed=GraphRuntime()',
 '    resumed.load_checkpoint(path)',
 '    g.train(steps=1,lr=.004,weight_decay=.005)',
 '    resumed.train(steps=1,lr=.004,weight_decay=.005)',
 '    cont={k:p.detach().tolist() for k,p in g.parameters.items()}',
 '    clone={k:p.detach().tolist() for k,p in resumed.parameters.items()}',
 'print(json.dumps({"grad":grad,"after":after,"history":history,"cont":cont,"clone":clone}))'
].join('\n');
const actual=run(compiled.source+'\n'+extra,'numpy');
const js=Training.steps(graph,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
for(const id of ['w1','w2','w3']){
 close(actual.grad[id],grads.gradients[id],'MoE gradient '+id,3e-6);
 close(actual.after[id],js.graph.nodes.find(n=>n.id===id).params.values,'MoE optimizer '+id,5e-6);
 close(actual.cont[id],actual.clone[id],'MoE checkpoint resume '+id,1e-10);
}
assert.equal(actual.history.length,2);
assert.ok(actual.history.every(h=>Number.isFinite(h.beforeLoss)&&Number.isFinite(h.afterLoss)));
assert.deepEqual(compiled.trainable.sort(),['w1','w2','w3']);
console.log('MOE_EXECUTABLE_CODEGEN_PASS '+count+' source tensor intermediates, 3 shared parameter banks, real Autograd, 2 AdamW steps, exact checkpoint resume');
