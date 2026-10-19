'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Torch=require('../torch_codegen.js');
const single=['row_mean','row_sum','row_var','row_logsumexp','log_softmax','l2norm','qk_norm',
  'logit_softcap','sliding_window_mask','temperature','residual_scale','topk','argmax','detach_demo'];
const node=(id,kind,params={},x=30,y=40)=>({id,kind,params,x,y});
const link=(from,to,port)=>({from,to,port});
function build(kind){
 const square=['sliding_window_mask','masked_softmax'].includes(kind),rows=square?4:3,width=4;
 const nodes=[node('x','input',{rows,width,seed:3})],edges=[],params={
   l2norm:{epsilon:1e-6},qk_norm:{epsilon:1e-5},logit_softcap:{cap:3},
   sliding_window_mask:{window:2},temperature:{temperature:.7},
   residual_scale:{factor:.35},topk:{k:2}
 };
 if(kind==='low_rank_project'){
   nodes.push(node('down','input',{rows:4,width:2,seed:4}),node('up','input',{rows:2,width:4,seed:5}));
   for(const p of ['x','down','up'])edges.push(link(p,'y',p));
 }else if(kind==='row_bias_add'){
   nodes.push(node('biasSource','input',{rows:1,width:4,seed:4}));
   edges.push(link('x','y','scores'),link('biasSource','y','bias'));
 }else if(kind==='weighted_sum'){
   nodes.push(node('values','input',{rows:4,width:3,seed:4}));
   edges.push(link('x','y','weights'),link('values','y','values'));
 }else if(kind==='mae_loss'){
   nodes.push(node('target','input',{rows:3,width:4,seed:4}));
   edges.push(link('x','y','prediction'),link('target','y','target'));
 }else if(kind==='masked_softmax'){
   nodes.push(node('mask','sliding_window_mask',{window:2}));
   edges.push(link('x','mask','x'),link('x','y','logits'),link('mask','y','mask'));
 }else edges.push(link('x','y','x'));
 nodes.push(node('y',kind,params[kind]||{},540,160));
 return L.validateGraph({schema:L.SCHEMA,title:'Numerical contract '+kind,nodes,edges});
}

function execute(script,extra='') {
 const result=execFileSync('python3',['-c',script.source+'\n'+extra],{timeout:45000,encoding:'utf8',maxBuffer:5*1024*1024});
 return JSON.parse(result.trim().split('\n').at(-1));
}
const shape=x=>Array.isArray(x)?[x.length,...shape(x[0])]:[];
const close=(label,a,b,tol=3e-8)=>{
 assert.deepEqual(shape(a),shape(b),label+' shape');
 const actual=a.flat(Infinity),expected=b.flat(Infinity);
 for(let i=0;i<actual.length;i++)assert.ok(Number.isFinite(actual[i])&&Math.abs(actual[i]-expected[i])<tol+Math.abs(expected[i])*1e-10,label+' element '+i);
};
const all=[...single,'low_rank_project','row_bias_add','weighted_sum','mae_loss','masked_softmax'];
let steps=0;
for(const kind of all){
 const graph=build(kind),compiled=Torch.compile(graph);
 const output=execute(compiled,'print(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))');
 const oracle=L.evaluate(graph);
 assert.equal(compiled.nodes.length,graph.nodes.length);
 for(const item of oracle.steps){assert.ok(Object.hasOwn(output,item.id));close(kind+'/'+item.id,output[item.id],item.values);steps++;}
 assert.ok(Torch.supportedKinds.includes(kind));
 console.log('Torch extended graph PASS '+kind);
}
const maeloss=L.preset('trainable_swiglu');
maeloss.nodes.find(n=>n.id==='loss').kind='mae_loss';
const model=L.validateGraph(maeloss);
const compiled=Torch.compile(model);
assert.equal(compiled.lossId,'loss');
const py=execute(compiled,[
'r=GraphRuntime()',
'r.forward()[r.loss_id].sum().backward()',
'grad={k:v.grad.detach().tolist() for k,v in r.parameters.items()}',
'history=r.train(steps=2,lr=.004,weight_decay=.005)',
'after={k:v.detach().tolist() for k,v in r.parameters.items()}',
'import tempfile,os',
'with tempfile.TemporaryDirectory() as t:',
'    path=os.path.join(t,"checkpoint.pt")',
'    r.save_checkpoint(path)',
'    reloaded=GraphRuntime()',
'    reloaded.load_checkpoint(path)',
'    r.train(steps=1,lr=.004,weight_decay=.005)',
'    reloaded.train(steps=1,lr=.004,weight_decay=.005)',
'    resumed={k:v.detach().tolist() for k,v in reloaded.parameters.items()}',
'    continuous={k:v.detach().tolist() for k,v in r.parameters.items()}',
'print(json.dumps({"grad":grad,"after":after,"history":history,"resumed":resumed,"continuous":continuous}))'
].join('\n'));
const Grad=require('../graph_autodiff.js'),T=require('../train_state.js');
const jsGrad=Grad.backward(model,L.evaluate(model));
const jsTrain=T.steps(model,{targetId:'loss',count:2,lr:.004,weightDecay:.005});
for(const [id,gradient] of Object.entries(py.grad)){
 close('mae/grad/'+id,gradient,jsGrad.gradients[id],4e-7);
 close('mae/2steps/'+id,py.after[id],jsTrain.graph.nodes.find(x=>x.id===id).params.values,3e-6);
 close('mae/checkpoint/'+id,py.resumed[id],py.continuous[id],1e-10);
}
assert.equal(py.history.length,2);
const bad=execute(Torch.compile(build('masked_softmax')),[
'try:',
'    _masked_softmax(torch.ones((2,4),dtype=DTYPE),torch.zeros((2,4),dtype=DTYPE))',
'except ValueError as e:',
'    print(json.dumps({"error":str(e)}))'
].join('\n'));
assert.match(bad.error,/fully masked/);
console.log('TORCH_EXTENDED_BLOCKS_PASS '+all.length+' operators, '+steps+' intermediary tensors, MAE full autograd+AdamW+resume');
