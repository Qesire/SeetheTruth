'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Py=require('../python_codegen.js');
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
function run(compiled,extra=''){
 const output=execFileSync('python3',['-c',compiled.source+'\n'+extra],{timeout:25000,encoding:'utf8',maxBuffer:4*1024*1024});
 return JSON.parse(output.trim().split('\n').at(-1));
}
const expected=[...single,'low_rank_project','row_bias_add','weighted_sum','mae_loss','masked_softmax'];
let checked=0;
for(const kind of expected){
 const graph=build(kind),compiled=Py.compile(graph),got=run(compiled),oracle=L.evaluate(graph);
 for(const item of oracle.steps){
  const a=item.values.flat(Infinity),v=got[item.id].flat(Infinity);
  assert.equal(a.length,v.length,kind+'/'+item.id+' tensor shape');
  a.forEach((x,j)=>assert.ok(Number.isFinite(v[j])&&Math.abs(x-v[j])<2e-9+Math.abs(x)*1e-11,
    kind+'/'+item.id+'['+j+'] JS '+x+' != Python '+v[j]));
  checked++;
 }
 assert.equal(compiled.nodes.length,graph.nodes.length);
 console.log('NumPy standalone extended contract PASS '+kind+' ['+oracle.steps.length+' nodes]');
}
const graph=build('masked_softmax'),compiled=Py.compile(graph);
const invalid=[
 'bad=run_graph({"x":[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]})',
 'bad["mask"][:]=0',
 'try:',
 '    _masked_softmax(bad["x"],bad["mask"])',
 'except ValueError as err:',
 '    print(json.dumps({"rejected":str(err)}))'
].join('\n');
const out=run(compiled,invalid);
assert.match(out.rejected,/fully masked/);
assert.ok(expected.every(k=>Py.supportedKinds.includes(k)),'public compiler whitelist must include all tested operators');
console.log('PYTHON_EXTENDED_BLOCKS_PASS '+expected.length+' independent operators, '+checked+' intermediate JS↔NumPy comparisons');
