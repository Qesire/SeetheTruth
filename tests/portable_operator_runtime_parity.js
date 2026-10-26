'use strict';
/* Execute emitted standalone NumPy/PyTorch, not JS imitations of their math. */
const assert=require('node:assert/strict'),{spawnSync}=require('node:child_process');
const L=require('../learning_lab.js'),NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const G=require('../graph_autodiff.js'),Rules=require('../portable_operator_rules.js');
const N=(id,kind,params={},x=50,y=50)=>({id,kind,params,x,y});
const E=(from,to,port)=>({from,to,port});
const special4=new Set(['nd_batched_matmul','nd_broadcast_add','nd_broadcast_mul','nd_scale','nd_cache_read']);
function make(kind){
 const inputs=[],edges=[],params={
  scale:{factor:0.25},residual_scale:{factor:-0.4},bias:{bias:0.14},
  temperature:{temperature:0.8},slice_cols:{start:1,width:2},
  slice_rows:{start:0,length:1},diffusion_ddim_step:{alphaT:0.36,alphaPrevious:0.64}
 };
 const add=(id,rows,width,seed)=>inputs.push(N(id,'input',{rows,width,seed},50,inputs.length*120+20));
 const link=(from,port)=>edges.push(E(from,'y',port));
 if(['matmul','weighted_sum'].includes(kind)){
  add('a',2,4,2);add('b',4,3,3);
  link('a',kind==='weighted_sum'?'weights':'a');
  link('b',kind==='weighted_sum'?'values':'b');
 }else if(kind==='low_rank_project'){
  add('x',2,4,1);add('down',4,2,2);add('up',2,3,3);
  for(const port of ['x','down','up'])link(port,port);
 }else if(kind==='row_bias_add'){
  add('scores',2,4,1);add('bias',1,4,2);link('scores','scores');link('bias','bias');
 }else if(kind==='diffusion_ddim_step'){
  add('x',2,4,1);add('epsilon',2,4,2);link('x','x');link('epsilon','epsilon');
 }else if(['add','mul','subtract','divide','nd_broadcast_add'].includes(kind)){
  add('a',2,4,1);add('b',2,4,2);link('a','a');link('b','b');
 }else{add('x',2,4,1);link('x','x');}
 return L.validateGraph({schema:L.SCHEMA,title:'Portable '+kind,
  nodes:[...inputs,N('y',kind,params[kind]||{},550,100)],edges});
}
const cases=[];
for(const kind of Rules.kinds){
 if(special4.has(kind))continue;
 const graph=make(kind);
 cases.push({name:kind,graph,target:'y'});
}
const cached=L.preset('nd_cached_gqa_decode'),attention=L.preset('nd_attention_lesson');
cases.push({name:'nd_batched_matmul',graph:attention,target:'batch_scores'});
cases.push({name:'nd_broadcast_add',graph:attention,target:'biased'});
cases.push({name:'nd_broadcast_mul',graph:require('./fixtures/nd_broadcast_reduction_graphs.js').create('nd_token_reduce_head_broadcast'),target:'gated'});
cases.push({name:'nd_scale',graph:cached,target:'scaled'});
cases.push({name:'nd_cache_read',graph:cached,target:'k_prefix'});
assert.deepEqual([...new Set(cases.map(x=>x.name))].sort(),Rules.kinds);
const payload=cases.map(item=>{
 const trace=L.evaluate(item.graph).steps.find(n=>n.id===item.target);
 assert.ok(trace,item.name+' expected target missing');
 return {name:item.name,target:item.target,numpy:NP.compile(item.graph).source,
  torch:Torch.compile(item.graph).source,expected:trace.values};
});
const trainable=L.preset('trainable_core'),g=G.backward(trainable,L.evaluate(trainable));
payload.push({name:'trainable_core_gradient',target:null,numpy:null,
 torch:Torch.compile(trainable).source,expectedGrad:g.gradients.shared_weight});
const runner=String.raw`import sys,json
items=json.loads(sys.stdin.read())
results=[]
for item in items:
    if item["numpy"] is not None:
        env={"__name__":"portable_execution"}
        exec(item["numpy"],env)
        nv=env["run_graph"]()[item["target"]].tolist()
    else:
        nv=None
    env={"__name__":"portable_execution"}
    exec(item["torch"],env)
    runtime=env["GraphRuntime"]()
    outputs=runtime.forward()
    if item["target"] is None:
        outputs[runtime.loss_id].sum().backward()
        tv=runtime.parameters["shared_weight"].grad.detach().tolist()
    else:
        tv=outputs[item["target"]].detach().tolist()
    results.append({"name":item["name"],"numpy":nv,"torch":tv})
print(json.dumps(results,allow_nan=False))
`;
const done=spawnSync('python3',['-c',runner],{input:JSON.stringify(payload),encoding:'utf8',
 timeout:220000,maxBuffer:12*1024*1024});
if(done.status!==0)throw Error('Portable runtime parity failure: '+(done.stderr||done.stdout).slice(-5000));
const output=JSON.parse(done.stdout.trim().split('\n').at(-1));
function close(label,expected,actual,eps=4e-8){
 const flat=a=>a.flat(Infinity);
 const e=flat(expected),v=flat(actual);
 assert.equal(e.length,v.length,label+' element count');
 e.forEach((x,i)=>assert.ok(Number.isFinite(v[i])&&Math.abs(x-v[i])<=eps+Math.abs(x)*1e-9,
  label+'['+i+'] '+x+' vs '+v[i]));
}
for(const item of payload){
 const result=output.find(o=>o.name===item.name);
 assert.ok(result,item.name+' result missing');
 if(item.expected)close(item.name+' NumPy',item.expected,result.numpy);
 close(item.name+' PyTorch',item.expected||item.expectedGrad,result.torch,item.expectedGrad?8e-6:4e-8);
}
console.log('PORTABLE_OPERATOR_RUNTIME_PARITY_PASS '+cases.length+' operators + real Torch autograd gradient');
