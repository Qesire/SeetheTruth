'use strict';
const assert=require('node:assert/strict'),{spawnSync}=require('node:child_process');
const L=require('../learning_lab.js'),NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const Grad=require('../graph_autodiff.js'),Shape=require('../shape_index_rules.js');
const n=(id,kind,params={},x=40,y=45)=>({id,kind,params,x,y});
const mk=(kind,params,rows,width)=>L.validateGraph({schema:L.SCHEMA,title:kind,
 nodes:[n('x','input',{rows,width,seed:3}),n('y',kind,params,350,60)],
 edges:[{from:'x',to:'y',port:'x'}]});
const nd=L.preset('nd_attention_lesson'),cached=L.preset('nd_cached_gqa_decode');
const cases=[
 ['transpose2d',mk('transpose2d',{},2,3),'y'],
 ['nd_reshape',mk('nd_reshape',{b:1,h:2,t:3,d:4},6,4),'y'],
 ['nd_permute',nd,'perm'],['nd_flatten',nd,'flat'],
 ['nd_expand4d',nd,'expand'],['nd_broadcast_add',nd,'biased'],
 ['nd_reduce_sum',nd,'summed'],['nd_repeat_kv',cached,'shared_k']
];
assert.deepEqual(cases.map(x=>x[0]).sort(),Shape.kinds);
const payload=cases.map(([kind,g,target])=>{
 const value=L.evaluate(g).steps.find(s=>s.id===target).values;
 return {name:kind,target,expected:value,
  np:NP.compile(g,{targetId:target}).source,
  pt:Torch.compile(g,{targetId:target}).source};
});
const graph=L.preset('nd_trainable_gqa'),trace=L.evaluate(graph),back=Grad.backward(graph,trace);
const spot=(g,id)=>{
 const shift=1e-5,loss=offset=>{
  const copy=JSON.parse(JSON.stringify(g)),node=copy.nodes.find(n=>n.id===id);
  node.params.values[0][0][0][0]+=offset;
  return L.evaluate(copy).steps.find(s=>s.id==='loss').values[0][0];
 };
 return (loss(shift)-loss(-shift))/(2*shift);
};
for(const id of ['query','key','value']){
 const g=back.gradients[id][0][0][0][0],fd=spot(graph,id);
 assert.ok(Math.abs(g-fd)<4e-5,id+' VJP finite difference');
}
payload.push({name:'gradient_gqa',np:null,pt:Torch.compile(graph).source,target:null,
 expectedGrad:Object.fromEntries(['query','key','value'].map(id=>[id,back.gradients[id]]))});
const code=String.raw`import json,sys
items=json.loads(sys.stdin.read())
results=[]
for i in items:
    numpy=None
    if i["np"] is not None:
        env={"__name__":"shape_parity_module"}
        exec(i["np"],env)
        numpy=env["run_graph"]()[i["target"]].tolist()
    env={"__name__":"shape_parity_module"}
    exec(i["pt"],env)
    r=env["GraphRuntime"]()
    o=r.forward()
    if i["target"] is None:
        o[r.loss_id].sum().backward()
        torch={k:v.grad.detach().tolist() for k,v in r.parameters.items() if v.grad is not None}
    else:
        torch=o[i["target"]].detach().tolist()
    results.append({"name":i["name"],"numpy":numpy,"torch":torch})
print(json.dumps(results,allow_nan=False))
`;
const run=spawnSync('python3',['-c',code],{input:JSON.stringify(payload),encoding:'utf8',timeout:230000,maxBuffer:20*1024*1024});
if(run.status!==0)throw Error('shape executable parity failed: '+(run.stderr||run.stdout||run.error).slice(-5000));
const actual=JSON.parse(run.stdout.trim().split('\n').at(-1));
const close=(label,a,b,tol=4e-8)=>{
 const shape=x=>Array.isArray(x)?[x.length,...shape(x[0])]:[];
 assert.deepEqual(shape(a),shape(b),label+' shape');
 const xs=a.flat(Infinity),ys=b.flat(Infinity);
 for(let i=0;i<xs.length;i++)assert.ok(Number.isFinite(ys[i])&&Math.abs(xs[i]-ys[i])<=tol+Math.abs(xs[i])*1e-9,label+'['+i+'] '+xs[i]+' != '+ys[i]);
};
for(const item of payload){
 const out=actual.find(v=>v.name===item.name);
 assert.ok(out,item.name+' output missing');
 if(item.expected){
  close(item.name+' NumPy',item.expected,out.numpy);
  close(item.name+' Torch',item.expected,out.torch);
 }else for(const [id,val] of Object.entries(item.expectedGrad))close('GQA '+id+' gradient',val,out.torch[id],8e-6);
}
console.log('SHAPE_INDEX_RUNTIME_PARITY_PASS 8 executable forward operations + GQA parameter VJP/finite-difference');
