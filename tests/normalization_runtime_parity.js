'use strict';
/* Independent Python NumPy/Torch execution against bounded JS forward values.
 * Gradients use central differences of the JS loss, not a duplicate Torch rule.
 */
const assert=require('node:assert/strict'),{spawnSync}=require('node:child_process');
const L=require('../learning_lab.js'),NP=require('../python_codegen.js'),PT=require('../torch_codegen.js'),S=require('../normalization_rules.js');
const node=(id,kind,params={},x=25,y=35)=>({id,kind,params,x,y});
const edge=(from,to,port)=>({from,to,port});
const parameter=()=>node('w','parameter',{
 values:[[0.24,-0.47,0.61],[-0.33,0.52,-0.28]],
 optimizer:{t:0,m:[[0,0,0],[0,0,0]],v:[[0,0,0],[0,0,0]]}
});
function makeForward(kind){
 if(kind==='nd_softmax_last'){
  return L.validateGraph({schema:L.SCHEMA,title:kind,
   nodes:[node('x','nd_input',{b:1,h:2,t:3,d:4,seed:3}),node('y',kind)],
   edges:[edge('x','y','x')]});
 }
 const nodes=[node('x','input',{rows:3,width:4,seed:2})],links=[],p={epsilon:1e-5};
 if(kind==='attention'){
  nodes.push(node('k','input',{rows:3,width:4,seed:5}),node('v','input',{rows:3,width:2,seed:4}));
  links.push(edge('x','y','q'),edge('k','y','k'),edge('v','y','v'));
 }else links.push(edge('x','y','x'));
 nodes.push(node('y',kind,p,550,90));
 return L.validateGraph({schema:L.SCHEMA,title:'Normalization '+kind,nodes,edges:links});
}
function makeGrad(kind){
 const nodes=[parameter()],links=[];
 if(kind==='attention'){
  nodes.push(node('k','input',{rows:2,width:3,seed:3}),node('v','input',{rows:2,width:2,seed:4}));
  links.push(edge('w','y','q'),edge('k','y','k'),edge('v','y','v'));
 }else links.push(edge('w','y','x'));
 nodes.push(node('y',kind,{epsilon:1e-5},480,40));
 const width=kind==='attention'?2:kind==='row_logsumexp'?1:3;
 nodes.push(node('teacher','input',{rows:2,width,seed:5},40,400),node('loss','mse',{},820,120));
 links.push(edge('y','loss','prediction'),edge('teacher','loss','target'));
 return L.validateGraph({schema:L.SCHEMA,title:'Gradient finite difference '+kind,nodes,edges:links});
}
function finiteDiff(raw,h=1e-5){
 const gradient=[];
 for(let row=0;row<2;row++){
  const values=[];
  for(let col=0;col<3;col++){
   const test=sign=>{
    const copy=JSON.parse(JSON.stringify(raw));
    copy.nodes.find(n=>n.id==='w').params.values[row][col]+=sign*h;
    const run=L.evaluate(copy);
    return run.steps.find(s=>s.id==='loss').values[0][0];
   };
   values.push((test(+1)-test(-1))/(2*h));
  }
  gradient.push(values);
 }
 return gradient;
}
const payload=[];
for(const kind of S.kinds){
 const graph=makeForward(kind),out=L.evaluate(graph).steps.find(s=>s.id==='y');
 payload.push({name:'fwd_'+kind,target:'y',numpy:NP.compile(graph).source,torch:PT.compile(graph).source,expected:out.values});
 if(kind!=='nd_softmax_last'){
  const g=makeGrad(kind),f=L.evaluate(g);
  payload.push({name:'grad_'+kind,target:'loss',torch:PT.compile(g).source,parameter:'w',
   expected:f.steps.find(s=>s.id==='loss').values,gradient:finiteDiff(g)});
 }
}
const python=String.raw`import json,sys
items=json.loads(sys.stdin.read())
results=[]
for item in items:
    a=None
    if item.get("numpy") is not None:
        n={"__name__":"parity_import"}
        exec(item["numpy"],n)
        a=n["run_graph"]()[item["target"]].tolist()
    t={"__name__":"parity_import"}
    exec(item["torch"],t)
    r=t["GraphRuntime"]()
    out=r.forward()
    v=out[item["target"]].detach().tolist()
    grad=None
    if item.get("parameter"):
        out[r.loss_id].sum().backward()
        grad=r.parameters[item["parameter"]].grad.detach().tolist()
    results.append({"name":item["name"],"numpy":a,"torch":v,"grad":grad})
print(json.dumps(results,allow_nan=False))
`;
const result=spawnSync('python3',['-c',python],{input:JSON.stringify(payload),encoding:'utf8',timeout:200000,maxBuffer:16*1024*1024});
if(result.status!==0)throw Error('Normalization CPU oracle failure: '+(result.stderr||result.stdout||result.error).slice(-5500));
const parsed=JSON.parse(result.stdout.trim().split('\n').at(-1));
function close(name,a,b,tol=5e-8){
 const shape=x=>Array.isArray(x)?[x.length,...shape(x[0])]:[];
 assert.deepEqual(shape(a),shape(b),name+' shape');
 const x=a.flat(Infinity),y=b.flat(Infinity);
 for(let i=0;i<x.length;i++)assert.ok(Number.isFinite(y[i])&&Math.abs(x[i]-y[i])<tol+Math.abs(x[i])*1e-9,name+'['+i+'] '+x[i]+' vs '+y[i]);
}
for(const item of payload){
 const got=parsed.find(x=>x.name===item.name);
 assert.ok(got,item.name+' missing');
 if(item.numpy)close(item.name+'/NumPy',item.expected,got.numpy);
 close(item.name+'/Torch',item.expected,got.torch);
 if(item.gradient)close(item.name+'/Autograd-finite-difference',item.gradient,got.grad,1e-4);
}
console.log('NORMALIZATION_RUNTIME_PARITY_PASS '+S.kinds.length+' operators NumPy/Torch + '+payload.filter(p=>p.gradient).length+' independent gradients');
