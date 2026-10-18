'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),A=require('../graph_autodiff.js'),B=require('../block_catalog.js');
const eq=(a,b,eps=2e-6)=>assert.ok(Math.abs(a-b)<eps,String(a)+' != '+String(b));
function forwardWith(graph,overrides={}){
 const nodes=L.order(L.validateGraph(graph)),data={},results=[];
 for(const n of nodes){
   const ins={};for(const port of L.defs[n.kind].ports){
     const e=graph.edges.find(e=>e.to===n.id&&e.port===port);
     ins[port]=data[e.from];
   }
   let values,flops=null,internals=null;
   if(Object.hasOwn(overrides,n.id))values=overrides[n.id];
   else if(n.kind==='input'){const t=L.evaluate({schema:L.SCHEMA,title:'',nodes:[n],edges:[]});values=t.steps[0].values;}
   else if(n.kind==='matmul'){const x=ins.a,y=ins.b;values=x.map(row=>y[0].map((_,j)=>row.reduce((s,v,k)=>s+v*y[k][j],0)));}
   else if(n.kind==='rmsnorm')values=ins.x.map(row=>{const inv=1/Math.sqrt(row.reduce((s,v)=>s+v*v,0)/row.length+n.params.epsilon);return row.map(v=>v*inv);});
   else if(n.kind==='swiglu')values=ins.gate.map((row,i)=>row.map((v,j)=>v/(1+Math.exp(-v))*ins.up[i][j]));
   else if(n.kind==='attention'){
     const q=ins.q,k=ins.k,v=ins.v,scale=1/Math.sqrt(q[0].length),
       scores=q.map(row=>k.map(other=>row.reduce((s,a,j)=>s+a*other[j],0)*scale)),
       prob=scores.map(row=>{const m=Math.max(...row),exp=row.map(x=>Math.exp(x-m)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(x=>x/sum);});
     values=prob.map(row=>v[0].map((_,j)=>row.reduce((s,a,i)=>s+a*v[i][j],0)));
   }
   else if(n.kind==='add')values=ins.a.map((row,i)=>row.map((v,j)=>v+ins.b[i][j]));
   else if(n.kind==='mse'){
     const x=ins.prediction,y=ins.target;values=[[x.reduce((acc,row,i)=>acc+row.reduce((s,v,j)=>s+(v-y[i][j])**2,0),0)/(x.length*x[0].length)]];
   }else throw Error('unsupported forward fixture node '+n.kind);
   data[n.id]=values;results.push({id:n.id,kind:n.kind,values,shape:[values.length,values[0].length],flops,internals});
 }
 return {steps:results};
}
const graph=L.preset('autodiff_core'),original=L.evaluate(graph),back=A.backward(graph,original);
assert.equal(graph.nodes.length,11);
assert.equal(back.targetId,'loss');
assert.equal(back.gradients.shared_weight.length,2);
assert.equal(back.pathIds.filter(id=>id==='swiglu').length,1);
assert.deepEqual(back.gradients.target,original.steps.find(x=>x.id==='target').values.map(row=>row.map(()=>0)));
assert.ok(back.cutEdges.some(x=>x.port==='target'&&x.nodeId==='loss'));
const seen=new Set(),grad=back.gradients;
const src=Object.fromEntries(original.steps.map(t=>[t.id,t.values]));
for(const name of ['shared_weight','features']){
 for(let i=0;i<2;i++)for(let j=0;j<2;j++){
   const h=1e-6,plus=src[name].map(row=>row.slice()),minus=src[name].map(row=>row.slice());
   plus[i][j]+=h;minus[i][j]-=h;
   const fwd=forwardWith(graph,{[name]:plus}),bwd=forwardWith(graph,{[name]:minus});
   const finite=(fwd.steps.at(-1).values[0][0]-bwd.steps.at(-1).values[0][0])/(2*h);
   eq(grad[name][i][j],finite,5e-6);
   seen.add(name+':'+i+':'+j);
 }
}
assert.equal(seen.size,8);
forwardWith(graph).steps.forEach((step,i)=>step.values.forEach((row,r)=>row.forEach((v,c)=>eq(v,original.steps[i].values[r][c],1e-11))));
const nonScalar=L.preset('autodiff_core');
assert.throws(()=>A.backward(nonScalar,original,{targetId:'project'}),/scalar/);
const broken=L.preset('autodiff_core');broken.nodes.find(n=>n.id==='attend').kind='gqa';
assert.throws(()=>A.backward(broken,original),/stale|invalid|trace/);
console.log('Unified graph autodiff PASS: 8 central differences, shared-weight gradient accumulation, Norm/SwiGLU/Attention and target barrier');
