'use strict';
const assert=require('node:assert/strict');
const V=require('../expert_ffn_vjp.js'),B=require('../block_catalog.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<2e-6,a+' != '+b);
const copy=a=>a.map(row=>row.slice()),dot=(a,b)=>a.reduce((z,row,i)=>z+row.reduce((s,x,j)=>s+x*b[i][j],0),0);
function fd(loss,t,i,j){const a=copy(t),b=copy(t),h=1e-6;a[i][j]+=h;b[i][j]-=h;return (loss(a)-loss(b))/(2*h);}
const x=[[1,2],[-1,3]],I=[[0,1],[0,0]],W=[[2,-1],[1,3],[4,1],[-2,1]],up=[[1,2],[.5,-1],[3,0],[2,1]];
const selected=B.execute({kind:'dispatch_routed_tokens',params:{}},{x,indices:I}).values;
const f=(x0,w)=>B.execute({kind:'expert_linear_bank',params:{}},{x:x0,indices:I,weights:w}).values;
const grad=V.expertLinear(selected,I,W,up);
assert.deepEqual(grad.load,[3,1]); // Expert 0 is hit by three token routes.
for(let i=0;i<W.length;i++)for(let j=0;j<W[0].length;j++)near(grad.gradWeights[i][j],fd(w=>dot(f(selected,w),up),W,i,j));
for(let i=0;i<selected.length;i++)for(let j=0;j<selected[0].length;j++)near(grad.gradSelected[i][j],fd(z=>dot(f(z,W),up),selected,i,j));
const disp=V.dispatch(I,grad.gradSelected),dispObjective=(xin)=>dot(f(B.execute({kind:'dispatch_routed_tokens',params:{}},{x:xin,indices:I}).values,W),up);
for(let i=0;i<x.length;i++)for(let j=0;j<x[0].length;j++)near(disp.gradInput[i][j],fd(dispObjective,x,i,j));
const v=[[1,-1],[2,0],[-2,3],[4,-3]],g=[[.2,.8],[.6,.4]],u=[[2,-2],[1,3]];
const red=V.weightedReduce(v,g,u);
const reduce=(vv,gg)=>B.execute({kind:'routed_weighted_reduce',params:{}},{values:vv,gates:gg}).values;
for(let i=0;i<v.length;i++)for(let j=0;j<v[0].length;j++)near(red.gradValues[i][j],fd(q=>dot(reduce(q,g),u),v,i,j));
for(let i=0;i<g.length;i++)for(let j=0;j<g[0].length;j++)near(red.gradGates[i][j],fd(q=>dot(reduce(v,q),u),g,i,j));
const gate=[[.4,-1],[-2,2]],upval=[[2,1],[.5,-3]],du=[[1,2],[3,-.2]];
const swi=V.swiglu(gate,upval,du);
const sw=(g0,u0)=>B.execute({kind:'swiglu',params:{}},{gate:g0,up:u0}).values;
for(let i=0;i<gate.length;i++)for(let j=0;j<gate[0].length;j++){
near(swi.gradGate[i][j],fd(q=>dot(sw(q,upval),du),gate,i,j));
near(swi.gradUp[i][j],fd(q=>dot(sw(gate,q),du),upval,i,j));}
assert.throws(()=>V.expertLinear(selected,[[0,9],[0,0]],W,up),/out of range/);
console.log('Expert VJPs finite difference PASS: reused weights accumulate across routes, dispatch and SwiGLU exact');
