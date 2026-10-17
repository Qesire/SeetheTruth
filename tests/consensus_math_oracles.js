/* Numerical checks for illustrative algebra; NOT model benchmarks. */
'use strict';
const assert=require('node:assert/strict');
const K=require('../consensus_knowledge.js');
const near=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<=eps,String(a)+' vs '+String(b));
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const rot=(v,t)=>[Math.cos(t)*v[0]-Math.sin(t)*v[1],Math.sin(t)*v[0]+Math.cos(t)*v[1]];
const q=[.4,-.7],k=[-.2,.5];
for(const i of [0,1,3,7])for(const j of [0,2,6])
  near(dot(rot(q,i*.41),rot(k,j*.41)),dot(q,rot(k,(j-i)*.41)));
assert.ok(K.byId.rope.formula.includes("R(j−i)"));
const heads=32,kv=8,bytes=2,seq=1024,d=128,batch=3;
const kvBytes=h=>2*batch*seq*h*d*bytes;
near(kvBytes(heads)/kvBytes(kv),heads/kv);
assert.ok(K.byId.gqa.objective.includes("缓存"));
const softmax=xs=>{const m=Math.max(...xs),e=xs.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b);return e.map(x=>x/z)};
const Q=[1,0],Keys=[[1,0],[0,1]],Values=[[4,-2],[-1,5]];
const weights=softmax(Keys.map(key=>dot(Q,key)/Math.sqrt(2)));
near(weights.reduce((a,b)=>a+b),1);
const O=[0,1].map(c=>weights.reduce((sum,w,t)=>sum+w*Values[t][c],0));
near(O[0],weights[0]*4-weights[1]); near(O[1],-weights[0]*2+weights[1]*5);
const sig=x=>1/(1+Math.exp(-x)),swiglu=(g,u)=>g*sig(g)*u;
near(swiglu(0,4),0);near(swiglu(1,2),2*sig(1));
const logits=[3,1,-2],prob=softmax(logits);
near(-Math.log(prob[0]),Math.log(logits.reduce((s,x)=>s+Math.exp(x-3),0)));
console.log("Attention, RoPE, GQA bytes, SwiGLU and NLL numeric oracles PASS");
