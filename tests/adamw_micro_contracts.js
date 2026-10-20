'use strict';
const assert=require('node:assert/strict'),A=require('../adamw_micro.js');
const near=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<eps,a+' != '+b);
const param=[[1,-2],[0.25,.5]],gradient=[[2,-3],[0,4]];
const before=JSON.stringify(param),gbefore=JSON.stringify(gradient);
const o={lr:.1,beta1:.9,beta2:.999,eps:1e-8,weightDecay:.02};
const first=A.step(param,gradient,null,o);
for(let i=0;i<2;i++)for(let j=0;j<2;j++){
  near(first.state.m[i][j],.1*gradient[i][j]);
  near(first.state.v[i][j],.001*gradient[i][j]**2);
  const direction=gradient[i][j]===0?0:gradient[i][j]/(Math.abs(gradient[i][j])+o.eps);
  near(first.parameter[i][j],param[i][j]-.1*(direction+.02*param[i][j]),2e-9);
}
assert.equal(first.state.t,1);
assert.equal(JSON.stringify(param),before,'AdamW preview mutated graph parameter');
assert.equal(JSON.stringify(gradient),gbefore,'AdamW preview mutated upstream gradient');
const second=A.step(first.parameter,gradient,first.state,o);
assert.equal(second.state.t,2);
for(let i=0;i<2;i++)for(let j=0;j<2;j++){
 const mhat=second.state.m[i][j]/(1-o.beta1**2),vhat=second.state.v[i][j]/(1-o.beta2**2);
 near(second.parameter[i][j],first.parameter[i][j]-.1*mhat/(Math.sqrt(vhat)+o.eps)-.1*.02*first.parameter[i][j]);
}
const zero=A.step([[1,-2]],[[0,0]],null,{lr:.1,weightDecay:.1});
near(zero.parameter[0][0],.99);near(zero.parameter[0][1],-1.98);
assert.throws(()=>A.step([[1,2]],[[1]]),/shape mismatch/);
assert.throws(()=>A.step([[1]],[[1]],null,{lr:0}),/hyperparameters/);
assert.throws(()=>A.step([[1]],[[1]],{t:1,m:[[1,2]],v:[[1]]}),/shape mismatch/);
console.log('AdamW micro-state PASS: bias correction, two steps, decay with zero gradient and immutable inputs');
