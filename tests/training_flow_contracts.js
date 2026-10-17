'use strict';
const assert=require('node:assert/strict');
const T=require('../training_flow.js');
assert.equal(T.contract.types,14);
assert.equal(Object.keys(T.BLOCKS).length,14);
const cfg={theta:0.4,x1:2,y1:1,x2:-1,y2:2,lr:.1,weightDecay:.02,
 optimizer:"adamw",beta1:.9,beta2:.999,eps:1e-8,microbatches:2,steps:2};
const near=(actual,expected,label,eps=1e-10)=>
 assert.ok(Math.abs(actual-expected)<eps,label+': '+actual+' differs from '+expected);
const symbolic=T.build(cfg,'concept');
const numerical=T.build(cfg,'scalar');
assert.equal(symbolic.events.length,numerical.events.length);
assert.equal(symbolic.final,null);
assert.equal(symbolic.evidence,'symbolic_state_contract_no_numeric_training');
assert.ok(symbolic.events.every(e=>e.value===null&&e.state===null));
assert.ok(numerical.events.length>30);
assert.equal(numerical.final.step,2);
for(const [i,e] of numerical.events.entries()){
 assert.equal(e.kind,symbolic.events[i].kind);
 assert.equal(e.symbol,symbolic.events[i].symbol);
 assert.ok(e.formula&&e.detail&&e.label);
 assert.equal(e.id,'e'+i);
 assert.ok(e.state&&typeof e.state.theta==='number');
}
for(const b of Object.values(T.BLOCKS)){
 for(const [field,min] of Object.entries({formula:5,how:18,practice:18,trap:18,test:18}))
   assert.ok(b[field]?.length>=min,field+' too short');
 for(const ref of b.refs)assert.ok(T.SOURCE[ref]?.url.startsWith('https://docs.pytorch.org'));
}
// Independent scalar analytic reference for microbatch 1 and microbatch 2.
const g1=(cfg.theta*cfg.x1-cfg.y1)*cfg.x1;
const g2=(cfg.theta*cfg.x2-cfg.y2)*cfg.x2;
const g=(g1+g2)/2;
near(g1,-0.4,'local g1');near(g2,2.4,'local g2');near(g,1,'mean grad');
const events=numerical.events;
const firstReady=events.find(e=>e.kind==='gradient_ready');
near(firstReady.value,g,'first buffer average');
for(const e of events.slice(0,events.findIndex(x=>x.kind==='update')))
 near(e.state.theta,cfg.theta,'theta frozen through backward');
const firstUpdate=events.find(e=>e.kind==='update');
const expectedTheta1=cfg.theta*(1-cfg.lr*cfg.weightDecay)-
 cfg.lr*g/(Math.abs(g)+cfg.eps);
near(firstUpdate.value,expectedTheta1,'first AdamW update');
near(firstUpdate.meta.old,cfg.theta,'old param');
assert.equal(firstUpdate.meta.optimizerStep,1);
const zeroEvents=events.filter(e=>e.kind==='zero_grad');
assert.equal(zeroEvents.length,2);
assert.equal(zeroEvents[0].state.grad,null);
assert.equal(zeroEvents[1].state.grad,null);
near(zeroEvents[1].state.theta,firstUpdate.value,'second batch uses updated theta');
const mom2=events.filter(e=>e.kind==='adam_moment')[1];
const ready2=events.filter(e=>e.kind==='gradient_ready')[1];
const gNext=(
 ((expectedTheta1*cfg.x1-cfg.y1)*cfg.x1)+
 ((expectedTheta1*cfg.x2-cfg.y2)*cfg.x2))/2;
near(ready2.value,gNext,'second batch average, new theta');
const m2=cfg.beta1*((1-cfg.beta1)*g)+(1-cfg.beta1)*gNext;
const v2=cfg.beta2*((1-cfg.beta2)*g*g)+(1-cfg.beta2)*gNext*gNext;
near(mom2.meta.m,m2,'persisted first moment');
near(mom2.meta.v,v2,'persisted second moment');
assert.equal(mom2.state.step,2);
const mhat2=m2/(1-cfg.beta1**2);
const vhat2=v2/(1-cfg.beta2**2);
const expectedTheta2=expectedTheta1*(1-cfg.lr*cfg.weightDecay)-
 cfg.lr*mhat2/(Math.sqrt(vhat2)+cfg.eps);
near(numerical.final.theta,expectedTheta2,'second AdamW step');
const SGD=T.build({...cfg,optimizer:'sgd',steps:2},'scalar');
near(SGD.events.find(e=>e.kind==='update').value,cfg.theta-cfg.lr*g,'SGD first update (no decay)');
assert.ok(!SGD.events.some(e=>e.kind==='adam_moment'||e.kind==='bias_correct'||e.kind==='decay'));
assert.equal(SGD.final.m,0);
assert.equal(SGD.final.v,0);
const noAccum=T.build({...cfg,microbatches:1,steps:1,optimizer:'sgd'},'scalar');
near(noAccum.events.find(e=>e.kind==='gradient_ready').value,g1,'one microbatch gradient');
near(noAccum.final.theta,cfg.theta-cfg.lr*g1,'single microbatch SGD');
assert.equal(T.build(cfg,'scalar').events.length,numerical.events.length);
assert.deepEqual(T.build(cfg,'scalar'),numerical,'deterministic replay');
const frozen=T.build({...cfg,x1:0,x2:0,y1:1,y2:1,steps:1},'scalar');
near(frozen.events.find(e=>e.kind==='gradient_ready').value,0,'zero-input gradient');
near(frozen.final.theta,cfg.theta*(1-cfg.lr*cfg.weightDecay),'AdamW decays even when g=0');
const finiteDiffTheta=0.325,delta=1e-6;
const lossAt=t=>.5*((t*cfg.x1-cfg.y1)**2+(t*cfg.x2-cfg.y2)**2)/2;
const finite=(lossAt(finiteDiffTheta+delta)-lossAt(finiteDiffTheta-delta))/(2*delta);
const analytic=((finiteDiffTheta*cfg.x1-cfg.y1)*cfg.x1+
 (finiteDiffTheta*cfg.x2-cfg.y2)*cfg.x2)/2;
near(finite,analytic,'independent finite-difference gradient',1e-9);
for(const [patch,rx] of [
 [{lr:0},/学习率/],[{lr:Infinity},/有限/],[{beta1:1},/参数/],
 [{weightDecay:-1},/参数/],[{microbatches:3},/microbatches/],
 [{steps:5},/steps/],[{optimizer:'adam'},/optimizer/],[{theta:NaN},/有限/]
])assert.throws(()=>T.build({...cfg,...patch},'scalar'),rx);
assert.throws(()=>T.build(cfg,'gpu'),/mode/);
console.log('TRAINING CONTRACTS PASS: 14 blocks, accumulation, finite differences, AdamW t=1/2 bias correction and decay, SGD, None gradients, frozen parameter, symbolic separation');
