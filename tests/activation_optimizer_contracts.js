'use strict';
const assert=require('node:assert/strict');
const A=require('../activation_math.js'),ND=require('../nd_tensor.js'),
 B=require('../block_catalog.js'),L=require('../learning_lab.js'),
 G=require('../graph_autodiff.js'),O=require('../optimizer_rules.js'),T=require('../train_state.js');
const near=(a,b,eps=4e-6)=>assert.ok(Math.abs(a-b)<eps,a+' != '+b);
const x=[-4,-2,-.5,-.1,.1,.7,2,5];
const acts=['gelu_erf','relu_squared','mish','leaky_relu','silu'];
for(const kind of acts)for(const v of x){
 const h=1e-5,y=A.value(kind,v),d=A.derivative(kind,v);
 near(d,(A.value(kind,v+h)-A.value(kind,v-h))/(2*h),2e-5);
 assert.ok(Number.isFinite(y));
}
near(A.value('gelu_erf',0),0);
near(A.value('gelu_erf',1),.8413447460685429,1e-12);
near(A.value('mish',-20),0,1e-7);
near(A.value('leaky_relu',-2,{slope:.1}),-.2);
assert.deepEqual(B.execute({kind:'relu_squared',params:{}},{x:[[-2,-1,0,2]]}).values,[[0,0,0,4]]);
for(const kind of ['gelu_erf','relu_squared','mish','leaky_relu']){
 const y=B.execute({kind,params:{}},{x:[[-2,-.5,.5,2]]});
 y.values[0].forEach((v,i)=>near(v,A.value(kind,[-2,-.5,.5,2][i])));
}
const four=ND.seedTensor([1,2,2,4],3);
for(const kind of ['nd_gelu_erf','nd_silu','nd_relu_squared','nd_mish']){
 const result=B.execute({kind,params:{}},{x:four});
 assert.deepEqual(result.shape,[1,2,2,4]);
 const grad=require('../nd_autodiff.js').vjp(kind,{x:four},{},ND.build([1,2,2,4],()=>1),result.values).x;
 ND.indices([1,2,2,4],idx=>{
   near(ND.at(result.values,idx),A.value(kind.slice(3),ND.at(four,idx)));
   const h=1e-5,v=ND.at(four,idx);
   near(ND.at(grad,idx),(A.value(kind.slice(3),v+h)-A.value(kind.slice(3),v-h))/(2*h),2e-5);
 });
}
const p=[[.5,-.7],[.1,-.3]],g=[[.2,-.4],[.5,.05]];
const strategies=O.KINDS;
assert.deepEqual(strategies,['adamw','sgd_momentum','rmsprop','lion']);
for(const optimizer of strategies){
 const h={optimizer,lr:.01,weightDecay:.02},a=O.step(p,g,null,h);
 assert.equal(a.state.kind,optimizer);assert.equal(a.state.t,1);
 assert.deepEqual(B.execute({kind:'opt_'+optimizer,params:{}},{parameter:p,gradient:g}).shape,[2,2]);
 assert.notDeepEqual(a.parameter,p);
 const b=O.step(a.parameter,g,a.state,h);
 assert.equal(b.state.t,2);
 assert.deepEqual(O.step(a.parameter,g,JSON.parse(JSON.stringify(a.state)),h),b);
 const rem=L.preset('trainable_core');
 const train=T.steps(rem,{targetId:'loss',count:2,optimizer,lr:.004,weightDecay:.01});
 assert.equal(train.history.length,2);
 assert.ok(Number.isFinite(train.finalLoss));
 assert.equal(train.graph.nodes.find(n=>n.id==='shared_weight').params.optimizer.kind,optimizer);
 const resumed=JSON.parse(JSON.stringify(train.graph));
 assert.deepEqual(T.step(resumed,{targetId:'loss',optimizer,lr:.004,weightDecay:.01}).graph,
                  T.step(train.graph,{targetId:'loss',optimizer,lr:.004,weightDecay:.01}).graph);
 const fourModel=L.preset('nd_trainable_mqa');
 const ndTrain=T.steps(fourModel,{targetId:'loss',count:2,optimizer,lr:.004,weightDecay:.01});
 assert.equal(ndTrain.history.at(-1).records.length,3);
 assert.ok(ndTrain.graph.nodes.filter(n=>n.kind==='nd_parameter').every(n=>n.params.optimizer.kind===optimizer&&n.params.optimizer.t===2));
 console.log(optimizer+' 2D/4D TRAIN PASS: '+train.firstLoss+' -> '+train.finalLoss+', '+ndTrain.firstLoss+' -> '+ndTrain.finalLoss);
}
const sgd=O.step([[.5]],[[.2]],null,{optimizer:'sgd_momentum',lr:.1,weightDecay:0,momentum:.9});
near(sgd.parameter[0][0],.48);near(sgd.state.m[0][0],.2);
near(O.step(sgd.parameter,[[.2]],sgd.state,{optimizer:'sgd_momentum',lr:.1,weightDecay:0}).state.m[0][0],.38);
const lion=O.step([[.5,-.5]],[[.001,-2]],null,{optimizer:'lion',lr:.1,weightDecay:0});
assert.deepEqual(lion.update,[[.1,-.1]]);
assert.ok(lion.state.v.flat().every(x=>x===0));
const rms=O.step([[1]],[[2]],null,{optimizer:'rmsprop',lr:.1,weightDecay:0,alpha:.99});
near(rms.state.v[0][0],.04);near(rms.parameter[0][0],.00000005,2e-7);
const old=O.step([[1]],[[.5]],null,{lr:.01});
assert.throws(()=>O.step(old.parameter,[[.5]],old.state,{optimizer:'lion'}),/reset moments/);
assert.throws(()=>O.step([[1]],[[1]],null,{optimizer:'not-an-optimizer'}),/unsupported/);
assert.throws(()=>O.step([[1]],[[1]],null,{optimizer:'sgd_momentum',lr:-1}),/hyperparameters/);
// 4D activation is not merely a catalog display: it participates in the same VJP+optimizer transaction.
const mix=L.preset('nd_trainable_mqa'),oldEdge=mix.edges.find(e=>e.from==='query'&&e.to==='scores');
assert.ok(oldEdge);oldEdge.from='activated_query';
mix.nodes.push({id:'activated_query',kind:'nd_mish',x:600,y:40,params:{}});
mix.edges.push({from:'query',to:'activated_query',port:'x'});
const fg=L.evaluate(mix),bg=G.backward(mix,fg);
assert.deepEqual(ND.shape(bg.gradients.query),[1,4,3,2]);
const updated=T.step(mix,{optimizer:'lion',targetId:'loss',lr:.004});
assert.equal(updated.records.length,3);
console.log('Activation/optimizer contracts PASS: scalar derivatives, four-axis VJPs, four optimizer paths, state persistence and composed 4D training');
