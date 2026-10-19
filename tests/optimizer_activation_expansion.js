'use strict';
const assert=require('node:assert/strict');
const A=require('../activation_math.js'), O=require('../optimizer_rules.js'),
 ND=require('../nd_tensor.js'),L=require('../learning_lab.js'),
 G=require('../graph_autodiff.js'),T=require('../train_state.js'),
 B=require('../block_catalog.js'),C=require('../concept_flow.js');
const near=(x,y,eps=5e-6)=>assert.ok(Math.abs(x-y)<eps,'expected '+x+' ≈ '+y);
const clone=x=>JSON.parse(JSON.stringify(x));
for(const k of ['elu','hardswish'])for(const x of [-5,-4,-2,-.8,-.2,.3,1,2,4,7]){
 const h=1e-5;near(A.derivative(k,x,{alpha:1.3}),
   (A.value(k,x+h,{alpha:1.3})-A.value(k,x-h,{alpha:1.3}))/(2*h));
}
near(A.value('elu',-1,{alpha:2}),2*Math.expm1(-1),1e-12);
near(A.value('hardswish',-4),0,1e-12);
near(A.value('hardswish',0),0,1e-12);
near(A.value('hardswish',4),4,1e-12);
for(const kind of ['elu','hardswish','nd_elu','nd_hardswish']){
 const isND=kind.startsWith('nd_'),x=isND?ND.seedTensor([1,2,2,4],3):[[-1,-.3,.4,3.2],[.2,1,-2,-4]];
 const n={id:'activation',kind,params:kind.endsWith('elu')?{alpha:1}:{}};
 const inputs={x},concept=C.analyze(n,{x:isND?ND.shape(x):[2,4]}),actual=B.execute(n,inputs);
 assert.deepEqual(concept.shape,actual.shape);
 const f=kind.slice(isND?3:0),flat=isND?ND.flatten(actual.values):actual.values.flat(),
  flatx=isND?ND.flatten(x):x.flat();
 flat.forEach((v,i)=>near(v,A.value(f,flatx[i])));
}
const p=[[.75,-.4],[.2,-.7]],g=[[.2,-.5],[-.35,.4]],lr=.025,decay=.08;
const a=O.step(p,g,null,{optimizer:'adam',lr,weightDecay:decay});
const w=O.step(p,g,null,{optimizer:'adamw',lr,weightDecay:decay});
assert.notDeepEqual(a.parameter,w.parameter,'coupled L2 Adam must not equal decoupled AdamW with decay');
assert.equal(a.state.kind,'adam');assert.equal(w.state.kind,'adamw');
const z=O.step(p,g,null,{optimizer:'adam',lr,weightDecay:0}),
 zw=O.step(p,g,null,{optimizer:'adamw',lr,weightDecay:0});
assert.deepEqual(z.parameter,zw.parameter,'Adam and AdamW without decay must be identical');
assert.deepEqual(z.state.m,zw.state.m);assert.deepEqual(z.state.v,zw.state.v);
const ad=O.step([[1]],[[2]],null,{optimizer:'adagrad',lr:.1,weightDecay:0,eps:1e-10});
near(ad.state.v[0][0],4);near(ad.parameter[0][0],.9);
const ad2=O.step(ad.parameter,[[2]],ad.state,{optimizer:'adagrad',lr:.1,weightDecay:0,eps:1e-10});
near(ad2.state.v[0][0],8);near(ad2.parameter[0][0],.9-.2/Math.sqrt(8));
const rms=O.step([[1]],[[2]],null,{optimizer:'rmsprop',lr:.1,weightDecay:0,eps:1e-10});
near(rms.state.v[0][0],.04);
assert.notEqual(rms.state.v[0][0],ad.state.v[0][0]);
const trainedBase=L.preset('trainable_core');
const adagradBase=T.steps(trainedBase,{targetId:'loss',optimizer:'adagrad',count:3,lr:.01,lrDecay:0});
const adagradDecay=T.steps(trainedBase,{targetId:'loss',optimizer:'adagrad',count:3,lr:.01,lrDecay:1});
assert.notDeepEqual(adagradBase.graph.nodes.find(n=>n.id==='shared_weight').params.values,
                    adagradDecay.graph.nodes.find(n=>n.id==='shared_weight').params.values,
                    'UI-controllable Adagrad lrDecay must alter actual multi-step parameter trajectory');
const sgdFlat=T.steps(trainedBase,{targetId:'loss',optimizer:'sgd_momentum',count:3,lr:.01,momentum:0});
const sgdMom=T.steps(trainedBase,{targetId:'loss',optimizer:'sgd_momentum',count:3,lr:.01,momentum:.9});
assert.notDeepEqual(sgdFlat.graph.nodes.find(n=>n.id==='shared_weight').params.values,
                    sgdMom.graph.nodes.find(n=>n.id==='shared_weight').params.values);
const rmsSlow=T.steps(trainedBase,{targetId:'loss',optimizer:'rmsprop',count:2,lr:.01,alpha:.9});
const rmsFast=T.steps(trainedBase,{targetId:'loss',optimizer:'rmsprop',count:2,lr:.01,alpha:.999});
assert.notDeepEqual(rmsSlow.graph.nodes.find(n=>n.id==='shared_weight').params.values,
                    rmsFast.graph.nodes.find(n=>n.id==='shared_weight').params.values);
const slowed=O.step(ad.parameter,[[2]],ad.state,{optimizer:'adagrad',lr:.1,weightDecay:0,lrDecay:1});
assert.ok(slowed.update[0][0]<ad2.update[0][0],'learning-rate decay must change update');
for(const type of ['adam','adagrad']){
 assert.ok(O.KINDS.includes(type));
 assert.deepEqual(B.execute({kind:'opt_'+type,params:{}},{parameter:p,gradient:g}).shape,[2,2]);
 const preset=L.preset('trainable_core'),orig=JSON.stringify(preset);
 const trained=T.steps(preset,{optimizer:type,targetId:'loss',count:2,lr:.004,weightDecay:.01});
 assert.equal(JSON.stringify(preset),orig,'training must preserve previous undo snapshot');
 assert.equal(trained.graph.nodes.find(n=>n.id==='shared_weight').params.optimizer.kind,type);
 assert.equal(trained.graph.nodes.find(n=>n.id==='shared_weight').params.optimizer.t,2);
 assert.deepEqual(T.step(clone(trained.graph),{optimizer:type,targetId:'loss',lr:.004}).graph,
                  T.step(trained.graph,{optimizer:type,targetId:'loss',lr:.004}).graph);
 const nd=L.preset('nd_trainable_mqa');
 const state=T.steps(nd,{optimizer:type,targetId:'loss',count:2,lr:.004,weightDecay:.01});
 assert.equal(state.history[0].records.length,3);
 assert.ok(state.graph.nodes.filter(n=>n.kind==='nd_parameter').every(n=>n.params.optimizer.kind===type&&n.params.optimizer.t===2));
 const switched=clone(state.graph);
 assert.throws(()=>T.step(switched,{optimizer:type==='adam'?'adagrad':'adam',targetId:'loss'}),/reset moments/);
}
// Value preserving shape is insufficient: VJP must drive parameter updates through both activations.
for(const kind of ['elu','hardswish']){
 const g2=L.preset('trainable_core'),source=g2.nodes.find(n=>n.id==='shared_weight');
 const graph={schema:L.SCHEMA,title:kind+' trainable 2D',nodes:[
   clone(source),{id:'act',kind,x:250,y:30,params:kind==='elu'?{alpha:1.3}:{}},
   {id:'target',kind:'input',x:250,y:320,params:{rows:2,width:2,seed:2}},
   {id:'loss',kind:'mse',x:540,y:90,params:{}}
 ],edges:[{from:'shared_weight',to:'act',port:'x'},
          {from:'act',to:'loss',port:'prediction'},{from:'target',to:'loss',port:'target'}]};
 const f=L.evaluate(graph),d=G.backward(graph,f);
 for(let i=0;i<2;i++)for(let j=0;j<2;j++){
   const h=1e-6,get=(sign)=>{const edit=clone(graph);edit.nodes[0].params.values[i][j]+=sign*h;return L.evaluate(edit).steps.at(-1).values[0][0];};
   near(d.gradients.shared_weight[i][j],(get(1)-get(-1))/(2*h),1e-5);
 }
 assert.equal(T.step(graph,{targetId:'loss',optimizer:'adagrad'}).records.length,1);
}
for(const name of ['activation_extension','optimizer_decay_compare']){
 const graph=L.preset(name);
 assert.deepEqual(C.evaluate(graph).steps.map(s=>s.shape),L.evaluate(graph).steps.map(s=>s.shape));
}
console.log('Expanded activation/optimizer PASS: 2D/4D ELU and HardSwish, Adam/AdamW decay, Adagrad accumulation and 2D/4D stateful training');
