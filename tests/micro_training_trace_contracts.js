'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const Core=require('../compute_microscope.js'),Training=require('../micro_training_flows.js'),Trace=require('../micro_training_trace.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10, a+' should equal '+b);
function shared(xValues,wValues){
 const t=new Training.Tape(),x=t.input([xValues],'input'),w=t.parameter('shared',wValues.map(z=>[z]));
 const a=t.matmul(x,w),b=t.matmul(x,t.parameter('shared',wValues.map(z=>[z])));
 const y=t.add(a,b),loss=t.meanSquare(y,[[0]]);t.backward(loss);
 return {t,x,w,a,b,y,loss};
}
for(const [x,w]of [[[1,2],[2,3]],[[-1,2],[3,-2]],[[.1,-.4],[.5,1]]]){
 const {t,a,b,w:wid}=shared(x,w);const frame=Trace.tape(t);
 assert.equal(Core.validate(frame),frame);
 assert.equal(frame.steps.length,12);
 assert.equal(frame.steps.filter(z=>z.kind==='vjp').length,6);
 assert.equal(t.parameters.size,1);
 near(Core.dot(frame,a,0,0).error,0);near(Core.dot(frame,b,0,0).error,0);
 const g=t.gradients().shared;
 const predicted=t.value(a)[0][0]*2*(x[0])*4;
 near(g[0][0],predicted);near(g[1][0],t.value(a)[0][0]*2*x[1]*4);
 const grad=frame.steps.find(s=>s.id==='grad_'+wid);
 assert.deepEqual(grad.out.values,g.flat());
 assert.equal(grad.dependencies.length,2);
 const corrupt=JSON.parse(JSON.stringify(frame));corrupt.steps.find(s=>s.id===a).out.values[0]+=1;
 assert.throws(()=>Core.validate(corrupt),/GEMM reconstruction contradicts displayed output/);
}
const examples=[
 {capacity:2,r:5},{capacity:1,r:5},{capacity:1,r:-2}
];
for(const {capacity,r}of examples){
 const args={tokens:[[1,2],[2,1]],expertWeights:[[[1],[0]],[[0],[1]]],
   routerLogits:[[r,0],[r,0]],target:[[0],[0]],options:{capacity}};
 const result=Training.moeBackward(args.tokens,args.expertWeights,args.routerLogits,args.target,args.options);
 const frame=Trace.moe(args,result);
 assert.equal(Core.validate(frame),frame);
 assert.equal(frame.steps.length,9);
 assert.deepEqual(frame.steps.find(s=>s.id==='dispatch').out.values,
  result.routed.flatMap(x=>[x.expert,x.accepted?1:0,x.probability]));
 if(capacity===1&&result.routed[0].expert===result.routed[1].expert){
   assert.equal(result.routed[1].accepted,false);
   assert.deepEqual(result.output[1],[0]);
 }
 const corrupt=JSON.parse(JSON.stringify(result));corrupt.gradExpert[0][0][0]+=.05;
 assert.throws(()=>Trace.moe(args,corrupt),/MoE results disagree/);
}
for(const adv of [-1,1])for(const logit of [-8,-1,0,1,8]){
 const args={logits:[logit,0],oldLogits:[0,0],action:0,advantage:adv,epsilon:.2};
 const result=Training.ppo(args.logits,args.oldLogits,args.action,args.advantage,args.epsilon);
 const frame=Trace.ppo(args,result);
 assert.equal(Core.validate(frame),frame);
 assert.equal(frame.steps.length,8);
 assert.equal(frame.steps.at(-1).kind,'vjp');
 near(frame.steps.at(-1).out.values[0],result.gradLogits[0]);
 const bad={...result,ratio:result.ratio+.2};assert.throws(()=>Trace.ppo(args,bad),/PPO results disagree/);
}
const html=fs.readFileSync(path.join(__dirname,'../micro_training.html'),'utf8');
for(const file of ['micro_training_trace.js','compute_microscope.js','compute_microscope_ui.js'])
 assert.ok(html.includes('src="'+file+'"'),'missing runtime '+file);
assert.ok(html.includes('id="computeMicroscope"'));
assert.ok(html.includes('microscope.show(frame)'));
const allow=fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/);
assert.ok(allow.includes('micro_training_trace.js'));
new vm.Script(fs.readFileSync(path.join(__dirname,'../micro_training_trace.js'),'utf8'));
console.log('TRAINING_OPERATOR_TRACE_PASS Tape forward/VJP accumulation, 3 MoE routing/capacity cases, 10 clipped PPO regimes, fail-closed data and shared UI');
