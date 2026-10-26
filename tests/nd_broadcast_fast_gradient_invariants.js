'use strict';
// Fast source-execution algebraic guard for real N-D broadcast/reduction graphs.
// PyTorch CPU independent evidence is checked separately by torch-export.yml.
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),ND=require('../nd_tensor.js');
const G=require('../graph_autodiff.js'),Torch=require('../torch_codegen.js');
const {cases,create}=require('./fixtures/nd_broadcast_reduction_graphs.js');
const near=(actual,expected,where)=>{
 assert(Number.isFinite(actual)&&Number.isFinite(expected),where+' nonfinite');
 assert.ok(Math.abs(actual-expected)<=3e-11+1e-10*Math.abs(expected),
  where+' '+actual+' != '+expected);
};
let checked=0;
for(const spec of cases){
 const graph=create(spec.id),untouched=JSON.stringify(graph);
 const run=L.evaluate(graph);
 const reverse=G.backward(graph,run,{targetId:'loss'});
 const byId=new Map(run.steps.map(s=>[s.id,s.values]));
 const original=ND.shape(byId.get('x'));
 assert.deepEqual(original,spec.shape,spec.id+' source axis identity');
 const [B,H,T,C]=original;
 for(const [name,sh] of [
  ['bias',[1,1,1,C]],['gain',[1,1,1,C]],
  ['biased',original],['gated',original],['summed',[B,H,1,C]],
  ['loss',[1,1]]
 ])assert.deepEqual(ND.shape(byId.get(name)),sh,spec.id+'/'+name);
 for(const name of ['x','bias','gain','biased','gated','summed','loss'])
  assert.ok(reverse.pathIds.includes(name),spec.id+'/'+name+' must lie on real Loss path');
 assert.ok(!reverse.pathIds.includes('target'),'fixed target cannot claim computed Loss adjoint');
 const g=reverse.gradients,values=byId;
 for(let c=0;c<C;c++){
  let biasExpected=0,gainExpected=0;
  for(let b=0;b<B;b++)for(let h=0;h<H;h++)
   for(let t=0;t<T;t++){
    const at=[b,h,t,c],up=ND.at(g.summed,[b,h,0,c]);
    const gated=ND.at(g.gated,at),biased=ND.at(g.biased,at),gain=ND.at(values.get('gain'),[0,0,0,c]);
    near(gated,up,spec.id+'/reduce keepdim '+at);
    near(biased,gated*gain,spec.id+'/mul local VJP '+at);
    near(ND.at(g.x,at),biased,spec.id+'/add source VJP '+at);
    biasExpected+=biased;
    gainExpected+=gated*ND.at(values.get('biased'),at);
    checked++;
   }
  near(ND.at(g.bias,[0,0,0,c]),biasExpected,spec.id+'/bias singleton axes sum '+c);
  near(ND.at(g.gain,[0,0,0,c]),gainExpected,spec.id+'/gain singleton axes sum '+c);
 }
 assert.equal(JSON.stringify(graph),untouched,spec.id+' original source graph mutated');
 const program=Torch.compile(graph,{lossId:'loss'});
 assert.equal(program.lossId,'loss',spec.id+' codegen kept native loss');
 assert.ok(program.source.includes('torch'),'original generated PyTorch source absent');
}
for(const name of ['nd_broadcast_add','nd_broadcast_mul','nd_reduce_sum','nd_mse'])
 assert.ok(Torch.supportedKinds.includes(name),'independent exporter removed '+name);
assert.throws(()=>ND.broadcastBinary(
 ND.seedTensor([1,2,3,2],2),ND.seedTensor([1,3,2,2],2),(a,b)=>a+b),
 /incompatible axes/,'disallow incompatible non-singleton broadcasting');
assert.equal(cases.length,2);
console.log('ND_BROADCAST_FAST_GRADIENT_INVARIANTS_PASS '+JSON.stringify({
 fixtures:cases.map(x=>x.id),recheckedCoordinates:checked,
 invariants:['reduction_keepdim','mul_vjp','add_vjp','broadcast_bias_sum','broadcast_gain_sum'],
 evidence:'SOURCE_EXECUTED_NOT_INDEPENDENT_TORCH'
}));
