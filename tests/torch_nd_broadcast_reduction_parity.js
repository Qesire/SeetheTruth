'use strict';
/* Independent real PyTorch CPU 4D broadcast+reduction autograd oracle.
 * No certificate is emitted until this is covered by source-pinned ledger.
 */
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),ND=require('../nd_tensor.js');
const Grad=require('../graph_autodiff.js'),Torch=require('../torch_codegen.js');
const {cases,create}=require('./fixtures/nd_broadcast_reduction_graphs.js');
const tol={forward:3e-10,backward:3e-9,finiteDifference:6e-6};
const eq=(a,b,context,t)=>{
 const aa=ND.flatten(a),bb=Array.isArray(b)?b.flat(Infinity):[b];
 assert.equal(aa.length,bb.length,context+' coordinate count');
 for(let i=0;i<aa.length;i++)
  assert.ok(Number.isFinite(bb[i])&&Math.abs(aa[i]-bb[i])<=t,
   context+'['+i+'] '+aa[i]+' vs '+bb[i]);
};
const flat=x=>Array.isArray(x)?x.flat(Infinity):[x];
let nForward=0,nBackward=0,nFD=0;
for(const spec of cases){
 const graph=create(spec.id),pristine=JSON.stringify(graph);
 const original=L.evaluate(graph),reverse=Grad.backward(graph,original,{targetId:'loss'});
 const emitted=Torch.compile(graph,{lossId:'loss'});
 assert.ok(!emitted.source.includes('import seethetruth'));
 const py=[
  'torch.set_num_threads(1)',
  'm=GraphRuntime()',
  'out=m.forward()',
  'for t in out.values():',
  '    if t.requires_grad: t.retain_grad()',
  'out["loss"].backward()',
  'print(json.dumps({"torch":torch.__version__,',
  ' "forward":{k:v.detach().tolist() for k,v in out.items()},',
  ' "grads":{k:v.grad.detach().tolist() for k,v in out.items() if v.grad is not None}},allow_nan=False))'
 ].join('\n');
 const output=execFileSync('python3',['-c',emitted.source+'\n'+py],{
  encoding:'utf8',timeout:90000,maxBuffer:8*1024*1024});
 const evidence=JSON.parse(output.trim().split('\n').at(-1));
 assert.equal(evidence.torch,'2.6.0+cpu');
 for(const step of original.steps){
  eq(step.values,evidence.forward[step.id],spec.id+'/'+step.id+'/forward',tol.forward);
  nForward+=ND.flatten(step.values).length;
  // GraphAutodiff publishes zero-filled arrays for inactive source nodes.
  // Only its actual visited pathIds are independently claimed gradients.
  if(reverse.pathIds.includes(step.id)){
   eq(reverse.gradients[step.id],evidence.grads[step.id],
     spec.id+'/'+step.id+'/backward',tol.backward);
   nBackward+=ND.flatten(reverse.gradients[step.id]).length;
  }
 }
 for(const node of graph.nodes.filter(x=>x.kind==='nd_parameter')){
  const grads=reverse.gradients[node.id],sh=ND.shape(node.params.values);
  assert.ok(grads,'missing trainable gradient '+node.id);
  ND.indices(sh,idx=>{
   const raw=ND.at(grads,idx),before=ND.at(node.params.values,idx);
   const plus=structuredClone(graph),minus=structuredClone(graph),eps=1e-6;
   let u=plus.nodes.find(n=>n.id===node.id).params.values;
   for(let i=0;i<idx.length-1;i++)u=u[idx[i]];
   u[idx.at(-1)]=before+eps;
   let v=minus.nodes.find(n=>n.id===node.id).params.values;
   for(let i=0;i<idx.length-1;i++)v=v[idx[i]];
   v[idx.at(-1)]=before-eps;
   const loss=g=>ND.flatten(L.evaluate(g).steps.find(s=>s.id==='loss').values)[0];
   const fd=(loss(plus)-loss(minus))/(2*eps);
   assert.ok(Number.isFinite(fd)&&Math.abs(fd-raw)<tol.finiteDifference,
    spec.id+'/'+node.id+JSON.stringify(idx)+' analytic='+raw+' FD='+fd);
   nFD++;
  });
 }
 // Proof of *aggregation*: bias and gain have singleton batch/head/token axes.
 // Their source VJP combines all contributing positions; it must equal
 // the sum of full-output gradients weighted by the corresponding local rule.
 const biased=original.steps.find(s=>s.id==='biased').values;
 const upstream=reverse.gradients.biased;
 const gateGradient=reverse.gradients.gated;
 const expectedBias=ND.build([1,1,1,spec.shape[3]],([b,h,t,c])=>{
  let total=0;
  for(let bi=0;bi<spec.shape[0];bi++)
   for(let hi=0;hi<spec.shape[1];hi++)
    for(let ti=0;ti<spec.shape[2];ti++)
     total+=ND.at(upstream,[bi,hi,ti,c]);
  return total;
 });
 const expectedGain=ND.build([1,1,1,spec.shape[3]],([b,h,t,c])=>{
  let total=0;
  for(let bi=0;bi<spec.shape[0];bi++)
   for(let hi=0;hi<spec.shape[1];hi++)
    for(let ti=0;ti<spec.shape[2];ti++)
     total+=ND.at(gateGradient,[bi,hi,ti,c])*ND.at(biased,[bi,hi,ti,c]);
  return total;
 });
 eq(reverse.gradients.bias,expectedBias,spec.id+'/broadcast-bias-aggregate',tol.backward);
 eq(reverse.gradients.gain,expectedGain,spec.id+'/broadcast-gain-aggregate',tol.backward);
 assert.equal(JSON.stringify(graph),pristine,'read-only fixture');
}
assert.equal(cases.length,2);
assert.ok(nForward>=30&&nBackward>=25&&nFD>=30);
console.log('INDEPENDENT_ND_BROADCAST_REDUCTION_PASS '+JSON.stringify({
 fixtures:cases.length,forwardCoordinates:nForward,backwardCoordinates:nBackward,
 independentFiniteDifferences:nFD,patterns:cases.map(x=>x.tags),torch:'2.6.0+cpu'
}));
