'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),M=require('../compute_microscope.js'),
 E=require('../element_lineage.js'),Bridge=require('../coordinate_vjp_bridge.js'),
 Cap=require('../block_capabilities.js'),ND=require('../nd_tensor.js');
const node=(id,kind,params={})=>({id,kind,params,x:20,y:30});
const edge=(from,to,port)=>({from,to,port});
const params=(values)=>({values,optimizer:{kind:'adamw',t:0,
 m:values.map(row=>row.map(()=>0)),v:values.map(row=>row.map(()=>0))}});
const g=L.validateGraph({schema:L.SCHEMA,title:'shared matrix two-route VJP',nodes:[
 node('x','parameter',params([[.25,-.4],[.6,.7]])),
 node('w','parameter',params([[.32,.19],[-.22,.44]])),
 node('project','matmul'),node('boost','scale',{factor:.4}),node('sum','add'),node('out','square')
],edges:[edge('x','project','a'),edge('w','project','b'),
 edge('project','boost','x'),edge('project','sum','a'),edge('boost','sum','b'),edge('sum','out','x')]});
const run=L.evaluate(g),frame=M.learning(g,run,L);
const result=Bridge.inspect(g,run,frame,{stepId:'out',coordinate:[0,1]});
assert.equal(result.schema,Bridge.SCHEMA);
assert.equal(result.backward.complete,true);
assert.equal(result.backward.boundaryNodes.length,0);
assert.ok(result.backward.edgeCount>=8,'shared source fans back through two routes');
assert.ok(result.selected.forwardReconstructable);
assert.equal(result.selected.kind,'square');
assert.ok(result.selected.forwardError<1e-11);
assert.equal(result.codegen.wholeGraphExport,'NOT_VERIFIED');
assert.equal(result.codegen.vjpExport,'NOT_CLAIMED');
const fromAudit=Cap.audit().rows.find(r=>r.kind==='square');
assert.equal(result.codegen.numpyOperatorEligible,fromAudit.numpyEligible);
assert.equal(result.codegen.torchOperatorEligible,fromAudit.torchEligible);
const outputAt=(graph,id,index)=>ND.at(L.evaluate(graph).steps.find(s=>s.id===id).values,index);
const h=1e-6;
const gradFD=(original,sourceId,idx,finalId,finalIdx)=>{
 const side=(sign)=>{
  const clone=structuredClone(original);
  let data=clone.nodes.find(n=>n.id===sourceId).params.values;
  for(const v of idx.slice(0,-1))data=data[v];
  data[idx.at(-1)]+=sign*h;
  return outputAt(clone,finalId,finalIdx);
 };
 return (side(1)-side(-1))/(2*h);
};
for(const p of result.backward.sources){
 assert.ok(['x','w'].includes(p.nodeId));
 const fd=gradFD(g,p.nodeId,p.index,'out',[0,1]);
 assert.ok(Math.abs(p.sensitivity-fd)<2e-6,
  'fanout gradient mismatch '+p.nodeId+' '+p.index+' '+p.sensitivity+' '+fd);
}
const direct=E.inspect(g,{targetId:'out',outputIndex:[0,1],trace:run});
assert.equal(result.backward.nodeCount,direct.vertices.length);
const native=L.validateGraph({schema:L.SCHEMA,title:'fixed Linear with traceable input derivative',nodes:[
 node('x','parameter',params([[.3,-.4,.1],[.6,.2,-.8]])),
 node('project','linear',{out:2}),node('act','silu')
],edges:[edge('x','project','x'),edge('project','act','x')]});
const nativeRun=L.evaluate(native),nativeFrame=M.learning(native,nativeRun,L);
const linear=Bridge.inspect(native,nativeRun,nativeFrame,{stepId:'project',coordinate:[1,1]});
assert.equal(linear.backward.complete,true,'native Linear cannot remain an artificial VJP boundary');
assert.equal(linear.backward.localEdges.length,3);
assert.equal(linear.backward.sources.length,3);
for(const s of linear.backward.sources){
 const fd=gradFD(native,'x',s.index,'project',[1,1]);
 assert.ok(Math.abs(s.sensitivity-fd)<2e-6,'fixed Linear local derivative '+s.index);
}
const activated=Bridge.inspect(native,nativeRun,nativeFrame,{stepId:'act',coordinate:[1,1]});
assert.equal(activated.backward.complete,true);
for(const s of activated.backward.sources)
 assert.ok(Math.abs(s.sensitivity-gradFD(native,'x',s.index,'act',[1,1]))<2e-6,
  'activation ∘ Linear derivative '+s.index);
const staticFrame=M.learning(native,require('../concept_flow.js').evaluate(native),L);
assert.throws(()=>Bridge.inspect(native,require('../concept_flow.js').evaluate(native),staticFrame,
 {stepId:'project',coordinate:[1,1]}),/only current bounded numerical execution/);
assert.throws(()=>Bridge.inspect(native,nativeRun,nativeFrame,{stepId:'project',coordinate:[10,1]}),
 /coordinate out of bounds/);
assert.throws(()=>Bridge.inspect(native,nativeRun,nativeFrame,{stepId:'project:fake',coordinate:[0,1]}),
 /selected graph node unavailable/);
const altered=structuredClone(nativeFrame);
altered.steps.find(s=>s.id==='project').out.values[0]+=.4;
assert.throws(()=>Bridge.inspect(native,nativeRun,altered,{stepId:'project',coordinate:[1,1]}),
 /microscope frame differs/);
const edited=structuredClone(native);
edited.nodes.find(n=>n.id==='x').params.values[1][0]+=.1;
assert.throws(()=>Bridge.inspect(edited,nativeRun,nativeFrame,{stepId:'project',coordinate:[1,1]}),
 /trace does not match current graph/);
// Rank-five parent projections are valid selections even when a specialist
// operator's internal VJP is not registered: never fake a gradient.
const video=L.preset('extension_video_time_composition'),vr=L.evaluate(video);
const vf=M.learning(video,vr,L);
const vrpt=Bridge.inspect(video,vr,vf,{stepId:'join',coordinate:[0,2,1,0,1]});
assert.deepEqual(vrpt.selected.index,[0,2,1,0,1]);
assert.equal(vrpt.selected.value,ND.at(vr.steps.find(s=>s.id==='join').values,[0,2,1,0,1]));
assert.equal(vrpt.codegen.wholeGraphExport,'NOT_VERIFIED');
const unknown=L.validateGraph({schema:L.SCHEMA,title:'unsupported VJP boundary',nodes:[
 node('x','parameter',params([[.3,.7],[.4,-.3]])),
 node('y','parameter',params([[.1,.2],[.8,-.5]])),node('loss','mse')
],edges:[edge('x','loss','prediction'),edge('y','loss','target')]});
const lossRun=L.evaluate(unknown),lossFrame=M.learning(unknown,lossRun,L);
const boundary=Bridge.inspect(unknown,lossRun,lossFrame,{stepId:'loss',coordinate:[0,0]});
assert.equal(boundary.backward.complete,false);
assert.deepEqual(boundary.backward.boundaryNodes,['loss@0,0']);
assert.equal(boundary.backward.sources.length,0);
console.log('COORDINATE_VJP_BRIDGE_PASS: verified forward + scalar VJP + fanout gradient accumulation + Linear and activation FD + source/capability boundaries');
