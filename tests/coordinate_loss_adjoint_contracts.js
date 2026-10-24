'use strict';
/* Distinguish d(selected scalar)/d(source) from dLoss/d(source).
 * Pure independent central differences use the actual graph forward interpreter,
 * never the backwards function or displayed microscope coordinate arithmetic. */
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),Bridge=require('../coordinate_vjp_bridge.js'),
 M=require('../compute_microscope.js'),Grad=require('../graph_autodiff.js'),ND=require('../nd_tensor.js');
const node=(id,kind,params={})=>({id,kind,params,x:0,y:0});
const edge=(from,to,port)=>({from,to,port});
const param=values=>({values,optimizer:{t:0,m:values.map(r=>r.map(()=>0)),v:values.map(r=>r.map(()=>0))}});
const base=L.validateGraph({schema:L.SCHEMA,title:'fanout loss VJP',nodes:[
 node('x','parameter',param([[.22,-.43],[.61,.74]])),
 node('w','parameter',param([[.32,.19],[-.22,.44]])),
 node('project','matmul'),node('scaled','scale',{factor:.4}),node('sum','add'),
 node('out','square'),node('target','parameter',param([[.12,-.1],[.3,.09]])),
 node('loss','mse')
],edges:[edge('x','project','a'),edge('w','project','b'),
 edge('project','scaled','x'),edge('project','sum','a'),edge('scaled','sum','b'),
 edge('sum','out','x'),edge('out','loss','prediction'),edge('target','loss','target')]});
const run=L.evaluate(base),frame=M.learning(base,run,L);
const fd=(graph,src,index,target='loss')=>{
 const step=sign=>{
  const raw=structuredClone(graph);
  let data=raw.nodes.find(n=>n.id===src).params.values;
  for(const i of index.slice(0,-1))data=data[i];
  data[index.at(-1)]+=sign*1e-6;
  return L.evaluate(raw).steps.find(s=>s.id===target).values[0][0];
 };
 return (step(1)-step(-1))/(2e-6);
};
const b=Grad.backward(base,run,{targetId:'loss'});
const reconstructed={x:[[0,0],[0,0]],w:[[0,0],[0,0]]};
for(let i=0;i<2;i++)for(let j=0;j<2;j++){
 const proof=Bridge.inspect(base,run,frame,{stepId:'project',coordinate:[i,j]});
 assert.equal(proof.backward.complete,true);
 const loss=proof.loss;
 assert.equal(loss.status,'VERIFIED_GRAPH_LOSS_VJP');
 assert.equal(loss.targetId,'loss');
 assert.equal(loss.completeSelectedPath,true);
 assert.equal(loss.throughSelectedEligible,true);
 assert.ok(Math.abs(loss.selectedAdjoint-ND.at(b.gradients.project,[i,j]))<1e-12);
 for(const c of loss.sourceContributions){
  assert.ok(Number.isFinite(c.throughSelected));
  reconstructed[c.nodeId][c.index[0]][c.index[1]]+=c.throughSelected;
  assert.ok(Math.abs(c.wholeLossGradient-ND.at(b.gradients[c.nodeId],c.index))<1e-12);
 }
 assert.equal(loss.cutEdges.filter(e=>e.from==='target').length,1);
}
// Chain rule: all project outputs form a complete cut between trainable inputs
// and loss, so summing the selected-output contributions must give the exact
// whole-loss parameter gradients, not just one selected output's gradient.
for(const id of ['x','w'])for(let i=0;i<2;i++)for(let j=0;j<2;j++){
 const exact=ND.at(b.gradients[id],[i,j]),sum=reconstructed[id][i][j];
 assert.ok(Math.abs(sum-exact)<2e-11,id+' lost shared-path gradient '+i+','+j);
 assert.ok(Math.abs(fd(base,id,[i,j])-exact)<2e-6,id+' independent FD '+i+','+j);
}
const atLoss=Bridge.inspect(base,run,frame,{stepId:'loss',coordinate:[0,0]});
assert.equal(atLoss.backward.complete,false,'loss is not fake scalar lineage');
assert.equal(atLoss.loss.status,'VERIFIED_GRAPH_LOSS_VJP');
assert.equal(atLoss.loss.selectedAdjoint,1);
assert.equal(atLoss.loss.cutEdges.some(e=>e.from==='target'),true);
assert.deepEqual(atLoss.loss.sourceContributions,[]);
// Two objective leaves: the target must be explicitly nominated when a graph
// has more than one valid scalar loss; no accidental selection by node position.
const two=L.validateGraph({...base,title:'two loss leaves, weighted sum',
 nodes:[...base.nodes,node('extra_target','parameter',param([[.01,.12],[-.17,.04]])),
  node('loss_extra','mse'),node('weighted','scale',{factor:.35}),node('total','add')],
 edges:[...base.edges,edge('out','loss_extra','prediction'),
  edge('extra_target','loss_extra','target'),edge('loss_extra','weighted','x'),
  edge('loss','total','a'),edge('weighted','total','b')]});
const twoRun=L.evaluate(two),twoFrame=M.learning(two,twoRun,L);
const t=Bridge.inspect(two,twoRun,twoFrame,{stepId:'out',coordinate:[0,1],lossId:'total'});
assert.equal(t.loss.status,'VERIFIED_GRAPH_LOSS_VJP');
assert.equal(t.loss.targetId,'total');
for(const c of t.loss.sourceContributions){
 assert.ok(Math.abs(c.wholeLossGradient-fd(two,c.nodeId,c.index,'total'))<3e-6);
}
assert.throws(()=>Bridge.inspect(two,twoRun,twoFrame,{stepId:'out',coordinate:[0,1],lossId:'out'}),
 /not a verified scalar training loss/);
const noLoss=L.preset('ffn'),nlRun=L.evaluate(noLoss),nlFrame=M.learning(noLoss,nlRun,L);
assert.equal(Bridge.inspect(noLoss,nlRun,nlFrame,{stepId:'act',coordinate:[0,1]}).loss.status,'NO_LOSS_OBJECTIVE');
// Unsupported *reachable* VJP remains a blocker, never a silently invented zero.
const unsupported=L.validateGraph({schema:L.SCHEMA,title:'reachable unsupported divide',nodes:[
 node('x','parameter',param([[.6,.7],[.9,1.2]])),
 node('w','parameter',param([[1.2,.8],[.6,2.1]])),
 node('division','divide'),node('target','parameter',param([[.1,.3],[.2,.4]])),
 node('loss','mse')
],edges:[edge('x','division','a'),edge('w','division','b'),
 edge('division','loss','prediction'),edge('target','loss','target')]});
const uRun=L.evaluate(unsupported),uFrame=M.learning(unsupported,uRun,L);
const u=Bridge.inspect(unsupported,uRun,uFrame,{stepId:'division',coordinate:[0,0]});
assert.equal(u.backward.complete,true,'scalar derivative of division is available');
assert.equal(u.loss.status,'UNAVAILABLE_GRAPH_LOSS_VJP');
assert.match(u.loss.reason,/unsupported ancestor divide/);
assert.equal(u.loss.selectedAdjoint,undefined);
const det=L.validateGraph({schema:L.SCHEMA,title:'detach semantics',nodes:[
 node('x','parameter',param([[.1,.4],[.8,.2]])),node('stop','detach_demo'),
 node('target','parameter',param([[.05,.02],[.2,.3]])),node('loss','mse')
],edges:[edge('x','stop','x'),edge('stop','loss','prediction'),edge('target','loss','target')]});
const detRun=L.evaluate(det),detFrame=M.learning(det,detRun,L);
const stopped=Bridge.inspect(det,detRun,detFrame,{stepId:'stop',coordinate:[0,0]});
assert.equal(stopped.loss.status,'VERIFIED_GRAPH_LOSS_VJP');
assert.equal(stopped.loss.cutEdges.some(e=>e.from==='x'),true);
assert.equal(stopped.loss.throughSelectedEligible,false);
assert.equal(ND.at(Grad.backward(det,detRun).gradients.x,[0,0]),0);
console.log('COORDINATE_LOSS_ADJOINT_PASS: full-loss vs seeded VJP, fan-out conservation, finite differences, weighted losses, target stops, Detach, unsupported operators');
