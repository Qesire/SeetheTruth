'use strict';
const assert=require('node:assert/strict');
const Ops=require('../deepinfer_runtime_ops.js'),L=require('../learning_lab.js');
const C=require('../concept_flow.js'),R=require('../graph_readiness.js'),K=require('../block_knowledge.js');
assert.equal(Ops.KINDS.length,3);
for(const kind of Ops.KINDS){
 assert.ok(L.KINDS.includes(kind),kind+' registered');
 assert.ok(K.get(kind)?.refs.length,kind+' source anchor');
}
const remap=[[0],[1],[-1]],rates=[[0.9,0.8],[0.5,0.5]];
const budget=Ops.execute('deepinfer_mtp_confidence_schedule',
 {remap,prev_confidence:rates},{totalQ:5});
assert.deepEqual(budget,[[3],[1],[1]],'global cumprod rank budget matches official reference');
assert.deepEqual(Ops.schedule(remap,rates,3),[[1],[1],[1]],'base one Q each');
assert.deepEqual(Ops.schedule([[0],[0]],[[1,1]],4),[[3],[1]],'stable flat index ties');
assert.deepEqual(Ops.schedule([[-1],[0]],[[.2,.1]],3),[[1],[2]],'remap=-1 rates zeros');
assert.throws(()=>Ops.execute('deepinfer_mtp_confidence_schedule',
 {remap:[[3]],prev_confidence:rates},{totalQ:2}),/remap/);
assert.throws(()=>Ops.execute('deepinfer_mtp_confidence_schedule',
 {remap:[[0]],prev_confidence:[[-.1,.2]]},{totalQ:2}),/confidence/);
assert.throws(()=>Ops.shape('deepinfer_mtp_confidence_schedule',
 {remap:[3,1],prev_confidence:[2,2]},{totalQ:10}),/budget/);
const p={window:2,keepTokens:3,backtracking:1};
const q=[[1,0],[0,1],[1,1],[2,0]],k=[[1,0],[0,1],[1,1],[1,2]],v=[[2,0],[0,3],[4,1],[5,2]];
const inp={q,k,v};
const mask=Ops.execute('deepinfer_yoco_swa_mask',inp,p);
const compact=Ops.execute('deepinfer_yoco_swa_compact',inp,p);
assert.deepEqual(Ops.shape('deepinfer_yoco_swa_mask',{q:[4,2],k:[4,2],v:[4,2]},p),[4,2]);
assert.deepEqual(Ops.shape('deepinfer_yoco_swa_compact',{q:[4,2],k:[4,2],v:[4,2]},p),[3,2]);
assert.deepEqual(mask[0],[0,0],'masked prefix sentinel is zero (not full model output)');
for(let i=0;i<3;i++)for(let j=0;j<2;j++)
 assert.ok(Math.abs(mask[i+1][j]-compact[i][j])<1e-14,'suffix equivalence at '+i+','+j);
assert.deepEqual(Ops.report(4,p),{
 totalQueryRows:4,retainedQueryRows:3,removedQueryRows:1,
 maskQuerySlots:4,compactQuerySlots:3,eligibleQKPairs:6,
 note:'mask reserves all Q rows but masked prefix is not semantically reused; compact omits prefix rows; no measured GPU FLOPs'
});
assert.throws(()=>Ops.shape('deepinfer_yoco_swa_compact',{q:[4,2],k:[4,2],v:[4,2]},
 {window:2,keepTokens:2,backtracking:1}),/keepTokens/);
assert.throws(()=>Ops.shape('deepinfer_yoco_swa_mask',{q:[4,2],k:[3,2],v:[3,2]},p),/aligned/);
for(const name of ['deepinfer_mtp_budget','deepinfer_yoco_replay']){
 const graph=L.preset(name),numerical=L.evaluate(graph),symbolic=C.evaluate(graph),ready=R.check(graph);
 assert.equal(ready.executable,true,name+' should be runnable');
 assert.equal(numerical.steps.length,graph.nodes.length,name+' all outputs');
 for(const step of numerical.steps){
  assert.deepEqual(step.shape,symbolic.steps.find(x=>x.id===step.id)?.shape,name+'/'+step.id);
  assert.ok(step.values.flat(Infinity).every(Number.isFinite));
 }
 if(name==='deepinfer_mtp_budget')
  assert.deepEqual(numerical.steps.find(x=>x.id==='mtp').values,[[3],[1],[1]]);
 else{
  const a=numerical.steps.find(x=>x.id==='mask').values,b=numerical.steps.find(x=>x.id==='compact').values;
  assert.deepEqual(a.slice(-b.length),b);
 }
}
console.log('DEEPINFER_RUNTIME_CONTRACTS_PASS reference MTP cumprod+stable budget, YoCo mask/compact suffix, math and sources');
