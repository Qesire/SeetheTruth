'use strict';
const assert=require('node:assert/strict');
const Core=require('../learning_experience_core.js');
const Worker=require('../deepinfer_worker_events.js');
const Scheduler=require('../deepinfer_scheduler_workflow.js');

function nativePair(domain,operation,inputA,inputB){
 const A=Core.executeLab(domain,operation,inputA);
 const B=Core.executeLab(domain,operation,inputB);
 assert.ok(A.stateTrace&&B.stateTrace,domain+' must provide native state lineage');
 assert.equal(Core.replayLab(A).reproduced,true);
 assert.equal(Core.replayLab(B).reproduced,true);
 const diff=Core.compareRecords(A,B);
 assert.equal(diff.replayVerified,true);
 assert.equal(diff.inputEqual,false);
 return {A,B,diff};
}
const operations=[{op:'arrive',id:'A',tokenLen:5},{op:'allocate',id:'A'}];
const kv=nativePair('deepinfer_kv','replay',
 {config:{numBlocks:6,blockSize:4},operations},
 {config:{numBlocks:7,blockSize:4},operations});
assert.ok(kv.diff.changedSteps.some(x=>x.kind==='state_transition'));
assert.equal(kv.A.value.timeline.at(-1).numBlocks,6);
assert.equal(kv.B.value.timeline.at(-1).numBlocks,7);
assert.deepEqual(kv.A.inputSnapshot.operations,kv.B.inputSnapshot.operations);

const scheduler=nativePair('deepinfer_scheduler','simulate',
 {spec:Scheduler.EXAMPLES.ample,options:{stopPreempted:false}},
 {spec:Scheduler.EXAMPLES.pressure,options:{stopPreempted:false}});
assert.ok(scheduler.diff.changedSteps.some(x=>x.kind==='state_transition'));
assert.ok(scheduler.B.value.executionProof.every(x=>typeof x.forwardEvent==='string'));
const inspect=Core.inspectObservationStep(scheduler.B,{
 kind:'state_transition',stepId:scheduler.B.stateTrace.steps[1].id});
assert.equal(inspect.replayVerified,true);

const steps=Worker.DEFAULT_STEPS.map(x=>({mode:x.mode,sequenceIds:[...x.sequenceIds]}));
const worker=nativePair('deepinfer_worker','simulate',
 {steps,options:{zeroOverhead:true}},
 {steps,options:{zeroOverhead:true,durations:{forward:Worker.DEFAULT_DURATIONS.forward+2}}});
assert.ok(worker.diff.changedSteps.some(x=>x.kind==='state_transition'));
assert.ok(worker.B.value.makespan>=worker.A.value.makespan);
assert.equal(worker.A.evidence,'ABSTRACT_TIMELINE');

// Explicitly setting the documented default is a legitimate distinct input
// with an unchanged output. Such negative findings must remain representable.
const control=nativePair('deepinfer_worker','simulate',
 {steps,options:{zeroOverhead:true}},
 {steps,options:{zeroOverhead:true,durations:{forward:Worker.DEFAULT_DURATIONS.forward}}});
assert.equal(control.diff.inputEqual,false);
assert.equal(control.diff.valueEqual,true);
assert.equal(control.diff.stateTraceEqual,true);
assert.deepEqual(control.diff.changedSteps,[]);
console.log('NATIVE_AB_GUIDE_PASS 3 engines and one replay-verified no-effect comparison');

const Handoff=require('../learning_handoff.js');
for(const [name,pair,target] of [
 ['deepinfer_kv',kv,'decode'],['deepinfer_scheduler',scheduler,'decode'],
 ['deepinfer_worker',worker,'attention'],['deepinfer_worker',control,'attention']
]){
 const selected=pair.B.stateTrace.steps[1].id;
 const handoff=Handoff.create(Core,pair.B,pair.A,selected,
  '哪些真实源事件由一个原生输入变化引起？');
 const storage=new Map();
 Handoff.store({setItem:(key,val)=>storage.set(key,val)},handoff);
 const imported=Handoff.load({getItem:key=>storage.get(key)});
 const restored=Handoff.restore(Core,imported);
 assert.equal(restored.domain,name);
 assert.equal(restored.course,target);
 assert.equal(restored.selectedStepId,selected);
 assert.equal(restored.current.checkpoint.stepId,selected);
 assert.equal(restored.comparison.replayVerified,true);
 assert.equal(restored.comparison.inputEqual,false);
 assert.equal(restored.baseline.record.evidence,pair.A.evidence);
 const changed=JSON.parse(JSON.stringify(imported));
 changed.active.cursor.stateDigest='fnv1a32:00000000:1';
 assert.throws(()=>Handoff.restore(Core,changed),/replay differs|cursor/);
 const mismatched=JSON.parse(JSON.stringify(imported));
 mismatched.domain=name==='deepinfer_kv'?'deepinfer_worker':'deepinfer_kv';
 assert.throws(()=>Handoff.restore(Core,mismatched),/mismatch/);
 assert.throws(()=>Handoff.load({getItem:()=>'{invalid'}),/invalid cross-page JSON/);
}
const single=Handoff.create(Core,worker.B,null,null,'单次实验进入计算图');
assert.equal(Handoff.restore(Core,single).baseline,null);
assert.equal(Handoff.restore(Core,single).current.record.id,'deepinfer_worker');
console.log('NATIVE_GRAPH_HANDOFF_PASS replayed A/B, negative/no-effect, cursor tamper, source mismatch');

const Learning=require('../learning_lab.js');
for(const [pair,name,course,focus] of [
 [kv,'deepinfer_kv','decode','gqa'],
 [scheduler,'deepinfer_scheduler','decode','gqa'],
 [worker,'deepinfer_worker','attention','attn'],
 [control,'deepinfer_worker','attention','attn']
]){
 const packet=Handoff.create(Core,pair.B,pair.A,pair.B.stateTrace.steps[1].id);
 const graph=Learning.preset(course);
 const staticProof=Handoff.explain(Core,packet,graph);
 assert.equal(staticProof.status,'REPLAYED_NATIVE_AND_GRAPH_STATIC');
 assert.equal(staticProof.graph.nodeId,focus);
 assert.equal(staticProof.graph.numeric,null);
 assert.deepEqual(staticProof.graph.ports.map(x=>x.port),['q','k','v']);
 assert.ok(staticProof.graph.ports.every(x=>typeof x.sourceId==='string'&&x.sourceShape.length>=1));
 assert.equal(staticProof.native.stepId,pair.B.stateTrace.steps[1].id);
 assert.equal(staticProof.native.selectedStepChangedInAB,
  pair.diff.changedSteps.some(x=>x.kind==='state_transition'&&x.stepId===staticProof.native.stepId));
 const numeric=Core.inspectGraph(graph,{id:course,numeric:true});
 assert.equal(numeric.numericOk,true);
 const computed=Handoff.explain(Core,packet,graph,numeric);
 assert.equal(computed.status,'REPLAYED_NATIVE_AND_GRAPH_NUMERIC');
 assert.equal(computed.graph.numeric.observedNode,focus);
 assert.ok(Array.isArray(computed.graph.numeric.sample));
 assert.throws(()=>Handoff.explain(Core,packet,graph,
  Core.inspectGraph(Learning.preset('ffn'),{id:'ffn',numeric:true})),/different or non-executed graph/);
 const edited=JSON.parse(JSON.stringify(graph));
 edited.nodes=edited.nodes.filter(n=>n.id!==focus);
 assert.throws(()=>Handoff.explain(Core,packet,edited),'missing target node must fail source-graph mechanism proof');
}
console.log('NATIVE_GRAPH_MECHANISM_PASS real Q/K/V port provenance, numeric replay, edited graph and stale numeric rejection');

for(const [pair,course,kind,heads] of [
 [worker,'attention','attn',1],
 [kv,'decode','gqa',2],
 [scheduler,'decode','gqa',2]
]){
 const graph=Learning.preset(course);
 const record=Core.inspectGraph(graph,{id:course,numeric:true});
 const packet=Handoff.create(Core,pair.B,pair.A,pair.B.stateTrace.steps[1].id);
 const first=Handoff.coordinate(Core,packet,graph,record);
 assert.equal(first.nodeId,kind);
 assert.equal(first.numericalEvidence,'SYNTHETIC_NUMERIC_REPLAY_VERIFIED');
 assert.equal(first.headCount,heads);
 assert.ok(Math.abs(first.probabilitySum-1)<1e-10);
 assert.ok(first.error<1e-10);
 assert.ok(first.originalSubstep.replayVerified);
 assert.ok(first.keyCount>=3);
 assert.equal(first.terms.length,first.keyCount);
 assert.ok(first.terms.every(x=>Number.isFinite(x.score)&&Number.isFinite(x.contribution)));
 for(const head of [0,heads-1]){
  const result=Handoff.coordinate(Core,packet,graph,record,
   {queryToken:first.outputShape[0]-1,head,
    channel:first.outputShape[1]/heads-1});
  assert.equal(result.queryHead,head);
  assert.equal(result.outputChannel,head*(first.outputShape[1]/heads)+first.outputShape[1]/heads-1);
  assert.ok(result.error<1e-10);
  if(kind==='gqa')assert.ok(result.terms.some(x=>x.masked===false));
 }
 if(kind==='gqa'){
  const observed=Handoff.coordinate(Core,packet,graph,record,{queryToken:0,head:0,channel:0});
  assert.ok(observed.terms.every(x=>!x.masked || x.probability===0));
  assert.ok(observed.terms.every(x=>!x.masked),
   'single final decode query may attend every key in the supplied prefix');
  assert.equal(observed.kvHead,0);
 }
 assert.throws(()=>Handoff.coordinate(Core,packet,graph,record,{queryToken:999}),/out of bounds/);
 assert.throws(()=>Handoff.coordinate(Core,packet,graph,record,{head:heads}),/out of bounds/);
 assert.throws(()=>Handoff.coordinate(Core,packet,graph,record,{channel:-1}),/out of bounds/);
 assert.throws(()=>Handoff.coordinate(Core,packet,graph,record,{channel:0.5}),/out of bounds/);
 assert.throws(()=>Handoff.coordinate(Core,packet,graph,
  Core.inspectGraph(Learning.preset('ffn'),{numeric:true,id:'ffn'})),/different or non-executed graph/);
}
// A multi-query causal GQA fixture actually excludes future keys for early queries.
const causalGraph=Learning.preset('gqa');
const causalRun=Core.inspectGraph(causalGraph,{id:'decode',numeric:true});
const causalPacket=Handoff.create(Core,kv.B,kv.A,kv.B.stateTrace.steps[1].id);
const maskedProof=Handoff.coordinate(Core,causalPacket,causalGraph,causalRun,
 {queryToken:0,head:1,channel:0});
assert.ok(maskedProof.terms.some(x=>x.masked&&x.probability===0));
assert.ok(maskedProof.terms.some(x=>!x.masked&&x.probability>0));
assert.ok(maskedProof.error<1e-10);
console.log('NATIVE_GRAPH_COORDINATE_PASS full original P×V evidence, attention/GQA, head sharing, causal mask, bounds and stale-graph rejection');

for(const [pair,course,expectRoPE,query,key,head] of [
 [worker,'attention',false,1,2,0],
 [kv,'decode',true,0,2,1],
 [scheduler,'decode',true,0,1,0]
]){
 const graph=Learning.preset(course),numeric=Core.inspectGraph(graph,{id:course,numeric:true});
 const packet=Handoff.create(Core,pair.B,pair.A,pair.B.stateTrace.steps[1].id);
 const proof=Handoff.score(Core,packet,graph,numeric,{
  queryToken:query,keyToken:key,head,channel:0});
 assert.equal(proof.rotary,expectRoPE);
 assert.equal(proof.keyToken,key);assert.equal(proof.queryToken,query);
 assert.equal(proof.features.length,proof.headWidth);
 assert.ok(proof.scoreError<1e-10);
 assert.ok(proof.softmax.error<1e-10);
 assert.ok(Math.abs(proof.scale-1/Math.sqrt(proof.headWidth))<1e-12);
 assert.ok(proof.features.every(x=>Number.isFinite(x.scaledContribution)));
 if(expectRoPE){
  assert.equal(proof.queryPositionOffset,2);
  assert.ok(proof.features.some(x=>Math.abs(x.kAngle)>0));
  assert.ok(proof.features.some(x=>Math.abs(x.rawK-x.positionedK)>1e-7));
 }else{
  assert.equal(proof.queryPositionOffset,0);
  assert.ok(proof.features.every(x=>x.rawQ===x.positionedQ&&x.rawK===x.positionedK));
 }
 assert.throws(()=>Handoff.score(Core,packet,graph,numeric,{keyToken:999}),/key token out of bounds/);
 assert.throws(()=>Handoff.score(Core,packet,graph,numeric,{keyToken:0.25}),/key token out of bounds/);
 assert.throws(()=>Handoff.score(Core,packet,graph,numeric,{
  queryToken:999,keyToken:0}),/query token out of bounds/);
}
const causalScore=Handoff.score(Core,causalPacket,causalGraph,causalRun,
 {queryToken:0,head:1,keyToken:2,channel:0});
assert.equal(causalScore.rotary,true);
assert.equal(causalScore.masked,true);
assert.equal(causalScore.probability,0);
assert.equal(causalScore.softmax.numerator,0);
assert.equal(causalScore.softmax.recomputed,0);
assert.ok(Number.isFinite(causalScore.sourceScore)); // raw score is real even if key is masked
assert.ok(causalScore.scoreError<1e-10);
const earlyScore=Handoff.score(Core,causalPacket,causalGraph,causalRun,
 {queryToken:0,head:1,keyToken:0,channel:0});
assert.equal(earlyScore.masked,false);
assert.ok(earlyScore.probability>0);
console.log('NATIVE_GRAPH_SCORE_PASS original Q/K rotation pair, scaled feature dot, masked stable softmax and provenance');

for(const [pair,course,port,sourceToken,feature,queryToken,keyToken,head] of [
 [worker,'attention','q',1,0,1,1,0],
 [worker,'attention','k',1,1,0,1,0],
 [kv,'decode','q',0,5,0,1,1],
 [scheduler,'decode','k',1,1,0,1,0]
]){
 const graph=Learning.preset(course),record=Core.inspectGraph(graph,{id:course,numeric:true});
 const packet=Handoff.create(Core,pair.B,pair.A,pair.B.stateTrace.steps[1].id);
 const experiment=Handoff.intervene(Core,packet,graph,record,{
  port,sourceToken,feature,delta:0.125,queryToken,keyToken,head,channel:0
 });
 assert.equal(experiment.schema,'seethetruth-coordinate-intervention-v1');
 assert.equal(experiment.source.port,port);
 assert.equal(experiment.source.token,sourceToken);
 assert.equal(experiment.source.feature,feature);
 assert.equal(experiment.source.after-experiment.source.before,0.125);
 assert.equal(experiment.graphComparison.replayVerified,true);
 assert.equal(experiment.graphComparison.inputEqual,false);
 assert.equal(experiment.graphComparison.numericCompared,true);
 assert.equal(experiment.baselineVerifiedSteps,record.session.trace.steps.length);
 assert.equal(experiment.changedInputScalars,1);
 const path=experiment.propagation;
 assert.equal(path.schema,'seethetruth-attention-causal-propagation-v1');
 assert.equal(path.observed,'REPLAYED_GRAPH_PARAMETER_AB');
 assert.equal(path.nativeEventId,experiment.sourceEventId);
 assert.equal(path.graphSteps.length,experiment.baselineVerifiedSteps+1);
 assert.equal(path.graphSteps.filter(x=>x.id===experiment.source.parameter)[0].changedCoordinates,1);
 assert.equal(path.keys.length,3);
 assert.ok(path.featureTerms.length>=2);
 assert.ok(path.graphSteps.some(x=>x.id===(course==='decode'?'gqa':'attn') && x.changed));
 assert.ok(path.graphSteps.some(x=>x.id===experiment.source.from && !x.changed));
 assert.ok(path.graphSteps.some(x=>x.id!==(course==='decode'?'gqa':'attn')&&x.id!==experiment.source.parameter&&!x.changed));
 assert.ok(Math.abs(path.score.delta-experiment.score.delta)<1e-12);
 assert.ok(Math.abs(path.probability.delta-experiment.probability.delta)<1e-12);
 assert.ok(Math.abs(path.output.delta-experiment.output.delta)<1e-12);
 assert.ok(Math.abs(path.row.reconstructedOutputDelta-experiment.output.delta)<1e-10);
 assert.ok(Math.abs(path.row.probabilitySumDelta)<1e-10);
 assert.ok(Math.abs(path.featureTerms.reduce((sum,x)=>sum+x.dot.delta,0)-experiment.score.delta)<1e-10);
 assert.ok(path.keys.every(x=>Number.isFinite(x.score.delta)&&Number.isFinite(x.probability.delta)&&Number.isFinite(x.contribution.delta)));
 assert.ok(path.graphSteps.every(x=>x.examples.length<=6));
 assert.equal(path.changedOutsideCausalCone,0);
 assert.ok(path.causalReachableCount>=2);
 assert.ok(path.graphSteps.filter(x=>x.changed).every(x=>x.reachable));
 assert.ok(path.graphSteps.some(x=>!x.reachable&&!x.changed));
 assert.ok(path.graphSteps.find(x=>x.id===(course==='decode'?'gqa':'attn')).incoming
  .some(x=>x.from===experiment.source.parameter&&x.port===port&&x.sourceChanged));
 assert.ok(path.graphSteps.every(step=>step.incoming.every(edge=>
  typeof edge.port==='string'&&path.graphSteps.some(s=>s.id===edge.from))));


 assert.equal(experiment.sourceEquivalent.maxAbsolute,0);
 assert.equal(experiment.sourceEquivalent.changedCoordinates,0);
 assert.ok(experiment.graphComparison.changedSteps.some(x=>x.stepId===experiment.source.parameter));
 assert.ok(experiment.outputChange.maxAbsolute>0);
 assert.ok(Math.abs(experiment.score.delta)>1e-12);
 assert.ok(Math.abs(experiment.probability.delta)>1e-12);
 assert.ok(Number.isFinite(experiment.output.delta));
 assert.notEqual(experiment.source.after,experiment.source.before);
 assert.throws(()=>Handoff.intervene(Core,packet,graph,record,{
  port:'v',sourceToken,feature,delta:0.1}),/must be Q or K/);
 assert.throws(()=>Handoff.intervene(Core,packet,graph,record,{
  port,sourceToken:999,feature,delta:0.1}),/source token coordinate out of bounds/);
 assert.throws(()=>Handoff.intervene(Core,packet,graph,record,{
  port,sourceToken,feature:999,delta:0.1}),/source feature coordinate out of bounds/);
 assert.throws(()=>Handoff.intervene(Core,packet,graph,record,{
  port,sourceToken,feature,delta:0}),/nonzero/);
 assert.throws(()=>Handoff.intervene(Core,packet,graph,record,{
  port,sourceToken,feature,delta:Infinity}),/finite/);
 assert.throws(()=>Handoff.intervene(Core,packet,graph,
  Core.inspectGraph(Learning.preset('ffn'),{id:'ffn',numeric:true}),{
   port,sourceToken,feature,delta:0.1}),/different or non-executed graph/);
}
// The same Q feature belongs to Hq=0; asking about Hq=1 should yield
// zero *selected-head* score/probability/output changes, even though another
// head's overall output changes. A meaningful localization control.
const headLocal=Handoff.intervene(Core,
 Handoff.create(Core,kv.B,kv.A,kv.B.stateTrace.steps[1].id),
 Learning.preset('decode'),
 Core.inspectGraph(Learning.preset('decode'),{id:'decode',numeric:true}),{
 port:'q',sourceToken:0,feature:2,delta:0.125,
 queryToken:0,keyToken:1,head:1,channel:0
});
assert.equal(headLocal.score.delta,0);
assert.equal(headLocal.probability.delta,0);
assert.equal(headLocal.output.delta,0);
assert.ok(headLocal.outputChange.maxAbsolute>0);
assert.equal(headLocal.propagation.selectedHeadIsolated,true);
assert.equal(headLocal.propagation.score.changed,false);
assert.equal(headLocal.propagation.probability.changed,false);
assert.equal(headLocal.propagation.output.changed,false);
assert.ok(headLocal.propagation.graphSteps.find(x=>x.id==='gqa').changed);
assert.ok(headLocal.propagation.keys.every(x=>!x.probability.changed));

// A future-K score can change while its causal-masked probability and
// forward output remain unchanged: a necessary negative control.
const negative=Handoff.intervene(Core,causalPacket,causalGraph,causalRun,{
 port:'k',sourceToken:2,feature:0,delta:0.25,queryToken:0,keyToken:2,head:1,channel:0
});
assert.ok(Math.abs(negative.score.delta)>1e-12);
assert.equal(negative.probability.before,0);
assert.equal(negative.probability.after,0);
assert.equal(negative.output.delta,0);
assert.ok(negative.outputChange.changedCoordinates>0,
 'later queries can observe a key masked for the selected early query');
assert.equal(negative.sourceEquivalent.maxAbsolute,0);
assert.equal(negative.changedInputScalars,1);
assert.equal(negative.propagation.maskedSelectedKey,true);
assert.equal(negative.propagation.selectedKeySuppressed,true);
assert.equal(negative.propagation.keys[2].score.changed,true);
assert.equal(negative.propagation.keys[2].probability.changed,false);
assert.equal(negative.propagation.keys[2].contribution.changed,false);
assert.equal(negative.propagation.row.probabilityChangedKeys,0);
assert.equal(negative.propagation.row.contributionChangedKeys,0);
assert.ok(negative.propagation.graphSteps.find(x=>x.id==='gqa').changed);
assert.equal(negative.propagation.changedOutsideCausalCone,0);
assert.equal(negative.propagation.output.delta,0);

console.log('NATIVE_GRAPH_INTERVENTION_PASS replayed one-element Q/K A/B, numeric cascade, masked negative, provenance and bounds');




console.log('NATIVE_GRAPH_PROPAGATION_PASS full-DAG stage deltas, per-feature Score/PV conservation, head isolation and causal-mask suppression');
