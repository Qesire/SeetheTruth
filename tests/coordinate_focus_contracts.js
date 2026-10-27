"use strict";
const assert=require("node:assert/strict");
const Focus=require("../coordinate_focus.js");
const Examples=require("../semantic_state_examples.js");
const Bridge=require("../state_learning_handoff.js");
const tiers=Focus.TIERS;
assert.equal(Focus.SCHEMA,"seethetruth-coordinate-focus-v1");
for(const name of Examples.names){
 const result=Examples.run(name);
 for(const branch of ["a","b"])for(const i of Bridge.coordinateList(name,result,branch)){
  const source=Bridge.coordinate(name,result,branch,i),focus=Focus.fromState(source);
  assert.equal(focus.executionId,"semantic_state/scenario/"+name);
  assert.equal(focus.flatIndex,i);
  assert.deepEqual(Focus.unflatten(focus.shape,i),source.coordinate);
  assert.equal(Focus.flatten(focus.shape,focus.coordinate),i);
  assert.equal(focus.value,source.value);
  assert.equal(focus.evidence,tiers.STATE);
  assert.ok(Object.isFrozen(focus.coordinate));
 }
}
const axis=Examples.run("axis"),a=Focus.fromState(Bridge.coordinate("axis",axis,"a",4));
assert.throws(()=>Focus.create({...a,coordinate:[2,1]}),/outside tensor shape/);
assert.throws(()=>Focus.create({...a,shape:[1,3]}),/outside tensor shape/);
assert.throws(()=>Focus.create({...a,axes:["T","T"]}),/axis metadata/);
assert.throws(()=>Focus.create({...a,executionId:""}),/execution identity/);
assert.throws(()=>Focus.create({...a,evidence:tiers.STATIC,value:25}),/numeric coordinate/);
assert.throws(()=>Focus.unflatten([2,3],6),/flat index/);
const x=Focus.create({domain:"guided_graph",executionId:"graph/one",nodeId:"add",
 shape:[2,3],coordinate:[1,1],evidence:tiers.STATIC,note:"structural only"});
assert.equal(x.value,undefined);
assert.deepEqual(x.coordinate,[1,1]);
assert.throws(()=>Focus.compare(a,x),/same native execution/);
const video=Examples.run("video"),left=Focus.fromState(Bridge.coordinate("video",video,"a",22)),
 right=Focus.fromState(Bridge.coordinate("video",video,"b",22));
assert.equal(Focus.compare(left,right).delta,0);
assert.throws(()=>Focus.compare(a,left),/same native execution/);
const stream=Examples.run("stream"),leftStream=Focus.fromState(Bridge.coordinate("stream",stream,"a",5)),
 rightStream=Focus.fromState(Bridge.coordinate("stream",stream,"b",5));
assert.equal(Focus.compare(leftStream,rightStream).delta,-54);
const frame={title:"Real tiny teaching",origin:"deterministic",evidenceTier:tiers.GRAPH};
const node={id:"linear",out:{shape:[1,2,3],axes:["B","T","D"]}};
const mic=Focus.fromMicroscope(frame,node,[0,1,2],1.5);
assert.equal(mic.domain,"computation_microscope");
assert.equal(mic.flatIndex,5);
assert.equal(mic.value,1.5);
assert.throws(()=>Focus.fromMicroscope({...frame,evidenceTier:tiers.STATIC},node,[0,1,2],1.5),/numeric coordinate/);
const accepted={kind:"graph",numericOk:true,id:"free_graph",
 path:{steps:[{id:"sum",shape:[2,3]}]}};
const inspected={kind:"tensor_computation",replayVerified:true,stepId:"sum",
 coordinate:[1,2],evidence:tiers.GRAPH,detail:{value:23}};
const observed=Focus.fromObservation(accepted,inspected);
assert.equal(observed.domain,"unified_observation");
assert.equal(observed.value,23);
assert.equal(observed.flatIndex,5);
assert.throws(()=>Focus.fromObservation({...accepted,numericOk:false},inspected),/replayed native/);
assert.throws(()=>Focus.fromObservation(accepted,{...inspected,replayVerified:false}),/replayed native/);
assert.throws(()=>Focus.fromObservation(accepted,{...inspected,detail:{value:null}}),/missing native coordinate/);
const store=Focus.store();const events=[],unsubscribe=store.subscribe(v=>events.push(v));
assert.equal(store.current,null);
const chosen=store.select(a);
assert.equal(chosen.value,25);
assert.equal(store.current.domain,"semantic_state");
assert.equal(events.length,1);
store.select(x);assert.equal(events.length,2);assert.equal(store.current.value,undefined);
unsubscribe();store.clear();assert.equal(store.current,null);assert.equal(events.length,2);
console.log("COORDINATE_FOCUS_PASS native shape/value provenance, same-run A/B, numeric/structural boundary, microscope selection, subscribers");
