"use strict";
const assert=require("node:assert/strict");
const L=require("../learning_lab.js"),M=require("../compute_microscope.js"),
 Focus=require("../coordinate_focus.js"),Drill=require("../coordinate_drilldown.js"),
 E=require("../element_lineage.js"),V=require("../coordinate_vjp_bridge.js"),
 Journey=require("../coordinate_journey.js"),Canvas=require("../coordinate_journey_canvas.js"),
 Core=require("../learning_experience_core.js"),Examples=require("../semantic_state_examples.js"),
 Bridge=require("../state_learning_handoff.js");
const node=(id,kind,params={})=>({id,kind,params,x:20,y:30});
const wire=(from,to,port)=>({from,to,port});
const para=(values)=>({values,optimizer:{kind:"adamw",t:0,
 m:values.map(row=>row.map(()=>0)),v:values.map(row=>row.map(()=>0))}});
const g=L.validateGraph({schema:L.SCHEMA,title:"focus drilldown two-route VJP",nodes:[
 node("x","parameter",para([[.25,-.4],[.6,.7]])),
 node("w","parameter",para([[.32,.19],[-.22,.44]])),
 node("project","matmul"),node("boost","scale",{factor:.4}),node("sum","add"),node("out","square")
],edges:[wire("x","project","a"),wire("w","project","b"),wire("project","boost","x"),
 wire("project","sum","a"),wire("boost","sum","b"),wire("sum","out","x")]});
const fs=require("node:fs"),path=require("node:path");
const html=fs.readFileSync(path.join(__dirname,"..","learn.html"),"utf8");
const order=["graph_readiness.js","element_lineage.js","compute_microscope.js",
 "coordinate_vjp_bridge.js","coordinate_focus.js","learning_experience_core.js","coordinate_drilldown.js","coordinate_journey.js","coordinate_journey_ui.js","coordinate_journey_canvas.js"];
const positions=Object.fromEntries(order.map(name=>[name,html.indexOf('<script src="'+name+'"></script>')]));
for(const [name,pos] of Object.entries(positions))assert.ok(pos>=0,"missing runtime "+name);
for(const name of order.filter(n=>n!=="coordinate_drilldown.js"))
 assert.ok(positions[name]<positions["coordinate_drilldown.js"],"graph engine must load before drilldown: "+name);
const before=JSON.stringify(g),run=L.evaluate(g),frame=M.learning(g,run,L);
const chosen=frame.steps.find(s=>s.id==="out"),coord=[0,1],flat=1;
const numerical=Focus.fromMicroscope(frame,chosen,coord,chosen.out.values[flat]);
const verified=Drill.inspect(numerical,{graph:g,run,frame});
assert.equal(verified.scope,"native_numeric_graph");
assert.equal(verified.schema,Drill.SCHEMA);
assert.equal(verified.lineage.mode,"bounded-numeric");
assert.equal(verified.vjp.schema,V.SCHEMA);
assert.equal(verified.vjp.backward.complete,true);
assert.equal(verified.vjp.selected.value,numerical.value);
assert.deepEqual(verified.lineage.outputIndex,coord);
assert.deepEqual(verified.vjp.selected.index,coord);
const timeline=Journey.fromDrilldown(verified);
const scalarEdge=timeline.steps.find(step=>step.kind==="edge");
assert.ok(scalarEdge);
const mapped=Canvas.resolve(timeline,scalarEdge,g);
assert.equal(mapped.kind,"edge");
assert.deepEqual(mapped.coordinate,scalarEdge.index);
assert.deepEqual(mapped.downstreamCoordinate,scalarEdge.sourceIndex);
assert.ok(g.edges.some(e=>Canvas.wireEquals(e,mapped.wire)));
assert.equal(mapped.upstreamKey,scalarEdge.targetKey);
assert.equal(mapped.downstreamKey,scalarEdge.sourceKey);
const mismatched=structuredClone(g);
mismatched.edges=mismatched.edges.filter(e=>!Canvas.wireEquals(e,mapped.wire));
assert.throws(()=>Canvas.resolve(timeline,scalarEdge,mismatched),/absent/);
const seeded=Canvas.resolve(timeline,timeline.steps[0],g);
assert.equal(seeded.kind,"node");
assert.equal(seeded.nodeId,"out");
assert.equal(seeded.wire,null);

assert.equal(timeline.schema,Journey.SCHEMA);
assert.equal(timeline.scope,"native_numeric_graph");
assert.equal(timeline.complete,true);
assert.equal(timeline.steps.length,verified.lineage.vertices.length+verified.lineage.edges.length);
assert.equal(timeline.steps[0].kind,"seed");
assert.equal(timeline.steps[0].sensitivity,1);
const sourceSteps=timeline.steps.filter(s=>s.kind==="merge"&&s.nodeId==="x");
assert.ok(sourceSteps.length>0);
const actualSources=verified.lineage.vertices.filter(v=>v.status==="source");
for(const source of actualSources){
 const step=timeline.steps.find(s=>s.kind==="merge"&&s.key===source.key);
 assert.ok(step,"native source has a timeline merge "+source.key);
 assert.ok(Math.abs(step.sensitivity-source.sensitivity)<1e-9);
}
const fanout=timeline.steps.filter(s=>s.kind==="edge"&&s.targetKey==="project@0,1");
assert.ok(fanout.length>=2,"shared projected coordinate should receive two path contributions");
assert.ok(fanout.some(e=>e.firstArrival===false),"second incoming path is tagged as accumulation");
const staticTimeline=Journey.fromDrilldown(Drill.inspect(Focus.create({
 domain:"guided_graph",executionId:"graph/shape",nodeId:"out",shape:[2,2],
 coordinate:coord,evidence:Focus.TIERS.STATIC,note:"static lesson"}),{graph:g}));
assert.equal(staticTimeline.scope,"native_structural_graph");
assert.ok(staticTimeline.steps.some(x=>x.kind==="edge"&&x.localDerivative===null),
 "unknown static derivative remains unevaluated");
const trainingInput=Core.executeLab("training","coordinate_proof",{
 graph:require("./fixtures/coordinate_training_graph.js").create(),
 options:{targetId:"loss",lr:0.01,weightDecay:0.001},
 selection:{parameterId:"w",index:[0,1],viaNodeId:"project",viaIndex:[0,1]}
});
const trainingFocus=Focus.fromTraining(trainingInput,Core.replay(trainingInput));
const trainingSteps=Journey.fromDrilldown(Drill.inspect(trainingFocus)).steps;
assert.deepEqual(trainingSteps.map(x=>x.kind),["parameter_before","loss_gradient",
 "optimizer_state","parameter_after","loss_transition","selected_path"]);
assert.equal(trainingSteps.find(x=>x.kind==="loss_gradient").value,
 trainingInput.value.wholeLossGradient);
assert.equal(trainingSteps.find(x=>x.kind==="parameter_after").value,
 trainingInput.value.parameter.after);
assert.ok(Object.isFrozen(trainingSteps.find(x=>x.kind==="optimizer_state").moment1));
assert.equal(JSON.stringify(g),before);
const structural=Focus.create({domain:"guided_graph",executionId:"current/shape",nodeId:"out",
 shape:[2,2],coordinate:coord,evidence:Focus.TIERS.STATIC,note:"native static graph"});
const pure=Drill.inspect(structural,{graph:g,run,frame});
assert.equal(pure.scope,"native_structural_graph");
assert.equal(pure.lineage.mode,"structural");
assert.equal(pure.vjp,null);
assert.equal(pure.lineage.complete,true);
assert.throws(()=>Drill.inspect({...numerical,value:numerical.value+1},{graph:g,run,frame}),/disagree/);
assert.throws(()=>Drill.inspect(numerical,{graph:g}),/bounded numeric/);
const edited=structuredClone(g);
edited.nodes.find(n=>n.id==="boost").params.factor=2;
assert.throws(()=>Drill.inspect(numerical,{graph:edited,run,frame}),/frame differs|trace does not match|execution/);
assert.throws(()=>Drill.inspect({...structural,coordinate:[4,0]},{graph:g}),/outside tensor/);
assert.throws(()=>Drill.inspect(Focus.create({...structural,nodeId:"unknown"}),{graph:g}),/node not in/);
const state=Focus.fromState(Bridge.coordinate("video",Examples.run("video"),"b",22));
const paired=Drill.inspect(state,{graph:g,run,frame});
assert.equal(Journey.fromDrilldown(paired).steps[0].kind,"correspondence");
const correspondence=Journey.fromDrilldown(paired);
assert.equal(Canvas.resolve(correspondence,correspondence.steps[0],g).kind,"separate_evidence");
assert.equal(paired.scope,"mechanism_correspondence");
assert.equal(paired.lineage,null);
assert.equal(paired.vjp,null);
assert.match(paired.note,/独立数值执行/);
console.log("COORDINATE_DRILLDOWN_PASS: real graph numerical vs structural scope, VJP authentication, stale trace rejection, distinct scenario execution");
