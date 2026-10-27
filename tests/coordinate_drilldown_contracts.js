"use strict";
const assert=require("node:assert/strict");
const L=require("../learning_lab.js"),M=require("../compute_microscope.js"),
 Focus=require("../coordinate_focus.js"),Drill=require("../coordinate_drilldown.js"),
 E=require("../element_lineage.js"),V=require("../coordinate_vjp_bridge.js"),
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
 "coordinate_vjp_bridge.js","coordinate_focus.js","learning_experience_core.js","coordinate_drilldown.js"];
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
assert.equal(paired.scope,"mechanism_correspondence");
assert.equal(paired.lineage,null);
assert.equal(paired.vjp,null);
assert.match(paired.note,/独立数值执行/);
console.log("COORDINATE_DRILLDOWN_PASS: real graph numerical vs structural scope, VJP authentication, stale trace rejection, distinct scenario execution");
